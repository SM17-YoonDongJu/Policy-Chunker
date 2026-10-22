# insurance-chunker

보험 약관 PDF → PostgreSQL(pgvector) RAG 파이프라인.

PDF를 받아 텍스트·표를 추출하고, 조항 단위로 청킹한 뒤 임베딩 벡터와 함께 DB에 저장한다.
저장된 청크는 BM25 키워드 검색 + 벡터 유사도 하이브리드 RAG의 검색 대상이 된다.

---

## 전체 흐름

```
약관.pdf
  │
  ├─ pdf_parser.py    텍스트 추출 (PyMuPDF, 표 영역 제외)
  │
  ├─ extractor.py     표 추출 — 아래 4가지 방법 중 자동 선택
  │     ├ PyMuPDF     괘선 표 (빠름)
  │     ├ pdfplumber  설치 시 자동 사용
  │     ├ camelot     설치 시 자동 사용 (ghostscript 필요)
  │     └ VLM         PyMuPDF가 못 닿은 페이지 (VLM_TARGET=gap)
  │                   local=OpenAI 호환 — 기본은 Ollama qwen3-vl
  │
  ├─ combine.py       페이지별 best-of 표 선택 (더블스페이스 기준)
  │
  ├─ boundaries.py    폰트 크기로 약관·별표 경계 감지
  │     └ assess()    신뢰도 판정 (ok / weak) — 조용히 틀린 청크 방지
  │
  ├─ rechunk.py       경계 라벨 부여 → 조항 병합 → 중복 제거 → 대형 표 분할
  │
  ├─ chunker.py       오케스트레이터 → InsuranceChunk + TableMeta 출력
  │
  ├─ embedder.py      임베딩 (Ollama qwen3-embedding:0.6b / BGE-M3)
  │
  ├─ db/storage.py    pgvector upsert (policy_chunks 테이블)
  │
  └─ S3               대형 표 원본 markdown 저장
                      (키: policy-tables/{table_id}.md)
```

---

## 빠른 시작

### 설치

```bash
pip install -e ".[dev]"
```

VLM 표 추출은 기본으로 켜져 있고, OpenAI 호환 `/v1/chat/completions`를 쓰는 서버에 붙는다
(`VLM_URL`). 로컬 개발은 Ollama(`qwen3-vl:8b-instruct`), 운영은 클러스터의 vLLM이다.
이미지에 추가 의존이 없다.

| 백엔드 | 준비물 |
|---|---|
| `local` (기본) | OpenAI 호환 서버 — Ollama에 VLM 모델, 또는 llama-server |
| `surya` | `pip install ".[ocr]"` |
| `off` | 없음 (VLM 단계를 건너뛴다) |

```bash
# 로컬에서 Ollama 없이 청킹만 볼 때
VLM_BACKEND=off python ingest.py --pdf 약관.pdf --insurer ... --product ...
```

### 단일 PDF — DB 저장

```bash
DATABASE_URL=postgresql://user:pass@host/db \
python ingest.py \
  --pdf 약관.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --effective-date 2025-06-01
```

같은 파일을 다시 ingestion하면 자동으로 건너뛴다. 덮어쓰려면 `--overwrite`.

### 여러 PDF 일괄 처리 (YAML manifest)

```bash
python ingest_many.py --manifest docs.yaml
```

`docs.yaml` 예시:

```yaml
documents:
  - pdf: 약관A.pdf
    insurer: 메리츠화재
    product_name: 단체안심생활보험
    effective_date: "2025-06-01"
    generation: "4세대"

  - pdf: 약관B.pdf
    insurer: KB손해보험
    product_name: 실손보험
    doc_type: policy_terms   # 없으면 파일명으로 자동 판별
```

---

## dry-run — DB 없이 청킹 결과만 확인

`--dry-run` 플래그를 붙이면 DB·S3·Ollama 없이 청킹 결과만 JSON으로 출력한다.  
청킹이 제대로 됐는지 확인하거나, 로컬 환경에서 빠르게 테스트할 때 사용한다.

```bash
# 결과를 파일로 저장
python ingest.py \
  --pdf 약관.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --no-embed --dry-run --dry-run-out out.json

# VLM(Ollama)도 없는 환경
python ingest.py \
  --pdf 약관.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --no-embed --no-vision --dry-run
```

`ingest_many.py`도 동일하게 `--dry-run` 플래그를 지원한다.

```bash
python ingest_many.py --manifest docs.yaml --dry-run --dry-run-dir out/
```

dry-run 결과 JSON 구조:

```jsonc
{
  "summary": {
    "chunk_count": 312,
    "chunk_type_counts": { "coverage": 88, "exclusion": 42, … },
    "token_stats": { "min": 24, "max": 980, "avg": 410 },
    "over_600": 15,      // 600 토큰 초과 청크 수
    "warnings": []
  },
  "chunks": [ … ]        // InsuranceChunk 목록
}
```

---

## doc_type 자동 판별 규칙

`--doc-type`을 지정하지 않으면 파일명 키워드로 자동 분류한다.

| 파일명 패턴 | doc_type |
|---|---|
| 요약서, 상품안내, 상품설명서 | `product_summary` |
| 약관, 보통약관, 특별약관 | `policy_terms` |
| 산출기준표, 지급률표, 장해분류표 | `schedule` |

`product_summary`는 ingest 대상이 아니며, 파이프라인 진입 전 차단된다.

---

## InsuranceChunk 스키마

DB에 저장되는 청크 단위. `insurance_chunker/models.py`에 dataclass로 정의된다.

```jsonc
{
  // ── 식별 ─────────────────────────────────────────────────────────────
  "chunk_id":       "6e533f8dc4204f3f…",   // SHA256[:24] — doc_hash + source_pdf + page + idx 조합
  "doc_hash":       "58c3c307…",           // PDF SHA256 — 중복 ingest 방지·삭제 기준
  "source_pdf":     "약관.pdf",            // 원본 파일명 (DB 저장 안 됨, 내부 키 생성에만 사용)

  // ── 상품 메타 ─────────────────────────────────────────────────────────
  "insurer":        "메리츠화재",
  "product_name":   "단체안심생활보험",
  "product_code":   "ABC-123",             // 선택
  "effective_date": "2025-06-01",          // 선택
  "generation":     "4세대",               // 선택

  // ── 문서 구조 ─────────────────────────────────────────────────────────
  "doc_type":       "policy_terms",        // policy_terms | schedule (내부용, DB 저장 안 됨)
  "section":        "암 진단특별약관",     // 폰트 기반 경계 라벨
  "page_number":    42,
  "chunk_index":    17,                    // 문서 내 전체 순서 — 조항 복원 시 ORDER BY에 사용

  // ── 조항 메타 ─────────────────────────────────────────────────────────
  // article_number가 null인 청크가 발생할 수 있다 (아래 한계 참고)
  "article_number": "제2조",
  "article_title":  "보험금의 지급",

  // ── 분류 ─────────────────────────────────────────────────────────────
  "chunk_type":     "coverage",            // 8종, 아래 표 참고
  "token_count":    412,                   // 글자 수 × 0.6 (Kiwi 형태소 토큰 근사값)

  // ── 검색 ─────────────────────────────────────────────────────────────
  "content":        "메리츠화재 | 단체안심생활보험 | 암 진단특별약관\n[제2조 보험금의 지급]\n① …",
  "content_tokens": "보험금 지급 암 진단 …",   // Kiwi 형태소 분리 결과 — BM25 색인용
  "embedding":      [0.021, …],               // 1024d 벡터

  // ── 표 row 청크 전용 (텍스트 청크는 null) ────────────────────────────
  "table_id":  "550e8400-e29b-…",   // UUID — S3 키: policy-tables/{table_id}.md
  "row_start": 1,                   // 이 청크가 표의 몇 번째 행부터 시작하는지
  "row_end":   20                   // 이 청크가 표의 몇 번째 행에서 끝나는지
}
```

### chunk_type 8종

regex 기반 자동 분류다. 오분류 가능성이 있으므로 MVP 단계에서는 필터 조건으로 사용하지 않는 것을 권장한다.

| 값 | 분류 기준 |
|---|---|
| `coverage` | 보장 범위, 지급사유, "보험금" |
| `exclusion` | 면책, 부지급, "지급하지 않" |
| `duty` | 알릴 의무, 고지의무, 통지 의무 |
| `claim` | 보험금 청구, 청구 절차 |
| `termination` | 해지, 효력상실, 소멸 |
| `special_clause` | 특약, 특별 |
| `definition` | 정의, 용어 |
| `schedule` | 별표, 장해분류표, 지급률표 (doc_type=schedule에서 주로 출현) |
| `general` | 위 분류 외 일반 조항 |

### article_number 구조적 한계

`article_number`는 `제N조(제목)` 패턴을 폰트·텍스트 기반으로 탐지해 채운다.  
아래 경우에는 탐지가 안 되어 `null`이 된다.

| 상황 | 이유 |
|---|---|
| 각 섹션(특약)의 첫 번째 `제1조` 이전 도입 문장 | 섹션 경계 마다 `cur_art`가 리셋됨 |
| 약관별로 고유한 비표준 조항 표기 | 패턴이 커버 못 하는 형식 존재 가능 |

`article_number=null` 청크는 RAG 검색에서 완전히 배제되지 않는다.  
조항 단위 복원이 필요한 경우 `section + chunk_index`로 ORDER BY 하면 된다.

---

## 대형 표 처리 (TableMeta + S3)

장해분류표처럼 수백 행짜리 표는 20행 단위로 분할해 각각 `InsuranceChunk`로 저장한다.

- 표 원본 전체 markdown은 S3(`policy-tables/{table_id}.md`)에 업로드
- 각 row 청크는 `table_id`(UUID)로 같은 표임을 식별
- DB에 FK 없음 — `table_id`는 S3 참조용 단순 UUID 컬럼

`S3_BUCKET` 환경변수가 없으면 `.table_cache/` 폴더에 로컬 저장한다.

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL 연결 문자열 (필수, dry-run 제외) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 서버 주소 |
| `EMBED_MODEL` | `qwen3-embedding:0.6b` | Ollama 임베딩 모델 |
| `EMBED_BATCH_SIZE` | `32` | 배치 크기 (키우면 `EMBED_BATCH_TIMEOUT`도 함께) |
| `EMBED_BATCH_TIMEOUT` | `120` | 배치 요청 타임아웃(초). 초과 시 건별 폴백 |
| `EMBED_MAX_CHARS` | `1800` | 장문 절단 상한 — **eval과 같은 값이어야 수치가 재현된다** |
| `EMBED_BACKEND` | `ollama` | `ollama` \| `sentence_transformers` (BGE-M3 전환) |
| `S3_BUCKET` | — | 대형 표 markdown 저장용 S3 버킷 (없으면 `.table_cache/` 로컬 저장) |
| `VLM_BACKEND` | `local` | `local` \| `surya` \| `off` |
| `VLM_URL` | `http://localhost:11434` | `local` 백엔드 서버. 운영은 차트가 `http://vllm-qwen3vl:8000` |
| `VLM_MODEL` | `qwen3-vl:8b-instruct` | Ollama는 필수. llama-server면 비운다. Paddle 전환은 `PaddleOCR-VL-1.6-0.9B` |
| `VLM_PROMPT` | 모델 따라 자동 | 모델명에 `paddleocr`가 있으면 `Table Recognition:`(태스크 토큰), 아니면 한국어 표 변환 지시. 명시하면 그 값이 이긴다 |
| `VLM_LOOP_CELLS` | `20` | 같은 셀이 이만큼 연속되면 반복 루프 출력으로 보고 폐기 |
| `VLM_DPI` | `150` | VLM 페이지 렌더링 해상도 |
| `VLM_TIMEOUT` | `300` | VLM **읽기** 타임아웃(초). GPU 경로 실측 21~36초 |
| `VLM_CONNECT_TIMEOUT` | `5` | 접속 타임아웃(초). 서버가 없으면 즉시 실패해야 한다 |
| `VLM_FAIL_STREAK` | `3` | 연속 실패 N회면 그 문서의 남은 페이지는 VLM 건너뜀 |
| `VLM_TARGET` | `gap` | VLM 투입 대상. `gap`=PyMuPDF가 못 닿은 페이지 \| `tables`=표 탐지 페이지(구동작) \| `both` |
| `VISION_MAX_PAGES` | `50` | 문서당 VLM 호출 상한. 넘는 페이지는 PyMuPDF/pdfplumber 결과만 쓴다 |
| `INGEST_STATE_DIR` | `/data/state` | 실행 이력 디렉터리 (쓰기 불가 시 `./.state` 폴백) |
| `INGEST_MAX_RETRY` | `3` | 0청크·오류 연속 N회면 문서를 격리 |
| `INGEST_CONCURRENCY` | `1` | 동시 처리 문서 수 (프로세스). 노드 vCPU 4 기준 2가 현실적 |
| `HEALTH_GRACE_FACTOR` | `1.5` | 헬스체크 허용 배수 (마지막 성공 기준) |
| `DISCORD_WEBHOOK_INGEST` | — | 사이클 결과 알림 웹훅 (없으면 알림만 생략) |
| `INGEST_NOTIFY` | `always` | `always` \| `failure` |
| `LOG_FORMAT` | `text` | `text` \| `json` (운영은 차트가 `json`) |
| `LOG_LEVEL` | `INFO` | 루트 로거 레벨 |
| `METRICS_PORT` | `9101` | Prometheus `/metrics` 포트 (`0`이면 노출 안 함) |

---

## 운영 계측

인덱싱은 7일 주기로 도는 상주 데몬이라(`worker.py`) 실패해도 눈에 잘 안 띈다. 그래서 매 실행을
파일로 남기고, 그걸 근거로 건강을 판정하고 지표를 뽑는다.

### 무엇이 남나

`INGEST_STATE_DIR`(기본 `/data/state`, `chunker-state` PVC라 파드를 다시 만들어도 유지):

| 파일 | 내용 |
|---|---|
| `items.jsonl` | 문서 1건 처리 = 1줄. 상태·청크 수·**단계별 소요 시간**(hash/parse/chunk/embed/store)·최고 상주 메모리(`rss_peak_mb`, 단계별은 `stages.*.rss_mb`) |
| `runs.jsonl` | 적재 CLI 1회 실행 = 1줄. 사이클 집계 |
| `attempts.json` | sha256 → 최근 시도 요약. 격리 판정에 쓴다. 처리를 **시작할 때** `STARTED`로 먼저 적으므로 OOM처럼 결과 기록 없이 죽은 시도도 센다 |
| `daemon.json` | 기동 시각 · 마지막 사이클 성패 · **마지막 성공 시각** |

DB 테이블이 아니라 파일인 이유: `corpus.*`의 DDL 진실원은 AI 레포 `migrations/corpus` 하나다.
운영 이력은 우리 쪽 데이터라 그 계약을 건드리지 않는 곳에 둔다.

### 지표 뽑기

```bash
python metrics.py             # 사람이 읽는 형태
python metrics.py --last 5    # 최근 5개 사이클
python metrics.py --json      # 대시보드·알림에 물릴 때
```

```
문서 5건 — {'OK': 1, 'EMPTY': 3, 'ERROR': 1}
  시도 대비 성공률   20.0%
  멱등 스킵률        40.0%  (doc_hash 중복 제거로 다운로드·파싱을 아예 안 한 비율)

문서당 처리 시간(n=1)  p50 41.3s  p95 41.3s  max 41.3s

단계별 비중 — 여기가 병목 후보다
  embed   55.2% ██████████████████████ (합 22.8s, p50 22.8s)
  parse   30.0% ████████████ (합 12.4s, p50 12.4s)
```

### Prometheus /metrics

데몬이 상주하므로 pull 모델이 그대로 성립한다(Pushgateway·textfile collector 불필요).
스크랩할 때마다 `runlog`를 읽어 렌더링하므로 앱이 상태를 들고 있지 않고, 적재 CLI가
subprocess로 돌아도(=프로세스가 죽어도) 지표가 정확하다.

```
insurance_chunker_last_success_timestamp_seconds   ← 신선도 SLI
insurance_chunker_last_cycle_success               1/0
insurance_chunker_last_cycle_duration_seconds
insurance_chunker_last_cycle_documents{status="ok|empty|skipped|quarantined|error"}
insurance_chunker_last_cycle_chunks_indexed
insurance_chunker_quarantined_documents
insurance_chunker_documents_processed_total{status}   ← 누적
insurance_chunker_chunks_indexed_total
insurance_chunker_phase_duration_seconds_total{phase="parse|chunk|embed|store"}
```

주기가 7일이라 대부분의 시간 동안 카운터는 안 움직인다. **`rate()`가 아니라 "마지막 실행
상태" 게이지가 주 신호다.** 활동량 기반 알림(예: "5분간 처리 0건")을 걸면 상시 발화한다.

히스토그램은 두지 않는다 — 주당 문서 수백 건이면 p95를 낼 표본이 안 된다. 분포는
`metrics.py`와 Loki가 맡는다.

수집은 메트릭 전용 Service(`insurance-chunker-metrics:9101`)를 ServiceMonitor가 60초마다
긁는 경로다. probe가 없는 워크로드라 **이게 좀비 판정의 유일한 경로**다 — 주기가 7일이어서
이것 없이는 인덱싱이 멈춰도 몇 주 동안 아무도 모른다.

```bash
# 파드에서 바로
kubectl exec deploy/insurance-chunker -- \
  python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:9101/metrics').read().decode())" \
  | grep insurance_chunker

# 로컬 docker 실행이면
curl -s localhost:9101/metrics | grep insurance_chunker
```

### 로그

운영에서는 JSON 한 줄 = 로그 하나다(`LOG_FORMAT=json`, 차트가 지정). 클러스터의 Alloy가
`/var/log/pods`에서 걷어 파싱하고 `level`을 Loki 라벨로 올린다. 로컬 기본값은 `text`라
사람이 읽기 좋다.

```bash
kubectl logs -f deploy/insurance-chunker
```

문서·사이클 결과에는 집계용 필드가 실린다 — `/metrics` 없이 로그만으로도 알림을 걸 수 있다.

```json
{"timestamp":"...","level":"INFO","service":"insurance-chunker","logger":"ingest_catalog",
 "message":"문서 처리 완료","event":"document_done","status":"OK","document":"약관.pdf",
 "chunks":687,"elapsed_s":41.3,"phases":{"parse":12.4,"embed":22.8,"store":2.8}}
```

라벨 카디널리티 정책상 `document`·`sha256` 같은 값은 **본문에만** 둔다(라벨로 올리지 않는다).

### 헬스체크

판정 기준은 **마지막 인덱싱 성공 시각** — 프로세스는 살아 있는데 매 주기 실패하는 좀비를
프로세스 생존만으로는 못 잡기 때문이다. 주기 × `HEALTH_GRACE_FACTOR`를 넘기면 불건강.

**클러스터에서는 이걸 probe로 쓰지 않는다.** 배치 워커라 대기 구간이 길어서 liveness를 걸면
적재 중에 재시작되고, k8s는 이미지의 `HEALTHCHECK`도 읽지 않는다. 그래서 차트는 probe를
두지 않고, 같은 판정을 Prometheus 쪽에서 한다 —
`time() - insurance_chunker_last_success_timestamp_seconds`가 주기 × GRACE를 넘으면 알림.
수집 경로는 메트릭 전용 Service와 ServiceMonitor다(아래 참고).

손으로 볼 때는 이렇게 부른다.

```bash
kubectl exec deploy/insurance-chunker -- python /app/healthcheck.py
```

로컬 `docker run`에서는 이미지의 `HEALTHCHECK`가 같은 스크립트를 돌려 `docker ps`의 STATUS에
띄운다. 어느 쪽이든 자가치유가 아니라 드러내기 위한 신호다.

### 배포 구성

`soma-k8s` 클러스터의 GPU 워커 노드 `brbs-etl`(g4dn.xlarge / T4 16GB / vCPU 4 / RAM 16GiB)에
파드 넷이 함께 뜬다. 노드는 `workload=etl` 라벨과 `workload=etl:NoSchedule` taint로 묶여 있어
ETL 워크로드만 들어온다. 모두 `default` 네임스페이스다.

| 파드 | 레포 | 역할 | GPU |
|---|---|---|---|
| `insurance-chunker` | 이 레포 | S3 → pgvector 인덱싱 (7일 주기 데몬) | 없음 — HTTP로 아래 둘을 쓴다 |
| `ollama` | — | 임베딩(`qwen3-embedding:0.6b`) | `nvidia.com/gpu: 1` |
| `vllm-qwen3vl` | — | 표 추출 VLM(`qwen3-vl:8b-instruct`) | `nvidia.com/gpu: 1` |
| `corpus-worker` | SM17-YoonDongJu/AI | Notion → S3 약관 스테이징 | 없음 |

물리 GPU는 한 장이고 device plugin의 time-slicing으로 2개로 광고해 ollama와 vLLM이 나눠 쓴다.
우리 파드는 GPU를 직접 잡지 않으므로 이미지에 CUDA 의존이 없다.

호출은 `localhost`가 아니라 **Service 이름**으로 한다 — 파드마다 네트워크 namespace가 따로라
`localhost`는 자기 파드를 가리킨다. 차트가 주소를 환경변수로 넣는다.

```
OLLAMA_URL = http://ollama:11434            임베딩
VLM_URL    = http://vllm-qwen3vl:8000       표 추출
```

적재 진행 상태(`/data/state`)는 `chunker-state` PVC(local-path, 1Gi)에 있다. 증분 처리의
근거라 유실되면 전체 재적재가 되므로, 이 앱의 Argo CD Application은 `prune`을 켜지 않는다.

vCPU가 4개이고 파드들이 나눠 쓰므로 `INGEST_CONCURRENCY` 상한이 낮다(아래 참고).

노드 전제·선행 리소스·인수 절차는 K8s-manifests의 `envs/dev/etl.md`에 있다.

### 배포와 롤백

GitOps다. **이 레포의 CI는 클러스터에 손을 대지 않는다** — 이미지를 올리고, 배포 설정
레포의 이미지 태그를 고쳐 커밋하는 것까지가 전부다. 적용은 클러스터 안의 Argo CD가 한다.

```
main 머지
  └─ deploy.yml
       ├─ ci.yml 재사용 (ruff · import 스모크 · pytest)
       ├─ ECR push  soma/insurance_chunker:<sha7>  (+ :latest)
       └─ K8s-manifests의 envs/dev/etl-chunker.values.yaml 의
          image.tag 를 <sha7> 로 커밋                 ← 여기까지가 CI
                │
                ▼
          Argo CD (etl-chunker-dev) 가 3분마다 확인 → 클러스터에 적용
```

태그는 **커밋 해시(sha7)**다. `:latest`처럼 떠다니는 태그를 쓰면 이미지를 덮어써도
매니페스트 문자열이 그대로여서 Argo CD가 아무 일도 하지 않는다. 태그가 바뀌는 커밋 자체가
배포 신호다.

K8s-manifests에 커밋할 때는 `brbs-manifests-ci` GitHub App 토큰을 쓴다 — 그 레포 하나의
contents 쓰기로만 범위가 좁혀져 있고, CI는 AWS·클러스터 자격증명을 들지 않는다.

`etl-chunker-dev` Application은 자동 Sync다(`selfHeal: true`, `prune: false`). `prune`을 끈
것은 `chunker-state` PVC 때문이다 — 파일이 git에서 잠깐 빠지는 실수로 날아가면 증분 처리의
근거를 잃고 전체 재적재가 된다.

`selfHeal`이 켜져 있으니 **`kubectl`로 파드나 설정을 직접 고치면 몇 분 안에 되돌아간다.**
급해도 K8s-manifests에 PR을 올리는 게 제일 빠른 길이다.

**롤백**은 K8s-manifests에서 태그 커밋을 revert하는 게 가장 빠르다 — 빌드도 CI도 거치지 않고
Argo CD가 바로 되돌린다. 배포 이력이 곧 그 레포의 git 로그다.

Actions에서 `Deploy`를 `workflow_dispatch`로 돌리고 `image_tag`에 옛 `sha7`을 넣는 길도
남아 있다. 이때는 빌드와 CI를 건너뛰고 그 태그를 커밋만 한다 — `main`이 깨져서 롤백하는
상황인데 CI가 막으면 복구를 못 하기 때문이다.

Discord 알림(`notify` 잡)은 **"태그를 커밋했다"까지**를 뜻한다. 실제로 파드가 떴는지는
Argo CD만 아니까, 초록불을 배포 완료로 읽지 말 것.

#### 시크릿

`DATABASE_URL`만 비밀값이다. AWS Secrets Manager의 `app/insurance-chunker-dev`에 두고,
External Secrets Operator가 1시간마다 읽어 `insurance-chunker-secrets` Secret으로 만든다.
차트는 그 Secret을 `secretKeyRef`로 참조할 뿐이라 **레포에는 "어떤 키를 가져올지"만 있고 값이
없다**(K8s-manifests의 `envs/dev/manifests/externalsecret-insurance-chunker-secrets.yaml`).

```bash
aws secretsmanager put-secret-value --secret-id app/insurance-chunker-dev \
  --secret-string '{"DATABASE_URL":"postgresql://..."}'
```

값을 바꿔도 **파드는 자동으로 재시작되지 않는다.** ESO가 Secret을 갱신해도 이미 뜬 파드는
기동 시점에 읽은 환경변수를 계속 쓴다. 교체했으면 `kubectl rollout restart
deploy/insurance-chunker`까지 해야 한다.

`S3_BUCKET`·`AWS_REGION`은 비밀이 아니라 차트의 평문 환경변수다. S3·Secrets Manager 접근은
노드 IAM 역할로 한다.

`DISCORD_WEBHOOK_INGEST`는 **현재 dev 클러스터에 설정돼 있지 않다** — 사이클 결과 알림이
조용히 생략된다(`notify.py`는 웹훅이 없으면 건너뛴다). 켜려면 Secrets Manager에 값을 넣고
차트에 환경변수를 추가해야 한다.

### 문서 동시 처리

`INGEST_CONCURRENCY`(기본 1 = 순차)로 문서를 동시에 처리한다. 실측 기준 `parse`가 전체
시간의 **47.7%**이고 CPU 바운드(PyMuPDF·pdfplumber)라, 문서를 겹치면 A가 임베딩을
기다리는 동안 B를 파싱할 수 있다. 스레드가 아니라 **프로세스**여야 하는 이유가 GIL이다.

```bash
INGEST_CONCURRENCY=2 python ingest_catalog.py
```

구조는 2단계다.

| 단계 | 어디서 | 왜 |
|---|---|---|
| 스킵·격리 판정, 이력 기록 | **부모(순차)** | `attempts.json`이 읽고-고쳐-쓰기라 여러 프로세스가 만지면 서로 덮어쓴다 |
| 다운로드 · 파싱 · 임베딩 · 저장 | 워커(동시) | 공유 상태를 안 건드린다 |

`spawn`을 명시한다 — `fork`면 부모의 psycopg2 커넥션과 boto3 상태를 물려받는데 둘 다
fork-safe하지 않다.

**동시성 상한은 노드가 정한다.** vCPU가 4개이고 `corpus-worker`·ollama·vLLM 파드와 나눠 쓰므로
2가 현실적이고 3부터는 실측이 필요하다. `embed`는 GPU가 큐잉하므로 여기를 올린다고
비례해서 빨라지지 않는다 — T4 실측에서 임베딩 부하가 이미 70W 캡을 넘겨 부스트 클럭이
1590→1200MHz로 깎인다.

### 경계 검출 신뢰도

`boundaries.assess()`가 문서마다 `ok` / `weak`을 판정한다. `weak`이면 **특약 경계를 못 잡았다**는
뜻이고, 그러면 이후 조번호가 전부 어긋난다(`eval/IMPROVEMENT_LOG.md` C-1).

문제는 **그래도 적재는 성공한다**는 점이다 — 경계가 없으면 단순 텍스트 청킹으로 폴백하므로
청크는 나오고 `status=OK`로 집계된다. 로그로만 흘려보내면 품질이 무너진 문서가 성공으로
잡힌다. 그래서 신뢰도를 상태와 **따로** 들고 간다.

```
items.jsonl                                boundary_confidence: "ok" | "weak" | "error"
/metrics   insurance_chunker_weak_boundary_documents
로그        event=boundary_weak (사유 포함)
```

재시도 카운터는 건드리지 않는다 — 품질 저하지 실패가 아니고, 격리하면 그 보험사 문서가
영영 안 들어온다. 특정 보험사에서 반복되면 경계 검출 로직을 봐야 한다는 신호다.

### 인덱스 SLO 점검

"적재됐다"와 "검색에 쓸 만하다"는 다르다. 임베딩 차원이 어긋나거나 문서 하나에 임베딩이
통째로 없어도 적재 자체는 성공으로 끝난다. `slo.py`가 사이클 끝마다 DB 상태를 본다.

| 점검 | 위반 시 |
|---|---|
| `embedding_dim` | `EMBED_DIM`과 다르면 **fail** — 모델 불일치, 재적재 필요 |
| `search_contract` | `content_tsv` 없으면 **fail** — AI 레포 RAG 검색이 0건이 된다 |
| `documents_embedded` | 임베딩이 0개인 문서가 있으면 **fail** — 벡터 검색에서 통째로 빠진다 |
| `freshness` | `max(ingested_at)`이 오래되면 **warn** |
| `catalog_coverage` | 카탈로그 대비 적재율이 하한 미만이면 **warn** — 격리·0청크 확인 |

위반이 있으면 사이클 알림에 실려 나간다. 단 **사이클 자체를 실패로 만들지는 않는다** —
적재는 이미 됐고, 여기서 실패로 처리하면 healthcheck까지 같이 울기 때문이다.

전체 NULL 비율을 안 쓰는 이유가 있다. boilerplate 청크는 일부러 임베딩을 건너뛰는데
(`embedder.embed_chunks`) `is_boilerplate`가 DB 컬럼이 아니라 **의도된 NULL과 실패한 NULL을
구분할 수 없다.** 대신 "한 문서에 임베딩 0개"라는 명백한 신호를 본다.

```bash
kubectl exec deploy/insurance-chunker -- python /app/slo.py
```

큰 변경 전 사람이 한 번 돌리는 `deploy_check.py`와는 역할이 다르다 — 그쪽은 asyncpg를
쓰는 배포 전 판정용이고, 이건 psycopg2로 컨테이너 안에서 상시 돈다.

### 실패 문서 격리

0청크 문서는 `policy_chunks`에 행이 안 생겨 `doc_already_ingested`가 영원히 `False`다 —
상한이 없으면 매 주기 S3에서 다시 받아 다시 파싱한다. `INGEST_MAX_RETRY`회 연속 실패하면
격리해 다운로드조차 하지 않는다. 성공 한 번이면 카운터가 0으로 돌아간다.

시도는 문서 처리를 **시작하기 전에** `STARTED`로 먼저 센다. OOM이나 강제 종료로 프로세스가
통째로 죽으면 결과 기록이 안 남아 다음 기동이 같은 문서로 돌아와 같은 자리에서 또 죽는데
(434쪽 문서 하나로 닷새 동안 111회 재시작), 선기록이 있으면 그 시도도 상한에 걸려 격리된다.
`attempts.json`에 `STARTED`가 남아 있는 문서가 그렇게 끝을 못 본 문서이고, 사이클 시작 로그에
`attempt_unfinished`로 찍힌다. 배포 재시작은 시작한 문서를 끝까지 보내므로 보통 여기 안 걸린다.

```bash
# 격리 목록
cat /data/state/attempts.json

# 격리 해제하고 다시 시도
python ingest_catalog.py --retry-quarantined
```

---

## 주요 파일 구조

```
Policy-Chunker/
├── ingest.py               단일 PDF CLI 진입점
├── ingest_many.py          YAML manifest 기반 일괄 처리
├── ingest_catalog.py       ai.corpus_file 카탈로그 + S3 기반 인덱싱
├── rebuild_search_terms.py BM25 쿼리 보정용 search_terms 테이블 재구축
│
├── worker.py               상시 인덱싱 데몬 (컨테이너 CMD)
├── runlog.py               실행 이력 기록·조회 (items/runs/attempts/daemon)
├── metrics.py              이력 → 운영 지표 (처리시간 분포·단계 비중·성공률)
├── healthcheck.py          마지막 성공 시각 기반 좀비 판정
├── notify.py               사이클 결과 Discord 알림
├── logging_setup.py        평문/JSON 로깅 설정
├── exporter.py             Prometheus /metrics (runlog를 읽는 커스텀 컬렉터)
├── slo.py                  인덱스 SLO 점검 (사이클 끝마다)
│
├── insurance_chunker/
│   ├── models.py           InsuranceChunk, TableMeta, DocMeta dataclass 정의
│   ├── chunker.py          doc_type별 오케스트레이터
│   ├── rechunk.py          병합·중복제거·표분할 (clean → merge → finalize)
│   ├── boundaries.py       폰트 기반 약관 경계 감지
│   ├── pdf_parser.py       PDF 텍스트 추출
│   ├── extractor.py        표 추출 (PyMuPDF/pdfplumber/camelot/VLM)
│   ├── combine.py          페이지별 best-of 표 선택
│   ├── embedder.py         Ollama / BGE-M3 임베딩
│   ├── tokenizer.py        Kiwi 형태소 분리
│   └── validator.py        청크 품질 검증
│
└── db/
    ├── schema.sql          policy_chunks + search_terms DDL + 마이그레이션 블록
    ├── storage.py          upsert_chunks, verify_upsert 등 DB 함수
    └── search_term.py      search_terms 재구축 로직
```
