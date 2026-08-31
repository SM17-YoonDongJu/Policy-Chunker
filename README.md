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
  │     └ VLM         PyMuPDF가 표를 감지한 페이지만 (VLM_BACKEND)
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

VLM 표 추출은 기본으로 켜져 있고, 같은 호스트의 **Ollama**(`qwen3-vl:8b-instruct`)에
OpenAI 호환 `/v1/chat/completions`로 붙는다. 이미지에 추가 의존이 없다.

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
| `VLM_URL` | `http://localhost:11434` | `local` 백엔드 서버 (같은 호스트 Ollama) |
| `VLM_MODEL` | `qwen3-vl:8b-instruct` | Ollama는 필수. llama-server면 비운다 |
| `VLM_PROMPT` | (한국어 표 변환 지시) | PaddleOCR-VL이면 `Table Recognition:` 로 |
| `VLM_DPI` | `150` | VLM 페이지 렌더링 해상도 |
| `VLM_TIMEOUT` | `600` | VLM 호출 타임아웃(초) |
| `VISION_MAX_PAGES` | `9999` | VLM 호출 상한 페이지 수 |
| `INGEST_STATE_DIR` | `/data/state` | 실행 이력 디렉터리 (쓰기 불가 시 `./.state` 폴백) |
| `INGEST_MAX_RETRY` | `3` | 0청크·오류 연속 N회면 문서를 격리 |
| `INGEST_CONCURRENCY` | `1` | 동시 처리 문서 수 (프로세스). 호스트 vCPU 4 기준 2가 현실적 |
| `HEALTH_GRACE_FACTOR` | `1.5` | 헬스체크 허용 배수 (마지막 성공 기준) |
| `DISCORD_WEBHOOK_INGEST` | — | 사이클 결과 알림 웹훅 (없으면 알림만 생략) |
| `INGEST_NOTIFY` | `always` | `always` \| `failure` |
| `LOG_FORMAT` | `text` | `text` \| `json` (운영은 compose가 `json`) |
| `LOG_LEVEL` | `INFO` | 루트 로거 레벨 |
| `METRICS_PORT` | `9101` | Prometheus `/metrics` 포트 (`0`이면 노출 안 함) |

---

## 운영 계측

인덱싱은 7일 주기로 도는 상주 데몬이라(`worker.py`) 실패해도 눈에 잘 안 띈다. 그래서 매 실행을
파일로 남기고, 그걸 근거로 건강을 판정하고 지표를 뽑는다.

### 무엇이 남나

`INGEST_STATE_DIR`(기본 `/data/state`, 호스트 볼륨이라 컨테이너를 다시 만들어도 유지):

| 파일 | 내용 |
|---|---|
| `items.jsonl` | 문서 1건 처리 = 1줄. 상태·청크 수·**단계별 소요 시간**(hash/parse/chunk/embed/store) |
| `runs.jsonl` | 적재 CLI 1회 실행 = 1줄. 사이클 집계 |
| `attempts.json` | sha256 → 최근 시도 요약. 격리 판정에 쓴다 |
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

```bash
curl -s localhost:9101/metrics | grep insurance_chunker
```

### 로그

운영에서는 JSON 한 줄 = 로그 하나다(`LOG_FORMAT=json`, compose가 지정). Alloy가 이걸
파싱해 `level`을 Loki 라벨로 올린다. 로컬 기본값은 `text`라 사람이 읽기 좋다.

문서·사이클 결과에는 집계용 필드가 실린다 — `/metrics` 없이 로그만으로도 알림을 걸 수 있다.

```json
{"timestamp":"...","level":"INFO","service":"insurance-chunker","logger":"ingest_catalog",
 "message":"문서 처리 완료","event":"document_done","status":"OK","document":"약관.pdf",
 "chunks":687,"elapsed_s":41.3,"phases":{"parse":12.4,"embed":22.8,"store":2.8}}
```

라벨 카디널리티 정책상 `document`·`sha256` 같은 값은 **본문에만** 둔다(라벨로 올리지 않는다).

수집 쪽(brbs-etl의 Alloy, Prometheus 스크랩 잡, 필요한 보안그룹)은
[`deploy/observability/`](deploy/observability/)에 있다. **CD가 배포하지 않는다** — 호스트
레벨 설정이라 사람이 적용한다.

### 헬스체크

`docker ps`의 STATUS 열에 뜬다. 판정 기준은 **마지막 인덱싱 성공 시각** —
프로세스는 살아 있는데 매 주기 실패하는 좀비를 프로세스 생존만으로는 못 잡기 때문이다.

```bash
docker inspect --format '{{.State.Health.Status}}' brbs-insurance-chunker
docker exec brbs-insurance-chunker python /app/healthcheck.py
```

주기 × `HEALTH_GRACE_FACTOR`를 넘기면 unhealthy. compose의 `restart` 정책은 unhealthy로
재시작하지 않으므로(그건 Swarm 기능) 자가치유가 아니라 드러내기 위한 신호다.

### 배포 구성

`brbs-etl`(g4dn.xlarge / T4 16GB / vCPU 4 / RAM 16GiB) 한 대에 컨테이너 넷이 함께 뜬다.

| 컨테이너 | 레포 | 역할 | GPU |
|---|---|---|---|
| `brbs-insurance-chunker` | 이 레포 | S3 → pgvector 인덱싱 (7일 주기 데몬) | 임베딩·VLM을 **Ollama 경유**로 사용 |
| `brbs-ollama` | — | 임베딩(`qwen3-embedding:0.6b`) · VLM(`qwen3-vl:8b-instruct`) | 직접 사용 |
| `brbs-corpus-worker` | SM17-YoonDongJu/AI | Notion → S3 약관 스테이징 | 없음 |
| `brbs-alloy` | 이 레포 (`deploy/observability/`) | 컨테이너 로그 → Loki. CD 대상 아님 | 없음 |

우리 컨테이너는 GPU를 직접 잡지 않는다 — HTTP로 `brbs-ollama`에 요청할 뿐이다. 그래서
compose에 GPU 예약이 없고, 이미지에도 CUDA 의존이 없다. 양쪽 다 `network_mode: host`라
`localhost:11434`로 닿는다.

vCPU가 4개이고 셋이 나눠 쓰므로 `INGEST_CONCURRENCY` 상한이 낮다(아래 참고).

### 배포와 롤백

`main`에 머지되면 `deploy.yml`이 이미지를 빌드해 ECR로 올리고, SSM으로 EC2에서
`deploy/remote-deploy.sh`를 실행한다. 그 스크립트가 pull → `up -d` → **검증 → 실패 시 롤백**까지 한다.

검증 항목(최대 90초 대기):

| 확인 | 왜 |
|---|---|
| `RestartCount` 증가 | 크래시루프. `docker ps`만 보면 잠깐 떠 있는 걸 성공으로 착각한다 |
| 컨테이너 `running` | 기본 |
| 로그에 `인덱싱 데몬 시작` | 프로세스는 살아 있는데 `main`에 못 들어간 경우(`.env` 오류 등)를 잡는다 |

실패하면 직전에 돌던 이미지 태그로 자동 롤백하고, **롤백이 성공해도 배포는 실패로 보고한다**
(초록불이 뜨면 아무도 안 본다). 첫 배포이거나 같은 태그 재배포라 되돌릴 지점이 없으면
`롤백 대상 없음`을 남기고 실패한다.

배포 성공 시 그 태그를 호스트 `.env`의 `IMAGE_TAG`에 기록한다 — 사람이 호스트에서
`docker compose up -d`를 쳐도 같은 이미지가 뜨게 하려는 것.

#### 시크릿

`DATABASE_URL` · `DISCORD_WEBHOOK_INGEST` · `S3_BUCKET`은 SSM Parameter Store(SecureString)에
두고, 배포할 때 `remote-deploy.sh`가 내려받아 호스트 `.env`에 쓴다.

```bash
aws ssm put-parameter --type SecureString --overwrite \
  --name /brbs/insurance-chunker/dev/DATABASE_URL --value 'postgresql://...'
```

디스크의 평문을 없애는 게 목적은 아니다 — compose가 `env_file`로 읽어야 하는 이상 기동
시점에 평문이 필요하다. 노리는 건 그 앞단이다: 사람이 호스트에 비밀번호를 손으로 넣지 않고,
로테이션이 "SSM 값 변경 + 재배포"로 끝나며, 열람이 CloudTrail에 남고, 호스트를 다시 만들어도
자동 복구된다.

파라미터가 없으면 기존 `.env` 값을 그대로 둔다 — 한 번에 이전하지 않아도 되게.

**수동 롤백**은 Actions에서 `Deploy` 워크플로를 `workflow_dispatch`로 실행하고
`image_tag`에 되돌릴 `sha7`을 넣으면 된다. 이때는 빌드도 CI도 건너뛴다 — `main`이 깨져서
롤백하는 상황인데 CI가 막으면 복구를 못 하기 때문이다.

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

**동시성 상한은 호스트가 정한다.** vCPU가 4개이고 `corpus_worker`·Ollama와 나눠 쓰므로
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
docker exec brbs-insurance-chunker python /app/slo.py
```

큰 변경 전 사람이 한 번 돌리는 `deploy_check.py`와는 역할이 다르다 — 그쪽은 asyncpg를
쓰는 배포 전 판정용이고, 이건 psycopg2로 컨테이너 안에서 상시 돈다.

### 실패 문서 격리

0청크 문서는 `policy_chunks`에 행이 안 생겨 `doc_already_ingested`가 영원히 `False`다 —
상한이 없으면 매 주기 S3에서 다시 받아 다시 파싱한다. `INGEST_MAX_RETRY`회 연속 실패하면
격리해 다운로드조차 하지 않는다. 성공 한 번이면 카운터가 0으로 돌아간다.

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
