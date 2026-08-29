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
  │     └ Claude CLI  VLM — PyMuPDF가 표를 감지한 페이지만
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

VLM 기능(표 추출)을 사용하려면 [Claude Code CLI](https://claude.ai/code) 설치·로그인 필요 (API 키 불필요).

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

# VLM(Claude CLI)도 없는 환경
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
| `EMBED_BACKEND` | `ollama` | `ollama` \| `sentence_transformers` (BGE-M3 전환) |
| `S3_BUCKET` | — | 대형 표 markdown 저장용 S3 버킷 (없으면 `.table_cache/` 로컬 저장) |
| `CLAUDE_BIN` | `claude` | VLM 표 추출에 사용하는 Claude CLI 실행 경로 |
| `VLM_DPI` | `150` | VLM 페이지 렌더링 해상도 |
| `VLM_TIMEOUT` | `600` | Claude CLI 호출 타임아웃(초) |
| `VISION_MAX_PAGES` | `9999` | VLM 호출 상한 페이지 수 |
| `INGEST_STATE_DIR` | `/data/state` | 실행 이력 디렉터리 (쓰기 불가 시 `./.state` 폴백) |
| `INGEST_MAX_RETRY` | `3` | 0청크·오류 연속 N회면 문서를 격리 |
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

### 헬스체크

`docker ps`의 STATUS 열에 뜬다. 판정 기준은 **마지막 인덱싱 성공 시각** —
프로세스는 살아 있는데 매 주기 실패하는 좀비를 프로세스 생존만으로는 못 잡기 때문이다.

```bash
docker inspect --format '{{.State.Health.Status}}' brbs-insurance-chunker
docker exec brbs-insurance-chunker python /app/healthcheck.py
```

주기 × `HEALTH_GRACE_FACTOR`를 넘기면 unhealthy. compose의 `restart` 정책은 unhealthy로
재시작하지 않으므로(그건 Swarm 기능) 자가치유가 아니라 드러내기 위한 신호다.

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
