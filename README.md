# insurance-chunker

보험 약관 RAG 청킹 파이프라인.

**Policy-Chunker**(팀 채택 코드)를 베이스로, 완전한 ingest 파이프라인과 풍부한 메타데이터를 보완한 버전.

---

## Policy-Chunker와의 차이

### 동일한 부분

| 구성요소 | 설명 |
|---|---|
| `boundaries.py` | PyMuPDF 폰트 크기 기반 약관/별표 경계 감지 — **코드 동일** |
| `combine.py` | 더블스페이스 지표로 PyMuPDF vs Vision 중 best 표 선택 — **로직 동일** |
| `rechunk.py` 뼈대 | 경계 라벨 부여 → clean → merge → 인용 조문 병합 → 중복 제거 |

> **약관(policy_terms) 청킹 결과**는 Policy-Chunker와 텍스트 내용 기준으로 거의 동일하다.
> 경계 감지·병합·필터 로직이 같기 때문. 메타데이터 필드와 contextual prefix만 다르다.

---

### 추가/변경된 부분

#### 1. `extractor.py` — 신규 (Policy-Chunker의 빈 칸)

Policy-Chunker의 `combine.py`는 "이미 추출된 두 소스를 비교"하는 로직만 갖고 있고,
실제 추출 코드는 없었다. `extractor.py`가 그 부분을 구현한다.

```
PyMuPDF (fitz.find_tables)  ──┐
                               ├→ combine.py → best 표 선택
Claude Vision (Anthropic API) ─┘
```

#### 2. `rechunk.py` — 세 가지 보완

| 항목 | Policy-Chunker | insurance-chunker |
|---|---|---|
| 토큰 계산 | `len(text)` 글자 수 | `tiktoken` cl100k 실측 |
| 출력 형식 | `list[dict]` | `list[InsuranceChunk]` |
| chunk_type | 없음 | 8종 자동 분류 |

> 글자 수 → 토큰 수 전환으로 한국어 특성상 병합 경계가 약간 달라질 수 있다.

#### 3. `pdf_parser.py` — 신규

Policy-Chunker는 PDF 파싱 코드를 제공하지 않는다.
pdfplumber 텍스트 추출 + PaddleOCR(스캔본) 지원.

#### 4. 완전한 ingest 파이프라인 — 신규

Policy-Chunker에 없는 구성요소:

- `tokenizer.py` — Kiwi 형태소 분석 → `content_tokens` (pg_trgm BM25용)
- `embedder.py` — Ollama(qwen3:embedding) / BGE-M3 듀얼 백엔드
- `validator.py` — 청크 품질 검증 (token 범위, 필수 필드, chunk_type 분포)
- `db/` — pgvector 스키마 + upsert 저장
- `ingest.py` / `ingest_many.py` — CLI

#### 5. 메타데이터 필드

`InsuranceChunk`가 추가하는 필드 (Policy-Chunker 출력 dict에 없음):

| 필드 | 설명 |
|---|---|
| `chunk_type` | coverage / exclusion / duty / claim / termination / schedule / definition / special_clause / general |
| `article_number` | 제N조 |
| `article_title` | 조항 제목 |
| `yakwan` | 약관명 (폰트 경계에서 추출) |
| `content_tokens` | Kiwi 형태소 토큰 (BM25 검색용) |
| `embedding` | 1024d 벡터 |
| contextual prefix | `"메리츠화재 \| 상품명 \| 약관명 \| 제N조(제목)"` 형식으로 content 앞에 붙임 |

---

## 파일 구조

```
insurance_chunker/
  boundaries.py   ← Policy-Chunker 동일
  combine.py      ← Policy-Chunker 동일 (시그니처 소폭 변경)
  extractor.py    ← 신규: PyMuPDF + Claude Vision 추출
  rechunk.py      ← Policy-Chunker 뼈대 + tiktoken + chunk_type + InsuranceChunk 출력
  chunker.py      ← 오케스트레이터 (policy_terms / product_summary / schedule)
  pdf_parser.py   ← 신규: pdfplumber 텍스트 + PaddleOCR
  embedder.py     ← 신규: Ollama / BGE-M3
  tokenizer.py    ← 신규: Kiwi 형태소
  validator.py    ← 신규: 청크 품질 검증
  models.py       ← 신규: InsuranceChunk, DocMeta, PageResult
db/
  schema.sql      ← pgvector + pg_trgm 스키마
  storage.py      ← upsert / verify
eval/
  eval_bm25.py    ← Policy-Chunker 동일
ingest.py         ← 단일 PDF CLI
ingest_many.py    ← YAML manifest 일괄 처리
```

---

## 빠른 시작

```bash
pip install -e ".[dev]"

# 단일 PDF (dry-run)
python ingest.py \
  --pdf 상해보험_단체안심생활보험_30327.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --dry-run --dry-run-out out.json

# DB 저장
DATABASE_URL=postgresql://... python ingest.py \
  --pdf 약관.pdf --insurer 메리츠화재 --product "..."

# Vision 비활성화 (빠른 테스트)
python ingest.py ... --no-vision

# 임베딩 백엔드 전환
EMBED_BACKEND=sentence_transformers python ingest.py ...
```

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Claude Vision 필수 |
| `VISION_MODEL` | `claude-sonnet-4-6` | Vision 모델 |
| `VISION_MAX_PAGES` | `9999` | 페이지당 Vision 호출 상한 |
| `EMBED_BACKEND` | `ollama` | `ollama` \| `sentence_transformers` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 서버 |
| `EMBED_MODEL` | `qwen3:embedding` | Ollama 임베딩 모델 |
| `DATABASE_URL` | — | pgvector 연결 문자열 |
