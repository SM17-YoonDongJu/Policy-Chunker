# insurance-chunker

보험 약관 PDF → RAG 청크 파이프라인.

**Policy-Chunker**(팀 채택 코드)의 경계 감지·청킹 로직을 그대로 유지하면서,
빠진 추출 레이어와 ingest 파이프라인을 추가한 버전.

---

## Policy-Chunker와 한눈에 비교

| 항목 | Policy-Chunker | insurance-chunker |
|---|---|---|
| **경계 감지** (`boundaries.py`) | 동일 + `assess()` 신뢰도 게이트 추가 | ← |
| **표 best-of 선택** (`combine.py`) | `table_sources: dict` 인터페이스, `"vlm"` 키 | 동일 (인터페이스 통일) |
| **청킹 뼈대** (`rechunk.py`) | clean → merge → dedup | 동일 |
| **VLM 방식** | `claude -p … --allowedTools Read` (구독 토큰) | 동일 |
| **표 추출** (`extract.py`) | PyMuPDF + pdfplumber + camelot + VLM | 동일 (`extractor.py`) |
| **PDF 텍스트 파싱** | PyMuPDF `get_text("blocks")`, 표 영역 제외 | 동일 |
| **청크 출력** | `list[dict]` | `list[InsuranceChunk]` (구조화 dataclass) |
| **chunk_type** | 없음 | 8종 자동 분류 |
| **contextual prefix** | 없음 | `"보험사 \| 상품명 \| 약관명 \| 제N조"` |
| **임베딩** | 없음 | Ollama / BGE-M3 |
| **DB 저장** | 없음 | pgvector upsert |
| **CLI** | `python -m policy_chunker pdf chunks.json` | `python ingest.py --pdf … --insurer … --product …` |

---

## 파이프라인

```
약관.pdf
  │
  ├─ pdf_parser.py   텍스트 추출 (PyMuPDF get_text, 표 영역 제외)
  │
  ├─ extractor.py    표 추출 (Policy-Chunker extract.py와 동일 전략)
  │     ├ PyMuPDF    (괘선 표, 빠름)
  │     ├ pdfplumber (설치 시 자동)
  │     ├ camelot    (설치 시 자동, ghostscript 필요)
  │     └ Claude CLI VLM (PyMuPDF 표 탐지 페이지만)
  │
  ├─ combine.py      페이지별 best-of 선택 (더블스페이스 적은 쪽)
  │
  ├─ boundaries.py   폰트 크기로 약관/별표 경계 감지
  │     └ assess()   신뢰도 판정 (ok / weak) — 조용히 틀린 청크 방지
  │
  ├─ rechunk.py      경계 라벨 부여 → 조항 병합 → 중복 제거
  │
  ├─ chunker.py      오케스트레이터 → InsuranceChunk 출력
  │
  ├─ embedder.py     Ollama(qwen3:embedding) / BGE-M3
  │
  └─ db/storage.py   pgvector upsert
```

---

## Policy-Chunker에서 달라진 점 상세

### 1. `boundaries.py` — assess() 신뢰도 게이트 추가

Policy-Chunker(main)에서 가져온 기능. 탐지 실패 시 폴백값을 조용히 채우는 대신
`None`을 반환하고 `assess()`가 신뢰도를 판정한다.

```
[신뢰도 OK]  특약 137 / 별표 17 경계, 제목폰트 12.9
[신뢰도 WEAK] 제목 폰트 신호 없음 → 청킹 결과를 확인하세요
```

weak 판정 조건: 제목 폰트 신호 없음 / `제1조` 미탐지 / 특약 경계 0개 중 하나.


### 2. `rechunk.py` — InsuranceChunk 출력 + 8종 chunk_type

Policy-Chunker는 `list[dict]`를 반환. insurance-chunker는 `InsuranceChunk` dataclass로
구조화하고, 내용 기반 regex로 8종 chunk_type을 자동 분류한다.

chunk_type 한계: regex 기반이므로 오분류 가능. 별표 섹션이 많은 문서는
`schedule` 비율이 높게 나타나는 것이 정상이다.

### 3. contextual prefix

임베딩 대상 `content` 앞에 prefix를 붙인다.

```
메리츠화재 | 단체안심생활보험 | 암 진단특별약관
[암 진단특별약관 > 제2조(보험금의 지급)]
① 회사는 피보험자가 …
```

prefix 없이 "보장 범위"만 검색해도 어느 상품·약관의 청크인지 벡터에 담겨 검색 정확도가 높아진다.

---

## 빠른 시작

VLM을 쓰려면 [Claude Code CLI](https://claude.com/claude-code) 설치·로그인 필요 (API 키 불필요).

```bash
pip install -e ".[dev]"

# 단일 PDF — dry-run (DB 저장 없이 청크 확인)
python ingest.py \
  --pdf 약관.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --no-embed --dry-run --dry-run-out out.json

# VLM 없이 빠른 테스트
python ingest.py … --no-vision

# DB 저장
DATABASE_URL=postgresql://... python ingest.py \
  --pdf 약관.pdf --insurer 메리츠화재 --product "..."

# 여러 PDF 일괄 처리 (YAML manifest)
python ingest_many.py --manifest docs.yaml
```

---

## InsuranceChunk 스키마

```jsonc
{
  // 식별
  "chunk_id":      "6e533f8dc4204f3f…",   // SHA256[:24]
  "source_pdf":    "약관.pdf",
  "doc_hash":      "58c3c307…",            // 중복 ingest 방지

  // 상품
  "insurer":       "메리츠화재",
  "product_name":  "단체안심생활보험",
  "effective_date": "2025-06-01",          // 선택

  // 문서 구조
  "doc_type":      "policy_terms",         // policy_terms | schedule
  "yakwan":        "암 진단특별약관",      // 보통약관·별표는 null
  "section_path":  ["암 진단특별약관"],
  "page_number":   42,

  // 조항
  "article_number": "제2조",
  "article_title":  "보험금의 지급",

  // 분류
  "chunk_type":    "coverage",             // 8종 참고
  "token_count":   412,                    // 글자 수

  // 검색
  "content":       "메리츠화재 | … \n[…]\n본문…",  // 임베딩 대상
  "content_tokens": "보험금 지급 암 진단 …",        // Kiwi 형태소 (BM25용)
  "structured_json": {"markdown": "| … |"},          // 표 청크만
  "embedding":     [0.021, …]             // 1024d
}
```

**chunk_type 8종:**

| 값 | 내용 |
|---|---|
| `coverage` | 보장 범위, 지급사유 |
| `exclusion` | 면책, 보험금 미지급 사유 |
| `duty` | 고지의무, 통지의무 |
| `claim` | 청구 절차, 제출서류 |
| `termination` | 계약 해지, 효력상실 |
| `schedule` | 별표, 장해분류표, 지급률표 |
| `definition` | 용어 정의 |
| `special_clause` | 특약 관련 |
| `general` | 위 분류 외 일반 조항 |

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `CLAUDE_BIN` | `claude` | Claude CLI 실행 경로 |
| `VLM_DPI` | `150` | 페이지 이미지 렌더링 해상도 |
| `VLM_TIMEOUT` | `300` | claude CLI 호출당 타임아웃(초) |
| `VISION_MAX_PAGES` | `9999` | VLM 호출 상한 페이지 수 |
| `EMBED_BACKEND` | `ollama` | `ollama` \| `sentence_transformers` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama 서버 |
| `EMBED_MODEL` | `qwen3:embedding` | Ollama 임베딩 모델 |
| `DATABASE_URL` | — | pgvector 연결 문자열 |
