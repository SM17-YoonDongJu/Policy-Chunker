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
  ├─ embedder.py      임베딩 (Ollama qwen3:embedding / BGE-M3)
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

### 단일 PDF — dry-run (DB 없이 청킹 결과 확인)

```bash
python ingest.py \
  --pdf 약관.pdf \
  --insurer 메리츠화재 \
  --product "단체안심생활보험" \
  --no-embed --dry-run --dry-run-out out.json
```

dry-run은 DB, S3, Ollama 없이 청킹 결과만 JSON으로 출력한다.  
`--no-vision`을 추가하면 Claude CLI 없이도 실행된다.

### DB 저장

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
| `EMBED_MODEL` | `qwen3:embedding` | Ollama 임베딩 모델 |
| `EMBED_BACKEND` | `ollama` | `ollama` \| `sentence_transformers` (BGE-M3 전환) |
| `S3_BUCKET` | — | 대형 표 markdown 저장용 S3 버킷 (없으면 `.table_cache/` 로컬 저장) |
| `CLAUDE_BIN` | `claude` | VLM 표 추출에 사용하는 Claude CLI 실행 경로 |
| `VLM_DPI` | `150` | VLM 페이지 렌더링 해상도 |
| `VLM_TIMEOUT` | `300` | Claude CLI 호출 타임아웃(초) |
| `VISION_MAX_PAGES` | `9999` | VLM 호출 상한 페이지 수 |

---

## 주요 파일 구조

```
Policy-Chunker/
├── ingest.py               단일 PDF CLI 진입점
├── ingest_many.py          YAML manifest 기반 일괄 처리
├── rebuild_search_terms.py BM25 쿼리 보정용 search_terms 테이블 재구축
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
