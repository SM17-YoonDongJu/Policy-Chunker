-- policy_chunks: 약관(policy_terms) 청크 테이블
-- 출력 컬럼 = DB 컬럼 원칙: InsuranceChunk의 DB 저장 대상 필드와 1:1 대응
--
-- 적용 범위: 로컬 개발용 빈 DB 전용.
-- 운영 RDS의 corpus.* DDL은 SM17-YoonDongJu/AI 레포 migrations/corpus/가 단일 관리한다
-- (.env.prod의 SKIP_INIT_SCHEMA=1이 이 파일의 실행을 막는다). 운영 스키마를 바꿔야 하면
-- 이 파일이 아니라 그쪽 마이그레이션에 추가할 것 — 여기만 고치면 운영에 반영되지 않는다.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── policy_chunks: 약관(policy_terms) 청크 테이블 ────────────────────────────
-- 출력 컬럼 = DB 컬럼 원칙: InsuranceChunk의 DB 저장 대상 필드와 1:1 대응
CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id        TEXT PRIMARY KEY,
    content         TEXT        NOT NULL,   -- 임베딩 원문
    content_tokens  TEXT,                   -- Kiwi 형태소 결과 (공백 구분) → tsvector 전문검색
    embedding       halfvec(1024),          -- qwen3:embedding 1024d / BGE-M3 1024d (float16: 용량·RAM 절반)
    token_count     INT,
    chunk_type      TEXT        NOT NULL,   -- coverage|exclusion|definition|special_clause|duty|claim|termination|schedule|general
    doc_hash        TEXT        NOT NULL,   -- PDF sha256, 중복 ingest 방지
    page_number     INT,
    ingested_at     TIMESTAMPTZ DEFAULT now(),

    -- 보험사·상품 메타 (검색 필터)
    insurer         TEXT        NOT NULL,
    product_name    TEXT        NOT NULL,
    product_code    TEXT,
    effective_date  DATE,

    -- 약관 구조 메타
    article_number  TEXT,                   -- "제12조"
    article_title   TEXT,                   -- "보험금을 지급하지 않는 사유"
    generation      TEXT,                   -- 세대 (예: "4세대")
    section         TEXT,                   -- 경계 라벨 또는 편/장 경로
    chunk_index     INT,                    -- 문서 전체 순서 (조항 복원 시 ORDER BY)

    -- 표 row 청크 전용 (텍스트 청크는 NULL)
    -- table_id: S3 key → policy-tables/{table_id}.md (FK 없음, S3 참조)
    table_id        UUID,
    row_start       SMALLINT,
    row_end         SMALLINT,

    -- 상품 FK (nullable — ingest 시 --product-id 미지정 시 NULL)
    -- REFERENCES insurance_products(id) 는 메인 앱 마이그레이션에서 관리
    product_id      UUID
);

-- ── 마이그레이션: 기존 DB에 신규 컬럼 추가 ──────────────────────────────────
-- CREATE TABLE IF NOT EXISTS는 기존 테이블에 컬럼을 추가하지 않으므로
-- 운영 DB 또는 이미 초기화된 DB에는 아래 ALTER TABLE을 별도 실행한다.
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS chunk_index INT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS table_id    UUID;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS row_start   SMALLINT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS row_end     SMALLINT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS product_id  UUID;

-- SM17-YoonDongJu/AI 레포(rag.search())가 content_tsv(tsvector 컬럼)를 직접
-- @@ 검색한다 — 우리는 지금까지 to_tsvector(content_tokens)를 매번 계산했으므로
-- 호출 계약을 맞추려면 이 생성 컬럼이 필요하다. content_tokens 갱신 시 자동 재계산.
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content_tokens, ''))) STORED;

-- embedding 타입을 vector → halfvec(float16)로 전환 (용량·RAM 절반).
-- 기존 운영 DB에만 수동 실행. 인덱스 연산자 클래스가 바뀌므로 인덱스 재생성 필요.
--   ALTER TABLE policy_chunks ALTER COLUMN embedding TYPE halfvec(1024);
--   DROP INDEX IF EXISTS idx_policy_hnsw;
--   (아래 CREATE INDEX idx_policy_hnsw 재실행)


-- ── 검색 인덱스 ──────────────────────────────────────────────────────────────

-- 벡터 검색 (ANN, cosine similarity) — HNSW
-- halfvec(float16) 전용 연산자 클래스 사용
CREATE INDEX IF NOT EXISTS idx_policy_hnsw
    ON policy_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 키워드 검색 — tsvector 전문검색 (GIN 인덱스)
-- 검색: WHERE to_tsvector('simple', content_tokens) @@ plainto_tsquery('simple', '보험금 지급')
-- 랭킹: ORDER BY ts_rank(to_tsvector('simple', content_tokens), plainto_tsquery('simple', '검색어')) DESC
CREATE INDEX IF NOT EXISTS idx_policy_fts
    ON policy_chunks
    USING gin (content_tsv);

-- 메타 필터 (보험사·청크타입·시행일)
CREATE INDEX IF NOT EXISTS idx_policy_meta
    ON policy_chunks (insurer, chunk_type, effective_date);

-- doc_hash 중복 방지 조회
CREATE INDEX IF NOT EXISTS idx_policy_doc_hash
    ON policy_chunks (doc_hash);

-- 표 row 청크 조회 (table_id로 child 전체 fetch)
CREATE INDEX IF NOT EXISTS idx_policy_table_id
    ON policy_chunks (table_id);

-- ── search_terms: 쿼리 단어 보정용 용어 사전 ────────────────────────────────
-- 적재: rebuild_search_terms.py 로 policy_chunks.content_tokens에서 자동 추출
-- 사용: 검색 쿼리 입력 → trigram 유사 term 조회 → 보정된 term으로 tsvector 검색

CREATE TABLE IF NOT EXISTS search_terms (
    term        TEXT PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_search_terms_trgm
    ON search_terms USING gin (term gin_trgm_ops);

