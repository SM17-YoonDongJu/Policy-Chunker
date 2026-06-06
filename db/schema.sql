-- policy_chunks: 약관(policy_terms) 청크 테이블
-- 출력 컬럼 = DB 컬럼 원칙: InsuranceChunk의 DB 저장 대상 필드와 1:1 대응

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id        TEXT PRIMARY KEY,
    content         TEXT        NOT NULL,   -- 임베딩 원문
    content_tokens  TEXT,                   -- Kiwi 형태소 결과 (공백 구분) → pg_search BM25 검색
    embedding       vector(1024),           -- qwen3:embedding 1024d / BGE-M3 1024d
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
    section         TEXT                    -- 경계 라벨 또는 편/장 경로
);

-- ── 검색 인덱스 ──────────────────────────────────────────────────────────────

-- 벡터 검색 (ANN, cosine similarity) — HNSW
CREATE INDEX IF NOT EXISTS idx_policy_hnsw
    ON policy_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 키워드 검색 — pg_search BM25 (tsvector + trigram 대체)
-- 검색: WHERE content_tokens @@@ '보험금 지급'
-- 랭킹: ORDER BY paradedb.score(chunk_id) DESC
CREATE INDEX IF NOT EXISTS idx_policy_bm25
    ON policy_chunks
    USING bm25 (chunk_id, content_tokens)
    WITH (key_field = 'chunk_id');

-- 메타 필터 (보험사·청크타입·시행일)
CREATE INDEX IF NOT EXISTS idx_policy_meta
    ON policy_chunks (insurer, chunk_type, effective_date);

-- doc_hash 중복 방지 조회
CREATE INDEX IF NOT EXISTS idx_policy_doc_hash
    ON policy_chunks (doc_hash);

-- ── search_terms: 쿼리 단어 보정용 용어 사전 ────────────────────────────────
-- 적재: rebuild_search_terms.py 로 policy_chunks.content_tokens에서 자동 추출
-- 사용: 검색 쿼리 입력 → trigram 유사 term 조회 → 보정된 term으로 BM25 검색

CREATE TABLE IF NOT EXISTS search_terms (
    term        TEXT PRIMARY KEY
);

CREATE INDEX IF NOT EXISTS idx_search_terms_trgm
    ON search_terms USING gin (term gin_trgm_ops);
