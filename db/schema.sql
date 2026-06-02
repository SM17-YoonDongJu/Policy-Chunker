-- insurance_chunks: 약관/요약서/장해분류표 등 보험 문서 청크 통합 테이블

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS insurance_chunks (
    chunk_id        TEXT PRIMARY KEY,
    parent_id       TEXT,
    source_pdf      TEXT        NOT NULL,
    doc_hash        TEXT        NOT NULL,

    insurer         TEXT        NOT NULL,
    product_name    TEXT        NOT NULL,
    product_code    TEXT,
    effective_date  DATE,

    doc_type        TEXT        NOT NULL,   -- product_summary | policy_terms | schedule
    chunk_type      TEXT        NOT NULL,   -- coverage | exclusion | definition | special_clause | duty | claim | termination | schedule | general

    section_path    TEXT[],
    page_number     INT,
    token_count     INT,

    article_number  TEXT,
    article_title   TEXT,
    yakwan          TEXT,                   -- 약관명 (폰트 기반 경계 감지 결과)

    content         TEXT        NOT NULL,
    content_tokens  TEXT,                   -- Kiwi 형태소 결과 → pg_trgm
    structured_json JSONB,

    embedding       vector(1024),

    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON insurance_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_trgm
    ON insurance_chunks USING gin (content_tokens gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_meta
    ON insurance_chunks (insurer, doc_type, chunk_type, effective_date);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash
    ON insurance_chunks (doc_hash);

CREATE INDEX IF NOT EXISTS idx_chunks_yakwan
    ON insurance_chunks (yakwan);
