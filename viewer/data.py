"""화면이 쓰는 데이터 — queries·files를 Streamlit 캐시로 감싼다.

캐시 수명은 데이터가 바뀌는 속도에 맞춘다. 적재는 7일 주기지만 사이클 도중에는 문서가 몇 분마다
들어오므로 목록은 짧게, 한 문서의 청크는 그보다 길게, 원본 파일 정보는 더 길게 둔다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from viewer import env, files, queries


@st.cache_data(ttl=600, show_spinner=False)
def catalog_access() -> bool:
    return queries.catalog_access()


@st.cache_data(ttl=30, show_spinner=False)
def recent_documents(limit: int, insurer: Optional[str], product: Optional[str],
                     name: Optional[str]) -> list[dict[str, Any]]:
    return queries.recent_documents(limit=limit, insurer=insurer, product=product, name=name,
                                    with_catalog=catalog_access())


@st.cache_data(ttl=300, show_spinner=False)
def insurers() -> list[str]:
    return queries.insurers()


@st.cache_data(ttl=300, show_spinner=False)
def chunk_types() -> list[str]:
    return queries.chunk_types()


@st.cache_data(ttl=120, show_spinner=False)
def document_meta(doc_hash: str) -> Optional[dict[str, Any]]:
    return queries.document_meta(doc_hash)


@st.cache_data(ttl=120, show_spinner=False, max_entries=20)
def document_chunks(doc_hash: str) -> list[dict[str, Any]]:
    return queries.document_chunks(doc_hash)


@st.cache_data(ttl=600, show_spinner=False)
def source_file(doc_hash: str) -> Optional[dict[str, Any]]:
    return queries.source_file(doc_hash) if catalog_access() else None


@st.cache_data(ttl=60, show_spinner=False)
def find_phrase(text: str, insurer: Optional[str], types: tuple[str, ...],
                limit: int) -> list[dict[str, Any]]:
    return queries.find_phrase(text, insurer=insurer, chunk_types=list(types), limit=limit)


@st.cache_data(ttl=120, show_spinner=False)
def chunk_locations(chunk_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return queries.chunk_locations(list(chunk_ids))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=200)
def table_markdown(table_id: str) -> Optional[str]:
    return files.table_markdown(table_id)


def pdf_path(doc_hash: str, src: Optional[dict[str, Any]]) -> Optional[Path]:
    """디스크 캐시가 있어 Streamlit 캐시는 두지 않는다(경로만 돌려주므로 싸다)."""
    if src is not None and not files.is_pdf_name(src.get("file_name")):
        return None
    return files.pdf_path(doc_hash, (src or {}).get("s3_key"))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=50)
def page_count(path: str) -> int:
    return files.page_count(Path(path))


@st.cache_data(ttl=3600, show_spinner=False, max_entries=300)
def render_page(path: str, page_number: int) -> bytes:
    return files.render_page(Path(path), page_number, env.RENDER_DPI)


@st.cache_resource(show_spinner=False)
def prod_search() -> Any:
    from viewer.prod_search import ProdSearch

    return ProdSearch(env.DATABASE_URL)
