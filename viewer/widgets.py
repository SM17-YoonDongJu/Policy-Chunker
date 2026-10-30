"""화면 조각 — 청크 카드, 문서 링크, 시각 표시."""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import streamlit as st

from viewer import data

KST = timezone(timedelta(hours=9))  # 서머타임이 없어 고정 오프셋으로 충분하다
DOCUMENT_PAGE = "views/document.py"

# chunk_type → 배지 색. 모르는 유형은 회색.
TYPE_COLORS = {
    "coverage": "blue", "exclusion": "red", "definition": "violet",
    "special_clause": "orange", "duty": "green", "claim": "green",
    "termination": "gray", "schedule": "orange", "general": "gray",
}

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")


def md_escape(text: Any) -> str:
    """markdown·LaTeX·Streamlit 지시어로 해석되지 않게 글자 그대로 보이도록 이스케이프한다."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def to_dt(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def kst(value: Any, fmt: str = "%m-%d %H:%M") -> str:
    dt = to_dt(value)
    return dt.astimezone(KST).strftime(fmt) if dt else "-"


def ago(value: Any, now: Optional[datetime] = None) -> str:
    dt = to_dt(value)
    if not dt:
        return "-"
    secs = int(((now or datetime.now(timezone.utc)) - dt).total_seconds())
    if secs < 60:
        return f"{max(secs, 0)}초 전"
    if secs < 3600:
        return f"{secs // 60}분 전"
    if secs < 86400:
        return f"{secs // 3600}시간 {secs % 3600 // 60}분 전"
    return f"{secs // 86400}일 전"


def duration(seconds: Any) -> str:
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "-"
    if s < 60:
        return f"{s}초"
    if s < 3600:
        return f"{s // 60}분 {s % 60}초"
    return f"{s // 3600}시간 {s % 3600 // 60}분"


def snippet(text: str, needle: str, width: int = 90) -> str:
    """needle 주변만 잘라 굵게 표시한다(대소문자 무시). 못 찾으면 앞부분."""
    idx = text.lower().find(needle.lower()) if needle else -1
    if idx < 0:
        return md_escape(text[: width * 2]) + ("…" if len(text) > width * 2 else "")
    start, end = max(0, idx - width), min(len(text), idx + len(needle) + width)
    return ("…" if start else "") + md_escape(text[start:idx]) \
        + f"**{md_escape(text[idx:idx + len(needle)])}**" \
        + md_escape(text[idx + len(needle):end]) + ("…" if end < len(text) else "")


def doc_link(doc_hash: str, page: Optional[int] = None, label: str = "문서 보기") -> None:
    params = {"doc": doc_hash}
    if page:
        params["page"] = str(page)
    st.page_link(DOCUMENT_PAGE, label=label, icon=":material/description:", query_params=params)


def doc_url(doc_hash: str, page: Optional[int] = None) -> str:
    """표의 링크 열(LinkColumn)용 상대 주소. 문서 보기 화면의 url_path와 맞춘다."""
    return f"document?doc={doc_hash}" + (f"&page={page}" if page else "")


def chunk_card(c: dict[str, Any]) -> None:
    with st.container(border=True):
        color = TYPE_COLORS.get(c.get("chunk_type") or "", "gray")
        head = [f"**#{c.get('chunk_index')}**", f":{color}-badge[{md_escape(c.get('chunk_type'))}]"]
        if c.get("article_number"):
            title = c.get("article_title") or ""
            head.append(md_escape(f"{c['article_number']} {title}".strip()))
        if c.get("token_count") is not None:
            head.append(f":gray[{c['token_count']} 토큰]")
        st.markdown(" · ".join(head))
        if c.get("section"):
            st.caption(md_escape(c["section"]))
        st.code(c.get("content") or "", language=None, wrap_lines=True)
        if c.get("table_id"):
            rows = ""
            if c.get("row_start") is not None:
                rows = f" — {c['row_start']}~{c['row_end']}행"
            with st.expander(f"표 원문{rows}"):
                try:
                    md = data.table_markdown(c["table_id"])
                except Exception as e:  # noqa: BLE001 - 표 하나 때문에 페이지가 깨지면 안 된다
                    st.warning(f"표 원문을 못 읽었어요: {e}")
                else:
                    if md:
                        st.markdown(md)
                    else:
                        st.caption("표 원문이 없어요.")
