"""문서 보기 — 원본 페이지 이미지와 그 쪽에서 나온 청크를 나란히 놓는다.

주소의 ?doc=<doc_hash>&page=<쪽>으로 열린다. 쪽은 1부터 센다(청크의 page_number와 같은 기준).
청크마다 쪽이 하나만 기록돼서, 여러 쪽에 걸친 청크는 시작한 쪽에만 보인다.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from viewer import data, env, files
from viewer import widgets as w

doc = st.query_params.get("doc")

if not doc:
    st.title("문서 보기")
    try:
        recent = data.recent_documents(50, None, None, None)
    except Exception as e:  # noqa: BLE001
        st.error(f"DB를 읽지 못했어요 — {e}")
        st.stop()
    if not recent:
        st.info("적재된 문서가 없어요.")
        st.stop()
    labels = {d["doc_hash"]: f"{w.kst(d['ingested_at'])} · {d['insurer']} · "
                             f"{d.get('file_name') or d['product_name']}" for d in recent}
    pick = st.selectbox("최근 적재 문서", list(labels), index=None, format_func=labels.get,
                        placeholder="문서를 고르세요")
    if pick:
        st.query_params["doc"] = pick
        st.rerun()
    st.stop()

try:
    meta = data.document_meta(doc)
    chunks = data.document_chunks(doc) if meta else []
    src = data.source_file(doc) if meta else None
except Exception as e:  # noqa: BLE001
    st.error(f"DB를 읽지 못했어요 — {e}")
    st.stop()
if not meta:
    st.error(f"이 문서의 청크가 없어요 (doc_hash {doc[:16]}…). "
             "적재 전이거나 0청크로 끝난 문서예요.")
    st.stop()

pdf, pdf_error = None, None
try:
    with st.spinner("원본 PDF를 가져오는 중…"):
        pdf = data.pdf_path(doc, src)
except Exception as e:  # noqa: BLE001 - PDF가 없어도 청크는 보여야 한다
    pdf_error = str(e)

by_page: dict[int, list[dict]] = {}
for c in chunks:
    by_page.setdefault(c["page_number"] or 0, []).append(c)
pages_with = sorted(p for p in by_page if p > 0)
n_pages = data.page_count(str(pdf)) if pdf else max(pages_with or [1])
empty_pages = [p for p in range(1, n_pages + 1) if p not in by_page] if pdf else []

st.title((src or {}).get("file_name") or meta["product_name"])
facts = [meta["insurer"], meta["product_name"],
         f"적재 {w.kst(meta['ingested_at'], '%Y-%m-%d %H:%M')}",
         f"청크 {meta['chunks']}개", f"{n_pages}쪽"]
if src and src.get("part_total"):
    facts.append(f"파트 {src['part_order'] + 1}/{src['part_total']}")  # part_order는 0부터
facts.append(f"doc_hash {doc[:12]}")
st.caption(" · ".join(w.md_escape(x) for x in facts))
if pdf_error:
    st.warning(f"원본 PDF를 못 가져왔어요 — {pdf_error}")
elif not pdf:
    if src and not files.is_pdf_name(src.get("file_name")):
        reason = f"원본이 PDF가 아니에요({src['file_name']})"
    elif src and not env.S3_BUCKET and not env.PDF_DIR:
        reason = "원본을 받을 S3_BUCKET이 설정되지 않았어요"
    elif env.PDF_DIR:
        reason = "VIEWER_PDF_DIR에 이 문서와 해시가 같은 PDF가 없어요"
    else:
        reason = "원본 파일 정보(ai.corpus_file_part)를 찾지 못했어요 — 없거나 읽기 권한이 없어요"
    st.info(f"{w.md_escape(reason)}. 청크만 쪽별로 보여요.")

page_key = f"page_{doc[:16]}"
if page_key not in st.session_state:
    try:
        wanted = int(st.query_params.get("page") or (pages_with[0] if pages_with else 1))
    except ValueError:
        wanted = 1
    st.session_state[page_key] = min(max(wanted, 1), n_pages)

with st.expander("문서 요약", expanded=bool(empty_pages)):
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**청크 유형**")
        counts = Counter(c["chunk_type"] for c in chunks).most_common()
        st.bar_chart(pd.DataFrame(counts, columns=["유형", "청크"]), x="유형", y="청크",
                     x_label="", y_label="청크 수", horizontal=True, height=220)
    with s2:
        tokens = [c["token_count"] for c in chunks if c.get("token_count") is not None]
        if tokens:
            st.markdown(f"**토큰** 최소 {min(tokens)} · 평균 {sum(tokens) // len(tokens)} · "
                        f"최대 {max(tokens)} · 600 초과 {sum(t > 600 for t in tokens)}개")
        unpaged = len(by_page.get(0, []))
        if unpaged:
            st.markdown(f"쪽 정보가 없는 청크 {unpaged}개")
        if pdf:
            st.markdown(f"**청크가 0개인 쪽 {len(empty_pages)}개** — "
                        "본문이 조용히 빠졌는지 원본과 대조해 보세요.")

            def _jump() -> None:
                if st.session_state.get("empty_pick"):
                    st.session_state[page_key] = st.session_state["empty_pick"]

            if empty_pages:
                st.pills("빈 쪽으로 이동", empty_pages[:80], key="empty_pick", on_change=_jump,
                         label_visibility="collapsed")
                if len(empty_pages) > 80:
                    st.caption(f"외 {len(empty_pages) - 80}쪽")


def _step(delta: int) -> None:
    cur = st.session_state[page_key]
    if st.session_state.get("only_chunked") and pages_with:
        seq = [p for p in pages_with if (p > cur if delta > 0 else p < cur)]
        st.session_state[page_key] = (seq[0] if delta > 0 else seq[-1]) if seq else cur
    else:
        st.session_state[page_key] = min(max(cur + delta, 1), n_pages)


n1, n2, n3, n4 = st.columns([1, 1, 1, 3], vertical_alignment="center")
n1.button("◀ 이전", on_click=_step, args=(-1,), width="stretch")
n2.number_input("쪽", min_value=1, max_value=n_pages, key=page_key, label_visibility="collapsed")
n3.button("다음 ▶", on_click=_step, args=(1,), width="stretch")
n4.toggle("청크 있는 쪽만 넘기기", key="only_chunked")
page = st.session_state[page_key]
st.query_params["page"] = str(page)  # 주소를 공유하거나 새로고침해도 같은 쪽

left, right = st.columns(2, gap="medium")
with left:
    if pdf:
        st.image(data.render_page(str(pdf), page), width="stretch")
    else:
        st.caption("원본 이미지 없음")
with right:
    here = by_page.get(page, [])
    st.markdown(f"**{page}쪽에서 나온 청크 {len(here)}개**")
    if not here:
        st.warning("이 쪽에서 나온 청크가 없어요."
                   + (" 앞쪽 청크가 이어졌거나 본문이 빠졌을 수 있어요 — 원본과 대조해 보세요."
                      if pdf else ""))
    for c in here:
        w.chunk_card(c)
