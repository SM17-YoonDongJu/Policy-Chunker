"""검색 — 운영 검색 재현(AI rag.search 그대로)과 문구 찾기(청크 본문 부분 일치)."""
from __future__ import annotations

import streamlit as st

from viewer import data
from viewer import widgets as w
from viewer.prod_search import ai_engine_ref

NAMESPACE_LABEL = {"terms": "약관(terms)", "case": "판례(case)", "level": "장해분류표(level)"}

st.title("검색")
try:
    insurer_options = ["(없음)", *data.insurers()]
except Exception as e:  # noqa: BLE001
    st.error(f"DB를 읽지 못했어요 — {e}")
    st.stop()

prod_tab, phrase_tab = st.tabs(["운영 검색 재현", "문구 찾기"])

with prod_tab:
    st.caption("리포트 워커가 쓰는 AI 레포 `rag.search`를 그대로 불러요"
               f"(ai-engine@{ai_engine_ref()}). "
               "라우팅·오타 교정·RRF 점수까지 운영과 같아요.")
    with st.form("prod_search"):
        query = st.text_input("질문", placeholder="예: 계단에서 넘어져 골절됐는데 보험금 나와?")
        c1, c2, c3 = st.columns([2, 2, 1])
        insurer = c1.selectbox("보험사 필터", insurer_options)
        product = c2.text_input("상품명 필터", help="운영과 같이 정확히 일치해야 걸려요")
        top_k = c3.number_input("top_k", min_value=1, max_value=30, value=8)
        with st.expander("고급"):
            namespaces = st.multiselect("검색 대상", list(NAMESPACE_LABEL),
                                        format_func=NAMESPACE_LABEL.get,
                                        help="비우면 운영처럼 라우터가 정해요(약관·판례).")
            insurance_type = st.text_input("보험 유형 힌트(insurance_type)",
                                           help="비신체보험이면 운영처럼 빈 결과가 나와요")
        submitted = st.form_submit_button("검색", type="primary")

    if submitted and query.strip():
        try:
            with st.spinner("운영 검색을 실행하는 중…"):
                result = data.prod_search().search(
                    query.strip(), insurer=None if insurer == "(없음)" else insurer,
                    product=product.strip() or None, insurance_type=insurance_type.strip() or None,
                    namespaces=namespaces or None, top_k=int(top_k))
        except Exception as e:  # noqa: BLE001
            st.error(f"운영 검색이 실패했어요 — {type(e).__name__}: {e}")
        else:
            st.session_state["prod_result"] = result

    result = st.session_state.get("prod_result")
    if result is not None:
        vector = {True: "벡터+키워드", False: "키워드만(임베딩 실패)", None: "확인 못 함"}
        info = [f"{result.elapsed_ms}ms", "대상 " + ", ".join(result.namespaces or ["-"]),
                vector[result.vector_used]]
        st.caption(" · ".join(info))
        if result.corrections:
            st.caption("오타 교정: " + ", ".join(
                f"{w.md_escape(c['original'])} → {w.md_escape(c['canonical'])}"
                for c in result.corrections))
        if result.keyword_query is not None:
            st.caption(f"키워드 질의: `{result.keyword_query}`")
        if not result.in_scope:
            st.warning(f"운영에서도 빈 결과예요 — {result.reason}")
        elif not result.chunks:
            st.info("결과가 없어요.")

        terms_ids = tuple(c["source_ref"] for c in result.chunks if c["namespace"] == "terms")
        try:
            where = data.chunk_locations(terms_ids)
        except Exception:  # noqa: BLE001 - 위치를 몰라도 결과는 보여 준다
            where = {}
        for rank, c in enumerate(result.chunks, 1):
            with st.container(border=True):
                head = [f"**{rank}**", f"RRF {c['score']:.4f}",
                        NAMESPACE_LABEL.get(c["namespace"], c["namespace"])]
                if c.get("article_number"):
                    head.append(w.md_escape(c["article_number"]))
                if c.get("chunk_type"):
                    head.append(f":gray-badge[{w.md_escape(c['chunk_type'])}]")
                st.markdown(" · ".join(head))
                loc = where.get(c["source_ref"])
                if loc:
                    w.doc_link(loc["doc_hash"], loc["page_number"],
                               label=f"{loc['insurer']} · {loc['product_name']} · "
                                     f"{loc['page_number']}쪽")
                st.code(c["text"], language=None, wrap_lines=True, height=160)
        if result.citations:
            st.caption("인용: " + ", ".join(
                w.md_escape(" ".join(x for x in (ct.get("clause_no"), ct.get("exhibit")) if x))
                for ct in result.citations))

with phrase_tab:
    st.caption("청크 본문에서 문구를 글자 그대로 찾아요(형태소·임베딩 없이). "
               "최근 적재 문서부터 나와요.")
    with st.form("phrase_search"):
        text = st.text_input("문구", placeholder="예: 보험금을 지급하지 않는 사유")
        p1, p2, p3 = st.columns([2, 3, 1])
        p_insurer = p1.selectbox("보험사", insurer_options, key="phrase_insurer")
        try:
            type_options = data.chunk_types()
        except Exception:  # noqa: BLE001
            type_options = []
        types = p2.multiselect("청크 유형", type_options)
        limit = p3.selectbox("개수", [30, 100, 300])
        go = st.form_submit_button("찾기", type="primary")

    if go and text.strip():
        try:
            hits = data.find_phrase(text.strip(), None if p_insurer == "(없음)" else p_insurer,
                                    tuple(types), int(limit))
        except Exception as e:  # noqa: BLE001
            st.error(f"찾지 못했어요 — {e}")
        else:
            st.caption(f"{len(hits)}건" + (" (상한)" if len(hits) == limit else ""))
            for h in hits:
                with st.container(border=True):
                    head = [w.md_escape(f"{h['insurer']} · {h['product_name']}"),
                            f"{h['page_number']}쪽 #{h['chunk_index']}",
                            f":gray-badge[{w.md_escape(h['chunk_type'])}]"]
                    if h.get("article_number"):
                        head.append(w.md_escape(h["article_number"]))
                    st.markdown(" · ".join(head))
                    st.markdown(w.snippet(h["content"], text.strip()))
                    w.doc_link(h["doc_hash"], h["page_number"])
