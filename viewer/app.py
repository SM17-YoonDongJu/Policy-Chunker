"""청크 뷰어 진입점 — `streamlit run viewer/app.py`

화면 셋:
  처리 현황  청커가 지금 처리 중인 문서·최근 결과(/status)와 적재된 문서 목록(최신순)
  문서 보기  원본 페이지 이미지와 그 쪽에서 나온 청크를 나란히
  검색       운영 검색 재현(AI rag.search 그대로)과 문구 찾기
"""
from __future__ import annotations

import sys
from pathlib import Path

# Streamlit은 이 파일의 디렉터리(viewer/)만 sys.path에 넣는다 — viewer 패키지로 import하려고
# 레포 루트를 더한다.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from viewer.prod_search import ai_engine_ref  # noqa: E402

st.set_page_config(page_title="청크 뷰어", page_icon=":material/plagiarism:", layout="wide")

nav = st.navigation([
    st.Page("views/status.py", title="처리 현황", icon=":material/monitoring:", default=True),
    st.Page("views/document.py", title="문서 보기", icon=":material/description:",
            url_path="document"),
    st.Page("views/search.py", title="검색", icon=":material/search:", url_path="search"),
])
with st.sidebar:
    st.caption(f"읽기 전용 · 운영 검색 코드 ai-engine@{ai_engine_ref()}")
nav.run()
