"""청크 뷰어 — 처리한 약관을 원본 페이지와 청크로 나란히 보고, 운영과 같은 검색을 재현한다.

진입점은 Streamlit 앱 app.py다. 어디에도 쓰지 않고 읽기만 한다.
  DB      corpus.policy_chunks(청크), ai.corpus_file·corpus_file_part(원본 파일·S3 키)
  S3      원본 PDF, 표 원문(policy-tables/{table_id}.md)
  청커    /status — 처리 중인 문서·최근 결과(exporter.status_snapshot)
  검색    AI 레포 rag.search를 운영과 같은 커밋으로(prod_search.py)
"""
