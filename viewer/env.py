"""뷰어 설정 — 전부 환경변수로 받는다(클러스터는 차트 values, 로컬은 .env).

운영 검색 재현(prod_search.py)이 쓰는 EMBEDDING_BASE_URL·EMBEDDING_MODEL·AI_TIMEOUT_SECONDS
같은 값은 여기가 아니라 AI 레포의 core.config가 직접 읽는다.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    pass

DATABASE_URL = os.environ.get("DATABASE_URL", "")
# RDS TLS 검증용 CA 번들. 클러스터에서는 rds-ca-bundle ConfigMap을 리포트 워커와 같은
# 경로에 마운트한다.
RDS_CA_PATH = os.environ.get("RDS_CA_PATH") or None
S3_BUCKET = os.environ.get("S3_BUCKET", "")
# boto3는 AWS_REGION을 보지 않는다 — ingest.py와 같은 이유로 직접 넘긴다.
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None
# 청커의 /status. 클러스터 안에서는 메트릭 전용 Service 주소다.
CHUNKER_STATUS_URL = os.environ.get(
    "CHUNKER_STATUS_URL", "http://insurance-chunker-metrics:9101/status")
# S3 대신 로컬 디렉터리에서 원본을 찾는다(로컬 개발용). 파일 해시를 doc_hash와 맞춘다.
PDF_DIR = os.environ.get("VIEWER_PDF_DIR") or None
# 표 원문을 S3 대신 찾을 로컬 디렉터리. ingest.py가 S3 없이 돌면 여기에 쓴다.
TABLE_DIR = Path(os.environ.get("VIEWER_TABLE_DIR", ".table_cache"))
CACHE_DIR = Path(os.environ.get("VIEWER_CACHE_DIR", "/tmp/chunk-viewer"))
CACHE_MAX_PDFS = int(os.environ.get("VIEWER_CACHE_MAX_PDFS", "40"))
# 약관 본문은 글씨가 작다 — 이미지 확대(전체 화면)로 읽을 수 있을 만큼 그린다.
RENDER_DPI = int(os.environ.get("VIEWER_RENDER_DPI", "144"))
STATUS_REFRESH_S = int(os.environ.get("VIEWER_STATUS_REFRESH_SECONDS", "10"))
