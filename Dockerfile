# insurance-chunker — 인덱싱 데몬 이미지.
# worker.py가 상주하며 주기(INGEST_INTERVAL_SECONDS)마다 인덱싱한다 → docker ps에 상시 노출.
# 일회성 실행이 필요하면 커맨드를 덮어쓴다:
#   docker compose run --rm chunker \
#     python ingest.py --pdf /data/약관.pdf --insurer 메리츠화재 --product "단체안심생활보험"
#
# base 의존만 설치한다: OCR(surya-ocr, GPU)·ST(sentence-transformers) extra는 무겁고
# 선택적이라 제외. VLM(claude CLI)은 컨테이너에 없다 → 기본 --no-vision 전제로 운용.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ca-certificates: Ollama/DB/외부 TLS 검증용. 그 외 시스템 의존은 wheel로 충족
# (pymupdf·kiwipiepy·psycopg2-binary·pdfplumber 모두 manylinux 휠 제공).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 패키지 코드 + 메타데이터. eval/tests/docs는 런타임 불필요라 이미지에서 제외(.dockerignore).
COPY pyproject.toml README.md ./
COPY insurance_chunker ./insurance_chunker
COPY db ./db
COPY ingest.py ingest_many.py ingest_catalog.py rebuild_search_terms.py worker.py ./

# base 의존 + 패키지 설치. setuptools packages.find가 insurance_chunker/db를 포함한다.
RUN pip install .

# 비루트 실행.
RUN useradd --create-home app && chown -R app:app /app
USER app

# 상시 데몬(worker.py) — 주기마다 ingest_many + rebuild_search_terms를 돌린다.
# SIGTERM에 우아하게 종료. 일회성 실행은 `docker compose run --rm chunker python ingest.py ...`.
CMD ["python", "worker.py"]
