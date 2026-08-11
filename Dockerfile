# insurance-chunker — 배치 파이프라인 이미지.
# 상시 데몬이 아니라 CLI(ingest.py / ingest_many.py / rebuild_search_terms.py)를
# 온디맨드로 실행하는 용도다:
#   docker compose --env-file .env.prod run --rm chunker \
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
COPY ingest.py ingest_many.py rebuild_search_terms.py ./

# base 의존 + 패키지 설치. setuptools packages.find가 insurance_chunker/db를 포함한다.
RUN pip install .

# 비루트 실행.
RUN useradd --create-home app && chown -R app:app /app
USER app

# 배치 도구라 상시 커맨드가 없다 — 기본은 사용법 출력. 실제 실행은 `docker compose run`으로 인자 전달.
CMD ["python", "ingest.py", "--help"]
