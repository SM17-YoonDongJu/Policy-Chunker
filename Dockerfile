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
# 운영 계측: 실행 이력(runlog) · 사이클 알림(notify) · 좀비 판정(healthcheck) · 지표(metrics).
COPY runlog.py notify.py healthcheck.py metrics.py logging_setup.py exporter.py ./
# Prometheus 스크랩 포트(METRICS_PORT). 접근은 보안그룹이 통제한다.
EXPOSE 9101

# base 의존 + 패키지 설치. setuptools packages.find가 insurance_chunker/db를 포함한다.
RUN pip install .

# 비루트 실행. uid를 1000으로 못박는다 — 호스트 ./data가 ubuntu(uid 1000) 소유로
# chown되므로(deploy.yml) 여기가 어긋나면 /data/state에 이력을 못 써 로컬 폴백으로 떨어진다.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# 상시 데몬(worker.py) — 주기마다 ingest_many + rebuild_search_terms를 돌린다.
# SIGTERM에 우아하게 종료. 일회성 실행은 `docker compose run --rm chunker python ingest.py ...`.
# 마지막 인덱싱 성공 시각으로 건강을 판정한다 — 프로세스는 살아 있는데 매 주기 실패하는
# 좀비를 프로세스 생존만으로는 못 잡기 때문. compose의 restart 정책은 unhealthy로 재시작하지
# 않으므로(그건 Swarm 기능) 이건 자가치유가 아니라 `docker ps` STATUS로 드러내는 신호다.
HEALTHCHECK --interval=5m --timeout=10s --start-period=1m --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

CMD ["python", "worker.py"]
