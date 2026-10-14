# insurance-chunker — 인덱싱 데몬 이미지.
# worker.py가 상주하며 주기(INGEST_INTERVAL_SECONDS)마다 인덱싱한다. 운영에서는 brbs-etl
# 노드의 Deployment(insurance-chunker, ns default)로 뜬다.
# 일회성 실행이 필요하면 도는 파드에서 커맨드를 덮어쓴다:
#   kubectl exec deploy/insurance-chunker -- \
#     python ingest.py --pdf /data/약관.pdf --insurer 메리츠화재 --product "단체안심생활보험"
#
# base 의존만 설치한다: OCR(surya-ocr)·ST(sentence-transformers) extra는 무겁고 선택적이라 제외.
#
# 표 추출 VLM과 임베딩은 이미지에 백엔드를 넣지 않고 클러스터의 vLLM·ollama Service를
# HTTP로 쓴다 — OpenAI 호환 /v1/chat/completions 한 번이라 추가 의존이 없다. 주소는 차트가
# VLM_URL·OLLAMA_URL로 넣고(K8s-manifests), 코드 기본값은 로컬 개발용 localhost다.
# VLM을 끄려면 VLM_BACKEND=off.
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
COPY runlog.py notify.py healthcheck.py metrics.py logging_setup.py exporter.py slo.py ./
# 정지 신호를 문서 경계까지 전달하는 공용 스위치 — worker와 두 CLI가 함께 쓴다.
COPY shutdown.py ./
# 벤치마크 정의와 러너. 클러스터에서 같은 이미지로 고정 문서셋을 돌린다
# (BENCH_DATA=공유 PVC 경로, BENCH_VERSION=이미지 태그). 데이터는 이미지에 넣지 않는다.
COPY eval/bench/bench.py eval/bench/README.md ./eval/bench/
COPY eval/bench/docsets ./eval/bench/docsets
COPY eval/bench/variants ./eval/bench/variants
# Prometheus 스크랩 포트(METRICS_PORT). 클러스터에서는 메트릭 전용 Service(ClusterIP)만
# 이 포트를 열고, ServiceMonitor가 그 Service를 긁는다 — 노드 포트로는 뜨지 않는다.
EXPOSE 9101

# base 의존 + 패키지 설치. setuptools packages.find가 insurance_chunker/db를 포함한다.
RUN pip install .

# 비루트 실행. uid를 1000으로 못박는다 — /data는 chunker-state PVC가 마운트되는 자리이고
# 쓰기 권한이 uid에 달려 있어서, 여기가 바뀌면 /data/state에 이력을 못 써 로컬 폴백으로
# 떨어진다(runlog.py). 폴백이 뜨면 이력이 파드와 함께 사라진다.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

# 상시 데몬(worker.py) — 주기마다 ingest_many + rebuild_search_terms를 돌린다.
# SIGTERM을 받으면 진행 중인 문서까지만 하고 멈춘다 — worker가 신호를 CLI에 전달하고
# CLI가 문서 경계에서 접는다. 유예는 파드의 terminationGracePeriodSeconds가 정한다.
#
# 아래 HEALTHCHECK는 로컬 docker 실행에서만 의미가 있다 — k8s는 이미지의 HEALTHCHECK를
# 읽지 않고, 차트는 probe를 일부러 두지 않는다(배치 워커라 대기 구간이 길어서 liveness를
# 걸면 적재 중에 재시작된다). 클러스터에서 좀비를 잡는 경로는 ServiceMonitor가 수집하는
# insurance_chunker_last_success_timestamp_seconds 알림이다.
HEALTHCHECK --interval=5m --timeout=10s --start-period=1m --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

CMD ["python", "worker.py"]
