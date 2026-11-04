# insurance-chunker — 인덱싱 데몬 이미지.
# worker.py가 상주하며 주기(INGEST_INTERVAL_SECONDS)마다 인덱싱한다. 운영에서는 brbs-etl
# 노드의 Deployment(insurance-chunker, ns default)로 뜬다.
# 일회성 실행이 필요하면 도는 파드에서 커맨드를 덮어쓴다:
#   kubectl exec deploy/insurance-chunker -- \
#     python ingest.py --pdf /data/약관.pdf --insurer 메리츠화재 --product "단체안심생활보험"
#
# base + [layout](레이아웃 모델) 의존을 설치한다. OCR(surya-ocr)·ST(sentence-transformers)
# extra는 무겁고 선택적이라 제외.
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

# ca-certificates: Ollama/DB/외부 TLS 검증용.
# libgl1·libglib2.0-0: paddlex가 GUI판 opencv-contrib-python을 버전까지 고정해 끌어오고, 그
# cv2가 import 시점에 libGL.so.1·libgthread를 연다. headless판으로 바꾸면 paddlex의 고정과
# 충돌하므로 라이브러리를 깐다.
# libgomp1: paddle 코어(libpaddle.so)가 OpenMP 런타임을 연다. 모델을 만들 때 로드된다.
# slim에는 셋 다 없다. 빠지면 layout.py가 레이아웃을 끈 채 페이지 전체 경로로 간다 — 적재는
# 계속되므로 로그 한 줄 말고는 드러나지 않는다.
# 그 외 시스템 의존은 wheel로 충족(pymupdf·kiwipiepy·psycopg2-binary·paddleocr 모두 manylinux 휠).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libglib2.0-0 libgomp1 \
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

# base 의존 + 패키지 + 레이아웃 모델 의존 설치. setuptools packages.find가
# insurance_chunker/db를 포함한다.
#
# paddle은 GPU 휠(paddlepaddle-gpu, CUDA 12.6)을 깐다. 레이아웃을 chunker 파드에서 GPU로 돌린다
# (LAYOUT_DEVICE=gpu:0) — T4에서 쪽당 4.77초 → 0.14초, 표 박스는 CPU와 같았다(2026-11-02 smoke).
# CUDA 런타임·cuDNN은 휠이 nvidia-* 의존으로 끌어오므로 CUDA 베이스 이미지는 필요 없고, 드라이버는
# 노드의 nvidia 런타임이 넣는다. 드라이버는 있는데 GPU를 못 열면 layout.py가 CPU로 내려간다.
# paddle 휠은 자체 인덱스에만 있고 nvidia-* 의존은 PyPI에 있어 두 인덱스를 함께 본다.
# [layout]을 먼저 깔면 numpy 등 공통 의존이 PyPI에서 해결된 뒤라 paddle 쪽은 휠과 CUDA 의존만 받는다.
# 주의: CUDA 의존으로 이미지가 GB 단위로 커진다(paddle GPU 휠만 1.9GB).
#
# GPU 휠의 paddle 코어(libpaddle.so)는 드라이버 라이브러리(libcuda.so.1)에 직접 링크돼 있어,
# 드라이버가 없으면 device=cpu로도 import 자체가 안 된다. 그래서 파드는 GPU를 받거나(레이아웃 GPU),
# GPU 없이 CPU로 돌릴 때도 NVIDIA_VISIBLE_DEVICES=none으로 nvidia 런타임이 드라이버 라이브러리만
# 넣게 해야 한다(K8s-manifests의 chunker 차트가 layoutDevice에 따라 둘 중 하나를 넣는다).
# 같은 이유로 빌드 머신(드라이버 없음)에서는 paddle을 import할 수 없다. 빌드 시점 검사는 대신
# paddle 코어의 동적 링크를 본다 — 드라이버 말고 빠진 라이브러리(libgomp·libGL 등)가 있으면
# 런타임에 레이아웃이 조용히 꺼지므로 여기서 빌드를 실패시킨다. paddleocr는 import해도 paddle
# 코어를 열지 않으므로 그대로 import로 본다. 모델은 받지 않는다.
RUN pip install ".[layout]" \
    && pip install paddlepaddle-gpu==3.2.1 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        --extra-index-url https://pypi.org/simple \
    && python -c "import importlib.metadata as m; from paddleocr import LayoutDetection; \
print('paddlepaddle-gpu', m.version('paddlepaddle-gpu'))" \
    && missing=$(ldd /usr/local/lib/python3.12/site-packages/paddle/base/libpaddle.so \
        | awk '/not found/ {print $1}' | grep -vx 'libcuda.so.1' || true) \
    && if [ -n "$missing" ]; then echo "paddle 코어가 못 여는 라이브러리: $missing"; exit 1; fi

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
