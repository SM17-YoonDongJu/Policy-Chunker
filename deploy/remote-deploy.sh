#!/usr/bin/env bash
# EC2에서 실행되는 배포 스크립트. deploy.yml이 SSM으로 이 파일을 보내고 실행한다.
#
# 인라인 heredoc 대신 파일로 둔 이유: SSM 커맨드는 러너 셸 → base64 → 호스트 셸을 거치는데,
# 인라인이면 호스트에서 평가돼야 할 $VAR를 전부 \$VAR로 이스케이프해야 해서 조금만 길어져도
# 어디서 전개되는지 읽을 수 없게 된다. 파일이면 그 층이 사라지고 리뷰·테스트도 된다.
#
# 하는 일: pull → up -d → 검증 → 실패 시 직전 태그로 롤백.
#
# 환경변수(호출측이 export):
#   EC2_DIR          compose·.env 위치
#   REGISTRY         ECR 레지스트리
#   REGION           AWS 리전
#   COMPOSE_SERVICE  compose 서비스명
#   CONTAINER        컨테이너명
#   IMAGE_TAG        배포할 태그
#   VERIFY_TIMEOUT   검증 대기 상한(초). 기본 90
set -euo pipefail

: "${EC2_DIR:?}" "${REGISTRY:?}" "${REGION:?}" "${COMPOSE_SERVICE:?}" "${CONTAINER:?}" "${IMAGE_TAG:?}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-90}"

cd "$EC2_DIR"

log() { echo "[deploy] $*"; }

current_tag() {
  # 지금 돌고 있는 컨테이너의 이미지 태그. 없으면 빈 문자열(첫 배포).
  docker inspect --format '{{.Config.Image}}' "$CONTAINER" 2>/dev/null | sed 's/.*://' || true
}

roll() {
  export AWS_ECR_REGISTRY="$REGISTRY" IMAGE_TAG="$1"
  docker compose pull "$COMPOSE_SERVICE"
  # --force-recreate: 태그만 같고 내용이 바뀐 :latest에서도 확실히 새 이미지로 뜨게 한다.
  docker compose up -d --no-deps --force-recreate "$COMPOSE_SERVICE"
}

verify() {
  # 컨테이너가 뜬 것만으로는 부족하다. worker.py가 main에 진입해 로그를 남기기 전에
  # 죽는 경우(.env 오류로 DB 연결 실패 등)가 실제 실패 모드다.
  local waited=0
  while [ "$waited" -lt "$VERIFY_TIMEOUT" ]; do
    sleep 5
    waited=$((waited + 5))

    local state restarts
    state=$(docker inspect --format '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
    restarts=$(docker inspect --format '{{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo 0)

    if [ "$restarts" -gt 0 ]; then
      log "크래시루프 감지 — 재시작 ${restarts}회"
      return 1
    fi
    if [ "$state" != "running" ]; then
      log "상태 ${state} (${waited}s) — 대기"
      continue
    fi
    # 데몬이 실제로 main에 들어갔는지. 평문·JSON 로그 어느 쪽이든 이 문구가 있다.
    if docker logs "$CONTAINER" 2>&1 | grep -q "인덱싱 데몬 시작"; then
      log "데몬 기동 확인 (${waited}s)"
      return 0
    fi
    log "기동 대기 중 (${waited}s)"
  done

  log "검증 시간 초과 ${VERIFY_TIMEOUT}s — 데몬 기동 로그를 못 봤다"
  return 1
}

report() {
  log "--- 컨테이너 ---"
  docker ps --filter "name=$CONTAINER" --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
  log "--- 최근 로그 ---"
  docker logs --tail 30 "$CONTAINER" 2>&1 || true
  # /metrics는 있으면 좋은 신호지 필수는 아니다(METRICS_PORT=0으로 끌 수 있다).
  local port="${METRICS_PORT:-9101}"
  if curl -sf --max-time 3 "http://127.0.0.1:${port}/metrics" >/dev/null 2>&1; then
    log "/metrics 응답 OK (:${port})"
  else
    log "/metrics 응답 없음 (:${port}) — 비활성이거나 아직 뜨지 않았다"
  fi
}

PREV_TAG=$(current_tag)
log "현재 태그: ${PREV_TAG:-(없음)} → 배포 태그: ${IMAGE_TAG}"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

roll "$IMAGE_TAG"

if verify; then
  # 배포된 태그를 .env에 남긴다 — 호스트에서 사람이 `docker compose up -d`를 쳐도 같은
  # 이미지가 뜨게 하려는 것. 이게 없으면 수동 실행이 :latest로 되돌아가 드리프트가 생긴다.
  if grep -q '^IMAGE_TAG=' .env 2>/dev/null; then
    # sed -i는 GNU와 BSD 문법이 달라 이식성이 없다. 그리고 .env에는 DB 비밀번호가 있어
    # 제자리 편집 중에 죽으면 곤란하다 — 임시 파일에 쓰고 내용만 덮는다(mv가 아니라
    # cat > 인 이유는 원본의 소유자·권한을 그대로 두기 위해서다).
    tmp=$(mktemp)
    sed "s|^IMAGE_TAG=.*|IMAGE_TAG=${IMAGE_TAG}|" .env > "$tmp" && cat "$tmp" > .env
    rm -f "$tmp"
  else
    echo "IMAGE_TAG=${IMAGE_TAG}" >> .env
  fi
  docker image prune -f >/dev/null 2>&1 || true
  report
  log "배포 완료 — ${IMAGE_TAG}"
  exit 0
fi

log "배포 검증 실패 — 롤백을 시도한다"
report

if [ -z "$PREV_TAG" ] || [ "$PREV_TAG" = "$IMAGE_TAG" ]; then
  # 첫 배포이거나 같은 태그(:latest 재배포)면 되돌릴 지점이 없다. :latest는 방금 덮였으므로
  # 이전 이미지가 로컬에 남아 있어도 어느 것인지 특정할 수 없다 → 사람이 판단해야 한다.
  log "롤백 대상 없음 (이전=${PREV_TAG:-없음}, 배포=${IMAGE_TAG}) — 수동 확인 필요"
  exit 1
fi

log "롤백: ${IMAGE_TAG} → ${PREV_TAG}"
if roll "$PREV_TAG" && verify; then
  log "롤백 완료 — ${PREV_TAG}로 복구됐다. 배포는 실패로 보고한다"
  exit 1
fi

log "롤백까지 실패 — 서비스가 내려가 있을 수 있다. 즉시 확인 필요"
exit 1
