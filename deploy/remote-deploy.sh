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
#   SSM_PREFIX       시크릿 파라미터 경로 접두사. 비면 SSM 동기화를 건너뛴다
#                    (예: /brbs/insurance-chunker/dev)
set -euo pipefail

: "${EC2_DIR:?}" "${REGISTRY:?}" "${REGION:?}" "${COMPOSE_SERVICE:?}" "${CONTAINER:?}" "${IMAGE_TAG:?}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-90}"

cd "$EC2_DIR"

log() { echo "[deploy] $*"; }

current_tag() {
  # 지금 돌고 있는 컨테이너의 이미지 태그. 없으면 빈 문자열(첫 배포).
  docker inspect --format '{{.Config.Image}}' "$CONTAINER" 2>/dev/null | sed 's/.*://' || true
}

upsert_env() {
  # .env의 키 하나를 덮거나 추가한다.
  #
  # sed로 치환하지 않는다 — 값에 |나 &가 있으면(DB 비밀번호에 충분히 있을 수 있다)
  # 치환식이 깨지거나 엉뚱한 값이 들어간다. 해당 줄을 지우고 다시 붙이는 게 안전하다.
  # 그 대가로 키가 파일 끝으로 밀려 주석과 떨어지지만, 호스트 .env는 배포가 관리하는
  # 파생물이므로 감수한다(원본 주석은 deploy/.env.example에 있다).
  local k="$1" v="$2" tmp
  tmp=$(mktemp)
  grep -v "^${k}=" .env > "$tmp" 2>/dev/null || true
  printf '%s=%s\n' "$k" "$v" >> "$tmp"
  cat "$tmp" > .env
  rm -f "$tmp"
}

sync_secrets() {
  # SSM Parameter Store(SecureString)의 값을 .env로 내려받는다.
  #
  # 여기서 노리는 건 "디스크에서 평문을 없애는 것"이 아니다 — compose가 env_file로 읽어야
  # 하는 이상 컨테이너 기동 시점에 평문이 필요하다. 노리는 건 그 앞단이다.
  #   · 사람이 호스트에 비밀번호를 손으로 넣지 않는다
  #   · 로테이션이 "SSM 값 변경 + 재배포"로 끝난다
  #   · 누가 언제 읽었는지 CloudTrail에 남는다
  #   · 호스트를 다시 만들어도 시크릿이 자동으로 복구된다
  #
  # 파라미터가 없으면 기존 .env 값을 그대로 둔다. 아직 SSM에 안 넣은 환경에서 배포가
  # 깨지지 않게 하려는 것 — 마이그레이션을 한 번에 안 해도 된다.
  [ -z "${SSM_PREFIX:-}" ] && { log "SSM_PREFIX 미설정 — 시크릿 동기화 건너뜀"; return 0; }

  local keys=(DATABASE_URL DISCORD_WEBHOOK_INGEST S3_BUCKET)
  local names=() k
  for k in "${keys[@]}"; do names+=("${SSM_PREFIX}/${k}"); done

  local found=0 missing=0
  # 값은 절대 로그로 내보내지 않는다 — SSM 커맨드 출력은 Actions 로그로 올라간다.
  for k in "${keys[@]}"; do
    local v
    v=$(aws ssm get-parameter --name "${SSM_PREFIX}/${k}" --with-decryption \
          --region "$REGION" --query 'Parameter.Value' --output text 2>/dev/null) || v=""
    if [ -n "$v" ] && [ "$v" != "None" ]; then
      upsert_env "$k" "$v"
      found=$((found + 1))
    else
      missing=$((missing + 1))
      log "SSM에 ${k} 없음 — 기존 .env 값 유지"
    fi
  done
  chmod 600 .env
  log "시크릿 동기화: ${found}건 갱신, ${missing}건 유지 (경로 ${SSM_PREFIX})"
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
    #
    # 파이프로 grep -q에 넘기지 않는다. grep -q는 첫 매치에서 바로 끝내며 파이프를 닫는데,
    # 로그가 크면 아직 쓰고 있던 docker logs가 SIGPIPE로 죽고 set -o pipefail이 그걸
    # 파이프라인 실패로 올린다 — 매치에 성공해도 종료코드 141이 된다. 실제로 이것 때문에
    # 정상 배포가 "검증 시간 초과"로 롤백됐다.
    local logs
    logs=$(docker logs "$CONTAINER" 2>&1) || true
    case "$logs" in
      *"인덱싱 데몬 시작"*)
        log "데몬 기동 확인 (${waited}s)"
        return 0 ;;
    esac
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

sync_secrets

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

roll "$IMAGE_TAG"

if verify; then
  # 배포된 태그를 .env에 남긴다 — 호스트에서 사람이 `docker compose up -d`를 쳐도 같은
  # 이미지가 뜨게 하려는 것. 이게 없으면 수동 실행이 :latest로 되돌아가 드리프트가 생긴다.
  upsert_env IMAGE_TAG "$IMAGE_TAG"
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
