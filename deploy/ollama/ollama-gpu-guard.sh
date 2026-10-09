#!/usr/bin/env bash
# brbs-ollama가 GPU 대신 CPU로 떠 있으면 재시작한다.
#
# 왜 필요한가
#   Ollama는 기동할 때 GPU 디스커버리에 실패해도 에러를 내지 않는다. 그냥 CPU 모드로
#   뜨고, 기동은 성공하고, 로그 레벨은 INFO고, API는 200을 돌려준다.
#
#     msg="failure during llama-server GPU discovery" error="context deadline exceeded"
#     msg="inference compute" id=cpu library=cpu total="15.4 GiB"
#     msg="vram-based default context" total_vram="0 B"
#
#   그 상태로 qwen3-vl:8b를 돌리면 페이지 한 장이 600초를 넘겨 VLM 호출이 전부
#   타임아웃하고, 가중치 12.6GiB가 호스트 RAM에 올라가 15.4GiB짜리 이 호스트를
#   OOM으로 밀어낸다. 2026-10-06~09에 그렇게 38시간을 태우고 호스트가 6시간 죽었다.
#
# 왜 부팅 레이스만 막지 않는가
#   관측된 계기는 "부팅 직후 드라이버가 준비되기 전에 컨테이너가 뜬 것"이지만, OOM으로
#   러너가 죽고 Ollama가 다시 뜨는 경로에서도 똑같이 CPU로 떨어진다. 그래서 기동 계기를
#   따지지 않고 "지금 CPU 모드인가"만 본다.
#
# 판정
#   현재 기동분 로그에 library=CUDA 줄이 있으면 정상.
#   inference compute 줄이 나왔는데 전부 library=cpu면 CPU 모드 — 재시작.
#   아직 아무 줄도 없으면 기동 중 — 판정을 미룬다.
set -uo pipefail

CONTAINER="${CONTAINER:-brbs-ollama}"
GRACE_SECONDS="${GRACE_SECONDS:-120}"   # 기동 후 디스커버리 로그를 기다려 주는 시간
MAX_RESTARTS="${MAX_RESTARTS:-3}"       # 연속 재시작 상한. 드라이버가 정말 죽었으면 멈춰야 한다
STATE_DIR="${STATE_DIR:-/var/lib/ollama-gpu-guard}"
COUNT_FILE="$STATE_DIR/consecutive_restarts"
VERDICT_FILE="$STATE_DIR/verdict"   # "<StartedAt> CUDA" — 기동 1회당 한 번만 판정하기 위한 캐시

log() { logger -t ollama-gpu-guard -- "$*"; echo "$*"; }

mkdir -p "$STATE_DIR"
[ -f "$COUNT_FILE" ] || echo 0 >"$COUNT_FILE"

status=$(docker inspect "$CONTAINER" --format '{{.State.Status}}' 2>/dev/null) || {
  log "컨테이너 $CONTAINER 없음 — 건너뜀"
  exit 0
}
[ "$status" = running ] || { log "상태=$status — 판정 보류"; exit 0; }

started_at=$(docker inspect "$CONTAINER" --format '{{.State.StartedAt}}')
# 파싱 실패를 -1(모름)로 흡수한다. set -u 아래서 빈 값이 그대로 산술식에 들어가면
# 가드가 판정이 아니라 문법 오류로 죽는다 — 그러면 CPU 모드를 영영 못 잡는다.
if started_epoch=$(date -d "$started_at" +%s 2>/dev/null) && [ -n "$started_epoch" ]; then
  age=$(( $(date +%s) - started_epoch ))
else
  age=-1
fi

# 이 기동을 이미 GPU로 확인했으면 로그를 다시 읽지 않는다.
#
# 디바이스 결정은 기동할 때 한 번 일어나고 그 뒤로 바뀌지 않는다. 그런데 brbs-ollama의
# json 로그는 로테이션 설정이 없어 수백 MB까지 자라고, docker logs --since는 파일 앞에서부터
# 훑으므로 한 번 읽는 데 10초가 넘는다. 그걸 매분 반복하면 가드가 코어를 상시로 갉아먹는다.
# 덤으로, 로그가 로테이션돼 기동 줄이 잘려나가도 판정을 잃지 않는다.
cached=$(cat "$VERDICT_FILE" 2>/dev/null || echo "")
if [ "$cached" = "$started_at CUDA" ]; then
  exit 0
fi

# 현재 기동분 로그만 본다 — 이전 기동의 CUDA 줄을 보고 "정상"이라 속으면 안 된다.
compute=$(docker logs "$CONTAINER" --since "$started_at" 2>&1 \
  | grep 'msg="inference compute"' || true)

if [ -z "$compute" ]; then
  if [ "$age" -lt 0 ]; then
    log "기동 시각을 못 읽었다($started_at) — 판정 보류"
  elif [ "$age" -lt "$GRACE_SECONDS" ]; then
    log "기동 ${age}s 경과 — 디스커버리 로그 대기 중"
  else
    # 로그 형식이 바뀌면 이 가드는 조용히 무력해진다. 그 사실만큼은 남긴다.
    log "경고: 기동 ${age}s인데 inference compute 줄이 없다 — Ollama 로그 형식 변경 의심"
  fi
  exit 0
fi

if grep -q 'library=CUDA' <<<"$compute"; then
  printf '%s CUDA\n' "$started_at" >"$VERDICT_FILE"
  prev=$(cat "$COUNT_FILE")
  [ "$prev" != 0 ] && log "GPU 정상 — 재시작 카운터 초기화 (직전 ${prev}회)"
  echo 0 >"$COUNT_FILE"
  log "GPU 정상 확인 (기동 $started_at) — 다음 기동까지 재판정 안 함"
  exit 0
fi

n=$(( $(cat "$COUNT_FILE") + 1 ))
if [ "$n" -gt "$MAX_RESTARTS" ]; then
  # 재시작으로 안 고쳐지는 상황(드라이버 사망, GPU 점유)에서 무한 재시작은 더 나쁘다.
  log "CPU 모드인데 재시작 ${MAX_RESTARTS}회를 넘겼다 — 자동 복구 중단. nvidia-smi와 드라이버를 확인할 것"
  exit 1
fi

echo "$n" >"$COUNT_FILE"
log "CPU 모드 감지 — $CONTAINER 재시작 (${n}/${MAX_RESTARTS})"
if docker restart "$CONTAINER" >/dev/null; then
  log "재시작 완료"
else
  log "재시작 실패"
  exit 1
fi
