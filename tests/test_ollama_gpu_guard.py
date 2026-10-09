"""deploy/ollama/ollama-gpu-guard.sh — CPU 폴백 감지와 재시작.

이 가드는 장애가 났을 때만 동작하는 코드다. 평소에는 "GPU 정상"만 찍고 끝나므로,
정작 필요한 순간에 처음 돌아보게 된다. remote-deploy.sh와 같은 방식으로 docker를
가짜로 갈아끼워 그 경로를 미리 밟아둔다.

가짜 docker는 상태를 파일로 들고 있다.
  state/status        docker inspect .State.Status
  state/started_at    docker inspect .State.StartedAt
  state/logs          docker logs 출력
  state/restarts_log  docker restart가 불린 기록
"""
from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "ollama" / "ollama-gpu-guard.sh"

# 운영 Ollama가 실제로 찍는 줄. 형식이 바뀌면 가드가 무력해지므로 원문 그대로 둔다.
_GPU_LINE = (
    'time=2026-10-09T07:32:49.631Z level=INFO source=types.go:32 msg="inference compute" '
    'id=0 filter_id=0 library=CUDA compute=7.5 name=CUDA0 description="Tesla T4" '
    'libdirs=ollama,cuda_v13 driver=13.2 pci_id=0000:00:1e.0 type=discrete '
    'total="14.6 GiB" available="14.5 GiB"'
)
_CPU_LINE = (
    'time=2026-10-09T07:32:09.084Z level=INFO source=types.go:50 msg="inference compute" '
    'id=cpu library=cpu compute="" name=cpu description=cpu libdirs=ollama driver="" '
    'pci_id="" type="" total="15.4 GiB" available="14.9 GiB"'
)

_FAKE_DOCKER = r"""#!/usr/bin/env bash
S="$STATE_DIR"
case "$1" in
  inspect)
    [ -f "$S/missing" ] && exit 1
    case "$*" in
      *State.Status*)    cat "$S/status" ;;
      *State.StartedAt*) cat "$S/started_at" ;;
    esac
    exit 0 ;;
  logs)
    # --since를 안 쓰면 이전 기동의 CUDA 줄을 보고 "정상"이라 속는다. 쓰는지 기록한다.
    for a in "$@"; do [ "$a" = "--since" ] && echo used > "$S/since_used"; done
    echo called >> "$S/logs_calls"
    cat "$S/logs"
    exit 0 ;;
  restart)
    echo "$2" >> "$S/restarts_log"
    exit "$(cat "$S/restart_rc" 2>/dev/null || echo 0)" ;;
esac
exit 0
"""

# journald가 없는 환경에서도 돌아야 한다.
_FAKE_LOGGER = "#!/usr/bin/env bash\nexit 0\n"

# 가드는 GNU date(-d)를 쓴다. 개발 머신이 BSD date여도 운영과 같은 경로를 밟게 한다.
_FAKE_DATE = r"""#!/usr/bin/env bash
if [ "${1:-}" = "-d" ]; then
  python3 -c '
import sys, datetime
s = sys.argv[1].replace("Z", "+00:00")
try:
    print(int(datetime.datetime.fromisoformat(s).timestamp()))
except Exception:
    sys.exit(1)
' "$2"
else
  /bin/date "$@"
fi
"""


@pytest.fixture
def guard(tmp_path):
    """가짜 호스트. run(logs=..., **상태) → CompletedProcess"""
    state = tmp_path / "state"
    state.mkdir()
    guard_state = tmp_path / "guard_state"
    guard_state.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # 가짜 docker의 상태 경로는 여기서 박아 넣는다. 가드도 STATE_DIR를 쓰기 때문에
    # 환경변수로 넘기면 둘이 같은 디렉터리를 보게 된다.
    for name, body in (("docker", _FAKE_DOCKER.replace('S="$STATE_DIR"', f'S="{state}"')),
                       ("logger", _FAKE_LOGGER), ("date", _FAKE_DATE)):
        f = bin_dir / name
        f.write_text(body, encoding="utf-8")
        f.chmod(0o755)

    def _run(*, logs="", status="running", started_ago_s=600, started_at=None,
             counter=0, restart_rc=0, missing=False, max_restarts=3, verdict=None):
        (state / "status").write_text(status)
        if started_at is None:
            started_at = (datetime.now(UTC) - timedelta(seconds=started_ago_s)) \
                .strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        (state / "started_at").write_text(started_at)
        (state / "logs").write_text(logs)
        (state / "restarts_log").write_text("")
        (state / "logs_calls").write_text("")
        (state / "restart_rc").write_text(str(restart_rc))
        if missing:
            (state / "missing").touch()

        (guard_state / "consecutive_restarts").write_text(str(counter))
        if verdict is not None:
            (guard_state / "verdict").write_text(verdict)

        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STATE_DIR": str(guard_state),
            "CONTAINER": "brbs-ollama",
            "MAX_RESTARTS": str(max_restarts),
        }
        proc = subprocess.run(["bash", str(_SCRIPT)], env=env, capture_output=True,
                              text=True, timeout=60)
        proc.restarts = (state / "restarts_log").read_text().split()  # type: ignore[attr-defined]
        proc.counter = int((guard_state / "consecutive_restarts").read_text())  # type: ignore[attr-defined]
        proc.since_used = (state / "since_used").exists()  # type: ignore[attr-defined]
        proc.log_reads = len((state / "logs_calls").read_text().split())  # type: ignore[attr-defined]
        vf = guard_state / "verdict"
        proc.verdict = vf.read_text().strip() if vf.exists() else None  # type: ignore[attr-defined]
        return proc

    return _run


def test_gpu_mode_does_nothing(guard):
    """정상일 때 건드리면 안 된다 — 가드가 스스로 장애를 만드는 게 최악이다."""
    p = guard(logs=_GPU_LINE)
    assert p.returncode == 0
    assert p.restarts == []


def test_gpu_mode_resets_the_counter(guard):
    """한 번 흔들렸다 복구되면 카운터가 남아선 안 된다. 다음 장애에서 상한이 일찍 걸린다."""
    p = guard(logs=_GPU_LINE, counter=2)
    assert p.counter == 0


def test_cpu_mode_restarts_the_container(guard):
    p = guard(logs=_CPU_LINE)
    assert p.restarts == ["brbs-ollama"]
    assert p.counter == 1
    assert p.returncode == 0


def test_restart_stops_at_the_limit(guard):
    """재시작으로 안 고쳐지는 상황(드라이버 사망)에서 무한 재시작은 더 나쁘다."""
    p = guard(logs=_CPU_LINE, counter=3, max_restarts=3)
    assert p.restarts == []
    assert p.returncode == 1, "systemd에 실패로 남아야 사람이 본다"


def test_young_container_without_logs_waits(guard):
    """기동 직후엔 디스커버리 로그가 아직 없다. 그걸 CPU 모드로 오판하면 재시작 루프가 된다."""
    p = guard(logs="", started_ago_s=10)
    assert p.restarts == []
    assert p.returncode == 0


def test_old_container_without_logs_warns(guard):
    """로그 형식이 바뀌면 가드가 조용히 무력해진다. 그 사실만큼은 남겨야 한다."""
    p = guard(logs="", started_ago_s=99999)
    assert p.restarts == []
    assert "로그 형식 변경 의심" in p.stdout


def test_unparsable_start_time_holds(guard):
    """시각을 못 읽어도 죽지 않고 보류한다 — 죽으면 CPU 모드를 영영 못 잡는다."""
    p = guard(logs="", started_at="not-a-date")
    assert p.returncode == 0
    assert p.restarts == []
    assert "판정 보류" in p.stdout


def test_stopped_container_is_skipped(guard):
    p = guard(logs=_CPU_LINE, status="exited")
    assert p.restarts == []
    assert p.returncode == 0


def test_missing_container_is_skipped(guard):
    """ollama가 아예 없는 호스트에서도 타이머가 돈다. 거기서 시끄러우면 안 된다."""
    p = guard(logs=_CPU_LINE, missing=True)
    assert p.restarts == []
    assert p.returncode == 0


def test_failed_restart_is_reported(guard):
    p = guard(logs=_CPU_LINE, restart_rc=1)
    assert p.returncode == 1


def test_only_current_boot_logs_are_read(guard):
    """--since 없이 읽으면 이전 기동의 CUDA 줄을 보고 '정상'이라 속는다."""
    p = guard(logs=_CPU_LINE)
    assert p.since_used, "docker logs에 --since를 넘겨야 한다"


def test_mixed_devices_count_as_healthy(guard):
    """CPU와 CUDA가 함께 찍히는 경우 — CUDA가 있으면 GPU를 쓰고 있는 것이다."""
    p = guard(logs=f"{_CPU_LINE}\n{_GPU_LINE}")
    assert p.restarts == []


# ── 판정 캐시 ────────────────────────────────────────────────────────────────
# brbs-ollama의 json 로그는 로테이션이 없어 수백 MB까지 자란다(실측 633MB/430만 줄).
# docker logs --since는 파일 앞에서부터 훑으므로 한 번에 10초가 넘는다. 1분 타이머가
# 그걸 매번 반복하면 가드 자신이 호스트 부하가 된다.

_STARTED = "2026-10-09T07:32:36.523603206Z"


def test_healthy_verdict_is_cached(guard):
    p = guard(logs=_GPU_LINE, started_at=_STARTED)
    assert p.verdict == f"{_STARTED} CUDA"


def test_cached_verdict_skips_the_log_scan(guard):
    """같은 기동이면 로그를 다시 읽지 않는다 — 디바이스 결정은 기동 때 한 번뿐이다."""
    p = guard(logs=_GPU_LINE, started_at=_STARTED, verdict=f"{_STARTED} CUDA")
    assert p.log_reads == 0
    assert p.returncode == 0


def test_restarted_container_is_judged_again(guard):
    """재기동하면 StartedAt이 바뀐다. 캐시를 그대로 믿으면 새 기동의 CPU 폴백을 놓친다."""
    p = guard(logs=_CPU_LINE, started_at="2026-10-09T09:00:00.000000000Z",
              verdict=f"{_STARTED} CUDA")
    assert p.log_reads == 1
    assert p.restarts == ["brbs-ollama"]


def test_rotated_away_logs_keep_the_cached_verdict(guard):
    """로그가 로테이션돼 기동 줄이 잘려도, 이미 확인한 기동이면 판정을 잃지 않는다."""
    p = guard(logs="", started_at=_STARTED, verdict=f"{_STARTED} CUDA")
    assert p.returncode == 0
    assert "로그 형식 변경 의심" not in p.stdout
