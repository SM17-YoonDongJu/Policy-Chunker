"""deploy/remote-deploy.sh 검증·롤백 로직.

배포 스크립트는 실패했을 때만 실행되는 경로(롤백)를 품고 있어서, 정작 필요한 순간에
처음 돌아본다. docker·aws를 가짜로 갈아끼워 그 경로를 미리 밟아둔다.

가짜 docker는 상태를 파일로 들고 있다.
  state/image     지금 떠 있는 컨테이너의 이미지 태그
  state/status    running | missing
  state/restarts  RestartCount
  state/logs      docker logs 출력
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "remote-deploy.sh"

_FAKE_DOCKER = r"""#!/usr/bin/env bash
S="$STATE_DIR"
case "$1" in
  inspect)
    [ "$(cat "$S/status")" = "missing" ] && exit 1
    case "$3" in
      *Config.Image*)  echo "registry/repo:$(cat "$S/image")" ;;
      *State.Status*)  cat "$S/status" ;;
      *RestartCount*)  cat "$S/restarts" ;;
    esac
    exit 0 ;;
  compose)
    if [ "$2" = "up" ]; then
      # up 횟수를 센다. 1회차 = 새 배포, 2회차 = 롤백.
      n=$(( $(cat "$S/ups" 2>/dev/null || echo 0) + 1 ))
      echo "$n" > "$S/ups"
      echo "$IMAGE_TAG" > "$S/image"
      echo "$IMAGE_TAG" >> "$S/deployed"
      if [ "$n" -ge 2 ] && [ "${ROLLBACK_OK:-}" = "1" ]; then
        # 롤백은 성공하는 시나리오 — 정상 상태로 되돌린다.
        echo running > "$S/status"; echo 0 > "$S/restarts"; cp "$S/ok_logs" "$S/logs"
      else
        cp "$S/on_up_status" "$S/status"
        cp "$S/on_up_restarts" "$S/restarts"
        cp "$S/on_up_logs" "$S/logs"
      fi
    fi
    exit 0 ;;
  logs)  cat "$S/logs" ;;
  ps)    echo "fake-ps" ;;
  image) exit 0 ;;
  # 실제 docker login은 stdin을 읽는다. 안 읽으면 앞 파이프가 SIGPIPE로 죽어
  # pipefail에 걸린다(테스트만의 문제지만 원인 파악이 오래 걸린다).
  login) cat > /dev/null 2>&1; exit 0 ;;
esac
exit 0
"""

_FAKE_AWS = r"""#!/usr/bin/env bash
# ecr get-login-password / ssm get-parameter 두 가지만 흉내낸다.
if [ "$1" = "ssm" ]; then
  # $4 = /prefix/KEY. state/ssm_<KEY> 파일이 있으면 그 값을, 없으면 실패.
  key="${4##*/}"
  f="$STATE_DIR/ssm_$key"
  [ -f "$f" ] || exit 1
  cat "$f"
  exit 0
fi
echo fake-token
"""
_FAKE_SLEEP = "#!/usr/bin/env bash\nexit 0\n"   # 테스트가 검증 대기로 늘어지지 않게
_FAKE_CURL = "#!/usr/bin/env bash\nexit 1\n"    # /metrics 없음 — 검증에 영향 없어야 한다

_OK_LOG = '{"level":"INFO","message":"인덱싱 데몬 시작 — 주기 604800s"}'
_CRASH_LOG = "Traceback: DB 연결 실패"


@pytest.fixture
def host(tmp_path):
    """가짜 호스트. run(tag, **상태) → CompletedProcess"""
    state = tmp_path / "state"
    state.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", _FAKE_DOCKER), ("aws", _FAKE_AWS),
                       ("sleep", _FAKE_SLEEP), ("curl", _FAKE_CURL)):
        f = bin_dir / name
        f.write_text(body, encoding="utf-8")
        f.chmod(0o755)

    (tmp_path / ".env").write_text("IMAGE_TAG=old111\nDATABASE_URL=x\n", encoding="utf-8")

    def _run(tag: str, *, current="old111", up_status="running",
             up_restarts="0", up_logs=_OK_LOG, rollback_ok=True,
             ssm: dict[str, str] | None = None, ssm_prefix=""):
        (state / "image").write_text(current)
        (state / "status").write_text("running")
        (state / "restarts").write_text("0")
        (state / "logs").write_text(_OK_LOG)
        # up 직후의 상태 — 배포가 성공했을 때/실패했을 때를 여기서 흉내낸다.
        (state / "on_up_status").write_text(up_status)
        (state / "on_up_restarts").write_text(up_restarts)
        (state / "on_up_logs").write_text(up_logs)
        (state / "ok_logs").write_text(_OK_LOG)
        (state / "deployed").write_text("")
        (state / "ups").write_text("0")
        for k, v in (ssm or {}).items():
            (state / f"ssm_{k}").write_text(v)

        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "STATE_DIR": str(state),
            "EC2_DIR": str(tmp_path), "REGISTRY": "registry", "REGION": "ap-northeast-2",
            "COMPOSE_SERVICE": "chunker", "CONTAINER": "brbs-insurance-chunker",
            "IMAGE_TAG": tag, "VERIFY_TIMEOUT": "10",
            "SSM_PREFIX": ssm_prefix,
        }
        if rollback_ok:
            # 롤백이 성공하는 시나리오: 두 번째 up 이후로는 정상 상태로 돌아온다.
            env["ROLLBACK_OK"] = "1"
        proc = subprocess.run(["bash", str(_SCRIPT)], env=env, capture_output=True,
                              text=True, timeout=60)
        proc.state_dir = state          # type: ignore[attr-defined]
        proc.env_file = tmp_path / ".env"  # type: ignore[attr-defined]
        return proc

    return _run


def test_healthy_deploy_succeeds(host):
    p = host("new222")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "데몬 기동 확인" in p.stdout
    assert "배포 완료 — new222" in p.stdout


def test_healthy_deploy_records_tag_in_env(host):
    """호스트에서 사람이 docker compose up을 쳐도 같은 이미지가 뜨게 한다."""
    p = host("new222")
    assert "IMAGE_TAG=new222" in p.env_file.read_text(encoding="utf-8")
    assert "DATABASE_URL=x" in p.env_file.read_text(encoding="utf-8")  # 다른 값은 보존


def test_crash_loop_triggers_rollback(host):
    """컨테이너가 뜬 직후 재시작을 반복하면 배포는 실패다 — docker ps만 보면 놓친다."""
    p = host("bad333", up_restarts="2")
    assert p.returncode == 1
    assert "크래시루프 감지" in p.stdout
    assert "롤백: bad333 → old111" in p.stdout


def test_daemon_never_starts_triggers_rollback(host):
    """프로세스는 살아 있는데 main에 못 들어간 경우(.env 오류 등)."""
    p = host("bad333", up_logs=_CRASH_LOG)
    assert p.returncode == 1
    assert "검증 시간 초과" in p.stdout
    assert "롤백" in p.stdout


def test_failed_deploy_reports_failure_even_when_rollback_succeeds(host):
    """롤백이 됐어도 배포는 실패로 보고해야 한다 — 초록불이 뜨면 아무도 안 본다."""
    p = host("bad333", up_restarts="1")
    assert p.returncode == 1
    assert "배포는 실패로 보고한다" in p.stdout


def test_first_deploy_has_nothing_to_roll_back_to(host):
    """이전 컨테이너가 없으면 되돌릴 지점이 없다 — 사람이 판단해야 한다."""
    p = host("new222", current="", up_restarts="1")
    assert p.returncode == 1
    assert "롤백 대상 없음" in p.stdout


def test_same_tag_redeploy_is_not_rollbackable(host):
    """:latest 재배포처럼 태그가 같으면 이전 이미지를 특정할 수 없다."""
    p = host("old111", up_restarts="1")
    assert p.returncode == 1
    assert "롤백 대상 없음" in p.stdout


def test_metrics_absence_does_not_fail_the_deploy(host):
    """/metrics는 있으면 좋은 신호지 필수가 아니다(METRICS_PORT=0으로 끌 수 있다)."""
    p = host("new222")
    assert p.returncode == 0
    assert "/metrics 응답 없음" in p.stdout


# ── 시크릿 동기화 (#19) ───────────────────────────────────────────────────────

def test_ssm_values_land_in_env(host):
    """사람이 호스트에 비밀번호를 넣지 않아도 되게 하는 게 목적이다."""
    p = host("new222", ssm_prefix="/brbs/insurance-chunker/dev",
             ssm={"DATABASE_URL": "postgresql://u:pw@rds/db", "S3_BUCKET": "brbs-corpus"})
    env = p.env_file.read_text(encoding="utf-8")
    assert "DATABASE_URL=postgresql://u:pw@rds/db" in env
    assert "S3_BUCKET=brbs-corpus" in env
    assert p.returncode == 0


def test_secret_values_never_reach_the_log(host):
    """SSM 커맨드 출력은 Actions 로그로 올라간다 — 값이 섞이면 안 된다."""
    p = host("new222", ssm_prefix="/p", ssm={"DATABASE_URL": "postgresql://u:SUPERSECRET@rds/db"})
    assert "SUPERSECRET" not in p.stdout
    assert "SUPERSECRET" not in p.stderr
    assert "시크릿 동기화: 1건 갱신" in p.stdout


def test_missing_parameter_keeps_the_existing_value(host):
    """아직 SSM에 안 넣은 환경에서 배포가 깨지면 안 된다 — 한 번에 이전할 필요가 없게."""
    p = host("new222", ssm_prefix="/p", ssm={})
    assert p.returncode == 0
    assert "DATABASE_URL=x" in p.env_file.read_text(encoding="utf-8")
    assert "기존 .env 값 유지" in p.stdout


def test_no_prefix_skips_sync_entirely(host):
    """SSM_PREFIX가 비면 아무것도 건드리지 않는다(기존 배포 동작 그대로)."""
    p = host("new222")
    assert p.returncode == 0
    assert "시크릿 동기화 건너뜀" in p.stdout


def test_special_characters_in_password_survive(host):
    """비밀번호에 |나 &가 있으면 sed 치환이 깨진다 — 그래서 sed를 안 쓴다."""
    secret = "postgresql://u:a|b&c$d@rds/db?options=-csearch_path%3Dcorpus,public"
    p = host("new222", ssm_prefix="/p", ssm={"DATABASE_URL": secret})
    assert f"DATABASE_URL={secret}" in p.env_file.read_text(encoding="utf-8")


def test_env_file_is_not_world_readable(host):
    """DB 비밀번호가 담긴 파일이다."""
    p = host("new222", ssm_prefix="/p", ssm={"DATABASE_URL": "x"})
    assert oct(p.env_file.stat().st_mode)[-3:] == "600"
