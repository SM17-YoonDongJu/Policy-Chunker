"""컨테이너 헬스체크 — 데몬이 '살아 있지만 매 주기 실패하는' 좀비 상태를 잡는다.

프로세스 생존만으로는 부족하다. worker.py는 사이클이 실패해도 예외를 삼키고 다음 주기를
기다리므로(데몬 생존 우선), 인덱싱이 몇 주째 안 되고 있어도 컨테이너는 running이다.
그래서 '마지막 성공 적재 시각'을 건강 기준으로 삼는다.

판정:
  마지막 성공이 있으면       now - last_success_at < 주기 x GRACE  이면 healthy
  아직 성공이 없으면(첫 기동) now - started_at      < 주기 x GRACE  이면 healthy (첫 사이클 유예)
  상태 파일 자체가 없으면    unhealthy (데몬이 안 떴거나 상태 디렉터리가 깨졌다)

환경변수:
  INGEST_INTERVAL_SECONDS  주기(초). worker.py와 같은 값을 본다.
  HEALTH_GRACE_FACTOR      허용 배수. 기본 1.5 (한 주기를 놓치는 정도는 봐주고, 두 주기는 아니다).

사용: docker-compose.prod.yml의 healthcheck가 `python /app/healthcheck.py`로 부른다.
"""
from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

import runlog


def main() -> int:
    interval = int(os.environ.get("INGEST_INTERVAL_SECONDS", "604800"))
    grace = float(os.environ.get("HEALTH_GRACE_FACTOR", "1.5"))
    limit = interval * grace

    state = runlog.daemon_state()
    if not state:
        print("unhealthy: 상태 파일 없음 — 데몬이 기동 이력을 남기지 않았다")
        return 1

    ref_key = "last_success_at" if state.get("last_success_at") else "started_at"
    ref = state.get(ref_key)
    if not ref:
        print("unhealthy: 기동 시각도 성공 시각도 없다")
        return 1

    age = (datetime.now(UTC) - datetime.fromisoformat(ref)).total_seconds()
    label = "마지막 성공" if ref_key == "last_success_at" else "기동(첫 사이클 대기)"
    if age < limit:
        print(f"healthy: {label} {ref} ({age / 3600:.1f}h 경과, 한도 {limit / 3600:.1f}h)")
        return 0
    print(f"unhealthy: {label} {ref} ({age / 3600:.1f}h 경과 > 한도 {limit / 3600:.1f}h) "
          f"— 인덱싱이 주기 안에 성공하지 못하고 있다")
    return 1


if __name__ == "__main__":
    sys.exit(main())
