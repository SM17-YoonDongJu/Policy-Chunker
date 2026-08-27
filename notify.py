"""Discord 알림 — 인덱싱 사이클 결과를 사람이 볼 수 있는 곳으로 내보낸다.

여태 Discord 알림은 CI/배포에만 붙어 있었다(.github/actions/discord-notify). 정작 매 주기
도는 인덱싱은 실패해도 컨테이너 로그에만 남았고, 주기가 7일이라 아무도 안 보면 장애 탐지가
최악 7일이었다. 이 모듈이 그 구멍을 메운다.

메시지 규격은 .github/actions/discord-notify와 맞춘다(색상·아이콘·한국어 라벨 동일) —
같은 채널에서 CI/배포 알림과 나란히 읽히게.

환경변수:
  DISCORD_WEBHOOK_INGEST  웹훅 URL. 없으면 조용히 건너뛴다(알림 미설정이 인덱싱을 깨지 않게).
  INGEST_NOTIFY           always | failure. 기본 always. failure면 실패했을 때만 보낸다.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_COLORS = {"success": 3066993, "failure": 15158332, "warning": 16776960}
_ICONS = {"success": "✅", "failure": "❌", "warning": "⚠️"}
_LABELS = {"success": "성공", "failure": "실패", "warning": "경고"}


def _should_send(status: str) -> bool:
    mode = os.environ.get("INGEST_NOTIFY", "always").strip().lower()
    return status != "success" if mode == "failure" else True


def notify(status: str, title: str, fields: dict[str, Any],
           webhook_url: Optional[str] = None) -> bool:
    """Embed 하나를 웹훅으로 보낸다. 성공 여부를 돌려주되 예외는 삼킨다."""
    url = webhook_url or os.environ.get("DISCORD_WEBHOOK_INGEST", "").strip()
    if not url:
        logger.info("Discord 웹훅 미설정 → 알림 건너뜀")
        return False
    if not _should_send(status):
        return False

    payload = {
        "embeds": [{
            "title": f"{_ICONS.get(status, 'ℹ️')} {title} {_LABELS.get(status, '알림')}",
            "color": _COLORS.get(status, 8421504),
            # Discord는 필드당 1024자 상한이 있다 — 넘치면 400으로 통째 거부되므로 미리 자른다.
            "fields": [{"name": k, "value": str(v)[:1024] or "-", "inline": len(str(v)) < 32}
                       for k, v in fields.items()],
        }]
    }
    try:
        import requests
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code >= 400:
            logger.warning(f"Discord 알림 실패 {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:  # noqa: BLE001 - 알림 실패가 인덱싱을 죽이면 안 된다
        logger.warning(f"Discord 알림 실패: {e}")
        return False
