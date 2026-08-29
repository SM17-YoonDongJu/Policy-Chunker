"""로깅 설정 — 평문/JSON을 한 곳에서 고른다.

팀 관측 스택은 Alloy가 컨테이너 stdout을 JSON으로 파싱해 level을 라벨로 승격한다
(backend는 Boot 4 네이티브 구조화 로깅). 평문이면 그 단계가 통째로 안 먹어서 Loki에서
level="error" 필터가 안 된다.

라벨 정책은 팀 규약을 그대로 따른다 — 저카디널리티만 라벨로 올라가고 문서명·sha256 같은
값은 본문에만 남는다. 그래서 이 포매터는 고정 필드(service·level·logger)와 가변 컨텍스트를
같은 평면에 두되, 라벨로 승격할 것은 Alloy 쪽 설정이 고른다.

환경변수:
  LOG_FORMAT   json | text. 기본 text(로컬에서 사람이 읽기 좋게).
               운영은 compose가 json으로 준다.
  LOG_LEVEL    기본 INFO.
  LOG_SERVICE  JSON의 service 필드. 기본 insurance-chunker.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
_TEXT_DATEFMT = "%H:%M:%S"

# LogRecord의 표준 속성 — 이 밖의 것만 사용자가 extra=로 넘긴 컨텍스트로 본다.
_STD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "asctime", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    """한 줄 = JSON 객체 하나. Alloy가 파싱해 level을 라벨로 올린다."""

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # ISO8601 UTC — 컨테이너 TZ(Asia/Seoul)와 무관하게 한 가지로 고정한다.
            # 로그 저장소가 시간대를 추측하지 않게 하는 게 목적.
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # logger.info("...", extra={"phase": "embed", ...}) 로 넘긴 값을 그대로 싣는다.
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                payload[k] = v

        # 직렬화 불가 값이 하나 섞였다고 로그 줄을 통째로 잃지 않게 default=str.
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(service: str | None = None) -> None:
    """루트 로거를 설정한다. 각 진입점(CLI·데몬)이 맨 위에서 한 번 부른다.

    logging.basicConfig는 핸들러가 이미 있으면 조용히 아무것도 안 하므로, 여기서는
    핸들러를 직접 갈아끼워 어느 진입점에서 불려도 같은 결과가 나오게 한다.
    """
    fmt = os.environ.get("LOG_FORMAT", "text").strip().lower()
    level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
    svc = service or os.environ.get("LOG_SERVICE", "insurance-chunker")

    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(svc) if fmt == "json"
        else logging.Formatter(_TEXT_FORMAT, datefmt=_TEXT_DATEFMT)
    )

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)
