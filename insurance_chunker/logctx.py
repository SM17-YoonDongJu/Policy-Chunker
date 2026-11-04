"""로그 문맥 — 지금 처리 중인 문서와 표를 그 안에서 찍히는 로그 줄마다 싣는다.

VLM 호출 로그(event=vlm_page)는 쪽 번호만 남겼다. 그래서 Loki에서 타임아웃 줄을 찾아도 어느
보험사의 어느 약관, 그 쪽의 몇째 표인지 알 수 없었다. 문서는 적재 루프(ingest_catalog)가,
표는 VLM 단계가 여기에 묶어 둔다. 그러면 logging_setup의 필터가 그 안에서 찍힌 줄에 필드로
붙인다. extra로 직접 넘긴 값이 있으면 그쪽이 이긴다.

contextvars라 스레드마다 따로다. ThreadPoolExecutor는 문맥을 넘기지 않으므로, 작업을 넘길 때
contextvars.copy_context().run으로 감싼다(vlm_stage). 문서 병렬의 프로세스 풀 워커는 자기 안에서
묶는다(_process_one).
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any, Iterator, Optional

_fields: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "log_fields", default=None)


def current() -> dict[str, Any]:
    """지금 문맥에 묶인 필드. 고치지 말 것 — 묶을 때마다 새 dict를 만든다."""
    return _fields.get() or {}


@contextmanager
def bind(**fields: Any) -> Iterator[None]:
    """블록 안에서 찍히는 로그에 fields를 싣는다. None인 값은 싣지 않고, 겹치면 안쪽이 이긴다."""
    token = _fields.set({**current(), **{k: v for k, v in fields.items() if v is not None}})
    try:
        yield
    finally:
        _fields.reset(token)
