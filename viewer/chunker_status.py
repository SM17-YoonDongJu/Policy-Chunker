"""청커의 /status를 읽는다(exporter.status_snapshot). 형식은 tests/test_status.py가 고정한다."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


def fetch(url: str, *, limit: int = 30, timeout: float = 3.0) -> dict[str, Any]:
    sep = "&" if "?" in url else "?"
    with urllib.request.urlopen(f"{url}{sep}limit={limit}", timeout=timeout) as r:  # noqa: S310
        return json.load(r)
