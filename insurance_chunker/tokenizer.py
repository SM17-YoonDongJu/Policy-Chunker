"""한국어 형태소 분석 (kiwipiepy) → pg_trgm 검색용 토큰 문자열.
출처: rag/pipeline/tokenizer.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "경우", "경우에", "따라", "따른", "대한", "관한", "관련", "있는", "되는",
    "하는", "하여", "하고", "하면", "하거나", "이상", "이하", "이내", "이후",
    "등", "및", "또는", "또한", "다만", "단", "단지",
}
_KEEP_TAGS = ("NN", "VV", "VA", "SN", "SL", "XR")

_kiwi = None
_kiwi_available: bool | None = None  # None=미확인, True=사용가능, False=불가


def _check_kiwi_subprocess() -> bool:
    """subprocess로 Kiwi() 초기화를 격리 테스트 — C 레벨 exit 오염 방지."""
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-c",
             "from kiwipiepy import Kiwi; Kiwi().tokenize('테스트'); print('ok')"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0 and r.stdout.strip() == "ok"
    except Exception:
        return False


def _get_kiwi():
    global _kiwi_available
    if _kiwi_available is False:
        return None
    if _kiwi_available is None:
        _kiwi_available = _check_kiwi_subprocess()
        if not _kiwi_available:
            logger.warning("kiwipiepy 사용 불가 (Python 버전 비호환 등) — raw text 폴백")
            return None
    try:
        from kiwipiepy import Kiwi  # type: ignore
        return Kiwi()
    except Exception:
        return None


def tokenize_korean(text: str) -> str:
    global _kiwi
    if not text:
        return ""
    if _kiwi is None:
        _kiwi = _get_kiwi()
    if _kiwi is None:
        return text
    try:
        tokens = _kiwi.tokenize(text)
        words = [
            t.form for t in tokens
            if any(t.tag.startswith(tag) for tag in _KEEP_TAGS)
            and t.form not in _STOPWORDS
            and len(t.form) > 1
        ]
        return " ".join(words)
    except Exception as e:
        logger.debug(f"형태소 분석 실패: {e}")
        return text
