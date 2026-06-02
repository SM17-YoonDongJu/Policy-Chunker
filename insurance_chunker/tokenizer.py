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


def _get_kiwi():
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
