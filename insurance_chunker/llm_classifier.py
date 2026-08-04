"""LLM 기반 chunk_type 분류.

키워드 매칭(_classify)의 오탐 보완용. 로컬 Ollama 서버의 A.X-4.0-Light로
조항 텍스트를 8종 chunk_type 중 하나로 분류한다.

CLASSIFY_BACKEND 환경변수:
  keyword (기본) → 기존 키워드 매칭만 사용
  llm            → Ollama LLM 분류, 실패 시 키워드 폴백

키워드 분류가 확신하는 케이스(제목에 명시적 키워드)는 LLM을 건너뛰고,
본문 키워드로만 판정된 애매한 케이스만 LLM에 보낸다.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

CLASSIFY_BACKEND = os.environ.get("CLASSIFY_BACKEND", "keyword")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
# 골든셋 42건 검증(2026-07-30): gemma4 90% > qwen3.6 83% > A.X 62% > 키워드 48%
CLASSIFY_MODEL = os.environ.get("CLASSIFY_MODEL", "gemma4:26b-a4b-it-qat")

_TIMEOUT = 60
_MAX_CHARS = 2000  # 조항이 길면 앞부분만 — 분류엔 서두가 결정적

_TYPES = [
    "coverage", "exclusion", "definition", "special_clause",
    "duty", "claim", "termination", "schedule", "general",
]

_TYPE_DESC = """\
- coverage: 보험금을 지급하는 사유·보장 내용 (무엇을 얼마나 보장하는지)
- exclusion: 보험금을 지급하지 않는 사유·면책 조항
- definition: 용어의 정의
- special_clause: 특약의 구성·적용 관련 조항
- duty: 계약자·피보험자의 의무 (고지의무, 알릴 의무, 통지 의무)
- claim: 보험금 청구 절차·서류·지급 절차
- termination: 계약 해지·효력상실·소멸
- schedule: 별표·부표·장해분류표 등 표 형식 기준
- general: 위 어디에도 해당하지 않는 일반 조항"""

_SYSTEM = f"""당신은 보험 약관 조항 분류기입니다. 주어진 조항을 아래 유형 중 정확히 하나로 분류하세요.

{_TYPE_DESC}

판단 기준:
- 조항의 핵심 기능으로 판단하세요. 키워드 등장 여부가 아니라 조항이 실제로 무엇을 규정하는지가 기준입니다.
- 예: "보상하지 않는 손해를 제외하고 지급합니다"는 지급 사유를 규정하므로 coverage입니다.
- 제목이 절차여도 본문이 면책 사유 나열이면 exclusion입니다.

JSON으로만 답하세요: {{"chunk_type": "<유형>"}}"""


# 정형 조항은 LLM 없이 확정 — 결정론적이고 모델 편차 없음
_JUYONG = re.compile(r"정하지\s*않은\s*사항.{0,20}(보통약관|약관).{0,10}따[릅른]")


def rule_based(text: str, title: Optional[str] = None) -> Optional[str]:
    """정규식으로 확정 가능한 유형. 해당 없으면 None."""
    if (title and "준용" in title) or _JUYONG.search(text[:300]):
        return "special_clause"
    return None


def llm_available() -> bool:
    """Ollama 서버 + 분류 모델이 준비됐는지 확인."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        return any(CLASSIFY_MODEL.split(":")[0] in m for m in models)
    except Exception:
        return False


def classify_llm(text: str, title: Optional[str] = None) -> Optional[str]:
    """chunk_type 판정: 정규식 규칙 → LLM. 실패 시 None (호출측에서 폴백)."""
    rb = rule_based(text, title)
    if rb:
        return rb

    body = text[:_MAX_CHARS]
    user = f"제목: {title}\n\n{body}" if title else body
    payload = {
        "model": CLASSIFY_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        "format": {
            "type": "object",
            "properties": {"chunk_type": {"type": "string", "enum": _TYPES}},
            "required": ["chunk_type"],
        },
        "stream": False,
        "think": False,  # gemma4/qwen3.6은 thinking 모델 — 안 끄면 content가 빔
        "options": {"temperature": 0, "num_predict": 30},
    }
    try:
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=_TIMEOUT)
        if resp.status_code == 400:  # thinking 미지원 모델이면 think 빼고 재시도
            payload.pop("think", None)
            resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        ct = json.loads(resp.json()["message"]["content"]).get("chunk_type")
        return ct if ct in _TYPES else None
    except Exception as e:
        logger.debug(f"LLM 분류 실패, 키워드 폴백: {e}")
        return None
