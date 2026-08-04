"""chunk_type 분류 A/B 테스트: A.X-4.0-Light vs Qwen3.6-35B-A3B vs Gemma4-26B-A4B.

키워드 분류가 오판하기 쉬운 면책/지급 복합문 위주 샘플로
각 모델의 분류 정확도와 응답 속도를 비교한다.

실행: python eval/classify_ab_test.py [모델명 ...]
      (모델명 생략 시 3종 전부)
"""
from __future__ import annotations

import json
import sys
import time

import requests

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M",
    "qwen3.6:35b-a3b",
    "gemma4:26b-a4b-it-qat",
]

sys.path.insert(0, ".")
from insurance_chunker.llm_classifier import _SYSTEM, _TYPES  # noqa: E402

# (제목, 본문, 정답) — 키워드 분류의 알려진 실패 케이스 + 기본 케이스
SAMPLES = [
    # 면책/지급 복합문 — 키워드 우선순위가 exclusion으로 오판하는 대표 케이스
    ("보험금의 지급사유",
     "회사는 피보험자에게 다음 중 어느 하나의 사유가 발생한 경우에는 보험수익자에게 약정한 "
     "보험금을 지급합니다. 다만, 제5조(보험금을 지급하지 않는 사유)에 해당하는 경우에는 "
     "그러하지 않습니다.",
     "coverage"),
    ("암진단보험금",
     "회사는 보상하지 않는 손해를 제외하고 피보험자가 암보장개시일 이후에 암으로 진단 확정된 "
     "경우 최초 1회에 한하여 암진단보험금을 지급합니다.",
     "coverage"),
    # 순수 면책
    ("보험금을 지급하지 않는 사유",
     "회사는 다음 중 어느 하나의 사유로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 "
     "않습니다. 1. 피보험자가 고의로 자신을 해친 경우. 2. 보험수익자가 고의로 피보험자를 "
     "해친 경우. 3. 계약자가 고의로 피보험자를 해친 경우.",
     "exclusion"),
    # 제목은 절차, 본문은 면책 — 본문 우선 케이스
    ("보험금 지급에 관한 세부규정",
     "전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동으로 인한 손해는 보상하지 않습니다. "
     "또한 핵연료물질 또는 방사능 오염으로 인한 손해도 보상하지 않습니다.",
     "exclusion"),
    # 정의
    ("용어의 정의",
     "이 계약에서 사용되는 용어의 정의는 다음과 같습니다. 1. 계약자: 회사와 계약을 체결하고 "
     "보험료를 납입할 의무를 지는 사람. 2. 피보험자: 보험사고의 대상이 되는 사람.",
     "definition"),
    # 의무
    ("계약 전 알릴 의무",
     "계약자 또는 피보험자는 청약할 때 청약서에서 질문한 사항에 대하여 알고 있는 사실을 "
     "반드시 사실대로 알려야 합니다.",
     "duty"),
    # 청구 절차
    ("보험금의 청구",
     "보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다. 1. 청구서(회사양식) "
     "2. 사고증명서(진단서 등) 3. 신분증.",
     "claim"),
    # 해지
    ("계약의 해지",
     "계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는 "
     "해지환급금을 계약자에게 지급합니다.",
     "termination"),
    # 지급 관련 절차 조항 — claim vs coverage 경계 케이스
    ("보험금 지급절차",
     "회사는 청구서류를 접수한 때에는 접수증을 드리고, 접수일부터 3영업일 이내에 보험금을 "
     "지급합니다.",
     "claim"),
    # coverage 기본
    ("장해보험금의 지급",
     "회사는 피보험자가 보험기간 중 상해로 장해분류표에서 정한 장해지급률 3% 이상에 해당하는 "
     "장해상태가 되었을 때 장해지급률에 해당하는 보험금을 지급합니다.",
     "coverage"),
]


def classify(model: str, title: str, body: str) -> tuple[str | None, float]:
    t0 = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"제목: {title}\n\n{body}"},
                ],
                "format": {
                    "type": "object",
                    "properties": {"chunk_type": {"type": "string", "enum": _TYPES}},
                    "required": ["chunk_type"],
                },
                "stream": False,
                "think": False,  # thinking 모델(qwen3.6/gemma4)용 — 미지원 모델이면 아래서 재시도
                "options": {"temperature": 0, "num_predict": 30},
            },
            timeout=300,
        )
        resp.raise_for_status()
        ct = json.loads(resp.json()["message"]["content"]).get("chunk_type")
        return ct, time.time() - t0
    except Exception as e:
        print(f"    오류: {e}")
        return None, time.time() - t0


def main() -> None:
    models = sys.argv[1:] or MODELS
    available = [
        m["name"] for m in
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).json().get("models", [])
    ]

    results: dict[str, dict] = {}
    for model in models:
        if not any(model.split(":")[0] in a for a in available):
            print(f"\n### {model} — 미설치, 건너뜀")
            continue
        print(f"\n### {model}")
        correct, times = 0, []
        for title, body, expected in SAMPLES:
            got, dt = classify(model, title, body)
            times.append(dt)
            ok = got == expected
            correct += ok
            mark = "O" if ok else "X"
            print(f"  [{mark}] {title[:20]:<22} 정답={expected:<13} 판정={got} ({dt:.1f}s)")
        results[model] = {
            "acc": f"{correct}/{len(SAMPLES)}",
            "avg_s": round(sum(times) / len(times), 1),
        }

    print("\n=== 요약 ===")
    for m, r in results.items():
        print(f"  {m}: 정확도 {r['acc']}, 평균 {r['avg_s']}s/건")


if __name__ == "__main__":
    main()
