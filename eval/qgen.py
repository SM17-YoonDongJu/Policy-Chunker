"""골든셋 질의 자동 생성기 — 청크 → 현실 질의 4형태 → 자동 골드 라벨 → 품질 필터.

기존 `eval/questions_30327.jsonl`(사람이 만든 47문항)의 한계를 푼다:
  - 유형당 6~9문항 → 1문항이 11~17%p를 좌우 (IMPROVEMENT_LOG 백로그 #34)
  - 문서 1건 전용, 확장하려면 사람이 다시 라벨링
  - "사용자 발화" 한 모양만 평가 — 실제 `db/search.py`에 들어오는 질의는
    랭그래프가 가공한 것이다

## 랭그래프 질의 4형태 (`variants`)

파트너 레포(SM17-YoonDongJu/AI)가 로컬에 없어 `db/search.py`가 구현한 계약
시그니처와 `contracts.md §5` 기준으로 **추정**한 모양이다. 실제 그래프 코드 확인 후
`_REPORT_TEMPLATES` / followup 프롬프트를 맞춰야 한다.

  utterance  챗봇 1턴 원문 발화 (구어·상황서술·조사생략) — 현재 평가가 재는 유일한 모양
  rewritten  rewrite 노드 통과 후 약관 용어 질의 — `query_rewrite_eval.py`의 검증된
             프롬프트를 그대로 재사용하므로 실제 노드 산출과 동일 분포
  followup   멀티턴 후속 발화 (대명사·생략) + 직전 턴. 그래프가 히스토리 압축 없이
             마지막 발화만 넘길 때의 최악 케이스를 잰다
  report     report_worker의 템플릿 질의 (말투 없는 키워드 나열) — 사람 발화와 어휘
             분포가 가장 다르다

## 골드 라벨 자동화

사람이 라벨링하던 `(section, article)` 쌍을 **정답 문장(answer_span) 역추적**으로
자동 생성한다. 생성 LLM이 청크 원문에서 정답 문장을 그대로 복사하게 하고, 그 문장을
포함하는 **모든** 청크의 (section, article)을 골드로 삼는다. "전쟁·폭동" 문구가 24개
특약에 verbatim 복제된 Q12형 문제(검색으로 변별 불가)가 라벨 단계에서 자동으로
multi-gold + `ambiguity=high` 태그로 드러난다.

## 필터 (통과 못 하면 폐기, 사유는 리포트에 집계)

  F1 span_anchor  정답 문장이 원문에 실제로 존재 (환각 질문 제거)
  F2 self_ref     "제3조", "이 조항", "위 표" 등 약관을 본 사람의 말투 제거
  F3 leak         질문이 원문을 14자 이상 그대로 복사 → BM25 공짜 정답 방지
  F4 keyword      lenient 판정용 키워드가 코퍼스에서 충분히 변별적(df ≤ 5%)
  F5 reachable    골드 청크가 BM25 또는 임베딩 top-50에 없으면 라벨 오류로 간주
  F6 dup          생성 질문 간 / 기존 질문셋과 코사인 0.93 이상 중복
  F7 judge        LLM 심판 2-way — 골드 청크로 답 가능(yes) & 하드네거티브로 불가(no)

## 사용법

    # 파일럿 (10문항, 5분)
    .venv/bin/python eval/qgen.py --n 10 --tag pilot
    # 본 생성
    .venv/bin/python eval/qgen.py --n 150 --tag v6
    # 다른 문서
    .venv/bin/python eval/qgen.py --chunks eval/chunks_삼성실손.jsonl --n 80 --tag silson
    # 생성 결과로 검색 평가 (variants 중 하나를 질의로)
    EVAL_QUESTIONS=eval/questions_gen_v6.jsonl .venv/bin/python eval/retrieval_eval.py --embed --run
    EVAL_QUESTIONS=eval/questions_gen_v6.jsonl EVAL_VARIANT=rewritten ... --run

중단해도 `eval/qgen_cache/`에 단계별로 append되므로 재실행하면 이어받는다.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from insurance_chunker.tokenizer import tokenize_korean  # noqa: E402
from insurance_chunker.embedder import QUERY_INSTRUCT  # noqa: E402
from eval.retrieval_eval import BM25, load_jsonl  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CACHE_DIR = EVAL_DIR / "qgen_cache"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
GEN_MODEL = os.environ.get("QGEN_MODEL", "gemma4:26b-a4b-it-qat")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")

MIN_BODY_CHARS = 60      # 정보밀도 하한 (본문 한글·숫자 기준)
LEAK_LCS_MAX = 14        # 질문↔원문 최장 공통부분문자열 상한
KEYWORD_DF_RATIO = 0.015  # 키워드 문서빈도 상한 (코퍼스 대비). lenient 판정이 같은
                          # section 안에서 무조건 참이 되는 걸 막는 값 — 파일럿에서
                          # '결손'(df=34)만 걸리고 변별력 있는 어구(df≤8)는 통과
DUP_COS = 0.93
CAND_K = 50              # reachable 판정 후보 폭
AMBIGUITY_HIGH = 8       # 골드 section이 이 수 이상이면 변별 불가로 태깅
# 하드네거티브를 골드로 흡수할 본문 유사도 하한. "보험금의 지급사유" 같은 조 제목은
# 거의 모든 특약이 공유하므로 제목 일치만으로는 무관한 특약이 골드에 섞인다
# (실측: 진짜 형제 조항 0.81~0.94 / 무관한 특약 0.21 — 0.8에서 깨끗이 갈린다).
ABSORB_SIM = 0.8
# 생성 프롬프트 판 번호 — 캐시 파일명에 들어간다. 프롬프트를 고치면 반드시 올릴 것
# (안 올리면 재실행 시 옛 프롬프트로 만든 질문을 그대로 이어받아 조용히 섞인다).
PROMPT_VERSION = 2


# ---------------------------------------------------------------- ollama

def chat(system: str, user: str, *, schema: dict | None = None,
         temperature: float = 0.2, num_predict: int = 600, retries: int = 3):
    """gemma4는 thinking 모델 — think:false 필수 (안 끄면 content가 빈 문자열)."""
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if schema:
        payload["format"] = schema
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
            r.raise_for_status()
            out = r.json()["message"]["content"].strip()
            if not schema:
                return out
            return json.loads(re.sub(r"^```(?:json)?|```$", "", out, flags=re.M).strip())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [경고] LLM 호출 실패: {last}")
    return None


def embed(texts: list[str]) -> list[list[float]]:
    """qwen3-embedding 러너가 장문·대량 입력에서 크래시 — 32건/1800자로 잘라 호출."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 32):
        batch = [(t[:1800] or " ") for t in texts[i:i + 32]]
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": batch}, timeout=300)
        r.raise_for_status()
        out += r.json()["embeddings"]
    return out


# ---------------------------------------------------------------- 텍스트 유틸

# [헤더]·조 제목·별표 제목 등 본문 앞에 반복되는 줄
_HEADER_PAT = re.compile(r"^(\[.*\]|제\s*\d+\s*조\([^)]*\)|【[^】]*】.*|별표\s*\d+\s*.*)$")
_NORM_PAT = re.compile(r"[\s　]+")


def body_of(content: str) -> str:
    """content = "보험사 | 상품 | 섹션 | 조" + [헤더] + 조 제목 + 본문 → 본문만."""
    lines = content.split("\n")
    # 첫 줄만 프리픽스 후보. markdown 표 행("| a | b |")과 구분하려고 시작 '|'는 제외
    i = 1 if lines and " | " in lines[0] and not lines[0].lstrip().startswith("|") else 0
    while i < len(lines) and _HEADER_PAT.match(lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip() or content


def norm(s: str) -> str:
    return _NORM_PAT.sub("", s)


def product_of(chunks: list[dict]) -> str:
    """프리픽스 "보험사 | 상품 | 섹션 | 조"에서 상품명. 표 청크(줄 시작이 '|')는 건너뛴다."""
    for c in chunks[:50]:
        head = c["content"].split("\n", 1)[0]
        if " | " in head and not head.lstrip().startswith("|"):
            parts = [p.strip() for p in head.split("|")]
            if len(parts) >= 2:
                return parts[1]
    return ""


def hangul_len(s: str) -> int:
    return sum(1 for c in s if "가" <= c <= "힣" or c.isdigit())


def body_ratio(a: dict, b: dict) -> float:
    """두 청크 본문의 유사도 — 형제 특약에 복제된 같은 조항인지 판정용."""
    return difflib.SequenceMatcher(None, norm(body_of(a["content"])),
                                   norm(body_of(b["content"])), autojunk=False).ratio()


def lcs_len(a: str, b: str) -> int:
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(
        0, len(a), 0, len(b))
    return m.size


def snap_span(span: str, content: str) -> str | None:
    """LLM이 살짝 바꿔 쓴 정답 문장을 원문 실제 부분문자열로 스냅. 없으면 None."""
    ns, nc = norm(span), norm(content)
    if len(ns) < 12:
        return None
    if ns in nc:
        return ns
    m = difflib.SequenceMatcher(None, ns, nc, autojunk=False).find_longest_match(
        0, len(ns), 0, len(nc))
    if m.size >= max(12, int(len(ns) * 0.7)):
        return nc[m.b:m.b + m.size]
    return None


# ---------------------------------------------------------------- 후보 청크

SECTION_KINDS = ("보통약관", "특별약관", "별표")


def section_kind(sec: str) -> str:
    if sec.startswith("별표") or "분류표" in sec:
        return "별표"
    if "특별약관" in sec or "특약" in sec:
        return "특별약관"
    return "보통약관"


def pick_candidates(chunks: list[dict], n: int, seed: int) -> list[dict]:
    """중복 문구 그룹 축약 → (chunk_type × section종류) 층화 라운드로빈 샘플링."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        if c.get("is_boilerplate"):
            continue
        b = body_of(c["content"])
        if hangul_len(b) < MIN_BODY_CHARS:
            continue
        groups[norm(b)[:400]].append(c)

    rng = random.Random(seed)
    reps = []
    for _, members in groups.items():
        rep = dict(members[0])
        rep["_dup_group"] = len(members)
        rep["_dup_pairs"] = sorted({(m["section"], m.get("article")) for m in members})
        reps.append(rep)

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for c in reps:
        strata[(c["chunk_type"], section_kind(c["section"]))].append(c)
    for v in strata.values():
        rng.shuffle(v)

    keys = sorted(strata, key=lambda k: -len(strata[k]))
    picked, i = [], 0
    while len(picked) < n and any(strata[k] for k in keys):
        k = keys[i % len(keys)]
        if strata[k]:
            picked.append(strata[k].pop())
        i += 1
    return picked


# ---------------------------------------------------------------- 생성

_PERSONAS = [
    ("계약자 본인", "반말 섞인 구어, 사고 상황부터 서술, 조사 생략 잦음"),
    ("피보험자 가족", "존댓말, '애가/아버지가' 같은 3인칭, 감정 섞임"),
    ("설계사·모집인", "존댓말, 약관 용어를 일부 정확히 씀, 지급 조건을 확정하려는 어투"),
    ("보험금 청구 담당자", "서류·기한·절차 중심의 짧은 실무 질문"),
]

_GEN_SYSTEM = """당신은 보험사 챗봇 로그를 만드는 시나리오 작가입니다.
주어진 약관 조항을 근거로, 그 조항을 **읽어본 적 없는** 사람이 챗봇에 실제로 칠 법한
질문을 만듭니다.

절대 규칙:
1. 질문자는 약관을 보지 않았다. "제3조", "이 조항", "위 표", "별표" 같은 말은 절대 쓰지 않는다.
2. 조항 문장을 그대로 베끼지 않는다. 일상어로 바꿔 묻는다.
   (예: "피보험자가 보험기간 중 상해의 직접결과로써 사망한 경우" → "다치고 나서 얼마 뒤에 죽어도 나와요?")
3. 답이 반드시 이 조항 안에 있어야 한다. 조항에 없는 내용을 묻지 않는다.
4. 질문은 **15~45자 한 문장**. 실제 채팅처럼 짧고 불완전해도 된다. 배경 설명을 길게 붙이지 않는다.
   (실제 로그 평균이 28자다 — 길게 쓰면 사람이 안 쓰는 문어체가 되고 검색 난이도가 왜곡된다.)
5. answer_span은 **조항 원문에서 그대로 복사한 한 문장**(20자 이상). 요약·의역 금지.
6. keywords는 정답 판정에 쓸 변별력 있는 어구 1~3개를 **원문 표기 그대로**. "보험금", "회사"
   같은 흔한 말 말고 그 조항을 특정하는 말로.

qtype은 다음 중 하나: coverage(지급사유), exclusion(면책·부지급), procedure(절차·서류·통지),
definition(용어 정의), cancel_refund(해지·환급·실효), table(표·지급률 조회), duty(알릴의무)."""

_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "qtype": {"type": "string"},
                    "question": {"type": "string"},
                    "answer_span": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["qtype", "question", "answer_span", "keywords"],
            },
        }
    },
    "required": ["questions"],
}


def gen_for_chunk(c: dict, persona: tuple[str, str], n_q: int) -> list[dict]:
    user = (f"[상품] {c['section']}\n"
            f"[조항] {c.get('article') or ''} {c.get('article_title') or ''}\n"
            f"[조항 원문]\n{body_of(c['content'])[:2000]}\n\n"
            f"질문자 유형: {persona[0]} — {persona[1]}\n"
            f"이 조항으로 답할 수 있는 질문 {n_q}개를 JSON으로 만드세요.")
    out = chat(_GEN_SYSTEM, user, schema=_GEN_SCHEMA, temperature=0.8, num_predict=700)
    if not out or not isinstance(out.get("questions"), list):
        return []
    return [q for q in out["questions"] if isinstance(q, dict) and q.get("question")]


# ---------------------------------------------------------------- 필터

_SELF_REF = re.compile(r"제\s*\d+\s*조|제\s*\d+\s*항|이\s*조항|위\s*표|본\s*약관|별표|위\s*조항|해당\s*조항")

try:  # 라우터 가드와 동일 목록 — 걸리는 질의는 운영에서 빈 결과가 나오므로 평가 대상 밖
    from db.search import NON_BODILY_HINTS  # noqa: E402
except Exception:  # asyncpg 등 미설치 환경 폴백 (db/search.py와 동기 유지 필요)
    NON_BODILY_HINTS = ("자동차", "자차", "차량", "화재", "재물", "배상책임",
                        "해상", "운송", "항공", "선박")


def cheap_filters(q: dict, c: dict) -> str | None:
    """LLM/임베딩 없이 거를 수 있는 것부터. 통과하면 None, 아니면 폐기 사유."""
    text = q["question"].strip()
    if not (10 <= len(text) <= 65):
        return "length"
    if _SELF_REF.search(text):
        return "self_ref"
    if any(h in text for h in NON_BODILY_HINTS):
        return "oos_guard"
    span = snap_span(q.get("answer_span", ""), c["content"])
    if not span:
        return "span_anchor"
    q["_span"] = span
    if lcs_len(norm(text), norm(body_of(c["content"]))) >= LEAK_LCS_MAX:
        return "leak"
    return None


def keyword_filter(q: dict, c: dict, df: Counter, n_docs: int) -> str | None:
    cap = max(2, int(n_docs * KEYWORD_DF_RATIO))
    kws = [k.strip() for k in q.get("keywords", []) if isinstance(k, str)]
    good = [k for k in kws
            if 2 <= len(k) <= 40 and k in c["content"] and df[k] <= cap]
    if not good:
        return "keyword"
    q["keywords"] = sorted(good, key=lambda k: df[k])[:2]  # 변별력 높은 것부터
    return None


def expand_gold(span: str, chunks: list[dict], norms: list[str], src: dict) -> list[dict]:
    """정답 문장을 포함하는 모든 청크의 (section, article) — verbatim 복제 자동 처리."""
    pairs = {(src["section"], src.get("article"))}
    for c, nc in zip(chunks, norms):
        if span in nc:
            pairs.add((c["section"], c.get("article")))
    return [{"section": s, "article": a} for s, a in sorted(pairs, key=lambda p: (p[0], p[1] or ""))]


# ---------------------------------------------------------------- 랭그래프 변형

_RW_SYSTEM = """당신은 보험 약관 검색 질의 변환기입니다. 사용자의 일상 질문을 약관 조문에 쓰이는 공식 용어로 바꿔 검색 질의 한 문장을 만드세요.

규칙:
- 구어를 약관 용어로: "넘어져 다쳤다"→"상해", "돈 나와?"→"보험금 지급사유", "안 나와?"→"보험금을 지급하지 않는 사유"
- 핵심 명사를 유지하고, 예상되는 조문 제목 표현을 포함
- 설명 없이 변환된 질의 한 줄만 출력"""

_FOLLOWUP_SYSTEM = """당신은 보험 챗봇 대화 로그를 만듭니다. 주어진 질문 하나를 **2턴 대화**로 쪼개세요.
- prev: 사용자가 먼저 던지는 넓은 첫 발화. 상황·가입 상품만 말하고 **정작 궁금한 조건은 아직 안 묻는다.**
- query: 그 다음 턴의 후속 발화. 앞 맥락(상품명·사고 경위)은 생략하고 "그럼", "그거" 같은 지시어를 쓴다.
  단, **핵심 명사 한 개는 반드시 남긴다** (예: "그럼 깁스만 해도 나와요?"의 '깁스').
  명사가 하나도 없는 "그거 되나요?" 류는 금지 — 검색 신호가 아예 없어져 로그로도 무의미하다.
JSON만 출력."""

_FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {"prev": {"type": "string"}, "query": {"type": "string"}},
    "required": ["prev", "query"],
}

_REPORT_INTENT = {
    "coverage": "보험금 지급사유 및 지급금액",
    "exclusion": "보험금을 지급하지 않는 사유",
    "procedure": "보험금 청구 절차 및 구비서류",
    "definition": "용어의 정의",
    "cancel_refund": "계약 해지 및 해약환급금",
    "table": "지급률 및 지급표",
    "duty": "계약 전후 알릴 의무",
}


def make_variants(q: dict, c: dict, product: str) -> dict:
    rw = chat(_RW_SYSTEM, q["question"], temperature=0.0, num_predict=80)
    fu = chat(_FOLLOWUP_SYSTEM, q["question"], schema=_FOLLOWUP_SCHEMA,
              temperature=0.7, num_predict=200)
    intent = _REPORT_INTENT.get(q["qtype"], "보험금 지급사유")
    report = f"{product} {c['section']} {intent}"
    return {
        "utterance": q["question"],
        "rewritten": (rw or q["question"]).split("\n")[0].strip(),
        "followup": fu if isinstance(fu, dict) else {"prev": "", "query": q["question"]},
        "report": report,
    }


# ---------------------------------------------------------------- 심판

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"answerable": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["answerable", "reason"],
}
_JUDGE_SYSTEM = """주어진 약관 조항 **하나만** 보고 질문에 답할 수 있는지 판정합니다.
조항에 답의 근거가 직접 있으면 answerable=true. 관련 주제일 뿐 답이 없거나, 다른 조항을
더 봐야 하면 false. 엄격하게 판정하세요. reason은 15자 이내."""


def judge(question: str, chunk_content: str) -> bool | None:
    out = chat(_JUDGE_SYSTEM,
               f"[조항]\n{chunk_content[:1800]}\n\n[질문] {question}",
               schema=_JUDGE_SCHEMA, temperature=0.0, num_predict=120)
    return None if not out else bool(out.get("answerable"))


# ---------------------------------------------------------------- 캐시

def cached_stage(path: Path, keys: list, fn):
    """단계별 append 캐시 — 중단 후 재실행하면 완료분을 건너뛴다."""
    done = {}
    if path.exists():
        for r in load_jsonl(path):
            done[r["key"]] = r
    todo = [k for k in keys if str(k) not in done]
    if todo:
        with path.open("a") as f:
            for i, k in enumerate(todo, 1):
                rec = fn(k)
                rec = {"key": str(k), **rec}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[str(k)] = rec
                if i % 5 == 0 or i == len(todo):
                    print(f"  {path.name}: {i}/{len(todo)}", flush=True)
    return done


# ---------------------------------------------------------------- 메인

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(EVAL_DIR / "chunks_30327_v6.jsonl"))
    ap.add_argument("--n", type=int, default=60, help="목표 질문 수(필터 통과 기준)")
    ap.add_argument("--per-chunk", type=int, default=2)
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--existing", default=str(EVAL_DIR / "questions_30327.jsonl"),
                    help="중복 제거 대상 기존 질문셋 (없으면 빈 문자열)")
    ap.add_argument("--emb-cache", default=str(EVAL_DIR / "emb_cache_v6.jsonl"),
                    help="청크 임베딩 캐시 (chunk_index 키). 다른 문서면 새 경로 지정")
    ap.add_argument("--no-judge", action="store_true", help="F7 LLM 심판 생략(빠른 파일럿)")
    a = ap.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)
    chunks = load_jsonl(Path(a.chunks))
    product = product_of(chunks)
    norms = [norm(c["content"]) for c in chunks]
    by_index = {c["chunk_index"]: c for c in chunks}
    n_docs = len(chunks)
    print(f"코퍼스: {Path(a.chunks).name} {n_docs}청크 / 상품 '{product}'")

    # 후보 청크 — 목표의 1.6배를 뽑아 폐기분을 흡수
    need_chunks = max(8, int(a.n * 1.6 / max(1, a.per_chunk)))
    cands = pick_candidates(chunks, need_chunks, a.seed)
    print(f"후보 청크 {len(cands)}개 (중복문구 축약·층화 샘플링)")

    # 1) 생성
    print("\n[1/5] 질문 생성")
    rng = random.Random(a.seed)
    persona_of = {c["chunk_index"]: _PERSONAS[rng.randrange(len(_PERSONAS))] for c in cands}
    by_idx = {c["chunk_index"]: c for c in cands}
    raw = cached_stage(
        CACHE_DIR / f"{a.tag}_raw_p{PROMPT_VERSION}.jsonl",
        [c["chunk_index"] for c in cands],
        lambda idx: {"persona": persona_of[idx][0],
                     "questions": gen_for_chunk(by_idx[idx], persona_of[idx], a.per_chunk)},
    )

    # 2) 저비용 필터
    print("\n[2/5] 규칙 필터")
    df = Counter()
    for r in raw.values():
        for q in r["questions"]:
            for k in q.get("keywords", []):
                if isinstance(k, str) and k.strip():
                    df[k.strip()] = sum(1 for nc in norms if norm(k) in nc)
    rejects = Counter()
    kept = []
    for idx_s, r in raw.items():
        c = by_idx[int(idx_s)]
        for q in r["questions"]:
            reason = cheap_filters(q, c) or keyword_filter(q, c, df, n_docs)
            if reason:
                rejects[reason] += 1
                continue
            q["_src"] = c
            q["_persona"] = r["persona"]
            kept.append(q)
    print(f"  통과 {len(kept)} / 폐기 {sum(rejects.values())} {dict(rejects)}")

    # 3) 도달성·난이도·중복 (BM25 + 임베딩)
    print("\n[3/5] 도달성·난이도·중복")
    bm25 = BM25([tokenize_korean(c["content"]).split() for c in chunks])
    # 청크 임베딩은 chunk_index로 키잉된 기존 캐시(retrieval_eval.py 산출)를 그대로 재사용
    cpath = Path(a.emb_cache) if a.emb_cache else EVAL_DIR / f"emb_cache_qgen_{a.tag}.jsonl"
    cache = {r["key"]: r["vec"] for r in load_jsonl(cpath)} if cpath.exists() else {}
    missing = [c for c in chunks if str(c["chunk_index"]) not in cache]
    if missing:
        print(f"  청크 임베딩 {len(missing)}건 생성 → {cpath.name}")
        with cpath.open("a") as f:
            for i in range(0, len(missing), 32):
                batch = missing[i:i + 32]
                for c, v in zip(batch, embed([c["content"] for c in batch])):
                    cache[str(c["chunk_index"])] = v
                    f.write(json.dumps({"key": str(c["chunk_index"]), "vec": v}) + "\n")
                f.flush()
                print(f"    {min(i + 32, len(missing))}/{len(missing)}", flush=True)
    cmat = np.array([cache[str(c["chunk_index"])] for c in chunks])
    cmat = cmat / np.linalg.norm(cmat, axis=1, keepdims=True)

    qvecs = np.array(embed([QUERY_INSTRUCT + q["question"] for q in kept])) if kept else np.zeros((0, 1))
    qvecs = qvecs / np.linalg.norm(qvecs, axis=1, keepdims=True)

    seen_vecs = []
    if a.existing and Path(a.existing).exists():
        ex = load_jsonl(Path(a.existing))
        ev = np.array(embed([QUERY_INSTRUCT + e["question"] for e in ex]))
        seen_vecs = list(ev / np.linalg.norm(ev, axis=1, keepdims=True))

    staged = []
    for q, qv in zip(kept, qvecs):
        gold = expand_gold(q["_span"], chunks, norms, q["_src"])
        gold_idx = {i for i, c in enumerate(chunks)
                    if any(c["section"] == g["section"] and c.get("article") == g["article"]
                           for g in gold)}
        bm_rank = list(np.argsort(-bm25.scores(tokenize_korean(q["question"]).split())))
        em_rank = list(np.argsort(-(cmat @ qv)))
        r_bm = next((i for i, d in enumerate(bm_rank[:CAND_K]) if d in gold_idx), None)
        r_em = next((i for i, d in enumerate(em_rank[:CAND_K]) if d in gold_idx), None)
        if r_bm is None and r_em is None:
            rejects["unreachable"] += 1
            continue
        if any(float(qv @ s) >= DUP_COS for s in seen_vecs):
            rejects["dup"] += 1
            continue
        seen_vecs.append(qv)
        best = min(x for x in (r_bm, r_em) if x is not None)
        n_sec = len({g["section"] for g in gold})
        q["_gold"] = gold
        q["_diag"] = {
            "bm25_rank": r_bm, "embed_rank": r_em,
            "difficulty": "easy" if best < 3 else ("medium" if best < 10 else "hard"),
            "gold_pairs": len(gold), "gold_sections": n_sec,
            "ambiguity": "high" if n_sec >= AMBIGUITY_HIGH else ("mid" if n_sec > 1 else "low"),
            "dup_group": q["_src"].get("_dup_group", 1),
            "leak_lcs": lcs_len(norm(q["question"]), norm(body_of(q["_src"]["content"]))),
        }
        # 하드 네거티브 = BM25 상위 중 골드가 아닌 첫 청크
        q["_hardneg"] = next((chunks[d] for d in bm_rank[:20] if d not in gold_idx), None)
        staged.append(q)
    print(f"  통과 {len(staged)} / 누적 폐기 {dict(rejects)}")

    # 4) LLM 심판 (2-way)
    print("\n[4/5] LLM 심판")
    final = []
    if a.no_judge:
        final = staged[:a.n]
    else:
        jpath = CACHE_DIR / f"{a.tag}_judge.jsonl"
        jdone = {r["key"]: r for r in load_jsonl(jpath)} if jpath.exists() else {}
        with jpath.open("a") as f:
            for i, q in enumerate(staged, 1):
                key = f"{q['_src']['chunk_index']}::{q['question'][:40]}"
                if key not in jdone:
                    hn = q["_hardneg"]
                    rec = {"key": key,
                           "pos": judge(q["question"], q["_src"]["content"]),
                           "neg": judge(q["question"], hn["content"]) if hn else False,
                           "hardneg": {"chunk_index": hn["chunk_index"], "section": hn["section"],
                                       "article": hn.get("article"),
                                       "article_title": hn.get("article_title")} if hn else None}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    jdone[key] = rec
                if i % 5 == 0 or i == len(staged):
                    print(f"  judge {i}/{len(staged)}", flush=True)
                r = jdone[key]
                if r["pos"] is not True:
                    rejects["judge_pos"] += 1
                    continue
                if r["neg"] is True:
                    # 하드네거티브로도 답이 된다 → 두 갈래다.
                    #  ① 형제 특약에 **같은 조항이 통째로 복제**된 경우: 그 청크도 진짜 정답이므로
                    #     골드에 흡수한다(버리면 어려운 케이스만 골라 빠져 셋이 조용히 쉬워진다).
                    #     제목 일치만으로는 부족 — 조 제목은 특약끼리 거의 다 같다.
                    #  ② 그 외: 질문이 조항을 특정하지 못한 것 → 폐기.
                    hn = r.get("hardneg") or {}
                    title = q["_src"].get("article_title")
                    hn_chunk = by_index.get(hn.get("chunk_index"))
                    sim = body_ratio(q["_src"], hn_chunk) if hn_chunk else 0.0
                    if title and hn.get("article_title") == title and sim >= ABSORB_SIM:
                        pair = {"section": hn["section"], "article": hn.get("article")}
                        if pair not in q["_gold"]:
                            q["_gold"].append(pair)
                            q["_gold"].sort(key=lambda g: (g["section"], g["article"] or ""))
                        n_sec = len({g["section"] for g in q["_gold"]})
                        q["_diag"].update(
                            gold_pairs=len(q["_gold"]), gold_sections=n_sec,
                            ambiguity="high" if n_sec >= AMBIGUITY_HIGH else "mid",
                            gold_from_judge=round(sim, 2))
                        rejects["judge_neg_absorbed"] += 1
                    else:
                        rejects["judge_neg"] += 1
                        continue
                final.append(q)
                if len(final) >= a.n:
                    break

    # 5) 랭그래프 변형 + 저장
    print(f"\n[5/5] 랭그래프 질의 변형 ({len(final)}문항)")
    vpath = CACHE_DIR / f"{a.tag}_variants.jsonl"
    vdone = {r["key"]: r["variants"] for r in load_jsonl(vpath)} if vpath.exists() else {}
    out_path = EVAL_DIR / f"questions_gen_{a.tag}.jsonl"
    with vpath.open("a") as vf, out_path.open("w") as out:
        for i, q in enumerate(final, 1):
            key = f"{q['_src']['chunk_index']}::{q['question'][:40]}"
            if key not in vdone:
                vdone[key] = make_variants(q, q["_src"], product)
                vf.write(json.dumps({"key": key, "variants": vdone[key]}, ensure_ascii=False) + "\n")
                vf.flush()
            src, vs = q["_src"], vdone[key]
            # 변형 후에도 라우터 가드에 걸리는 게 있는지 (걸리면 운영에서 빈 결과)
            q["_diag"]["oos_variants"] = [
                n for n, t in (("utterance", vs["utterance"]), ("rewritten", vs["rewritten"]),
                               ("followup", vs["followup"].get("query", "")), ("report", vs["report"]))
                if any(h in t for h in NON_BODILY_HINTS)]
            out.write(json.dumps({
                "qid": f"G{i:03d}",
                "qtype": q["qtype"],
                "question": q["question"],          # = variants.utterance (기존 평가 호환)
                "gold": q["_gold"],
                "keywords": q["keywords"],
                "variants": vs,
                "answer_span": q["_span"],
                "persona": q["_persona"],
                "source": {"chunk_index": src["chunk_index"], "page": src["page"],
                           "section": src["section"], "article": src.get("article"),
                           "article_title": src.get("article_title"),
                           "chunk_type": src["chunk_type"]},
                "diag": q["_diag"],
            }, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(final):
                print(f"  variants {i}/{len(final)}", flush=True)

    write_report(a, final, rejects, out_path, n_docs)


def write_report(a, final: list[dict], rejects: Counter, out_path: Path, n_docs: int) -> None:
    L = [f"# 생성 질의셋 리포트 — {a.tag}", "",
         f"- 코퍼스: `{Path(a.chunks).name}` ({n_docs}청크) / 생성 모델: `{GEN_MODEL}`",
         f"- 산출: `{out_path.name}` **{len(final)}문항** (목표 {a.n})", ""]
    L += ["## 필터 집계", "", "| 사유 | 건수 |", "|---|---|"]
    for k, v in rejects.most_common():
        L.append(f"| {k} | {v} |")
    L += ["", f"> `judge_neg_absorbed`는 폐기가 아니라 **골드 확장** — 형제 특약에 같은 조항이",
          f"> 통째로 복제(본문 유사도 ≥{ABSORB_SIM})돼 하드네거티브도 진짜 정답이었던 경우.", ""]
    L += ["## 유형 분포", "", "| qtype | 건수 |", "|---|---|"]
    for k, v in Counter(q["qtype"] for q in final).most_common():
        L.append(f"| {k} | {v} |")
    L += ["", "## 난이도 (골드 청크의 최상위 랭크)", "", "| 난이도 | 건수 |", "|---|---|"]
    for k, v in Counter(q["_diag"]["difficulty"] for q in final).most_common():
        L.append(f"| {k} | {v} |")
    L += ["", "## 골드 모호성 (정답 문구가 여러 특약에 복제된 정도)", "",
          "| ambiguity | 건수 |", "|---|---|"]
    for k, v in Counter(q["_diag"]["ambiguity"] for q in final).most_common():
        L.append(f"| {k} | {v} |")
    L += ["", f"- 커버 section 수: {len({q['_src']['section'] for q in final})}",
          f"- 평균 질문 길이: {np.mean([len(q['question']) for q in final]):.1f}자" if final else "",
          f"- 평균 leak LCS: {np.mean([q['_diag']['leak_lcs'] for q in final]):.1f}자 (상한 {LEAK_LCS_MAX})" if final else "",
          "", "## 샘플", ""]
    for q in final[:5]:
        L += [f"- **{q['qtype']}** ({q['_persona']}) — {q['question']}",
              f"  - 골드: {q['_gold'][:2]}{' …' if len(q['_gold']) > 2 else ''}",
              f"  - 정답문장: {q['_span'][:60]}…"]
    p = EVAL_DIR / f"qgen_report_{a.tag}.md"
    p.write_text("\n".join(x for x in L if x is not None))
    print("\n" + "\n".join(x for x in L if x is not None))
    print(f"\n저장: {out_path}\n리포트: {p}")


if __name__ == "__main__":
    main()
