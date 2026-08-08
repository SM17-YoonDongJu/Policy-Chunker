"""rechunk.py — 경계 라벨 부여 + 조번호 재추출 + 호→조 병합 + InsuranceChunk 출력.

Policy-Chunker의 rechunk.py 뼈대를 유지하되 아래를 보완:
  - 8종 chunk_type 분류 + contextual prefix (rag/ 방식)
  - article_number / article_title 메타데이터
  - 출력: dict → InsuranceChunk
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid as _uuid
from typing import Optional

from .boundaries import Boundary, label_for
from .models import DocMeta, InsuranceChunk, TableMeta

logger = logging.getLogger(__name__)

ROWS_PER_TABLE_CHUNK = 20  # 이 행 수를 초과하면 표를 child 청크로 분할

def _tok(text: str) -> int:
    return int(len(text) * 0.6)  # 글자 수 × 0.6 ≈ Kiwi 형태소 토큰 수 근사

# ── 범용 패턴 (한국 약관 공통) ────────────────────────────────────────────────
FOOTER = re.compile(r"^\s*[-‐–—]\s*\d{1,3}\s*[-‐–—]\s*$")
# 괄호 변형: ()  （）  【】 [] 모두 허용 — DB류는 대괄호 제목을 쓴다
# ("제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지]")
_ART_NUM   = r"제\s*(\d+)\s*조(?:의\s*\d+)?\s*"
# 제목 내 중첩 괄호 1단계 허용 — 예: "특별부활(효력회복)", "납입최고(독촉)",
# "(암(유사암, 뇌·수막의 양성신생물 제외)) 산정특례대상"(중첩 내용이 길다)
_ART_TITLE = (r"(?:[\(（]((?:[^()（）]|[\(（][^()（）]{0,40}[\)）]){1,80})[\)）]"
              r"|【([^】]{1,60})】|\[([^\]]{1,80})\])")
_PARTICLES = (
    # "제\s*\d+\s*항"만 인용 신호로 인정 — "제\s*\d"까지 넓히면 진짜 헤딩 뒤에
    # 조 인용이 바로 오는 문장("제5조(보험금의 분담) 제1조(…)의 비용")을 죽인다.
    r"(?:의|에|는|은|을|를|와|과|도|만|부터|이라|라고|에서|에도|제\s*\d+\s*항|"
    r"를\s*준용|의\s*죄|에\s*따|에\s*의|에\s*기재)"
)

ART       = re.compile(r"^" + _ART_NUM + _ART_TITLE)
# 표로 추출된 영역에 흡수된 조 헤딩("| 제9조(…의 정의 및 장소) …") — 표 조각은
# 분할하지 않지만 조 상태는 전진시켜야 뒤 조각들이 옳은 조 번호를 받는다.
TABLE_ART = re.compile(r"(?m)^[|\s]*" + _ART_NUM + _ART_TITLE)
# 제목이 다음 줄에 있는 형식: "제1조\n보험금의 지급"
ART_NEXTLINE = re.compile(
    r"^" + _ART_NUM + r"\n([^①②③④⑤\(（【\n]{2,40})\s*(?:\n|$)"
)
# 조사 검사는 제목과 같은 줄로 제한 — 줄바꿈 뒤는 본문 시작(인용 아님)
CITE      = re.compile(r"^" + _ART_NUM + _ART_TITLE + r"[^\S\n]*" + _PARTICLES)
TITLE_ONLY = re.compile(r"^제\d+조(?:의\d+)?\s*" + _ART_TITLE + r"\s*$")
TOC = re.compile(r"[·.]{6,}|…{3,}")

# ── chunk_type 분류 (rag/ 방식) ───────────────────────────────────────────────
_TYPE_OVERRIDE: dict[str, str] = {
    "알릴 의무": "duty", "고지의무": "duty", "통지": "duty",
    "청구": "claim", "지급 절차": "claim",
    "해지": "termination", "소멸": "termination", "효력": "termination",
    "면책": "exclusion", "지급하지 않": "exclusion",
    "별표": "schedule", "부표": "schedule", "장해분류": "schedule",
}
_TYPE_MAP: dict[str, str] = {
    "면책": "exclusion", "부지급": "exclusion", "지급하지 않": "exclusion",
    "보상하지 않": "exclusion",
    "알릴 의무": "duty", "고지의무": "duty", "통지 의무": "duty",
    "보험금 청구": "claim", "청구 절차": "claim", "청구서류": "claim",
    "해지": "termination", "효력상실": "termination", "소멸": "termination",
    "특약": "special_clause", "특별": "special_clause",
    "정의": "definition", "용어": "definition",
    "보장": "coverage", "지급": "coverage", "보험금": "coverage",
}


_EXCLUSION_KW = ("면책", "부지급", "지급하지 않", "보상하지 않")
_COVERAGE_KW = ("보장", "지급", "보험금")


def _is_ambiguous(combined: str, keyword_result: str) -> bool:
    """면책·지급 키워드가 동시에 걸리는 복합문 — 키워드 우선순위가 오판할 수 있는 구간."""
    if keyword_result not in ("exclusion", "coverage"):
        return False
    return any(k in combined for k in _EXCLUSION_KW) and any(k in combined for k in _COVERAGE_KW)


def _classify(text: str, label: Optional[str] = None) -> str:
    combined = (label or "") + " " + text
    result = "general"
    for kw, ct in _TYPE_OVERRIDE.items():
        if kw in combined:
            result = ct
            break
    else:
        for kw, ct in _TYPE_MAP.items():
            if kw in combined:
                result = ct
                break

    # 골든셋 검증(2026-07-30): 키워드 48% vs gemma4 90% — llm 백엔드에선
    # 전 청크를 LLM이 판정하고 키워드는 LLM 실패 시 폴백으로만 사용
    from .llm_classifier import CLASSIFY_BACKEND, classify_llm
    if CLASSIFY_BACKEND == "llm":
        llm_result = classify_llm(text, title=label)
        if llm_result:
            return llm_result
    return result


def _prefix(meta: DocMeta, section_label: Optional[str], art: Optional[int], atitle: Optional[str]) -> str:
    parts = [meta.insurer, meta.product_name]
    if section_label:
        parts.append(section_label)
    if art is not None:
        parts.append(f"제{art}조({atitle})" if atitle else f"제{art}조")
    return " | ".join(parts)


def _yname(lab: Optional[str], kind: str) -> Optional[str]:
    if kind in ("base", "byeolpyo") or not lab:
        return None
    return lab


def _header(lab: Optional[str], kind: str, art: Optional[int], atitle: Optional[str]) -> str:
    parts = []
    yn = _yname(lab, kind)
    if yn:
        parts.append(yn)
    if art is not None:
        parts.append(f"제{art}조({atitle})" if atitle else f"제{art}조")
    return "[" + " > ".join(parts) + "]" if parts else "[기타]"


def _art_from_match(m: re.Match) -> tuple[int, str | None]:
    """ART / ART_NEXTLINE 매치 → (조번호, 제목). 제목 없으면 None."""
    num = int(m.group(1))
    title = next((g for g in m.groups()[1:] if g), None)
    # 감긴 제목의 개행이 제목 안에 들어올 수 있다("부활(효\n력회복)")
    return num, re.sub(r"\s+", " ", title).strip() if title else None


def _pageseq(cid: str) -> tuple:
    p = re.search(r"#p(\d+)#", cid)
    pg = int(p.group(1)) if p else 10 ** 9
    t = re.search(r"#t(\d+)$", cid)
    if t:
        return (pg, 1, int(t.group(1)))
    s = re.search(r"#(\d+)$", cid)
    return (pg, 0, int(s.group(1)) if s else 0)


def _is_toc(t: str) -> bool:
    return bool(TOC.search(t)) or "- 목 차 -" in t


# 앞 줄이 조사·연결어로 끝나면 다음 줄의 "제N조(…)"는 줄바꿈으로 밀린 인용이다
# ("…본인부담의료비는\n제3조(보장종목별 보상내용)에 따라"). 진짜 조 헤딩은 완결된
# 문장(또는 표·빈 줄) 다음에 온다.
_PREV_OPEN_END = re.compile(
    r"(?:[는은이가을를의와과도만면서고며여든지]|또는|및|아래|다음|위한|따라|대한|[,·:]|-)\s*$"
)


_ART_OPEN_NUM = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*[\(（](.{10,})$")


def _art_open(ln: str) -> re.Match | None:
    """닫는 괄호가 같은 조각에 없는 감긴 헤딩("제3조(암(유사암제외), …경계성종" —
    제목이 다음 페이지로 넘어감). 여는 괄호가 닫는 괄호보다 많으면 미완 제목이다."""
    m = _ART_OPEN_NUM.match(ln)
    if m and ln.count("(") + ln.count("（") > ln.count(")") + ln.count("）"):
        return m
    return None


def _split_articles(body: str) -> list[str]:
    """조각 내부의 조 헤딩 줄에서 분할 — 첫 줄만 보는 ART 매치가 조각 중간에서
    시작하는 조(제5조 헤딩이 제4조 본문과 같은 조각에 붙는 밀집 조판)를 통째로
    앞 조에 흡수하는 문제를 막는다."""
    lines = body.split("\n")
    cuts = [0]
    for i in range(1, len(lines)):
        # 조 제목이 컬럼 폭에 잘려 닫는 괄호 없이 감기는 경우
        # ("제20조(…부활(효" + "력회복))")가 있어 세 줄까지 붙여 검사한다.
        probe = "\n".join(lines[i:i + 3])
        ok = ART.match(probe) and not CITE.match(probe)
        if not ok and not _art_open(lines[i]):
            continue
        prev = lines[i - 1].rstrip()
        # "제1관 일반사항 및 용어의 정의"처럼 '의'로 끝나는 관(款) 헤더는 완결된
        # 제목이다 — 미완 문장으로 취급하면 바로 아래 조 헤딩 분할이 막힌다.
        if (prev and _PREV_OPEN_END.search(prev)
                and not re.match(r"^제\s*\d+\s*관\b", prev)):
            continue
        cuts.append(i)
    if len(cuts) == 1:
        return [body]
    return ["\n".join(lines[a:b]).strip()
            for a, b in zip(cuts, cuts[1:] + [len(lines)])]


def _parse_markdown_table(body: str) -> tuple[list[str], list[str]]:
    """markdown 표에서 헤더 행(컬럼명 + 구분선)과 데이터 행 분리."""
    lines = [l for l in body.split("\n") if l.strip().startswith("|")]
    if not lines:
        return [], []
    sep_idx = next(
        (i for i, l in enumerate(lines) if re.match(r"^\|[-| :]+\|?\s*$", l.strip())),
        None,
    )
    if sep_idx is not None:
        return lines[: sep_idx + 1], lines[sep_idx + 1:]
    return lines[:1], lines[1:]


# ── clean ─────────────────────────────────────────────────────────────────────

def _dense(s: str) -> str:
    return re.sub(r"\s+", "", s)


def clean(data: list[dict], bounds: list[Boundary]) -> list[dict]:
    """약관 라벨 부여 + 조 재추출(인용 가드) + 푸터/목차/제목누수 제거."""
    for c in data:
        c["_key"] = _pageseq(c["chunk_id"])
    data.sort(key=lambda c: c["_key"])

    # 한 페이지에 경계가 2개 이상이면(밀집 조판 — 특약 2~3개/페이지) 페이지
    # 단위 라벨은 페이지 전체를 마지막 특약으로 칠한다. 경계 라벨 텍스트가
    # 나타나는 조각부터 전환하도록 페이지 내 경계 목록을 따로 관리한다.
    page_bounds: dict[int, list[Boundary]] = {}
    for b in bounds:
        page_bounds.setdefault(b.page, []).append(b)

    cleaned: list[dict] = []
    cur_label = object()
    cur_art = cur_title = None
    cur_pg = None
    todo: list[Boundary] = []       # 이 페이지에서 아직 전환 안 된 경계들
    lab, kind = None, "front"

    for c in data:
        pg = c["_key"][0]
        if pg != cur_pg:
            cur_pg = pg
            todo = list(page_bounds.get(pg, []))
            if todo:
                # 페이지 시작은 직전 페이지까지의 라벨 — 경계 위쪽 조각(이전
                # 특약의 끝부분)이 새 특약으로 오염되지 않게 한다. 전환은
                # 아래에서 경계 라벨이 나타나는 조각부터 적용한다.
                lab, kind = label_for(bounds, pg - 1)
            else:
                lab, kind = label_for(bounds, pg)
        if todo:
            frag = _dense(c["text"])
            switched = None
            for b in todo:
                # 별표 라벨은 "별표3 골절 분류표"로 재구성돼 원문(【 별표3 】…)과
                # 다르다 — 번호 뺀 제목부로도 찾아본다.
                probes = (_dense(b.label)[:10],
                          _dense(re.sub(r"^별표[\d\-]+\s*", "", b.label))[:10])
                if any(p and p in frag for p in probes):
                    switched = b
            if switched is not None:
                lab, kind = switched.label, switched.kind
                todo = todo[todo.index(switched) + 1:]
        if lab != cur_label:
            cur_label, cur_art, cur_title = lab, None, None

        raw = c["text"]
        is_table = c.get("is_table", False)

        if not is_table and _is_toc(raw):
            continue

        if not is_table:
            def drop_line(ln: str) -> bool:
                s = ln.strip()
                if FOOTER.match(ln):
                    return True
                if s == lab:
                    return True
                if re.match(r"^\d+\.\s.*담보$", s):
                    return True
                return False

            body = "\n".join(l for l in raw.split("\n") if not drop_line(l)).strip()
            if not body:
                continue
        else:
            body = raw

        segments = ([body] if is_table or kind not in ("base", "yak")
                    else _split_articles(body))
        for si, seg in enumerate(segments):
            if kind in ("base", "yak") and is_table:
                for tm in TABLE_ART.finditer(seg):
                    line = seg[tm.start():].split("\n", 1)[0].lstrip("| \t")
                    if CITE.match(line):
                        continue
                    new_art, new_title = _art_from_match(tm)
                    if new_art == 1 or (cur_art is not None
                                        and cur_art < new_art <= cur_art + 10):
                        cur_art, cur_title = new_art, new_title
            elif kind in ("base", "yak"):
                m = ART.match(seg)
                if not m:
                    m = ART_NEXTLINE.match(seg)
                if not m and not CITE.match(seg):
                    # 제목이 다음 페이지로 넘어가 이 조각 안에서 닫히지 않는 헤딩
                    mo = _art_open(seg.split("\n", 1)[0])
                    if mo:
                        m = mo
                if m and not CITE.match(seg):
                    new_art, new_title = _art_from_match(m)
                    # 단조증가 가드: 형법 등 타 법령 인용("제297조(강간)")이
                    # 조 번호를 오염시키지 못하게 큰 점프·역행은 무시.
                    # 제1조는 항상 허용 — 경계 감지가 실패한 문서(DB류)에서 새 특약
                    # 시작을 놓치지 않기 위함. 단 cur_art가 막 리셋된(None) 직후엔
                    # 새 라벨 첫 조각이 형법 조문 나열("제297조(강간)")일 수 있으므로
                    # new_art==1이 아니면 받지 않고 진짜 제1조가 나올 때까지 기다린다.
                    if new_art == 1 or (cur_art is not None and cur_art < new_art <= cur_art + 10):
                        cur_art, cur_title = new_art, new_title

            cc = c if si == 0 else {**c, "chunk_id": f"{c['chunk_id']}~{si}"}
            cc.update(
                _label=lab, _kind=kind,
                _art=(cur_art if kind in ("base", "yak") else None),
                _atitle=(cur_title if kind in ("base", "yak") else None),
                text=(seg if not is_table else raw),
            )
            cleaned.append(cc)
    return cleaned


# ── merge ─────────────────────────────────────────────────────────────────────

def merge(
    cleaned: list[dict],
    meta: DocMeta,
    target: int = 500,
    hard_max: int = 1000,
) -> list[dict]:
    """라벨+조 경계로 병합 → hard_max 분할 → 표 추가 → 인용조문 병합 → 중복 제거."""
    merged: list[dict] = []
    buf: list[dict] = []
    cur_key = object()

    def flush():
        nonlocal buf
        if not buf:
            return
        f0 = buf[0]
        body = "\n".join(b["text"].strip() for b in buf).strip()
        pages = [b["_key"][0] for b in buf]
        h = _header(f0["_label"], f0["_kind"], f0["_art"], f0["_atitle"])
        yn = _yname(f0["_label"], f0["_kind"])
        pfx = _prefix(meta, yn, f0["_art"], f0["_atitle"])
        ctype = _classify(body, f0["_label"])
        merged.append({
            "page_start": min(pages), "page_end": max(pages),
            "section": f0["_label"], "section_kind": f0["_kind"],
            "article_no": f0["_art"], "article_title": f0["_atitle"],
            "chunk_type": ctype, "is_table": False,
            "header": h, "prefix": pfx, "body": body,
            "text": pfx + "\n" + h + "\n" + body,
            "member_ids": [b["chunk_id"] for b in buf],
        })
        buf.clear()

    def buf_tok():
        return sum(_tok(b["text"]) for b in buf)

    tables = []
    for c in cleaned:
        if c.get("is_table"):
            tables.append(c)
            continue
        key = (c["_label"], c["_art"])
        if key != cur_key and buf:
            flush()
        cur_key = key
        buf.append(c)
        if buf_tok() >= target:
            flush()
    flush()

    # hard_max 분할
    final = []
    for m in merged:
        if _tok(m["body"]) <= hard_max:
            final.append(m)
            continue
        parts, cur = [], ""
        for line in m["body"].split("\n"):
            if cur and _tok(cur + "\n" + line) > target:
                parts.append(cur)
                cur = line
            else:
                cur = (cur + "\n" + line).strip()
        if cur:
            parts.append(cur)
        for p in parts:
            mm = dict(m)
            mm["body"] = p
            mm["text"] = m["prefix"] + "\n" + m["header"] + "\n" + p
            final.append(mm)

    # 표 청크 추가
    for t in tables:
        h = _header(t["_label"], t["_kind"], t["_art"], t["_atitle"])
        yn = _yname(t["_label"], t["_kind"])
        pfx = _prefix(meta, yn, t["_art"], t["_atitle"])
        body = t["text"].strip()
        ctype = _classify(body, t["_label"])
        final.append({
            "page_start": t["_key"][0], "page_end": t["_key"][0],
            "section": t["_label"], "section_kind": t["_kind"],
            "article_no": t["_art"], "article_title": t["_atitle"],
            "chunk_type": ctype, "is_table": True,
            "header": h, "prefix": pfx, "body": body,
            "text": pfx + "\n" + h + "\n" + body,
            "member_ids": [t["chunk_id"]],
            "table_source": t.get("table_source"),
        })

    final.sort(key=lambda m: (m["page_start"], m["is_table"]))
    final = _merge_title_runs(final)

    # 중복 제거
    seen, dedup = {}, []
    for m in final:
        k = (m["header"], m["body"].strip())
        if k in seen:
            seen[k]["member_ids"].extend(m["member_ids"])
            continue
        seen[k] = m
        dedup.append(m)
    dedup.sort(key=lambda m: (m["page_start"], m["is_table"]))
    return dedup


def _merge_title_runs(final: list[dict]) -> list[dict]:
    def title_only(m: dict) -> bool:
        return (not m["is_table"]) and bool(TITLE_ONLY.match(m["body"].strip()))

    out, i = [], 0
    while i < len(final):
        m = final[i]
        if title_only(m):
            run, j = [], i
            while (j < len(final) and title_only(final[j])
                   and final[j]["section"] == m["section"]):
                run.append(final[j])
                j += 1
            if len(run) >= 2:
                mm = dict(run[0])
                mm["article_no"] = None
                mm["article_title"] = "인용 조문"
                mm["body"] = "\n".join(r["body"].strip() for r in run)
                mm["page_end"] = run[-1]["page_end"]
                mm["member_ids"] = [x for r in run for x in r["member_ids"]]
                h = _header(mm["section"], mm["section_kind"], None, "인용 조문")
                mm["header"] = h
                mm["text"] = mm["prefix"] + "\n" + h + "\n" + mm["body"]
                out.append(mm)
                i = j
                continue
        out.append(m)
        i += 1
    return out


# ── finalize → InsuranceChunk ─────────────────────────────────────────────────

def finalize(
    dedup: list[dict],
    meta: DocMeta,
    rows_per_table_chunk: int = ROWS_PER_TABLE_CHUNK,
) -> tuple[list[InsuranceChunk], list[TableMeta]]:
    """dict 목록 → (InsuranceChunk 목록, TableMeta 목록).

    큰 표(rows_per_table_chunk 초과)는 row 단위 child 청크로 분할하고
    TableMeta를 반환한다. TableMeta.markdown은 호출 측에서 S3에 업로드한다.
    """
    from .tokenizer import tokenize_korean

    chunks: list[InsuranceChunk] = []
    table_metas: list[TableMeta] = []
    chunk_idx = 1  # 문서 전체 순서 카운터

    for i, m in enumerate(dedup, 1):
        # ── 표 청크 ──────────────────────────────────────────────────────────
        if m["is_table"]:
            header_lines, data_rows = _parse_markdown_table(m["body"])
            if len(data_rows) > rows_per_table_chunk and header_lines:
                # 큰 표 → TableMeta 생성 + row child 청크 분할
                table_id = str(_uuid.uuid4())
                col_count = max(0, len(header_lines[0].split("|")) - 2)
                table_metas.append(TableMeta(
                    doc_hash=meta.doc_hash,
                    source_pdf=meta.source_pdf,
                    insurer=meta.insurer,
                    product_name=meta.product_name,
                    effective_date=meta.effective_date,
                    section=m["section"] or None,
                    page_number=m["page_start"],
                    caption=m.get("article_title") or None,
                    extractor=m.get("table_source", "pdfplumber"),
                    row_count=len(data_rows),
                    col_count=col_count,
                    markdown=m["body"],
                    table_id=table_id,
                ))
                header_text = "\n".join(header_lines)
                for j in range(0, len(data_rows), rows_per_table_chunk):
                    batch = data_rows[j: j + rows_per_table_chunk]
                    row_start = j + 1
                    row_end = j + len(batch)
                    child_body = header_text + "\n" + "\n".join(batch)
                    text = m["prefix"] + "\n" + m["header"] + "\n" + child_body
                    key = f"{meta.doc_hash}:{meta.source_pdf}:{m['page_start']}:tbl:{table_id}:{row_start}"
                    chunk_id = hashlib.sha256(key.encode()).hexdigest()[:24]
                    chunks.append(InsuranceChunk(
                        chunk_id=chunk_id,
                        content=text,
                        content_tokens=tokenize_korean(text),
                        token_count=_tok(text),
                        section=m["section"] or "",
                        page_number=m["page_start"],
                        doc_type="policy_terms",
                        chunk_type=m["chunk_type"],
                        source_pdf=meta.source_pdf,
                        doc_hash=meta.doc_hash,
                        insurer=meta.insurer,
                        product_name=meta.product_name,
                        product_code=meta.product_code,
                        effective_date=meta.effective_date,
                        article_number=f"제{m['article_no']}조" if m.get("article_no") else None,
                        article_title=m.get("article_title"),
                        generation=meta.generation,
                        chunk_index=chunk_idx,
                        table_id=table_id,
                        row_start=row_start,
                        row_end=row_end,
                    ))
                    chunk_idx += 1
                continue

        # ── 텍스트 청크 + 작은 표 (분할 불필요) ──────────────────────────────
        text = m["text"]
        key = f"{meta.doc_hash}:{meta.source_pdf}:{m['page_start']}:{i}"
        chunk_id = hashlib.sha256(key.encode()).hexdigest()[:24]
        sec = m["section"] or ""
        chunks.append(InsuranceChunk(
            chunk_id=chunk_id,
            content=text,
            content_tokens=tokenize_korean(text),
            token_count=_tok(text),
            section=sec,
            page_number=m["page_start"],
            doc_type="policy_terms",
            chunk_type=m["chunk_type"],
            source_pdf=meta.source_pdf,
            doc_hash=meta.doc_hash,
            insurer=meta.insurer,
            product_name=meta.product_name,
            product_code=meta.product_code,
            effective_date=meta.effective_date,
            article_number=f"제{m['article_no']}조" if m["article_no"] else None,
            article_title=m["article_title"],
            generation=meta.generation,
            chunk_index=chunk_idx,
        ))
        chunk_idx += 1

    return chunks, table_metas


# ── 중복 정형문구 마킹 ────────────────────────────────────────────────────────

_NORM_RE = re.compile(r"[\s|․·⋅‧,.\-()\[\]【】〔〕:;'\"0-9]+")
_LEGAL_APPENDIX = re.compile(r"법률조문\s*\|\s*용어 해설|법률조문.{0,30}용어\s*해설")


def mark_boilerplate(chunks: list[InsuranceChunk]) -> list[InsuranceChunk]:
    """동일 본문 반복 청크와 법령 해설 부록에 검색 제외 플래그.

    - 본문(prefix/헤더 제외)이 같은 청크가 2개 이상이면 첫 번째를 대표로 두고
      나머지에 is_boilerplate=True + canonical_id 연결. 특약별 조항 복원을 위해
      삭제하지 않고 전부 보관한다.
    - 법령 해설 부록(신용정보법 등 표)은 대표 여부와 무관하게 검색 제외.
    """
    groups: dict[str, list[InsuranceChunk]] = {}
    for c in chunks:
        body = c.content.split("\n", 2)[-1]
        key = _NORM_RE.sub("", body)[:300]
        groups.setdefault(key, []).append(c)

    n_dup = n_legal = 0
    for grp in groups.values():
        for c in grp[1:]:
            c.is_boilerplate = True
            c.canonical_id = grp[0].chunk_id
            n_dup += 1
    for c in chunks:
        if _LEGAL_APPENDIX.search(c.content[:300]):
            c.is_boilerplate = True
            n_legal += 1
    if n_dup or n_legal:
        tok = sum(c.token_count for c in chunks if c.is_boilerplate)
        logger.info(f"[boilerplate] 반복 {n_dup} + 법령부록 {n_legal}청크 검색 제외 "
                    f"({tok}tok — 보관은 유지)")
    return chunks


def report(chunks: list[InsuranceChunk], bounds: list[Boundary]) -> dict:
    import statistics
    toks = [c.token_count for c in chunks]
    return {
        "n_chunks": len(chunks),
        "n_special_bounds": sum(1 for b in bounds if b.kind in ("yak", "base")),
        "n_byeolpyo_bounds": sum(1 for b in bounds if b.kind == "byeolpyo"),
        "tok_median": int(statistics.median(toks)) if toks else 0,
        "tok_mean": int(statistics.mean(toks)) if toks else 0,
        "tok_max": max(toks) if toks else 0,
        "over_600": sum(1 for t in toks if t > 600),
        "n_unique_yakwan": len({b.label for b in bounds if b.kind == "yak"}),
    }
