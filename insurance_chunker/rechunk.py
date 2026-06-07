"""rechunk.py — 경계 라벨 부여 + 조번호 재추출 + 호→조 병합 + InsuranceChunk 출력.

Policy-Chunker의 rechunk.py 뼈대를 유지하되 아래를 보완:
  - 8종 chunk_type 분류 + contextual prefix (rag/ 방식)
  - article_number / article_title 메타데이터
  - 출력: dict → InsuranceChunk
"""
from __future__ import annotations

import hashlib
import re
import uuid as _uuid
from typing import Optional

from .boundaries import Boundary, label_for
from .models import DocMeta, InsuranceChunk, TableMeta

ROWS_PER_TABLE_CHUNK = 20  # 이 행 수를 초과하면 표를 child 청크로 분할

def _tok(text: str) -> int:
    return int(len(text) * 0.6)  # 글자 수 × 0.6 ≈ Kiwi 형태소 토큰 수 근사

# ── 범용 패턴 (한국 약관 공통) ────────────────────────────────────────────────
FOOTER = re.compile(r"^\s*[-‐–—]\s*\d{1,3}\s*[-‐–—]\s*$")
ART = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*\(([^)]+)\)")
# 인용 가드: "제N조(...)에 따라" 등 — 조항 시작이 아니라 인용
CITE = re.compile(
    r"^제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]+\)\s*"
    r"(?:의|에|는|은|을|를|와|과|도|만|부터|이라|라고|에서|에도|제\s*\d|"
    r"를\s*준용|의\s*죄|에\s*따|에\s*의|에\s*기재)"
)
TITLE_ONLY = re.compile(r"^제\d+조(?:의\d+)?\s*[\(（][^)）]*[\)）]\s*$")
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


def _classify(text: str, label: Optional[str] = None) -> str:
    combined = (label or "") + " " + text
    for kw, ct in _TYPE_OVERRIDE.items():
        if kw in combined:
            return ct
    for kw, ct in _TYPE_MAP.items():
        if kw in combined:
            return ct
    return "general"


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

def clean(data: list[dict], bounds: list[Boundary]) -> list[dict]:
    """약관 라벨 부여 + 조 재추출(인용 가드) + 푸터/목차/제목누수 제거."""
    for c in data:
        c["_key"] = _pageseq(c["chunk_id"])
    data.sort(key=lambda c: c["_key"])

    cleaned: list[dict] = []
    cur_label = object()
    cur_art = cur_title = None

    for c in data:
        pg = c["_key"][0]
        lab, kind = label_for(bounds, pg)
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

        if kind in ("base", "yak"):
            m = ART.match(body)
            if m and not CITE.match(body):
                cur_art = int(m.group(1))
                cur_title = m.group(2).strip()

        c.update(
            _label=lab, _kind=kind,
            _art=(cur_art if kind in ("base", "yak") else None),
            _atitle=(cur_title if kind in ("base", "yak") else None),
            text=(body if not is_table else raw),
        )
        cleaned.append(c)
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
            "source": f0["source"],
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
            "source": t["source"],
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
