#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rechunk.py — 결합 청크(본문+표) + 경계 → 임베딩용 최종 청크.

v4.2 파이프라인의 핵심 로직을 함수로 정리한 것:
  - 약관 라벨 부여 + 조번호 재추출(구간마다 리셋) + 인용 가드
  - 페이지 푸터/목차 제거
  - 호→항→조 목표크기 병합, HARD_MAX 분할
  - 헤더 prepend([약관 > 제N조(제목)]), 연속 '제목만' 청크 병합, 중복 제거

대부분 한국 보험약관 전반에 통하는 범용 규칙이다.
TARGET/HARD_MAX는 하드코딩이 아니라 임베딩 모델에 맞춰 조절하는 노브.
"""
from __future__ import annotations

import re
import collections

from .boundaries import Boundary, label_for

# ---- 범용 패턴 (한국 약관 공통) ----
FOOTER = re.compile(r"^\s*[-‐–—]\s*\d{1,3}\s*[-‐–—]\s*$")
ART = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*\(([^)]+)\)")
# 인용 가드: 제N조(..) 뒤에 조사/연결어가 오면 '제목'이 아니라 '인용' → 라벨 갱신 금지
CITE = re.compile(
    r"^제\s*\d+\s*조(?:의\s*\d+)?\s*\([^)]+\)\s*"
    r"(?:의|에|는|은|을|를|와|과|도|만|부터|이라|라고|에서|에도|제\s*\d|"
    r"를\s*준용|의\s*죄|에\s*따|에\s*의|에\s*기재)"
)
TITLE_ONLY = re.compile(r"^제\d+조(?:의\d+)?\s*[\(（][^)）]*[\)）]\s*$")


def is_toc(t: str) -> bool:
    return bool(re.search(r"[·.]{6,}", t) or re.search(r"…{3,}", t)) or "- 목 차 -" in t


def _pageseq(cid: str):
    """combined 청크의 #pN#NNNN / #tN 태그로 페이지 내 순서를 복원."""
    p = re.search(r"#p(\d+)#", cid)
    pg = int(p.group(1)) if p else 10**9
    t = re.search(r"#t(\d+)$", cid)
    if t:
        return (pg, 1, int(t.group(1)))
    s = re.search(r"#(\d+)$", cid)
    return (pg, 0, int(s.group(1)) if s else 0)


def _yname(lab, kind):
    # 보통약관/별표/없음 → 약관명 없음
    if kind in ("base", "byeolpyo") or not lab:
        return None
    return lab


def _header(lab, kind, art, atitle):
    if kind == "byeolpyo":
        return f"[{lab}]"
    parts = []
    yn = _yname(lab, kind)
    if yn:
        parts.append(yn)
    if art is not None:
        parts.append(f"제{art}조({atitle})" if atitle else f"제{art}조")
    return "[" + " > ".join(parts) + "]" if parts else "[기타]"


def clean(data: list[dict], bounds: list[Boundary]) -> list[dict]:
    """약관 라벨 부여 + 조 재추출 + 푸터/목차/제목누수 제거."""
    for c in data:
        c["_key"] = _pageseq(c["chunk_id"])
    data.sort(key=lambda c: c["_key"])

    cleaned: list[dict] = []
    cur_label = object()
    cur_art = cur_title = None
    for c in data:
        pg = c["_key"][0]
        lab, kind = label_for(bounds, pg)
        if lab != cur_label:               # 경계 → 조 컨텍스트 리셋
            cur_label, cur_art, cur_title = lab, None, None

        raw = c["text"]
        if not c["is_table"] and is_toc(raw):
            continue

        def drop_line(ln):
            s = ln.strip()
            if FOOTER.match(ln):
                return True
            if s == lab:                                  # 약관 제목 누수
                return True
            if re.match(r"^\d+\.\s.*담보$", s):           # 섹션 divider
                return True
            if re.match(r"^【\s*(별표|부표).*】", s) and s == lab:
                return True
            return False

        if not c["is_table"]:
            body = "\n".join(l for l in raw.split("\n") if not drop_line(l)).strip()
            if not body:
                continue
        else:
            body = raw

        # 조 재추출 (인용 가드 적용)
        if kind in ("base", "yak"):
            m = ART.match(body)
            if m and not CITE.match(body):
                cur_art = int(m.group(1))
                cur_title = m.group(2).strip()

        c.update(
            _label=lab, _kind=kind,
            _art=(cur_art if kind in ("base", "yak") else None),
            _atitle=(cur_title if kind in ("base", "yak") else None),
            text=(body if not c["is_table"] else c["text"]),
        )
        cleaned.append(c)
    return cleaned


def merge(cleaned: list[dict], target: int, hard_max: int) -> list[dict]:
    """라벨+조 경계로 병합 → HARD_MAX 분할 → 표 추가 → 제목런 병합 → 중복 제거."""
    merged: list[dict] = []
    buf: list[dict] = []
    cur_key = object()

    def flush():
        nonlocal buf
        if not buf:
            return
        f0 = buf[0]
        types = [b["chunk_type"] for b in buf]
        ctype = next((t for t in types if t != "general"), types[0])
        body = "\n".join(b["text"].strip() for b in buf).strip()
        pages = [b["_key"][0] for b in buf]
        h = _header(f0["_label"], f0["_kind"], f0["_art"], f0["_atitle"])
        merged.append({
            "source": f0["source"], "doc_type": f0["doc_type"],
            "page_start": min(pages), "page_end": max(pages),
            "yakwan": _yname(f0["_label"], f0["_kind"]),
            "section": f0["_label"], "section_kind": f0["_kind"],
            "article_no": f0["_art"], "article_title": f0["_atitle"],
            "chunk_type": ctype, "is_table": False,
            "path": h, "header": h, "body": body, "text": h + "\n" + body,
            "member_ids": [b["chunk_id"] for b in buf],
        })
        buf = []

    def buf_len():
        return sum(len(b["text"]) + 1 for b in buf)

    tables = []
    for c in cleaned:
        if c["is_table"]:
            tables.append(c)
            continue
        key = (c["_label"], c["_art"])
        if key != cur_key and buf:
            flush()
        cur_key = key
        buf.append(c)
        if buf_len() >= target:
            flush()
    flush()

    # HARD_MAX 분할 (항/줄 경계로)
    final = []
    for m in merged:
        if len(m["body"]) <= hard_max:
            final.append(m)
            continue
        parts, cur = [], ""
        for line in m["body"].split("\n"):
            if cur and len(cur) + len(line) > target:
                parts.append(cur)
                cur = line
            else:
                cur = (cur + "\n" + line).strip()
        if cur:
            parts.append(cur)
        for p in parts:
            mm = dict(m)
            mm["body"] = p
            mm["text"] = m["header"] + "\n" + p
            final.append(mm)

    # 표 청크 추가
    for t in tables:
        h = _header(t["_label"], t["_kind"], t["_art"], t["_atitle"])
        final.append({
            "source": t["source"], "doc_type": t["doc_type"],
            "page_start": t["_key"][0], "page_end": t["_key"][0],
            "yakwan": _yname(t["_label"], t["_kind"]),
            "section": t["_label"], "section_kind": t["_kind"],
            "article_no": t["_art"], "article_title": t["_atitle"],
            "chunk_type": t["chunk_type"], "is_table": True,
            "path": h, "header": h, "body": t["text"].strip(),
            "text": h + "\n" + t["text"].strip(),
            "member_ids": [t["chunk_id"]], "table_source": t.get("table_source"),
        })

    # 연속 '제목만' 청크(인용 법조문 등) 병합
    final.sort(key=lambda m: (m["page_start"], m["is_table"]))
    final = _merge_title_runs(final)

    # 중복 제거 (header+body)
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
    def title_only(m):
        return (not m["is_table"]) and bool(TITLE_ONLY.match(m["body"].strip()))

    out, i = [], 0
    while i < len(final):
        m = final[i]
        if title_only(m):
            run, j = [], i
            while (j < len(final) and title_only(final[j])
                   and final[j]["yakwan"] == m["yakwan"]
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
                hh = _header(mm["section"], mm["section_kind"], None, "인용 조문")
                mm["header"] = mm["path"] = hh
                mm["text"] = hh + "\n" + mm["body"]
                out.append(mm)
                i = j
                continue
        out.append(m)
        i += 1
    return out


def finalize(dedup: list[dict], source: str) -> list[dict]:
    """안정적 chunk_id 부여 + char_len 기록 + 내부키 제거."""
    for i, m in enumerate(dedup, 1):
        m["chunk_id"] = f"{source}#rc{i:04d}"
        m["char_len"] = len(m["text"])
        m.pop("_key", None)
    return dedup


def report(dedup: list[dict], bounds: list[Boundary]) -> dict:
    import statistics
    L = [m["char_len"] for m in dedup]
    yk = collections.Counter(m["yakwan"] for m in dedup if m["yakwan"])
    return {
        "n_chunks": len(dedup),
        "n_yakwan_bounds": sum(1 for b in bounds if b.kind in ("yak", "base")),
        "n_byeolpyo_bounds": sum(1 for b in bounds if b.kind == "byeolpyo"),
        "char_median": int(statistics.median(L)) if L else 0,
        "char_mean": int(statistics.mean(L)) if L else 0,
        "char_max": max(L) if L else 0,
        "under_50": sum(1 for x in L if x < 50),
        "footer_left": sum(1 for m in dedup if FOOTER.search(m["body"])),
        "n_unique_yakwan": len(yk),
        "n_byeolpyo_chunks": sum(1 for m in dedup if m["section_kind"] == "byeolpyo"),
        "art47_leak": sum(1 for m in dedup if m["article_no"] == 47 and m["yakwan"]),
    }
