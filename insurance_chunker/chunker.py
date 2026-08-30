"""doc_type별 청킹 오케스트레이터.

policy_terms  : boundaries.py(폰트 기반) + rechunk.py(병합·메타) → InsuranceChunk
product_summary: 표 행 병합 + contextual prefix (rag/ 방식)
schedule       : 표 행 단위 청크
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .models import DocMeta, InsuranceChunk, PageResult, TableMeta, make_chunk_id
from .tokenizer import tokenize_korean

logger = logging.getLogger(__name__)

def _tok(text: str) -> int:
    return int(len(text) * 0.6)  # 글자 수 × 0.6 ≈ Kiwi 형태소 토큰 수 근사

_TYPE_MAP: dict[str, str] = {
    "면책": "exclusion", "부지급": "exclusion", "지급하지 않": "exclusion",
    "보상하지 않": "exclusion",
    "알릴 의무": "duty", "고지의무": "duty", "통지 의무": "duty",
    "보험금 청구": "claim", "청구 절차": "claim",
    "해지": "termination", "효력상실": "termination", "소멸": "termination",
    "특약": "special_clause", "특별": "special_clause",
    "정의": "definition", "용어": "definition",
    "보장": "coverage", "지급": "coverage", "보험금": "coverage",
}

_DOC_TYPE_PATTERNS = [
    (r"요약서|상품안내|상품설명서", "product_summary"),
    (r"약관|보통약관|특별약관", "policy_terms"),
    (r"산출기준표|보험료표|지급률표|장해분류표", "schedule"),
]


def auto_doc_type(filename: str) -> str:
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if re.search(pattern, filename):
            return doc_type
    return "product_summary"


def _classify(text: str) -> str:
    for kw, ct in _TYPE_MAP.items():
        if kw in text:
            return ct
    return "general"


def _prefix(meta: DocMeta, section: Optional[str] = None) -> str:
    parts = [meta.insurer, meta.product_name]
    if section:
        parts.append(section)
    return " | ".join(parts)


def _make(
    content: str,
    page_num: int,
    idx: int,
    section: str,
    chunk_type: str,
    meta: DocMeta,
    article_number: Optional[str] = None,
    article_title: Optional[str] = None,
    chunk_index: int = 0,
) -> InsuranceChunk:
    return InsuranceChunk(
        chunk_id=make_chunk_id(meta.source_pdf, page_num, idx, meta.doc_hash),
        content=content,
        content_tokens=tokenize_korean(content),
        token_count=_tok(content),
        section=section,
        page_number=page_num,
        doc_type=meta.doc_type,
        chunk_type=chunk_type,
        source_pdf=meta.source_pdf,
        doc_hash=meta.doc_hash,
        insurer=meta.insurer,
        product_name=meta.product_name,
        product_code=meta.product_code,
        effective_date=meta.effective_date,
        article_number=article_number,
        article_title=article_title,
        generation=meta.generation,
        chunk_index=chunk_index,
        product_id=meta.product_id,
    )


# ── 공개 API ──────────────────────────────────────────────────────────────────

def chunk_document(
    pages: list[PageResult],
    meta: DocMeta,
    pdf_path: Optional[str] = None,
    target: int = 500,
    hard_max: int = 1000,
    report: Optional[dict] = None,
) -> tuple[list[InsuranceChunk], list[TableMeta]]:
    """PDF 페이지 목록 → InsuranceChunk 목록.

    Args:
        pdf_path: policy_terms에서 폰트 기반 경계 감지에 사용. None이면 건너뜀.
        target: 목표 토큰 수 (병합 기준)
        hard_max: 강제 분할 상한 토큰 수
        report: 넘기면 경계 검출 품질을 채워준다(boundary_confidence 등).
            반환값이 아니라 out-param인 이유는 호출부가 13곳이라 튜플을 늘리면
            전부 깨지기 때문. 관심 있는 쪽만 dict를 넘긴다.

    Note:
        product_summary(요약서)는 ingest 대상이 아님. ingest.py에서 진입 전 차단.
    """
    if meta.doc_type == "policy_terms":
        return _chunk_policy_terms(pages, meta, pdf_path, target, hard_max, report)
    elif meta.doc_type == "schedule":
        return _chunk_schedule(pages, meta), []
    else:
        return _chunk_plain_text(pages, meta), []


# ── policy_terms: boundaries + rechunk ────────────────────────────────────────

def _chunk_policy_terms(
    pages: list[PageResult],
    meta: DocMeta,
    pdf_path: Optional[str],
    target: int,
    hard_max: int,
    report: Optional[dict] = None,
) -> tuple[list[InsuranceChunk], list[TableMeta]]:
    from .boundaries import find as find_bounds

    # rechunk.report는 청킹 통계 함수다. out-param `report`(dict)와 이름이 겹치면
    # dict인 줄 알고 대입하다 TypeError로 문서 전체가 죽는다 — 별칭으로 갈라 둔다.
    from .rechunk import clean, finalize, mark_boilerplate, merge
    from .rechunk import report as rechunk_stats

    # 폰트 기반 경계 감지
    bounds = None
    det = None
    toc_titles: list[str] = []
    if pdf_path:
        try:
            import pymupdf as fitz  # 'fitz'는 구 이름 — 그대로 쓰면 임포트마다 경고가 찍힌다

            from .boundaries import assess
            from .toc import extract_toc_titles_ordered
            with fitz.open(pdf_path) as doc:
                bounds, det = find_bounds(doc)
                level, reasons = assess(doc, det, bounds)
                # 목차는 문서가 스스로 선언한 특약 목록 — 조판 관습과 무관하다.
                # 경계 검출이 놓친 특약의 이름을 여기서 찾는다(rechunk.clean).
                try:
                    toc_titles = extract_toc_titles_ordered(doc, det.front_end_page)
                    logger.info(f"목차 제목 수확: {len(toc_titles)}개")
                except Exception as e:
                    logger.warning(f"목차 추출 실패(경계 검출만으로 진행): {e}")
            logger.info(f"폰트 경계 감지: {len(bounds)}개 "
                        f"(본문폰트={det.body_size}, 제목폰트={det.title_size})")
            if report is not None:
                report["boundary_confidence"] = level
                report["boundary_reasons"] = list(reasons)
                report["boundaries"] = len(bounds)
            if level == "weak":
                for r in reasons:
                    logger.warning(f"[신뢰도 WEAK] {r}")
                # 여기서 멈추지 않는 건 의도적이다 — 경계를 못 잡아도 텍스트 청킹으로
                # 폴백해 적재는 된다. 다만 섹션이 안 갈려 조번호가 어긋나므로 결과를
                # '성공'으로만 세면 안 된다. report로 올려 지표·SLO가 보게 한다.
                logger.warning("경계 신뢰도가 낮습니다 — 청킹 결과를 확인하세요",
                               extra={"event": "boundary_weak", "reasons": reasons})
            else:
                logger.info(f"[신뢰도 OK] {reasons[0]}")
        except Exception as e:
            logger.warning(f"폰트 경계 감지 실패, 경계 없이 진행: {e}")
            if report is not None:
                report["boundary_confidence"] = "error"
                report["boundary_reasons"] = [f"{type(e).__name__}: {e}"]

    # 서식 페이지 스킵 (제1조 이전 표지·목차·안내문 페이지)
    if det and det.base_start_page:
        front = [p for p in pages if p.page_num < det.base_start_page]
        pages = [p for p in pages if p.page_num >= det.base_start_page]
        if front:
            logger.info(f"서식 페이지 스킵: p1~p{front[-1].page_num} ({len(front)}페이지)")

    # 페이지 텍스트 → base_chunks (rechunk 입력 형식)
    base_chunks = _pages_to_base_chunks(pages, meta)

    if not bounds:
        # 경계 없으면 단순 텍스트 청킹으로 폴백
        logger.warning("경계 없음 → 단순 페이지 청킹으로 폴백")
        return _chunk_plain_text(pages, meta), []

    cleaned = clean(base_chunks, bounds, toc_titles=toc_titles)
    merged = merge(cleaned, meta, target=target, hard_max=hard_max)
    chunks, table_metas = finalize(merged, meta)
    chunks = mark_boilerplate(chunks)
    stats = rechunk_stats(chunks, bounds)
    logger.info(
        f"[policy_terms] {stats['n_chunks']}청크 | "
        f"tok_mean={stats['tok_mean']} | over_600={stats['over_600']} | "
        f"특약경계={stats['n_special_bounds']}종 | 표분할={len(table_metas)}건"
    )
    return chunks, table_metas


# 조 제목 단독 줄 — "제17조(보험계약의 성립)" 형태. 긴 제목이 줄바꿈으로 감싸이면
# 닫는 괄호가 다음 줄로 넘어가므로 닫는 괄호는 선택. 인용("…제5조(…)에 따라")은
# 제목 뒤에 조사가 붙어 한 줄을 다 못 채우므로 $ 앵커로 배제된다.
# 두 형태 허용: ① 제목 단독 줄(닫는 괄호가 다음 줄로 감쌀 수 있음),
# ② DB식 같은-줄 조판 "제4조의2(제목) ① 본문…" — 제목 뒤 항 마커(①/1.)로 본문 시작.
# 인용("제5조(…)에 따라")은 제목 뒤에 조사가 붙어 어느 형태에도 안 걸림.
_ART_HEAD_LINE = re.compile(
    r"^제\d+조(?:의\d+)?\s*\((?:[^()\n]|\([^()\n]{0,20}\)){1,50}"
    r"(?:\)?\s*$|\)\s*(?=①|1\.\s))", re.M)


_MIN_HEAD_BODY_CHARS = 4  # 헤더 매치 사이 실질 텍스트 최소치(공백 제외)


def _split_at_article_heads(para: str) -> list[str]:
    """빈 줄 없이 이어진 단락을 조 제목 줄 앞에서 분할.

    헤더로 보이는 줄이 바로 앞/뒤에 실질 본문 없이 다른 헤더와 곧장 붙어 있으면
    (예: 정의 조항 안에 "형법 제297조(강간)/제298조(강제추행)/…"처럼 타법령
    조문을 목록으로 나열) 진짜 조 제목이 아니라 인용 목록이다 — 진짜 조는 항상
    본문 문장을 동반하므로, 그런 매치는 분할 지점에서 제외해 나열 전체를
    이웃 조각에 붙여 둔다.
    """
    matches = list(_ART_HEAD_LINE.finditer(para))
    if not matches:
        return [para]

    def gap_len(a_end: int, b_start: int) -> int:
        return len(re.sub(r"\s+", "", para[a_end:b_start]))

    starts = []
    for i, m in enumerate(matches):
        if m.start() == 0:
            continue
        prev_trivial = i > 0 and gap_len(matches[i - 1].end(), m.start()) < _MIN_HEAD_BODY_CHARS
        next_trivial = (i + 1 < len(matches)
                        and gap_len(m.end(), matches[i + 1].start()) < _MIN_HEAD_BODY_CHARS)
        if prev_trivial or next_trivial:
            continue
        starts.append(m.start())

    if not starts:
        return [para]
    bounds = [0] + starts + [len(para)]
    return [para[a:b].strip() for a, b in zip(bounds, bounds[1:]) if para[a:b].strip()]


def _pages_to_base_chunks(pages: list[PageResult], meta: DocMeta) -> list[dict]:
    """PageResult → rechunk이 소비할 base_chunk dict 목록.

    텍스트: 단락(빈 줄) 단위 분할, chunk_id에 #pN# 태그.
    표: is_table=True 청크로 별도 추가.
    """
    chunks = []
    source = meta.source_pdf

    for page in pages:
        pno = page.page_num

        # 텍스트 단락 분할
        paras = [p.strip() for p in re.split(r"\n{2,}", page.text) if p.strip()]
        if not paras and page.text.strip():
            paras = [page.text.strip()]
        paras = [sub for p in paras for sub in _split_at_article_heads(p)]

        for seq, para in enumerate(paras):
            chunks.append({
                "chunk_id": f"{source}#p{pno}#{seq:04d}",
                "source": source,
                "doc_type": meta.doc_type,
                "chunk_type": "general",
                "text": para,
                "is_table": False,
                "page": pno,
            })

        # 표 청크
        for ti, tbl in enumerate(page.tables):
            md = tbl.get("markdown", "")
            src = tbl.get("source", "unknown")
            if md.strip():
                chunks.append({
                    "chunk_id": f"{source}#p{pno}#t{ti:02d}",
                    "source": source,
                    "doc_type": meta.doc_type,
                    "chunk_type": "general",
                    "text": md,
                    "is_table": True,
                    "table_source": src,
                    "page": pno,
                })

    return chunks


# ── schedule ──────────────────────────────────────────────────────────────────

def _chunk_schedule(pages: list[PageResult], meta: DocMeta) -> list[InsuranceChunk]:
    chunks: list[InsuranceChunk] = []
    pfx = _prefix(meta)

    for page in pages:
        for tbl in page.tables:
            md = tbl.get("markdown", "")
            if not md.strip():
                continue
            content = f"{pfx}\n{md}"
            chunks.append(_make(
                content=content, page_num=page.page_num, idx=len(chunks),
                section=meta.product_name, chunk_type="schedule",
                meta=meta, chunk_index=len(chunks),
            ))

    logger.info(f"[schedule] {len(chunks)}청크")
    return chunks


# ── plain text fallback ───────────────────────────────────────────────────────

def _chunk_plain_text(pages: list[PageResult], meta: DocMeta) -> list[InsuranceChunk]:
    chunks: list[InsuranceChunk] = []
    pfx = _prefix(meta)
    for page in pages:
        for para in _split_paragraphs(page.text, max_tok=450):
            if not para.strip() or _tok(para) < 10:
                continue
            chunks.append(_make(
                content=f"{pfx}\n{para}",
                page_num=page.page_num, idx=len(chunks),
                section=meta.product_name, chunk_type=_classify(para), meta=meta,
                chunk_index=len(chunks),
            ))
    logger.info(f"[plain_text] {len(chunks)}청크")
    return chunks


def _split_paragraphs(text: str, max_tok: int = 450) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    result, cur, cur_tok = [], [], 0
    for para in paras:
        pt = _tok(para)
        if cur and cur_tok + pt > max_tok:
            result.append("\n".join(cur))
            cur, cur_tok = [], 0
        cur.append(para)
        cur_tok += pt
    if cur:
        result.append("\n".join(cur))
    return result
