"""파이프라인 단계 — 문서 1건을 처리하는 과정을 따로 부를 수 있는 단계로 나눈다.

예전엔 parse_pdf 한 번이 텍스트·CPU 표·VLM·best-of를 전부 했다. 그래서 runlog의 phases에
`parse` 하나로만 남고, 그 안에서 VLM(GPU)이 얼마를 먹는지 알 수 없었다. 병목을 지목하려면
단계마다 시간을 재야 하고, 병렬화하려면 단계 경계가 있어야 한다.

  S1 layout      PDF → 쪽별 표 박스 (PP-DocLayoutV3, 백지 선필터)   CPU
  S2 text        PDF → 페이지 텍스트(표 영역 제외 — 레이아웃 박스,
                 레이아웃이 꺼지면 find_tables)                  CPU
  S3 tables_cpu  PDF → PyMuPDF 표(폴백) + VLM 대상(박스 또는 gap 쪽)  CPU
  S4 vlm         PDF + 대상 → VLM 표 (crop 또는 페이지 전체)      GPU
  S5 combine     박스 쪽은 VLM, 그 밖은 PyMuPDF (off면 best-of)    CPU
  S6 chunk       청크(경계 검출·청킹·검증)                      CPU
  S7 embed       벡터                                        GPU

layout이 text보다 앞인 이유: 텍스트의 표 영역 제외 기준이 레이아웃 박스라서다(이슈 #72의
스케치는 text 다음에 두었지만, 그러면 제외할 박스가 아직 없다).

저장(S7)은 DB 커넥션과 S3 업로드가 CLI 쪽에 있어 호출부(ingest_many._run_one)가 맡는다.

각 단계는 StageResult를 돌려준다 — 출력과 함께 소요 시간, 쓰는 자원, 단계별 규모 지표를
담는다. 자원 태그는 단계 합계를 CPU·GPU로 나눠 '병렬화로 얻을 수 있는 하한'을 계산하는 데 쓴다.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import DocMeta, InsuranceChunk, PageResult, TableMeta

logger = logging.getLogger(__name__)

CPU, GPU, IO = "cpu", "gpu", "io"


def rss_mb() -> Optional[float]:
    """이 프로세스의 현재 상주 메모리(MB). Linux는 /proc, 그 밖은 최고점(ru_maxrss)으로 대신한다.

    어느 단계가 메모리를 먹는지는 이것 없이는 추측뿐이다 — pdfplumber가 2GB를 쌓아 OOM으로
    111번 재시작하는 동안 죽는 순간은 VLM 단계여서, 로그만으로는 엉뚱한 단계를 의심했다.
    """
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            resident_pages = int(f.read().split()[1])
        return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)
    except (OSError, ValueError, IndexError):
        return peak_rss_mb()


def peak_rss_mb() -> Optional[float]:
    """이 프로세스가 지금까지 쓴 최고 상주 메모리(MB). 문서 한 건의 메모리 예산을 재는 기준."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # noqa: BLE001 - 계측이 파이프라인을 죽이면 안 된다
        return None
    # ru_maxrss 단위가 플랫폼마다 다르다: Linux는 KB, macOS는 bytes.
    return round(peak / 1048576 if sys.platform == "darwin" else peak / 1024, 1)


@dataclass
class StageResult:
    name: str
    output: Any
    seconds: float
    resource: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 단계가 끝난 직후의 상주 메모리를 지표에 끼운다. 이력(items.jsonl의 stages)과 로그에
        # 단계별로 남아, 어느 단계에서 메모리가 뛰는지 사후에 보인다.
        self.metrics.setdefault("rss_mb", rss_mb())

    def summary(self) -> dict[str, Any]:
        """runlog·로그에 남길 형태. 출력 본문은 빼고 시간·자원·지표만."""
        return {"seconds": self.seconds, "resource": self.resource, **self.metrics}


@dataclass
class LayoutResult:
    """S1 출력. enabled=False면 아래 모든 단계가 기존 경로(find_tables 제외·gap·페이지
    전체)로 동작한다 — LAYOUT_BACKEND=off뿐 아니라 paddleocr 미설치·모델 로드 실패도
    여기로 수렴하므로, 의존이 빠진 이미지가 'VLM이 조용히 꺼진 상태'가 되지 않는다."""
    enabled: bool
    boxes: dict[int, list[dict]]  # 쪽 → [{"bbox_pdf", "bbox_px", "score"}], 읽기 순서

    def pdf_boxes(self) -> Optional[dict[int, list[list[float]]]]:
        """쪽 → bbox(PDF 포인트) 목록. 꺼져 있으면 None — 호출부가 기존 경로로 간다."""
        if not self.enabled:
            return None
        return {pno: [b["bbox_pdf"] for b in bs] for pno, bs in self.boxes.items()}


@dataclass
class TablesCpu:
    """S3 출력. sources는 CPU 소스(PyMuPDF), targets는 VLM 대상(순서 = 예산 소진 순서) —
    레이아웃이 켜져 있으면 {"page","bbox"} crop 목록, 꺼져 있으면 쪽 번호 목록."""
    sources: dict[str, dict[int, str]]
    targets: list
    page_numbers: list[int]
    info: dict[str, int]


@dataclass
class Chunked:
    """S5 출력."""
    chunks: list[InsuranceChunk]
    table_metas: list[TableMeta]
    warnings: list[str]
    report: dict


def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, round(time.perf_counter() - t0, 2)


# ── S1~S5: 파싱 ──────────────────────────────────────────────────────────────

def layout(pdf_path: str) -> StageResult:
    """쪽별 표 박스. LAYOUT_BACKEND=off거나 모델을 못 열면 enabled=False(기존 경로)."""
    from . import layout as layout_mod

    if layout_mod.LAYOUT_BACKEND == "off":
        return StageResult("layout", LayoutResult(enabled=False, boxes={}), 0.0, CPU,
                           {"skipped": True, "boxes": 0})
    (boxes, m), sec = _timed(layout_mod.detect_tables, pdf_path)
    out = LayoutResult(enabled=layout_mod.available(), boxes=boxes)
    if not out.enabled:
        # 박스가 일부 쌓였어도 버린다 — 반쯤 켜진 상태로 내려가면 그 문서만 제외 기준이
        # 섞여서, 같은 설정의 두 실행이 다른 결과를 낸다.
        out.boxes = {}
    return StageResult("layout", out, sec, CPU, {"enabled": out.enabled, **m})


def text(pdf_path: str, use_ocr: bool = True,
         layout_result: Optional[LayoutResult] = None) -> StageResult:
    from .pdf_parser import parse_text

    table_boxes = layout_result.pdf_boxes() if layout_result else None
    (pages, page_numbers), sec = _timed(parse_text, pdf_path, use_ocr=use_ocr,
                                        table_boxes=table_boxes)
    return StageResult("text", (pages, page_numbers), sec, CPU, {
        "pages": len(pages),
        "chars": sum(len(p.text) for p in pages),
        "empty_pages": sum(1 for p in pages if not p.text.strip()),
    })


def tables_cpu(pdf_path: str, page_numbers: list[int], use_vision: bool = True,
               layout_result: Optional[LayoutResult] = None) -> StageResult:
    from .extractor import extract_cpu_tables

    layout_boxes = layout_result.pdf_boxes() if layout_result else None
    (sources, targets, info), sec = _timed(
        extract_cpu_tables, pdf_path, page_numbers, use_vision, layout_boxes)
    out = TablesCpu(sources=sources, targets=targets, page_numbers=page_numbers, info=info)
    return StageResult("tables_cpu", out, sec, CPU, {
        "table_pages": {k: len(v) for k, v in sources.items()},
        "gap_pages": info.get("gap_pages", 0),
        "vlm_targets": len(targets),
    })


def vlm(pdf_path: str, cpu: TablesCpu, use_vision: bool = True) -> StageResult:
    """VLM을 안 쓰는 설정이면 호출 없이 빈 결과를 돌려준다(단계 자체는 기록된다)."""
    from . import extractor

    if not use_vision or extractor.VLM_BACKEND == "off":
        extractor.reset_vision_counter()
        return StageResult("vlm", {}, 0.0, GPU, {"skipped": True, "tables": 0})
    tables, sec = _timed(extractor.extract_vlm_tables,
                         pdf_path, cpu.targets, len(cpu.page_numbers), cpu.info)
    return StageResult("vlm", tables, sec, GPU, {
        "targets": len(cpu.targets),
        "concurrency": extractor.VLM_CONCURRENCY,
        "calls": extractor._vision_call_count,
        "tables": len(tables),
        "circuit_open": extractor.vision_circuit_open(),
        **vlm_call_summary(extractor.vision_page_stats()),
    })


def _pct(values: list[float], q: float) -> Optional[float]:
    """가장 가까운 순위 방식 백분위수(metrics.py와 같은 규칙)."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def vlm_call_summary(calls: list[dict]) -> dict[str, Any]:
    """VLM 호출 기록 → 단계 지표.

    latencies는 원값 목록으로 남긴다. 문서별 p95를 모아 다시 p95를 내면 틀린 값이 되므로,
    문서셋 전체 분포는 metrics.py가 이 목록을 합쳐서 낸다. 문서당 최대 VISION_MAX_PAGES개라
    이력 파일이 크게 불지 않는다.
    """
    lat = [c["seconds"] for c in calls]
    outcomes: dict[str, int] = {}
    for c in calls:
        outcomes[c["outcome"]] = outcomes.get(c["outcome"], 0) + 1
    prompt = [c["prompt_tokens"] for c in calls if c.get("prompt_tokens") is not None]
    gen = [c["completion_tokens"] for c in calls if c.get("completion_tokens") is not None]
    px = [c["image_px"] for c in calls if c.get("image_px")]
    return {
        "outcomes": outcomes,
        "truncated": sum(1 for c in calls if c.get("finish_reason") == "length"),
        "latencies": lat,
        "latency_p50": _pct(lat, 0.5),
        "latency_p95": _pct(lat, 0.95),
        "prompt_tokens_p50": _pct(prompt, 0.5),
        "completion_tokens_p50": _pct(gen, 0.5),
        "image_px_p50": _pct(px, 0.5),
    }


def table_sources(cpu: TablesCpu, vlm_tables: dict[int, str]) -> dict[str, dict[int, str]]:
    """S2·S3 출력을 extract_tables_for_doc과 같은 모양으로 합친다."""
    sources = dict(cpu.sources)
    if vlm_tables:
        sources["vlm"] = vlm_tables
    return sources


def combine(pages: list[PageResult], cpu: TablesCpu, vlm_tables: dict[int, str],
            layout_result: Optional[LayoutResult] = None) -> StageResult:
    """페이지별 표를 고르고 PageResult에 붙인다. pages를 제자리에서 고친다.

    레이아웃이 켜져 있으면 VLM(crop) 우선·PyMuPDF 폴백이고, 꺼져 있으면 예전
    더블스페이스 best-of다.
    """
    from .combine import select_best_tables, select_layout_tables

    t0 = time.perf_counter()
    if layout_result and layout_result.enabled:
        best = select_layout_tables(cpu.sources.get("pymupdf", {}), vlm_tables)
    else:
        best = select_best_tables(table_sources(cpu, vlm_tables))
    chosen: dict[str, int] = {}
    for page in pages:
        entry = best.get(page.page_num)
        if entry:
            md, src = entry
            page.tables = [{"markdown": md, "source": src}]
            chosen[src] = chosen.get(src, 0) + 1

    table_pages = sum(1 for p in pages if p.tables)
    return StageResult("combine", pages, round(time.perf_counter() - t0, 2), CPU, {
        "table_pages": table_pages,
        "chosen": chosen,
    })


def table_fingerprint(pages: list[PageResult]) -> dict[str, dict[str, Any]]:
    """페이지별로 최종 채택된 표의 출처와 해시.

    병렬화처럼 결과가 바뀌면 안 되는 변경은 이 해시가 전부 같아야 하고, DPI처럼 결과를 바꾸는
    변경은 몇 페이지가 달라졌는지를 여기서 센다. 본문은 싣지 않는다(원문 보관 금지 원칙).
    """
    import hashlib

    out: dict[str, dict[str, Any]] = {}
    for p in pages:
        if not p.tables:
            continue
        md = p.tables[0]["markdown"]
        out[str(p.page_num)] = {"source": p.tables[0]["source"],
                                "sha256": hashlib.sha256(md.encode()).hexdigest()[:16],
                                "chars": len(md),
                                # VLM 프롬프트가 '읽기 어려운 셀'에 쓰게 한 표식
                                "unknown_cells": md.count("[?]")}
    return out


def _markdown_cells(md: str) -> list[str]:
    """markdown 표에서 대조할 셀 문자열(공백 제거). 빈 칸·구분선은 뺀다.

    셀 안 <br>(항목 줄바꿈)은 조각으로 나눠 따로 센다 — 텍스트 레이어는 줄 단위라
    여러 항목이 든 셀을 통짜로 대조하면 전부 불일치로 나온다.
    """
    out: list[str] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        for cell in line.strip("|").split("|"):
            c = cell.strip()
            if not c or set(c) <= {"-", ":"}:
                continue
            for frag in c.split("<br>"):
                frag = re.sub(r"\s+", "", frag)
                if frag:
                    out.append(frag)
    return out


def table_cell_match(pages: list[PageResult], pdf_path: str) -> dict[str, Any]:
    """최종 채택된 표의 셀이 같은 쪽 텍스트 레이어에 그대로 나타나는 비율. 벤치 가드레일용.

    VLM 오독(실측: 관혈→관협)을 재는 지표다. 공백을 걷어낸 셀 문자열이 같은 쪽 텍스트
    레이어(공백 제거)의 부분 문자열이면 일치. 텍스트 레이어가 없는 쪽(스캔)은 잴 수 없어
    뺀다. PyMuPDF 표는 텍스트 레이어에서 나오므로 1.0에 가깝다 — 기준선 구실을 한다.
    [?] 셀 수는 qwen 프롬프트 전용이라 소스가 바뀌면 무의미해지는 것과 달리, 이 지표는
    소스와 무관하게 유지된다.
    """
    import pymupdf as fitz

    per_page: dict[str, float] = {}
    total = matched = 0
    with fitz.open(pdf_path) as doc:
        for p in pages:
            if not p.tables:
                continue
            hay = re.sub(r"\s+", "", doc[p.page_num - 1].get_text())
            if not hay:
                continue
            cells = _markdown_cells(p.tables[0]["markdown"])
            if not cells:
                continue
            hit = sum(1 for c in cells if c in hay)
            per_page[str(p.page_num)] = round(hit / len(cells), 3)
            total += len(cells)
            matched += hit
    return {"rate": round(matched / total, 3) if total else None,
            "cells": total, "per_page": per_page}


def parse(pdf_path: str, use_ocr: bool = True,
          use_vision: bool = True) -> tuple[list[PageResult], list[StageResult]]:
    """S1~S5를 이어 부른다 — parse_pdf와 같은 결과에 단계별 기록이 붙는다."""
    s1 = layout(pdf_path)
    s2 = text(pdf_path, use_ocr=use_ocr, layout_result=s1.output)
    pages, page_numbers = s2.output
    s3 = tables_cpu(pdf_path, page_numbers, use_vision=use_vision, layout_result=s1.output)
    s4 = vlm(pdf_path, s3.output, use_vision=use_vision)
    counts = {k: len(v) for k, v in table_sources(s3.output, s4.output).items()}
    logger.info(f"표 추출 완료: {counts}")
    s5 = combine(pages, s3.output, s4.output, layout_result=s1.output)
    logger.info(f"표 부착 완료: {s5.metrics['table_pages']}페이지")
    return s5.output, [s1, s2, s3, s4, s5]


# ── S5~S6 ────────────────────────────────────────────────────────────────────

def chunk(pages: list[PageResult], meta: DocMeta, pdf_path: Optional[str],
          target: int = 500, hard_max: int = 1000) -> StageResult:
    from .chunker import chunk_document
    from .validator import validate_chunks

    t0 = time.perf_counter()
    report: dict = {}
    chunks, table_metas = chunk_document(pages, meta, pdf_path=pdf_path, target=target,
                                         hard_max=hard_max, report=report)
    vr = validate_chunks(chunks)
    out = Chunked(chunks=vr.valid_chunks, table_metas=table_metas,
                  warnings=vr.warnings, report=report)
    tokens = [c.token_count for c in out.chunks]
    return StageResult("chunk", out, round(time.perf_counter() - t0, 2), CPU, {
        "chunks": len(out.chunks),
        "removed": vr.removed,
        "warnings": len(vr.warnings),
        "over_600": sum(1 for t in tokens if t > 600),
        "boundary_confidence": report.get("boundary_confidence"),
    })


def embed(chunks: list[InsuranceChunk], ollama_url: Optional[str] = None,
          model: Optional[str] = None) -> StageResult:
    from .embedder import embed_chunks

    out, sec = _timed(embed_chunks, chunks, ollama_url=ollama_url, model=model)
    return StageResult("embed", out, sec, GPU, {
        "chunks": len(out),
        "embedded": sum(1 for c in out if c.embedding is not None),
    })
