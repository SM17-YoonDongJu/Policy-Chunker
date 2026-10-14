"""파이프라인 단계 — 문서 1건을 처리하는 과정을 따로 부를 수 있는 단계로 나눈다.

예전엔 parse_pdf 한 번이 텍스트·CPU 표·VLM·best-of를 전부 했다. 그래서 runlog의 phases에
`parse` 하나로만 남고, 그 안에서 VLM(GPU)이 얼마를 먹는지 알 수 없었다. 병목을 지목하려면
단계마다 시간을 재야 하고, 병렬화하려면 단계 경계가 있어야 한다.

  S1 text        PDF → 페이지 텍스트(표 영역 제외)          CPU
  S2 tables_cpu  PDF → PyMuPDF·pdfplumber·camelot 표 + VLM 대상  CPU
  S3 vlm         PDF + 대상 → VLM 표                        GPU
  S4 combine     S1 + S2 + S3 → 페이지별 best-of 표 부착       CPU
  S5 chunk       S4 → 청크(경계 검출·청킹·검증)               CPU
  S6 embed       S5 → 벡터                                 GPU

저장(S7)은 DB 커넥션과 S3 업로드가 CLI 쪽에 있어 호출부(ingest_many._run_one)가 맡는다.

각 단계는 StageResult를 돌려준다 — 출력과 함께 소요 시간, 쓰는 자원, 단계별 규모 지표를
담는다. 자원 태그는 단계 합계를 CPU·GPU로 나눠 '병렬화로 얻을 수 있는 하한'을 계산하는 데 쓴다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import DocMeta, InsuranceChunk, PageResult, TableMeta

logger = logging.getLogger(__name__)

CPU, GPU, IO = "cpu", "gpu", "io"


@dataclass
class StageResult:
    name: str
    output: Any
    seconds: float
    resource: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """runlog·로그에 남길 형태. 출력 본문은 빼고 시간·자원·지표만."""
        return {"seconds": self.seconds, "resource": self.resource, **self.metrics}


@dataclass
class TablesCpu:
    """S2 출력. sources는 설치된 CPU 소스만, targets는 VLM을 부를 페이지(순서 = 우선순위)."""
    sources: dict[str, dict[int, str]]
    targets: list[int]
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


# ── S1~S4: 파싱 ──────────────────────────────────────────────────────────────

def text(pdf_path: str, use_ocr: bool = True) -> StageResult:
    from .pdf_parser import parse_text

    (pages, page_numbers), sec = _timed(parse_text, pdf_path, use_ocr=use_ocr)
    return StageResult("text", (pages, page_numbers), sec, CPU, {
        "pages": len(pages),
        "chars": sum(len(p.text) for p in pages),
        "empty_pages": sum(1 for p in pages if not p.text.strip()),
    })


def tables_cpu(pdf_path: str, page_numbers: list[int], use_vision: bool = True) -> StageResult:
    from .extractor import extract_cpu_tables

    (sources, targets, info), sec = _timed(
        extract_cpu_tables, pdf_path, page_numbers, use_vision)
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


def combine(pages: list[PageResult], cpu: TablesCpu, vlm_tables: dict[int, str]) -> StageResult:
    """페이지별 best-of 표를 고르고 PageResult에 붙인다. pages를 제자리에서 고친다."""
    from .combine import select_best_tables

    t0 = time.perf_counter()
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


def parse(pdf_path: str, use_ocr: bool = True,
          use_vision: bool = True) -> tuple[list[PageResult], list[StageResult]]:
    """S1~S4를 이어 부른다 — parse_pdf와 같은 결과에 단계별 기록이 붙는다."""
    s1 = text(pdf_path, use_ocr=use_ocr)
    pages, page_numbers = s1.output
    s2 = tables_cpu(pdf_path, page_numbers, use_vision=use_vision)
    s3 = vlm(pdf_path, s2.output, use_vision=use_vision)
    counts = {k: len(v) for k, v in table_sources(s2.output, s3.output).items()}
    logger.info(f"표 추출 완료: {counts}")
    s4 = combine(pages, s2.output, s3.output)
    logger.info(f"표 부착 완료: {s4.metrics['table_pages']}페이지 (best-of)")
    return s4.output, [s1, s2, s3, s4]


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
