"""단계 분리 실행 — 단계마다 중간 산출물을 남기고, 원하는 단계부터 다시 돌린다.

운영 경로(ingest_many._run_one)는 메모리에서 단계를 이어 부르고 디스크에 아무것도 쓰지
않는다. 이 모듈은 벤치마크·디버깅용이다. 같은 단계 함수(stages.py)를 부르므로 여기서 잰
시간과 결과가 운영과 같은 코드에서 나온다.

쓰임:
  # 전 단계를 돌리고 산출물을 남긴다
  python -m insurance_chunker.stage_runner --pdf 약관.pdf --artifacts out/
  # VLM 설정만 바꿔 VLM부터 다시 — 텍스트·CPU 표는 캐시에서 읽는다
  VLM_DPI=200 python -m insurance_chunker.stage_runner --pdf 약관.pdf --artifacts out/ \\
      --from vlm --to combine

캐시 키는 (문서 해시, 단계, 그 단계 결과를 바꾸는 설정, 상위 단계 키, 코드 지문)이다.
설정이나 코드가 바뀌면 키가 달라져 그 단계와 하위 단계가 다시 돈다. --from은 '여기부터는
캐시가 있어도 다시 돈다'는 뜻이고, 그보다 앞 단계는 캐시가 없을 때만 돈다. 어느 단계든 새로
돌면 그 뒤 단계는 전부 다시 돈다.

산출물 배치: <artifacts>/<doc_hash 앞 16자>/<단계>-<키>.json (임베딩은 .npy를 함께 둔다)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

from . import stages
from .models import DocMeta, InsuranceChunk, PageResult, TableMeta

logger = logging.getLogger(__name__)

# 산출물 형식이 바뀌면 올린다 — 예전 산출물을 잘못 읽는 대신 키가 달라져 다시 돈다.
# 2: layout 단계 추가, tables_cpu.targets가 crop({"page","bbox"}) 목록일 수 있다.
ARTIFACT_VERSION = 2

# layout이 text보다 앞이다 — 텍스트의 표 영역 제외 기준이 레이아웃 박스라서.
ORDER = ["layout", "text", "tables_cpu", "vlm", "combine", "chunk", "embed"]
UPSTREAM = {
    "layout": [],
    "text": ["layout"],
    "tables_cpu": ["text", "layout"],
    "vlm": ["tables_cpu"],
    "combine": ["text", "tables_cpu", "vlm", "layout"],
    "chunk": ["combine"],
    "embed": ["chunk"],
}


# ── 캐시 키 ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """패키지 소스 전체의 해시. 코드가 바뀌면 모든 단계 키가 달라진다.

    단계별로 의존 파일을 골라내면 더 정밀하지만, 빠뜨리면 낡은 산출물을 조용히 재사용한다.
    벤치마크에서 그건 '개선 효과'로 둔갑하므로 넓게 잡는다.
    """
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def stage_config(name: str, opts: dict[str, Any]) -> dict[str, Any]:
    """그 단계의 결과를 바꾸는 설정만 고른다. 서버 주소처럼 결과와 무관한 것은 뺀다."""
    from . import embedder, extractor

    if name == "layout":
        from . import layout as layout_mod
        if layout_mod.LAYOUT_BACKEND == "off":
            return {"skipped": True}
        import importlib.util
        return {"backend": layout_mod.LAYOUT_BACKEND, "model": layout_mod.LAYOUT_MODEL,
                "min_score": layout_mod.LAYOUT_MIN_SCORE, "dpi": layout_mod.LAYOUT_DPI,
                # paddleocr가 없으면 enabled=False 산출물이 만들어진다. 설치 여부를 키에
                # 넣어야 나중에 의존을 깔았을 때 그 '꺼진' 산출물을 재사용하지 않는다.
                "installed": importlib.util.find_spec("paddleocr") is not None}
    if name == "tables_cpu":
        return {"use_vision": opts["use_vision"], "backend_off": extractor.VLM_BACKEND == "off",
                "target": extractor.VLM_TARGET,
                "hints": [extractor._HINT_DRAWINGS, extractor._HINT_MIN_TEXT,
                          extractor._HINT_IMG_FRAC]}
    if name == "vlm":
        if not opts["use_vision"] or extractor.VLM_BACKEND == "off":
            return {"skipped": True}
        return {"backend": extractor.VLM_BACKEND, "model": extractor.VLM_MODEL,
                "dpi": extractor.VLM_DPI, "max_pages": extractor.VISION_MAX_PAGES,
                "max_crops": extractor.VISION_MAX_CROPS,
                "fail_streak": extractor.VLM_FAIL_STREAK, "timeout": extractor.VLM_TIMEOUT,
                "max_tokens": extractor.VLM_MAX_TOKENS,
                "loop_cells": extractor.VLM_LOOP_CELLS,
                "snap": [extractor.VLM_SNAP_TEXT, extractor.VLM_SNAP_RATIO],
                # 결과는 바꾸지 않지만 시간을 바꾼다. 키에서 빼면 동시도만 바꾼 변형이 기준선의
                # VLM 산출물(과 그때 잰 시간)을 재사용해 '개선 없음'으로 보인다.
                "concurrency": extractor.VLM_CONCURRENCY,
                "prompt": hashlib.sha256(extractor.VLM_PROMPT.encode()).hexdigest()[:12]}
    if name == "chunk":
        return {"meta": asdict(opts["meta"]), "target": opts["target"],
                "hard_max": opts["hard_max"]}
    if name == "embed":
        return {"backend": embedder.EMBED_BACKEND,
                "model": opts.get("embed_model") or embedder.OLLAMA_MODEL,
                "max_chars": embedder._MAX_CHARS}
    return {}


def stage_key(name: str, doc_hash: str, opts: dict[str, Any], upstream_keys: list[str]) -> str:
    blob = json.dumps({"v": ARTIFACT_VERSION, "code": code_fingerprint(), "stage": name,
                       "doc": doc_hash, "config": stage_config(name, opts),
                       "up": upstream_keys}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── 직렬화 ───────────────────────────────────────────────────────────────────
# JSON 키는 문자열이라 페이지 번호(int) 키를 되돌려야 한다.

def _int_keys(d: dict) -> dict[int, Any]:
    return {int(k): v for k, v in d.items()}


def _pages(rows: list[dict]) -> list[PageResult]:
    return [PageResult(**r) for r in rows]


def _dump_output(name: str, output: Any, path: Path) -> Any:
    if name == "layout":
        return asdict(output)
    if name == "text":
        pages, page_numbers = output
        return {"pages": [asdict(p) for p in pages], "page_numbers": page_numbers}
    if name == "tables_cpu":
        return asdict(output)
    if name == "vlm":
        return output
    if name == "combine":
        return [asdict(p) for p in output]
    if name == "chunk":
        return asdict(output)
    if name == "embed":
        # 벡터는 JSON으로 두면 청크 500개에 10MB를 넘는다. 따로 .npy에 둔다.
        import numpy as np
        vecs = [c.embedding for c in output]
        dim = next((len(v) for v in vecs if v is not None), 0)
        arr = np.full((len(vecs), dim), np.nan, dtype=np.float32)
        for i, v in enumerate(vecs):
            if v is not None:
                arr[i] = v
        np.save(path.with_suffix(".npy"), arr)
        return {"chunks": [{**asdict(c), "embedding": None} for c in output],
                "has_embedding": [v is not None for v in vecs]}
    raise ValueError(name)


def _load_output(name: str, data: Any, path: Path) -> Any:
    if name == "layout":
        return stages.LayoutResult(enabled=data["enabled"], boxes=_int_keys(data["boxes"]))
    if name == "text":
        return _pages(data["pages"]), data["page_numbers"]
    if name == "tables_cpu":
        return stages.TablesCpu(
            sources={k: _int_keys(v) for k, v in data["sources"].items()},
            targets=data["targets"], page_numbers=data["page_numbers"], info=data["info"])
    if name == "vlm":
        return _int_keys(data)
    if name == "combine":
        return _pages(data)
    if name == "chunk":
        return stages.Chunked(
            chunks=[InsuranceChunk(**c) for c in data["chunks"]],
            table_metas=[TableMeta(**t) for t in data["table_metas"]],
            warnings=data["warnings"], report=data["report"])
    if name == "embed":
        import numpy as np
        arr = np.load(path.with_suffix(".npy"))
        chunks = [InsuranceChunk(**c) for c in data["chunks"]]
        for c, row, has in zip(chunks, arr, data["has_embedding"]):
            c.embedding = row.tolist() if has else None
        return chunks
    raise ValueError(name)


# ── 실행 ─────────────────────────────────────────────────────────────────────

def _call(name: str, pdf_path: str, opts: dict[str, Any],
          up: dict[str, Any]) -> stages.StageResult:
    if name == "layout":
        return stages.layout(pdf_path)
    if name == "text":
        return stages.text(pdf_path, layout_result=up["layout"])
    if name == "tables_cpu":
        _, page_numbers = up["text"]
        return stages.tables_cpu(pdf_path, page_numbers, use_vision=opts["use_vision"],
                                 layout_result=up["layout"])
    if name == "vlm":
        return stages.vlm(pdf_path, up["tables_cpu"], use_vision=opts["use_vision"])
    if name == "combine":
        pages, _ = up["text"]
        return stages.combine(pages, up["tables_cpu"], up["vlm"], layout_result=up["layout"])
    if name == "chunk":
        meta: DocMeta = opts["meta"]
        return stages.chunk(up["combine"], meta,
                            pdf_path=pdf_path if meta.doc_type == "policy_terms" else None,
                            target=opts["target"], hard_max=opts["hard_max"])
    if name == "embed":
        return stages.embed(up["chunk"].chunks, ollama_url=opts.get("ollama_url"),
                            model=opts.get("embed_model"))
    raise ValueError(name)


def run(
    pdf_path: str,
    artifacts: Path,
    meta: DocMeta,
    *,
    from_stage: Optional[str] = None,
    to_stage: str = "embed",
    use_vision: bool = True,
    target: int = 500,
    hard_max: int = 1000,
    ollama_url: Optional[str] = None,
    embed_model: Optional[str] = None,
    on_stage: Optional[Callable[[dict], None]] = None,
) -> list[dict]:
    """to_stage까지 돌리고 단계별 기록을 돌려준다.

    각 기록: {stage, key, cached, seconds, resource, metrics, path}
    cached=True인 단계의 seconds는 그 산출물을 만들 때 잰 값이다(이번에 읽은 시간이 아니다).
    """
    if to_stage not in ORDER:
        raise ValueError(f"알 수 없는 단계: {to_stage}")
    if from_stage is not None and ORDER.index(from_stage) > ORDER.index(to_stage):
        raise ValueError(f"--from {from_stage}가 --to {to_stage}보다 뒤다")
    force_from = ORDER.index(from_stage) if from_stage else len(ORDER)

    opts = {"use_vision": use_vision, "meta": meta, "target": target, "hard_max": hard_max,
            "ollama_url": ollama_url, "embed_model": embed_model}
    doc_dir = Path(artifacts) / meta.doc_hash[:16]
    doc_dir.mkdir(parents=True, exist_ok=True)

    keys: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    records: list[dict] = []
    recomputed = False
    for i, name in enumerate(ORDER[:ORDER.index(to_stage) + 1]):
        key = stage_key(name, meta.doc_hash, opts, [keys[u] for u in UPSTREAM[name]])
        keys[name] = key
        path = doc_dir / f"{name}-{key}.json"

        # 앞 단계를 새로 돌렸으면 뒤 단계 캐시는 믿지 않는다. 키는 상위 '키'로 만들지
        # 상위 '내용'으로 만들지 않아서, VLM처럼 같은 설정에서도 결과가 달라질 수 있는
        # 단계 뒤에 낡은 산출물이 남아 있으면 그대로 재사용되기 때문이다.
        if i < force_from and not recomputed and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            outputs[name] = _load_output(name, saved["output"], path)
            rec = {**saved["record"], "cached": True}
        else:
            up = {u: outputs[u] for u in UPSTREAM[name]}
            res = _call(name, pdf_path, opts, up)
            outputs[name] = res.output
            rec = {"stage": name, "key": key, "seconds": res.seconds,
                   "resource": res.resource, "metrics": res.metrics, "path": str(path)}
            # 저장은 다음 단계를 부르기 전에 한다 — combine이 text의 PageResult를 제자리에서
            # 고치므로, 나중에 쓰면 text 산출물에 표가 섞여 들어간다.
            body = {"version": ARTIFACT_VERSION, "record": rec,
                    "config": stage_config(name, opts),
                    "output": _dump_output(name, res.output, path)}
            path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            rec = {**rec, "cached": False}
            recomputed = True
        records.append(rec)
        if on_stage:
            on_stage(rec)
    return records


# ── 리포트 ───────────────────────────────────────────────────────────────────

# 2 = ingest_many.DRY_RUN_CONTRACT와 같은 형식. 3 = table_cell_match(표 셀의 텍스트 레이어
# 일치율, 가드레일용) 추가 — pdf를 다시 열어야 해서 dry-run에는 없다. 필드는 추가만 됐으므로
# 벤치 비교는 2 리포트와도 짝지을 수 있다(없는 쪽은 판정을 건너뛴다).
REPORT_CONTRACT = 3


def write_report(path: Path, artifacts: Path, meta: DocMeta, records: list[dict],
                 pdf_path: Optional[str] = None) -> dict:
    """단계 기록 + 페이지별 표 지문 + 청크 요약. 벤치마크는 이 파일만 읽는다.

    청크 본문은 싣지 않는다 — 약관 원문을 벤치마크 결과에 복제하지 않기 위해서다.
    pdf_path를 주면 표 셀의 텍스트 레이어 일치율(table_cell_match)도 싣는다.
    """
    by = {r["stage"]: r for r in records}
    report: dict[str, Any] = {
        "contract": REPORT_CONTRACT, "doc_hash": meta.doc_hash, "source_pdf": meta.source_pdf,
        "code": code_fingerprint(),
        "stages": {r["stage"]: {"seconds": r["seconds"], "resource": r["resource"],
                                "cached": r["cached"], **r["metrics"]} for r in records},
        # 캐시에서 읽은 단계도 그 산출물을 만들 때 잰 시간이 들어 있으므로 합이 곧 처리 시간이다.
        "elapsed_s": round(sum(r["seconds"] for r in records), 2),
    }
    if "combine" in by:
        cpath = Path(by["combine"]["path"])
        pages = _load_output("combine", json.loads(cpath.read_text())["output"], cpath)
        report["tables_by_page"] = stages.table_fingerprint(pages)
        if pdf_path:
            report["table_cell_match"] = stages.table_cell_match(pages, pdf_path)
    if "chunk" in by:
        cpath = Path(by["chunk"]["path"])
        chunked = _load_output("chunk", json.loads(cpath.read_text())["output"], cpath)
        tokens = [c.token_count for c in chunked.chunks]
        report["summary"] = {
            "chunk_count": len(chunked.chunks),
            "tok_avg": round(sum(tokens) / len(tokens)) if tokens else 0,
            "over_600": sum(1 for t in tokens if t > 600),
            "warnings": len(chunked.warnings),
            "boundary_confidence": chunked.report.get("boundary_confidence"),
            "articles": sorted({c.article_number for c in chunked.chunks if c.article_number}),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="파이프라인 단계 분리 실행 (중간 산출물 저장·재사용)")
    p.add_argument("--pdf", required=True)
    p.add_argument("--artifacts", required=True, help="중간 산출물 디렉터리")
    p.add_argument("--from", dest="from_stage", choices=ORDER, default=None,
                   help="이 단계부터는 캐시가 있어도 다시 돈다")
    p.add_argument("--to", dest="to_stage", choices=ORDER, default="embed")
    p.add_argument("--doc-type", default=None, help="없으면 파일명으로 판정")
    p.add_argument("--insurer", default="미상")
    p.add_argument("--product", default=None, help="없으면 파일명")
    p.add_argument("--no-vision", action="store_true")
    p.add_argument("--target-tokens", type=int, default=500)
    p.add_argument("--hard-max-tokens", type=int, default=1000)
    p.add_argument("--ollama-url", default=None)
    p.add_argument("--embed-model", default=None)
    p.add_argument("--json", action="store_true", help="단계 기록을 JSON으로 출력")
    p.add_argument("--report", default=None,
                   help="dry-run 결과와 같은 계약(contract 2)의 요약 JSON을 이 경로에 쓴다")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    from .chunker import auto_doc_type
    from .models import compute_doc_hash

    args = _parse_args(argv)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    pdf = Path(args.pdf)
    meta = DocMeta(source_pdf=pdf.name, doc_hash=compute_doc_hash(str(pdf)),
                   doc_type=args.doc_type or auto_doc_type(pdf.name),
                   insurer=args.insurer, product_name=args.product or pdf.stem)

    records = run(str(pdf), Path(args.artifacts), meta,
                  from_stage=args.from_stage, to_stage=args.to_stage,
                  use_vision=not args.no_vision, target=args.target_tokens,
                  hard_max=args.hard_max_tokens, ollama_url=args.ollama_url,
                  embed_model=args.embed_model)
    if args.report:
        write_report(Path(args.report), Path(args.artifacts), meta, records,
                     pdf_path=str(pdf))
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return
    print(f"\n{pdf.name} (doc_hash {meta.doc_hash[:16]}, code {code_fingerprint()})")
    for r in records:
        mark = "캐시" if r["cached"] else "실행"
        print(f"  {r['stage']:<11} {r['resource']:<4} {r['seconds']:>8.2f}s  {mark}  "
              f"{json.dumps(r['metrics'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
