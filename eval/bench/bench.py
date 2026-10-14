"""벤치마크 — 고정 문서셋을 변형(코드 버전 + 환경변수)별로 돌리고 짝지어 비교한다.

운영 이력(runlog)은 7일 주기에 이미 적재된 문서를 건너뛰므로 기간마다 처리한 문서가 다르다.
그걸로 개선 전후를 비교하면 문서 구성이 바뀐 것이 개선 효과처럼 보인다. 여기서는 같은
문서들을 변형 A·B로 돌려 문서별로 짝지어 본다.

chunker 내부는 import하지 않는다. `python -m insurance_chunker.stage_runner --report`를
subprocess로 부르고 그 리포트(contract 2)만 읽는다. 변형이 다른 git ref를 가리키면 그 ref의
워크트리에서 부른다 — 비교하는 두 코드가 서로를 import하지 않게 하기 위해서다.

  python eval/bench/bench.py fetch   --docset core            # 카탈로그 sha256 → PDF 내려받기
  python eval/bench/bench.py run     --docset core --variant baseline [--from vlm] [--to chunk]
  python eval/bench/bench.py compare --docset core --base baseline --variant vlm-c2 [--out r.md]

배치:
  eval/bench/docsets/<이름>.yaml   문서 목록 (PDF는 커밋하지 않는다 — sha256으로 지정)
  eval/bench/variants/<slug>.yaml  변형 정의 (ref·env·expect_identical_to·status·applied_at)
  eval/bench/results/<slug>/<docset>/<id>.json   문서별 리포트 (본문 없음, 수치·해시만)
  eval/bench/.artifacts/  단계 산출물 캐시(변형끼리 공유 — 키에 코드 지문·설정이 들어간다)
  eval/bench/.pdfs/       fetch로 받은 PDF
  eval/bench/.worktrees/  ref별 워크트리
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
# 정의·결과·캐시가 놓이는 곳. 테스트나 별도 실험은 BENCH_HOME으로 옮긴다.
BENCH = Path(os.environ.get("BENCH_HOME") or Path(__file__).resolve().parent)
WORKTREE = "WORKTREE"  # ref 자리에 쓰면 커밋 안 된 변경까지 포함한 현재 작업 트리

sys.path.insert(0, str(REPO))


# ── 정의 읽기 ────────────────────────────────────────────────────────────────

def load_docset(name: str) -> dict[str, Any]:
    data = yaml.safe_load((BENCH / "docsets" / f"{name}.yaml").read_text(encoding="utf-8"))
    data.setdefault("name", name)
    for d in data["documents"]:
        d.setdefault("type", "plain")
        d.setdefault("doc_type", "policy_terms")
        d.setdefault("insurer", "미상")
    return data


def load_variant(slug: str) -> dict[str, Any]:
    data = yaml.safe_load((BENCH / "variants" / f"{slug}.yaml").read_text(encoding="utf-8"))
    data.setdefault("slug", slug)
    data.setdefault("ref", WORKTREE)
    data.setdefault("env", {})
    return data


def resolve_pdf(doc: dict[str, Any]) -> Path:
    """로컬 경로가 있으면 그것, 없으면 fetch로 받아 둔 .pdfs/<sha256>.pdf."""
    if doc.get("pdf"):
        p = Path(doc["pdf"])
        p = p if p.is_absolute() else REPO / p
        if p.exists():
            return p
    if doc.get("sha256"):
        p = BENCH / ".pdfs" / f"{doc['sha256']}.pdf"
        if p.exists():
            return p
    raise FileNotFoundError(f"{doc['id']}: PDF 없음 — 경로를 확인하거나 `bench.py fetch`를 먼저")


# ── fetch ────────────────────────────────────────────────────────────────────

_PART_SQL = """
SELECT s3_key FROM ai.corpus_file_part
WHERE sha256 = %s AND status = 'uploaded' AND s3_key IS NOT NULL LIMIT 1
"""


def cmd_fetch(args: argparse.Namespace) -> None:
    """카탈로그(ai.corpus_file_part)에서 sha256으로 S3 키를 찾아 내려받는다.

    DATABASE_URL·S3_BUCKET·AWS 자격증명이 필요하다(ingest_catalog와 같은 변수).
    """
    import boto3
    import psycopg2

    docset = load_docset(args.docset)
    todo = [d for d in docset["documents"] if d.get("sha256")
            and not (BENCH / ".pdfs" / f"{d['sha256']}.pdf").exists()]
    if not todo:
        print("받을 문서 없음")
        return
    (BENCH / ".pdfs").mkdir(exist_ok=True)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        for d in todo:
            cur.execute(_PART_SQL, (d["sha256"],))
            row = cur.fetchone()
            if not row:
                print(f"  {d['id']}: 카탈로그에 없음 ({d['sha256'][:12]})")
                continue
            dest = BENCH / ".pdfs" / f"{d['sha256']}.pdf"
            s3.download_file(bucket, row[0], str(dest))
            print(f"  {d['id']}: 받음 ({dest.stat().st_size // 1024}KB)")


# ── run ──────────────────────────────────────────────────────────────────────

def _git(*argv: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *argv], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def checkout(ref: str) -> tuple[Path, str]:
    """ref를 실행할 디렉터리와 기록용 버전 문자열. WORKTREE면 현재 트리(+dirty 표시)."""
    if ref == WORKTREE:
        sha = _git("rev-parse", "--short", "HEAD")
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
        return REPO, f"{sha}{'+dirty' if dirty else ''}"
    sha = _git("rev-parse", "--short", ref)
    wt = BENCH / ".worktrees" / sha
    if not wt.exists():
        _git("worktree", "add", "--detach", str(wt), sha)
    return wt, sha


def run_doc(code_dir: Path, doc: dict[str, Any], variant: dict[str, Any], report: Path,
            from_stage: Optional[str], to_stage: str) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "insurance_chunker.stage_runner",
           "--pdf", str(resolve_pdf(doc)), "--artifacts", str(BENCH / ".artifacts"),
           "--to", to_stage, "--doc-type", doc["doc_type"], "--insurer", doc["insurer"],
           "--report", str(report)]
    if from_stage:
        cmd += ["--from", from_stage]
    env = {**os.environ, **{k: str(v) for k, v in variant["env"].items()},
           "LOG_LEVEL": os.environ.get("BENCH_LOG_LEVEL", "WARNING")}
    t0 = time.time()
    r = subprocess.run(cmd, cwd=code_dir, env=env, capture_output=True, text=True)
    return {"id": doc["id"], "ok": r.returncode == 0, "wall_s": round(time.time() - t0, 1),
            "error": (r.stderr or "")[-800:] if r.returncode else None}


def cmd_run(args: argparse.Namespace) -> None:
    docset, variant = load_docset(args.docset), load_variant(args.variant)
    code_dir, version = checkout(variant["ref"])
    out_dir = BENCH / "results" / variant["slug"] / docset["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{variant['slug']} @ {version} × {docset['name']} ({len(docset['documents'])}건)")

    runs = []
    for d in docset["documents"]:
        res = run_doc(code_dir, d, variant, out_dir / f"{d['id']}.json",
                      args.from_stage, args.to_stage)
        runs.append(res)
        mark = "ok" if res["ok"] else "실패"
        print(f"  {d['id']:<24} {mark} {res['wall_s']}s")
        if res["error"]:
            print("    " + res["error"].strip().splitlines()[-1])
    meta = {"variant": variant, "version": version, "docset": docset["name"],
            "from": args.from_stage, "to": args.to_stage,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": runs}
    (out_dir / "_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    if not all(r["ok"] for r in runs):
        sys.exit(1)


# ── compare ──────────────────────────────────────────────────────────────────

def load_results(slug: str, docset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out_dir = BENCH / "results" / slug / docset["name"]
    out = {}
    for d in docset["documents"]:
        p = out_dir / f"{d['id']}.json"
        if p.exists():
            out[d["id"]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def table_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """페이지별 채택 표 비교. 해시가 같으면 같은 표, 출처가 바뀌었는지도 센다."""
    ta, tb = a.get("tables_by_page") or {}, b.get("tables_by_page") or {}
    pages = set(ta) | set(tb)
    same = sum(1 for p in pages if p in ta and p in tb and ta[p]["sha256"] == tb[p]["sha256"])
    return {
        "pages": len(pages), "same": same,
        "changed": sorted((p for p in pages if p in ta and p in tb
                           and ta[p]["sha256"] != tb[p]["sha256"]), key=int),
        "only_base": sorted(set(ta) - set(tb), key=int),
        "only_variant": sorted(set(tb) - set(ta), key=int),
        "source_moves": sum(1 for p in pages if p in ta and p in tb
                            and ta[p]["source"] != tb[p]["source"]),
        "match_rate": round(same / len(pages), 3) if pages else 1.0,
        "unknown_cells": (sum(v.get("unknown_cells", 0) for v in ta.values()),
                          sum(v.get("unknown_cells", 0) for v in tb.values())),
    }


def doc_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    sa, sb = a.get("summary") or {}, b.get("summary") or {}
    arts_a, arts_b = set(sa.get("articles") or []), set(sb.get("articles") or [])
    stages = sorted(set(a.get("stages") or {}) | set(b.get("stages") or {}))
    return {
        "elapsed": (a.get("elapsed_s", 0.0), b.get("elapsed_s", 0.0)),
        "stages": {s: ((a["stages"].get(s) or {}).get("seconds", 0.0),
                       (b["stages"].get(s) or {}).get("seconds", 0.0)) for s in stages},
        "tables": table_diff(a, b),
        "chunks": (sa.get("chunk_count"), sb.get("chunk_count")),
        "articles_lost": sorted(arts_a - arts_b),
        "articles_gained": sorted(arts_b - arts_a),
        "boundary": (sa.get("boundary_confidence"), sb.get("boundary_confidence")),
    }


def identical(d: dict[str, Any]) -> bool:
    """결과가 바뀌면 안 되는 변경(병렬화·단계 겹치기)의 판정 — 표·청크·조번호가 전부 같아야 한다."""
    t = d["tables"]
    return (t["match_rate"] == 1.0 and not t["only_base"] and not t["only_variant"]
            and d["chunks"][0] == d["chunks"][1]
            and not d["articles_lost"] and not d["articles_gained"])


def compare(docset: dict[str, Any], base: dict[str, Any], variant: dict[str, Any],
            ra: dict[str, dict], rb: dict[str, dict]) -> dict[str, Any]:
    import metrics  # 운영 지표와 같은 하한 계산을 쓴다(입력은 리포트 dict라 chunker와 무관)

    docs = {d["id"]: d for d in docset["documents"]}
    paired = sorted(set(ra) & set(rb))
    diffs = {i: doc_diff(ra[i], rb[i]) for i in paired}
    groups: dict[str, list[str]] = defaultdict(list)
    for i in paired:
        groups[docs[i]["type"]].append(i)

    expect = variant.get("expect_identical_to") == base["slug"]
    return {
        "base": base["slug"], "variant": variant["slug"], "docset": docset["name"],
        "n": len(paired), "missing": sorted(set(docs) - set(paired)),
        "bound": (metrics.stage_breakdown([ra[i] for i in paired])["bound"],
                  metrics.stage_breakdown([rb[i] for i in paired])["bound"]),
        "vlm": (metrics.stage_breakdown([ra[i] for i in paired])["vlm"],
                metrics.stage_breakdown([rb[i] for i in paired])["vlm"]),
        "groups": {g: {"n": len(ids),
                       "elapsed": (round(sum(diffs[i]["elapsed"][0] for i in ids), 1),
                                   round(sum(diffs[i]["elapsed"][1] for i in ids), 1)),
                       "table_match": round(
                           sum(diffs[i]["tables"]["same"] for i in ids)
                           / max(1, sum(diffs[i]["tables"]["pages"] for i in ids)), 3)}
                   for g, ids in sorted(groups.items())},
        "docs": diffs,
        "expect_identical": expect,
        "verdict": (("PASS" if all(identical(d) for d in diffs.values()) else "FAIL")
                    if expect else None),
    }


def _pct_change(a: float, b: float) -> str:
    return f"{(b - a) / a:+.1%}" if a else "-"


def _lat(v: dict[str, Any]) -> str:
    return f"{v['latency_p50_s']}s / {v['latency_p95_s']}s" if v["calls"] else "-"


def _rate(r: Optional[float]) -> str:
    return "-" if r is None else f"{r:.1%}"


def render(c: dict[str, Any]) -> str:
    ba, bb = c["bound"]
    va, vb = c["vlm"]
    lines = [f"## {c['variant']} vs {c['base']} — {c['docset']} (n={c['n']})", ""]
    if c["verdict"]:
        lines += [f"**결과 동일성: {c['verdict']}** (expect_identical_to={c['base']})", ""]
    if c["missing"]:
        lines += [f"짝이 없어 뺀 문서: {', '.join(c['missing'])}", ""]
    lines += ["| | base | variant | 변화 |", "|---|---|---|---|",
              f"| 실제 시간 합 | {ba['t_actual_s']}s | {bb['t_actual_s']}s | "
              f"{_pct_change(ba['t_actual_s'], bb['t_actual_s'])} |",
              f"| 하한 | {ba['t_bound_s']}s | {bb['t_bound_s']}s | "
              f"{_pct_change(ba['t_bound_s'], bb['t_bound_s'])} |",
              f"| 차이(병렬화 몫) | {ba['gap_s']}s | {bb['gap_s']}s | |",
              f"| 병목 | {ba['bottleneck']} | {bb['bottleneck']} | |",
              f"| VLM 호출 | {va['calls']} | {vb['calls']} | |",
              f"| VLM p50 / p95 | {_lat(va)} | {_lat(vb)} | |",
              f"| VLM 타임아웃 | {_rate(va['timeout_rate'])} | {_rate(vb['timeout_rate'])} | |",
              ""]
    lines += ["| 유형 | n | 시간 base → variant | 표 해시 일치 |", "|---|---|---|---|"]
    for g, s in c["groups"].items():
        lines.append(f"| {g} | {s['n']} | {s['elapsed'][0]}s → {s['elapsed'][1]}s | "
                     f"{s['table_match']:.1%} |")
    lines += ["", "| 문서 | 시간 | 표 일치 | 바뀐 페이지 | 청크 | 조번호 손실 | 경계 |",
              "|---|---|---|---|---|---|---|"]
    for i, d in c["docs"].items():
        t = d["tables"]
        changed = t["changed"] + [f"-{p}" for p in t["only_base"]] + \
            [f"+{p}" for p in t["only_variant"]]
        lines.append(
            f"| {i} | {d['elapsed'][0]}→{d['elapsed'][1]}s | {t['match_rate']:.0%} | "
            f"{', '.join(map(str, changed[:8])) or '-'} | {d['chunks'][0]}→{d['chunks'][1]} | "
            f"{', '.join(d['articles_lost'][:5]) or '-'} | {d['boundary'][0]}→{d['boundary'][1]} |")
    return "\n".join(lines) + "\n"


def cmd_compare(args: argparse.Namespace) -> None:
    docset = load_docset(args.docset)
    base, variant = load_variant(args.base), load_variant(args.variant)
    c = compare(docset, base, variant, load_results(base["slug"], docset),
                load_results(variant["slug"], docset))
    md = render(c)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    if c["verdict"] == "FAIL":
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    stages = ["text", "tables_cpu", "vlm", "combine", "chunk", "embed"]
    p = argparse.ArgumentParser(description="고정 문서셋 × 변형 벤치마크")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="카탈로그 sha256 → PDF 내려받기")
    f.add_argument("--docset", required=True)
    f.set_defaults(fn=cmd_fetch)

    r = sub.add_parser("run", help="변형 하나로 문서셋을 돌린다")
    r.add_argument("--docset", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--from", dest="from_stage", choices=stages, default=None,
                   help="이 단계부터 캐시를 무시하고 다시 돈다(같은 설정 반복 측정 등)")
    r.add_argument("--to", dest="to_stage", choices=stages, default="chunk",
                   help="기본 chunk — embed는 Ollama가 필요하다")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="두 변형의 결과를 문서별로 짝지어 비교")
    c.add_argument("--docset", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--variant", required=True)
    c.add_argument("--out", default=None, help="마크다운 리포트 저장 경로")
    c.set_defaults(fn=cmd_compare)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
