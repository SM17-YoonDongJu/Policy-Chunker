"""벤치마크 — 고정 문서셋을 변형(코드 버전 + 환경변수)별로 돌리고 짝지어 비교한다.

운영 이력(runlog)은 7일 주기에 이미 적재된 문서를 건너뛰므로 기간마다 처리한 문서가 다르다.
그걸로 개선 전후를 비교하면 문서 구성이 바뀐 것이 개선 효과처럼 보인다. 여기서는 같은
문서들을 변형 A·B로 돌려 문서별로 짝지어 본다.

chunker 내부는 import하지 않는다. `python -m insurance_chunker.stage_runner --report`를
subprocess로 부르고 그 리포트(contract 2)만 읽는다. 변형이 다른 git ref를 가리키면 그 ref의
워크트리에서 부른다 — 비교하는 두 코드가 서로를 import하지 않게 하기 위해서다.

  python eval/bench/bench.py fetch   --docset core            # sha256 → S3에서 PDF 받기
  python eval/bench/bench.py run     --docset core --variant baseline [--from vlm] [--to chunk]
  python eval/bench/bench.py compare --docset core --base baseline --variant vlm-c2 [--out r.md]

배치:
  정의 (코드와 함께)
    eval/bench/docsets/<이름>.yaml   문서 목록 (PDF는 커밋하지 않는다 — sha256으로 지정)
    eval/bench/variants/<slug>.yaml  변형 정의 (ref·env·expect_identical_to·status·applied_at)
  데이터 (BENCH_DATA, 기본은 eval/bench — 클러스터에서는 팀 공유 PVC)
    results/<slug>/<docset>/<id>.json   문서별 리포트 (본문 없음, 수치·해시만)
    .artifacts/  단계 산출물 캐시(변형끼리 공유 — 키에 코드 지문·설정이 들어간다)
    .pdfs/       fetch로 받은 PDF
    .worktrees/  ref별 워크트리
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
# 정의(docsets·variants)는 코드와 함께 다니고, 데이터(캐시·PDF·결과)는 팀이 공유하는 볼륨에 둔다.
#   BENCH_DATA  데이터 위치. 클러스터에서는 공유 PVC 마운트 경로(예: /bench)
#   BENCH_HOME  정의와 데이터를 한꺼번에 옮긴다(테스트·별도 실험용)
DEFS = Path(os.environ.get("BENCH_HOME") or Path(__file__).resolve().parent)
BENCH = Path(os.environ.get("BENCH_DATA") or DEFS)
WORKTREE = "WORKTREE"  # ref 자리에 쓰면 커밋 안 된 변경까지 포함한 현재 작업 트리

sys.path.insert(0, str(REPO))


# ── 정의 읽기 ────────────────────────────────────────────────────────────────

def load_docset(name: str) -> dict[str, Any]:
    data = yaml.safe_load((DEFS / "docsets" / f"{name}.yaml").read_text(encoding="utf-8"))
    data.setdefault("name", name)
    for d in data["documents"]:
        d.setdefault("type", "plain")
        d.setdefault("doc_type", "policy_terms")
        d.setdefault("insurer", "미상")
    return data


def load_variant(slug: str) -> dict[str, Any]:
    data = yaml.safe_load((DEFS / "variants" / f"{slug}.yaml").read_text(encoding="utf-8"))
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

# corpus_worker는 S3 키를 파일 바이트 sha256으로 짓는다(corpus/terms/<sha256>.pdf). 그래서
# 대개 카탈로그 조회 없이 sha256만으로 받을 수 있다. 다른 키 규칙이면 docset에 s3_key를 쓴다.
_S3_PREFIX = os.environ.get("BENCH_S3_PREFIX", "corpus/terms/")

_PART_SQL = """
SELECT s3_key FROM ai.corpus_file_part
WHERE sha256 = %s AND status = 'uploaded' AND s3_key IS NOT NULL LIMIT 1
"""


def _catalog_key(sha256: str) -> Optional[str]:
    """키 규칙으로 못 찾을 때만 카탈로그를 본다(DATABASE_URL 필요)."""
    if not os.environ.get("DATABASE_URL"):
        return None
    import psycopg2
    with psycopg2.connect(os.environ["DATABASE_URL"]) as conn, conn.cursor() as cur:
        cur.execute(_PART_SQL, (sha256,))
        row = cur.fetchone()
    return row[0] if row else None


def _download(s3, bucket: str, key: str, dest: Path) -> Optional[str]:
    try:
        s3.download_file(bucket, key, str(dest))
    except Exception as e:  # noqa: BLE001 - 사유를 돌려주고 호출부가 다음 후보를 고른다
        return f"{type(e).__name__}: {e}"
    return None


def cmd_fetch(args: argparse.Namespace) -> None:
    """문서셋의 sha256(또는 s3_key)으로 S3에서 PDF를 받는다. S3_BUCKET·AWS 자격증명이 필요하다.

    받은 파일의 sha256을 다시 계산해 문서셋 값과 대조한다 — 키 규칙이 어긋나 다른 파일을
    받았다면 그 문서의 비교는 전부 무의미해진다.
    """
    import hashlib

    import boto3

    docset = load_docset(args.docset)
    todo = [d for d in docset["documents"] if d.get("sha256")
            and not (BENCH / ".pdfs" / f"{d['sha256']}.pdf").exists()]
    if not todo:
        print("받을 문서 없음")
        return
    (BENCH / ".pdfs").mkdir(parents=True, exist_ok=True)
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    bucket = os.environ["S3_BUCKET"]
    for d in todo:
        dest = BENCH / ".pdfs" / f"{d['sha256']}.pdf"
        primary = d.get("s3_key") or f"{_S3_PREFIX}{d['sha256']}.pdf"
        error = _download(s3, bucket, primary, dest)
        if error:  # 키 규칙이 어긋났을 수 있다 — 카탈로그에서 실제 키를 찾아 한 번 더
            key = _catalog_key(d["sha256"])
            error = _download(s3, bucket, key, dest) if key else error
        if error:
            print(f"  {d['id']}: 못 받음 ({d['sha256'][:12]}) {error}")
            continue
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != d["sha256"]:
            dest.unlink()
            print(f"  {d['id']}: sha256 불일치 — 받은 파일 {got[:12]}, 버림")
            continue
        print(f"  {d['id']}: 받음 ({dest.stat().st_size // 1024}KB)")


# ── run ──────────────────────────────────────────────────────────────────────

def _git(*argv: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *argv], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def _in_git() -> bool:
    return (REPO / ".git").exists()


def checkout(ref: str) -> tuple[Path, str]:
    """ref를 실행할 디렉터리와 기록용 버전 문자열. WORKTREE면 현재 트리(+dirty 표시).

    컨테이너 이미지에는 .git이 없다. 거기서는 이미지에 든 코드가 곧 변형의 코드이고, 버전은
    BENCH_VERSION(배포가 넣는 이미지 태그)으로 남긴다. 다른 ref를 비교하려면 그 태그의
    이미지로 따로 돌린다.
    """
    if not _in_git():
        if ref != WORKTREE:
            raise SystemExit(f"git이 없는 환경(이미지)에서는 ref={ref}를 돌릴 수 없다 — "
                             "그 ref의 이미지로 돌리고 변형의 ref는 WORKTREE로 둔다")
        return REPO, os.environ.get("BENCH_VERSION", "image")
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


# 결과가 바뀌는 변형(DPI·탐지 기준·엔진)의 품질 가드레일 — 목표는 시간 단축이고 품질은
# 기준선 대비 크게 나빠지지 않았는지만 본다. 값은 초안이다(설계 문서 2.4). 변형 정의의
# guardrail:로 항목별로 덮어쓰고, guardrail: false면 판정하지 않는다.
DEFAULT_GUARDRAIL: dict[str, Any] = {
    "max_articles_lost": 0,            # 빠진 조는 검색되지 않는다
    "boundary_no_downgrade": True,     # weak면 섹션이 안 갈려 조번호가 어긋난다
    "min_table_pages_ratio": 0.95,     # 호출 수를 줄이는 변경이 잃는 표의 폭
    "unknown_cells_no_increase": True,  # VLM이 읽지 못한 셀([?])
    "cpu_pages_unchanged": True,       # CPU 소스가 읽던 페이지를 VLM이 덮어쓰면 안 된다(ICD 사고)
}
_CPU_SOURCES = {"pymupdf", "pdfplumber", "camelot"}
_BOUNDARY_RANK = {"error": 0, "weak": 1, "ok": 2}


def guardrail_violations(a: dict[str, Any], b: dict[str, Any], d: dict[str, Any],
                         rules: dict[str, Any]) -> list[str]:
    """문서 하나의 가드레일 위반 목록. a=base 리포트, b=variant 리포트, d=doc_diff 결과."""
    out: list[str] = []
    if len(d["articles_lost"]) > rules["max_articles_lost"]:
        out.append(f"조번호 {len(d['articles_lost'])}개 손실({', '.join(d['articles_lost'][:5])})")
    ra, rb = (_BOUNDARY_RANK.get(x) for x in d["boundary"])
    if rules["boundary_no_downgrade"] and ra is not None and rb is not None and rb < ra:
        out.append(f"경계 신뢰도 하락({d['boundary'][0]}→{d['boundary'][1]})")
    na, nb = len(a.get("tables_by_page") or {}), len(b.get("tables_by_page") or {})
    if na and nb < na * rules["min_table_pages_ratio"]:
        out.append(f"표 페이지 {na}→{nb}({(nb - na) / na:+.1%}, 허용 "
                   f"{rules['min_table_pages_ratio'] - 1:+.0%})")
    ua, ub = d["tables"]["unknown_cells"]
    if rules["unknown_cells_no_increase"] and ub > ua:
        out.append(f"[?] 셀 {ua}→{ub}")
    if rules["cpu_pages_unchanged"]:
        ta, tb = a.get("tables_by_page") or {}, b.get("tables_by_page") or {}
        hit = sorted((p for p, v in ta.items() if v["source"] in _CPU_SOURCES
                      and (p not in tb or tb[p]["sha256"] != v["sha256"])), key=int)
        if hit:
            out.append(f"CPU 소스가 읽던 페이지의 표가 바뀜/사라짐: {', '.join(hit[:8])}")
    return out


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
    rules = variant.get("guardrail", True)
    rules = None if rules is False or expect else {
        **DEFAULT_GUARDRAIL, **(rules if isinstance(rules, dict) else {})}
    violations = ({i: v for i in paired
                   if (v := guardrail_violations(ra[i], rb[i], diffs[i], rules))}
                  if rules else {})
    if expect:
        verdict = "PASS" if all(identical(d) for d in diffs.values()) else "FAIL"
    elif rules:
        verdict = "FAIL" if violations else "PASS"
    else:
        verdict = None
    sa = metrics.stage_breakdown([ra[i] for i in paired])
    sb = metrics.stage_breakdown([rb[i] for i in paired])
    return {
        "base": base["slug"], "variant": variant["slug"], "docset": docset["name"],
        "n": len(paired), "missing": sorted(set(docs) - set(paired)),
        "bound": (sa["bound"], sb["bound"]),
        "vlm": (sa["vlm"], sb["vlm"]),
        "stages": (sa["stages"], sb["stages"]),
        "groups": {g: {"n": len(ids),
                       "elapsed": (round(sum(diffs[i]["elapsed"][0] for i in ids), 1),
                                   round(sum(diffs[i]["elapsed"][1] for i in ids), 1)),
                       "table_match": round(
                           sum(diffs[i]["tables"]["same"] for i in ids)
                           / max(1, sum(diffs[i]["tables"]["pages"] for i in ids)), 3)}
                   for g, ids in sorted(groups.items())},
        "docs": diffs,
        "expect_identical": expect,
        "guardrail": rules,
        "violations": violations,
        "verdict": verdict,
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
    if c["verdict"] and c["expect_identical"]:
        lines += [f"**결과 동일성: {c['verdict']}** (expect_identical_to={c['base']})", ""]
    elif c["verdict"]:
        lines += [f"**품질 가드레일: {c['verdict']}**", ""]
        for i, vs in c["violations"].items():
            lines += [f"- {i}: " + "; ".join(vs)]
        if c["violations"]:
            lines += [""]
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
    st_a, st_b = c["stages"]
    lines += [f"단계별 — ★는 variant에서 전체 하한을 결정하는 단계(병목 {bb['bottleneck']} → "
              f"{bb['limiting_stage']})", "",
              "| 단계 | 자원 | base 합 | variant 합 | 변화 | variant 단계 하한 | 하한 몫 |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted(set(st_a) | set(st_b), key=lambda k: -(st_b.get(k) or st_a[k])["total_s"]):
        a, b = st_a.get(k) or {}, st_b.get(k) or {}
        mark = "★ " if b.get("critical") else ""
        share = f"{b['bound_share']:.1%}" if b.get("critical") else "-"
        lines.append(f"| {mark}{k} | {(b or a).get('resource')} | {a.get('total_s', 0)}s | "
                     f"{b.get('total_s', 0)}s | "
                     f"{_pct_change(a.get('total_s', 0), b.get('total_s', 0))} | "
                     f"{b.get('bound_s', 0)}s | {share} |")
    lines += [""]
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
