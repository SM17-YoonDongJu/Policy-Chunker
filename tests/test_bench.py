"""벤치마크(eval/bench) 계약.

개선 판단은 '같은 문서를 두 변형으로 돌려 짝지어 본다'에 기댄다. 짝짓기가 틀리거나, 결과가
바뀌면 안 되는 변경(병렬화)에서 바뀐 것을 못 잡으면 벤치마크가 거짓말을 한다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_HOME", str(tmp_path / "bench"))
    spec = importlib.util.spec_from_file_location("bench", REPO / "eval" / "bench" / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (mod.BENCH / "docsets").mkdir(parents=True)
    (mod.BENCH / "variants").mkdir(parents=True)
    return mod


def _write(bench, kind: str, name: str, data: dict) -> None:
    (bench.BENCH / kind / f"{name}.yaml").write_text(yaml.safe_dump(data, allow_unicode=True))


def _report(tables: dict, chunks=5, articles=("제1조", "제2조"), elapsed=10.0, vlm=6.0):
    return {"contract": 2, "elapsed_s": elapsed,
            "stages": {"text": {"seconds": elapsed - vlm, "resource": "cpu"},
                       "vlm": {"seconds": vlm, "resource": "gpu", "latencies": [vlm],
                               "outcomes": {"table": 1}}},
            "tables_by_page": {p: {"source": s, "sha256": h, "chars": 10}
                               for p, (s, h) in tables.items()},
            "summary": {"chunk_count": chunks, "articles": list(articles),
                        "boundary_confidence": "strong"}}


DOCSET = {"name": "t", "documents": [{"id": "a", "type": "table"}, {"id": "b", "type": "plain"},
                                     {"id": "c", "type": "plain"}]}


# ── 비교 판정 ─────────────────────────────────────────────────────────────────

def test_identical_results_pass(bench):
    ra = {"a": _report({"1": ("pymupdf", "h1"), "3": ("vlm", "h3")})}
    rb = {"a": _report({"1": ("pymupdf", "h1"), "3": ("vlm", "h3")}, elapsed=6.0, vlm=2.0)}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v", "expect_identical_to": "base"},
                      ra, rb)
    assert c["verdict"] == "PASS"
    assert c["bound"][1]["t_actual_s"] == 6.0


def test_one_changed_table_fails_an_identity_variant(bench):
    """병렬화가 페이지 하나의 표를 바꿨다면 버그다 — 평균 일치율이 높아도 FAIL이어야 한다."""
    ra = {"a": _report({str(p): ("vlm", f"h{p}") for p in range(1, 21)})}
    tb = {str(p): ("vlm", f"h{p}") for p in range(1, 21)}
    tb["7"] = ("vlm", "different")
    rb = {"a": _report(tb)}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v", "expect_identical_to": "base"},
                      ra, rb)
    assert c["verdict"] == "FAIL"
    assert c["docs"]["a"]["tables"]["changed"] == ["7"]
    assert c["docs"]["a"]["tables"]["match_rate"] == 0.95


def test_lost_article_fails_even_if_tables_match(bench):
    ra = {"a": _report({"1": ("pymupdf", "h1")})}
    rb = {"a": _report({"1": ("pymupdf", "h1")}, articles=("제1조",))}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v", "expect_identical_to": "base"},
                      ra, rb)
    assert c["verdict"] == "FAIL" and c["docs"]["a"]["articles_lost"] == ["제2조"]


def test_vlm_overwriting_a_cpu_page_fails_the_guardrail(bench):
    """결과가 바뀌는 변형도 가드레일은 본다.

    PyMuPDF가 읽던 페이지를 VLM이 덮어쓴 건 ICD 코드 사고(I00 → 100)와 같은 경우다.
    """
    ra = {"a": _report({"1": ("pymupdf", "h1")})}
    rb = {"a": _report({"1": ("vlm", "h2"), "2": ("vlm", "h3")})}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v"}, ra, rb)
    t = c["docs"]["a"]["tables"]
    assert t["changed"] == ["1"] and t["only_variant"] == ["2"] and t["source_moves"] == 1
    assert c["verdict"] == "FAIL"
    assert "CPU 소스가 읽던 페이지" in c["violations"]["a"][0]
    assert "품질 가드레일: FAIL" in bench.render(c)


def test_guardrail_can_be_turned_off(bench):
    ra = {"a": _report({"1": ("pymupdf", "h1")})}
    rb = {"a": _report({"1": ("vlm", "h2")})}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v", "guardrail": False}, ra, rb)
    assert c["verdict"] is None and c["violations"] == {}


def test_changes_inside_vlm_pages_pass_the_guardrail(bench):
    """VLM이 맡은 페이지의 표가 바뀌는 건 허용한다 — DPI·모델을 바꾸면 당연히 바뀐다."""
    ra = {"a": _report({"1": ("pymupdf", "h1"), "5": ("vlm", "v5")})}
    rb = {"a": _report({"1": ("pymupdf", "h1"), "5": ("vlm", "v5-new")})}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v"}, ra, rb)
    assert c["verdict"] == "PASS"


@pytest.mark.parametrize("case,expect", [
    ("articles", "조번호"), ("boundary", "경계 신뢰도"), ("tables", "표 페이지"),
    ("unknown", "[?] 셀")])
def test_each_guardrail_rule(bench, case, expect):
    base_tables = {str(p): ("vlm", f"v{p}") for p in range(1, 21)}
    ra = {"a": _report(base_tables)}
    rb = {"a": _report(dict(base_tables))}
    if case == "articles":
        rb["a"]["summary"]["articles"] = ["제1조"]
    elif case == "boundary":
        ra["a"]["summary"]["boundary_confidence"] = "ok"
        rb["a"]["summary"]["boundary_confidence"] = "weak"
    elif case == "tables":  # 20 → 18쪽 = -10% (허용 -5%)
        rb["a"]["tables_by_page"].pop("19")
        rb["a"]["tables_by_page"].pop("20")
    else:
        rb["a"]["tables_by_page"]["3"]["unknown_cells"] = 2
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v"}, ra, rb)
    assert c["verdict"] == "FAIL"
    assert any(expect in v for v in c["violations"]["a"])


def test_guardrail_thresholds_can_be_overridden(bench):
    """허용 기준은 초안이다. 변형마다 덮어쓸 수 있어야 합의 전에도 실험할 수 있다."""
    base_tables = {str(p): ("vlm", f"v{p}") for p in range(1, 21)}
    ra = {"a": _report(base_tables)}
    rb = {"a": _report({k: v for k, v in base_tables.items() if k not in ("19", "20")})}
    loose = {"slug": "v", "guardrail": {"min_table_pages_ratio": 0.85}}
    assert bench.compare(DOCSET, {"slug": "base"}, loose, ra, rb)["verdict"] == "PASS"


def test_only_paired_documents_are_compared(bench):
    """한쪽에만 있는 문서를 섞으면 문서 구성 차이가 시간 차이로 보인다."""
    ra = {"a": _report({}), "b": _report({}, elapsed=100.0)}
    rb = {"a": _report({})}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v"}, ra, rb)
    assert c["n"] == 1 and c["missing"] == ["b", "c"]
    assert c["bound"][0]["t_actual_s"] == 10.0


def test_groups_carry_their_own_n(bench):
    ra = {"a": _report({}), "b": _report({}), "c": _report({})}
    c = bench.compare(DOCSET, {"slug": "base"}, {"slug": "v"}, ra, dict(ra))
    assert c["groups"]["table"]["n"] == 1 and c["groups"]["plain"]["n"] == 2
    assert "| plain | 2 |" in bench.render(c)


# ── 끝에서 끝까지 (실제 stage_runner를 subprocess로) ──────────────────────────

@pytest.fixture
def table_pdf(tmp_path):
    import pymupdf
    doc = pymupdf.open()
    for k in range(3):
        p = doc.new_page()
        p.insert_text((70, 60), f"Article {k + 1} body", fontsize=12)
        for r in range(3):
            for c in range(3):
                rect = pymupdf.Rect(60 + c * 140, 100 + r * 28, 60 + (c + 1) * 140,
                                    100 + (r + 1) * 28)
                p.draw_rect(rect, color=(0, 0, 0), width=0.8)
                p.insert_text((rect.x0 + 5, rect.y0 + 18), f"p{k}r{r}c{c}", fontsize=9)
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def test_run_and_compare_end_to_end(bench, table_pdf, capsys):
    """VLM을 끈 두 변형(같은 설정)은 동일해야 한다. 두 번째는 전 단계를 캐시에서 읽는다."""
    _write(bench, "docsets", "smoke", {"name": "smoke", "documents": [
        {"id": "t1", "pdf": table_pdf, "type": "table", "doc_type": "plain"}]})
    _write(bench, "variants", "base", {"slug": "base", "ref": "WORKTREE",
                                       "env": {"VLM_BACKEND": "off"}})
    _write(bench, "variants", "same", {"slug": "same", "ref": "WORKTREE",
                                       "env": {"VLM_BACKEND": "off"},
                                       "expect_identical_to": "base"})
    bench.main(["run", "--docset", "smoke", "--variant", "base"])
    bench.main(["run", "--docset", "smoke", "--variant", "same"])

    rep = json.loads((bench.BENCH / "results" / "same" / "smoke" / "t1.json").read_text())
    assert all(s["cached"] for s in rep["stages"].values()), "같은 설정이면 캐시를 써야 한다"
    assert rep["tables_by_page"], "표 3쪽이 잡혀야 의미 있는 비교다"

    out = bench.BENCH / "r.md"
    bench.main(["compare", "--docset", "smoke", "--base", "base", "--variant", "same",
                "--out", str(out)])
    assert "결과 동일성: PASS" in out.read_text()
    meta = json.loads((bench.BENCH / "results" / "base" / "smoke" / "_run.json").read_text())
    assert meta["runs"][0]["ok"] and meta["version"]


# ── 클러스터(공유 PVC·git 없는 이미지) ────────────────────────────────────────

def test_data_dir_can_live_apart_from_definitions(tmp_path, monkeypatch):
    """정의는 이미지에서, 캐시·PDF·결과는 팀이 공유하는 PVC에서 읽는다."""
    monkeypatch.delenv("BENCH_HOME", raising=False)
    monkeypatch.setenv("BENCH_DATA", str(tmp_path / "pvc"))
    spec = importlib.util.spec_from_file_location("bench2", REPO / "eval" / "bench" / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.DEFS == REPO / "eval" / "bench"
    assert mod.BENCH == tmp_path / "pvc"
    assert mod.load_docset("smoke")["documents"], "정의는 이미지(레포)에 든 것을 읽어야 한다"


def test_image_without_git_runs_its_own_code_only(bench, monkeypatch):
    """이미지엔 .git이 없다. 다른 ref는 그 ref의 이미지로 돌려야 한다.

    조용히 현재 이미지 코드로 돌면 '변형 B'라는 이름으로 A를 잰 결과가 남는다.
    """
    monkeypatch.setattr(bench, "_in_git", lambda: False)
    monkeypatch.setenv("BENCH_VERSION", "sha-abc1234")
    code_dir, version = bench.checkout(bench.WORKTREE)
    assert code_dir == bench.REPO and version == "sha-abc1234"
    with pytest.raises(SystemExit):
        bench.checkout("main")
