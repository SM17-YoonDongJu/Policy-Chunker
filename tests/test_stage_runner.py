"""단계 분리 실행(stage_runner) 계약.

벤치마크는 '설정 하나만 바꿔 그 단계부터 다시 돌리는 것'에 기댄다. 그러려면 셋이 지켜져야
한다. 1) 저장했다 읽은 산출물이 메모리 경로와 같다. 2) 바꾼 설정이 닿는 단계와 그 아래만
다시 돈다. 3) 낡은 산출물을 조용히 재사용하지 않는다 — 그러면 개선 효과로 둔갑한다.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_stages import _as_tuples, ex, fake_vlm, mixed_pdf  # noqa: E402,F401 - 픽스처 공유


@pytest.fixture
def runner(ex):  # noqa: F811
    def _load(**env):
        ex(**{"VLM_BACKEND": "local", "VLM_MODEL": "m", **env})
        from insurance_chunker import stage_runner
        return stage_runner
    return _load


def _meta(pdf: str, doc_type: str = "plain"):
    from insurance_chunker.models import DocMeta, compute_doc_hash
    return DocMeta(source_pdf=Path(pdf).name, doc_hash=compute_doc_hash(pdf),
                   doc_type=doc_type, insurer="테스트", product_name="상품")


def _by_stage(records):
    return {r["stage"]: r for r in records}


# ── 저장·재사용 ───────────────────────────────────────────────────────────────

def test_second_run_reads_everything_from_cache(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    sr = runner()
    first = sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="chunk")
    calls = fake_vlm["n"]
    second = sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="chunk")

    assert not any(r["cached"] for r in first)
    assert all(r["cached"] for r in second)
    assert fake_vlm["n"] == calls, "캐시가 있는데 VLM을 다시 불렀다"
    assert [r["key"] for r in first] == [r["key"] for r in second]


def test_cached_output_matches_the_in_memory_path(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    """저장했다 읽은 combine 결과가 stages.parse(운영 경로)와 같아야 한다."""
    sr = runner()
    from insurance_chunker import stages

    in_memory, _ = stages.parse(mixed_pdf)
    fake_vlm["n"] = 0
    sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine")
    path = Path(_by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf),
                                 to_stage="combine"))["combine"]["path"])
    loaded = sr._load_output("combine", json.loads(path.read_text())["output"], path)
    assert _as_tuples(loaded) == _as_tuples(in_memory)


def test_text_artifact_is_not_polluted_by_combine(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    """combine은 text의 PageResult에 표를 제자리에서 붙인다. 저장 시점이 늦으면 섞인다."""
    sr = runner()
    recs = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine"))
    saved = json.loads(Path(recs["text"]["path"]).read_text())["output"]
    assert all(p["tables"] == [] for p in saved["pages"])
    assert recs["combine"]["metrics"]["table_pages"] > 0


def test_chunk_round_trip(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    sr = runner()
    from insurance_chunker import stages

    recs = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="chunk"))
    path = Path(recs["chunk"]["path"])
    loaded = sr._load_output("chunk", json.loads(path.read_text())["output"], path)
    pages, _ = stages.parse(mixed_pdf)
    direct = stages.chunk(pages, _meta(mixed_pdf), pdf_path=None).output
    assert [c.content for c in loaded.chunks] == [c.content for c in direct.chunks]
    assert loaded.warnings == direct.warnings


def test_embeddings_round_trip_through_npy(runner, fake_vlm, mixed_pdf, tmp_path,  # noqa: F811
                                           monkeypatch):
    """벡터는 .npy로 따로 둔다. 임베딩을 건너뛴 청크(boilerplate)는 None으로 돌아와야 한다."""
    sr = runner()
    from insurance_chunker import embedder

    def _fake_embed(chunks, ollama_url=None, model=None):
        for i, c in enumerate(chunks):
            c.embedding = None if i == 0 else [float(i), 0.5, -1.0]
        return chunks

    monkeypatch.setattr(embedder, "embed_chunks", _fake_embed)
    recs = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="embed"))
    path = Path(recs["embed"]["path"])
    assert path.with_suffix(".npy").exists()
    loaded = sr._load_output("embed", json.loads(path.read_text())["output"], path)
    assert loaded[0].embedding is None
    assert loaded[1].embedding == pytest.approx([1.0, 0.5, -1.0])


# ── 어디부터 다시 도는가 ──────────────────────────────────────────────────────

def test_from_reruns_that_stage_and_below_only(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    sr = runner()
    sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="chunk")
    fake_vlm["n"] = 0
    recs = sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), from_stage="vlm", to_stage="chunk")
    cached = {r["stage"]: r["cached"] for r in recs}
    assert cached == {"text": True, "tables_cpu": True, "vlm": False,
                      "combine": False, "chunk": False}
    assert fake_vlm["n"] > 0


def test_vlm_setting_change_invalidates_vlm_and_below(runner, fake_vlm, mixed_pdf,  # noqa: F811
                                                      tmp_path):
    """DPI만 바꾼 변형은 텍스트·CPU 표를 재사용하고 VLM부터 다시 돈다 — 벤치마크의 기본 동작."""
    sr = runner(VLM_DPI="150")
    base = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine"))
    sr = runner(VLM_DPI="200")
    after = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine"))

    assert after["text"]["cached"] and after["tables_cpu"]["cached"]
    assert not after["vlm"]["cached"] and not after["combine"]["cached"]
    assert after["vlm"]["key"] != base["vlm"]["key"]
    assert after["combine"]["key"] != base["combine"]["key"]


def test_target_mode_change_invalidates_from_tables_cpu(runner, fake_vlm, mixed_pdf,  # noqa: F811
                                                        tmp_path):
    """VLM 대상 선정은 S2에서 한다. 모드를 바꾸면 S2부터 다시 돌아야 한다."""
    sr = runner(VLM_TARGET="gap")
    sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="vlm")
    sr = runner(VLM_TARGET="tables")
    recs = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="vlm"))
    assert recs["text"]["cached"]
    assert not recs["tables_cpu"]["cached"] and not recs["vlm"]["cached"]


def test_code_change_invalidates_everything(runner, fake_vlm, mixed_pdf, tmp_path,  # noqa: F811
                                            monkeypatch):
    sr = runner()
    sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine")
    monkeypatch.setattr(sr, "code_fingerprint", lambda: "changed")
    recs = sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine")
    assert not any(r["cached"] for r in recs)


def test_recomputed_upstream_forces_downstream(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    """VLM 산출물만 지워졌는데 combine 캐시가 남아 있으면, combine도 다시 돌아야 한다.

    키는 상위 '내용'이 아니라 상위 '키'로 만든다. 같은 설정에서도 결과가 달라지는 VLM 뒤에
    낡은 combine이 남아 있으면 그대로 재사용될 수 있다.
    """
    sr = runner()
    recs = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine"))
    Path(recs["vlm"]["path"]).unlink()
    again = _by_stage(sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), to_stage="combine"))
    assert again["tables_cpu"]["cached"]
    assert not again["vlm"]["cached"] and not again["combine"]["cached"]


def test_from_after_to_is_rejected(runner, mixed_pdf, tmp_path):  # noqa: F811
    sr = runner()
    with pytest.raises(ValueError):
        sr.run(mixed_pdf, tmp_path, _meta(mixed_pdf), from_stage="chunk", to_stage="vlm")


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_prints_stage_records(runner, fake_vlm, mixed_pdf, tmp_path, capsys):  # noqa: F811
    sr = runner()
    importlib.reload(sr)
    sr.main(["--pdf", mixed_pdf, "--artifacts", str(tmp_path), "--to", "combine",
             "--doc-type", "plain", "--json"])
    recs = json.loads(capsys.readouterr().out)
    assert [r["stage"] for r in recs] == ["text", "tables_cpu", "vlm", "combine"]
    assert recs[2]["resource"] == "gpu"


def test_report_follows_the_dry_run_contract(runner, fake_vlm, mixed_pdf, tmp_path):  # noqa: F811
    """벤치마크는 chunker 내부를 import하지 않고 이 파일만 읽는다. 청크 본문은 없어야 한다."""
    sr = runner()
    report_path = tmp_path / "r.json"
    sr.main(["--pdf", mixed_pdf, "--artifacts", str(tmp_path / "a"), "--to", "chunk",
             "--doc-type", "plain", "--report", str(report_path)])
    rep = json.loads(report_path.read_text())
    import ingest_many
    assert rep["contract"] == ingest_many.DRY_RUN_CONTRACT
    assert {"doc_hash", "stages", "tables_by_page", "summary", "elapsed_s"} <= rep.keys()
    assert "chunks" not in rep and "content" not in json.dumps(rep["summary"])
    assert rep["elapsed_s"] == pytest.approx(sum(s["seconds"] for s in rep["stages"].values()),
                                             abs=0.05)
