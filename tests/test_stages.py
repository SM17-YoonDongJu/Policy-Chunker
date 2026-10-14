"""파이프라인 단계 분리 계약.

parse_pdf 한 번이 텍스트·CPU 표·VLM·best-of를 전부 해서 runlog에는 `parse` 하나로만 남았다.
VLM(GPU)이 그 안에서 얼마를 먹는지 알 수 없었고, 단계 경계가 없으니 병렬화할 자리도 없었다.

단계를 나누는 건 순수한 리팩터링이어야 한다 — 나눈 경로와 예전 경로가 같은 페이지·표를
내야 하고, 운영 지표(phases)의 이름도 그대로여야 한다. 여기서 그 둘을 고정한다.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Resp:
    status_code = 200

    def __init__(self, content):
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}

    def raise_for_status(self):
        pass


@pytest.fixture
def ex(monkeypatch):
    """환경변수를 바꿔 extractor를 다시 읽는다. stages는 extractor를 모듈째로 참조한다."""
    def _load(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        from insurance_chunker import extractor
        importlib.reload(extractor)
        return extractor
    return _load


@pytest.fixture
def fake_vlm(monkeypatch):
    """페이지마다 다른 표를 돌려준다 — 같은 응답이면 페이지가 뒤섞여도 테스트가 못 잡는다."""
    import requests
    calls = {"n": 0}

    def _post(url, json=None, timeout=None):
        calls["n"] += 1
        return _Resp(f"| v | n |\n|---|---|\n| vlm | {calls['n']} |")

    monkeypatch.setattr(requests, "post", _post)
    return calls


@pytest.fixture
def mixed_pdf(tmp_path):
    """괘선 표 / 본문 / 이미지만 / 괘선 많은 본문 — VLM 대상 선정의 세 갈래가 다 나온다."""
    import pymupdf
    doc = pymupdf.open()

    p = doc.new_page()
    p.insert_text((70, 60), "제1조(목적) 이 약관은 보험계약의 내용을 정한다.", fontsize=11)
    for r in range(4):
        for c in range(3):
            rect = pymupdf.Rect(60 + c * 140, 80 + r * 28, 60 + (c + 1) * 140, 80 + (r + 1) * 28)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p.insert_text((rect.x0 + 5, rect.y0 + 18), f"r{r}c{c}", fontsize=9)

    p = doc.new_page()
    p.insert_text((70, 100), "제2조(정의) 본문만 있는 페이지입니다.", fontsize=11)

    img = doc.new_page()
    pix = doc[0].get_pixmap(dpi=72)
    img.insert_image(img.rect, stream=pix.tobytes("png"))

    p = doc.new_page()
    p.insert_text((70, 90), "괘선은 많지만 표로는 안 잡히는 페이지", fontsize=11)
    for i in range(30):
        p.draw_line(pymupdf.Point(60, 110 + i * 20), pymupdf.Point(520, 110 + i * 20))

    path = tmp_path / "mixed.pdf"
    doc.save(path)
    doc.close()
    return str(path)


def _legacy_parse(pdf_path: str, use_vision: bool):
    """단계 분리 전 parse_pdf 본문 그대로. 비교 기준으로만 쓴다."""
    from insurance_chunker.combine import select_best_tables
    from insurance_chunker.extractor import extract_tables_for_doc
    from insurance_chunker.pdf_parser import parse_text

    pages, page_numbers = parse_text(pdf_path)
    best = select_best_tables(extract_tables_for_doc(pdf_path, page_numbers,
                                                     use_vision=use_vision))
    for page in pages:
        entry = best.get(page.page_num)
        if entry:
            md, src = entry
            page.tables = [{"markdown": md, "source": src}]
    return pages


def _as_tuples(pages):
    return [(p.page_num, p.text, p.tables, p.is_ocr) for p in pages]


# ── 동일성: 나눈 경로 = 예전 경로 ─────────────────────────────────────────────

@pytest.mark.parametrize("target", ["gap", "tables", "both"])
def test_staged_parse_matches_the_legacy_path(ex, fake_vlm, mixed_pdf, target):
    """VLM 대상 선정 세 모드 모두에서 페이지 텍스트와 부착된 표가 같아야 한다.

    both는 VLM과 PyMuPDF가 같은 페이지를 두고 경쟁하므로 best-of 선택까지 검증된다.
    """
    ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET=target)
    from insurance_chunker import stages

    legacy = _legacy_parse(mixed_pdf, use_vision=True)
    legacy_calls = fake_vlm["n"]
    fake_vlm["n"] = 0
    staged, _ = stages.parse(mixed_pdf, use_vision=True)

    assert legacy_calls > 0, "VLM이 한 번도 안 불리면 의미 있는 비교가 아니다"
    assert fake_vlm["n"] == legacy_calls, "VLM 호출 수가 달라졌다"
    assert _as_tuples(staged) == _as_tuples(legacy)


def test_staged_parse_matches_without_vision(ex, fake_vlm, mixed_pdf):
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    legacy = _legacy_parse(mixed_pdf, use_vision=False)
    staged, _ = stages.parse(mixed_pdf, use_vision=False)
    assert fake_vlm["n"] == 0
    assert _as_tuples(staged) == _as_tuples(legacy)


def test_parse_pdf_goes_through_the_stages(ex, fake_vlm, mixed_pdf):
    """parse_pdf는 호출부가 eval까지 여럿이라 남겨 두되, 경로는 하나여야 한다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages
    from insurance_chunker.pdf_parser import parse_pdf

    via_stages, _ = stages.parse(mixed_pdf)
    fake_vlm["n"] = 0  # 가짜 응답이 호출 순번을 담으므로 같은 출발점에서 비교한다
    assert _as_tuples(parse_pdf(mixed_pdf)) == _as_tuples(via_stages)


# ── 단계 기록 ─────────────────────────────────────────────────────────────────

def test_stages_are_recorded_in_order_with_resources(ex, fake_vlm, mixed_pdf):
    """자원 태그로 CPU·GPU 합계를 따로 내야 병렬화 하한을 계산할 수 있다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    _, results = stages.parse(mixed_pdf)
    assert [(r.name, r.resource) for r in results] == [
        ("text", "cpu"), ("tables_cpu", "cpu"), ("vlm", "gpu"), ("combine", "cpu")]
    assert all(r.seconds >= 0 for r in results)


def test_vlm_stage_is_measured_on_its_own(ex, fake_vlm, mixed_pdf):
    """이번 분리의 핵심 — VLM 호출 수와 대상 수가 parse 전체가 아니라 VLM 단계에 붙는다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    _, results = stages.parse(mixed_pdf)
    s2, s3 = results[1], results[2]
    assert s3.metrics["targets"] == s2.metrics["vlm_targets"] > 0
    assert s3.metrics["calls"] == fake_vlm["n"]
    assert s3.metrics["tables"] == len(s3.output)


def test_vlm_stage_is_recorded_even_when_skipped(ex, fake_vlm, mixed_pdf):
    """VLM을 끈 실행도 단계는 남아야 한다 — 빠지면 '0초'와 '안 돌림'을 구분 못 한다."""
    ex(VLM_BACKEND="off", VLM_MODEL="m")
    from insurance_chunker import stages

    _, results = stages.parse(mixed_pdf)
    s3 = results[2]
    assert s3.name == "vlm" and s3.metrics["skipped"] is True
    assert s3.output == {} and fake_vlm["n"] == 0


def test_summary_carries_time_and_resource_but_not_the_output(ex, fake_vlm, mixed_pdf):
    """runlog에 실릴 형태 — 페이지 본문이 들어가면 items.jsonl이 문서 크기만큼 불어난다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    _, results = stages.parse(mixed_pdf)
    for r in results:
        s = r.summary()
        assert {"seconds", "resource"} <= s.keys()
        assert "output" not in s


# ── 호출부: 운영 지표 이름은 그대로 ───────────────────────────────────────────

def test_run_one_keeps_phase_names_and_adds_stages(ex, fake_vlm, mixed_pdf, tmp_path):
    """exporter·대시보드가 phases의 parse/chunk/... 이름으로 집계한다. 바꾸면 시계열이 끊긴다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    import ingest_many

    args = argparse.Namespace(
        dry_run=True, no_ocr=True, no_vision=False, no_embed=True,
        target_tokens=500, hard_max_tokens=1000, db_url=None,
        ollama_url=None, embed_model=None, overwrite=False, no_init_schema=True)
    doc = {"pdf": mixed_pdf, "doc_type": "policy_terms", "insurer": "테스트",
           "product_name": "테스트상품"}
    result = ingest_many._run_one(doc, args, tmp_path)

    assert result["status"] == "OK"
    assert set(result["phases"]) == {"hash", "parse", "chunk"}
    assert list(result["stages"]) == ["hash", "text", "tables_cpu", "vlm", "combine", "chunk"]
    assert result["stages"]["vlm"]["resource"] == "gpu"
    # 단계 합이 parse 구간을 넘으면 어딘가 이중으로 재고 있는 것이다.
    parse_stages = sum(result["stages"][k]["seconds"]
                       for k in ("text", "tables_cpu", "vlm", "combine"))
    assert parse_stages <= result["phases"]["parse"] + 0.05


# ── VLM 호출 계측 ─────────────────────────────────────────────────────────────
# 단건 지연이 렌더 해상도(비전 토큰)에서 오는지 생성 길이에서 오는지 가르려면 호출마다
# 소요·결과·이미지 크기·토큰이 함께 있어야 한다.

def test_vlm_stage_carries_per_call_stats(ex, monkeypatch, mixed_pdf):
    ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="both")
    import requests

    from insurance_chunker import stages

    class _UsageResp(_Resp):
        def json(self):
            return {**super().json(), "usage": {"prompt_tokens": 1200,
                                                "completion_tokens": 80}}

    n = {"i": 0}

    def _post(url, json=None, timeout=None):
        n["i"] += 1
        return _UsageResp("| a |\n|---|\n| 1 |" if n["i"] % 2 else "표 없음")

    monkeypatch.setattr(requests, "post", _post)
    _, results = stages.parse(mixed_pdf)
    v = results[2].metrics
    assert len(v["latencies"]) == v["calls"] == n["i"]
    assert v["outcomes"]["table"] + v["outcomes"]["empty"] == n["i"]
    assert v["prompt_tokens_p50"] == 1200 and v["completion_tokens_p50"] == 80
    assert v["image_px_p50"] > 0, "실제 렌더 크기가 실려야 비전 토큰을 추정할 수 있다"


def test_timeout_is_told_apart_from_other_failures(ex, monkeypatch, mixed_pdf):
    """타임아웃 비율이 VLM_TIMEOUT을 정하는 근거다 — 다른 실패와 섞이면 못 쓴다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="both", VLM_FAIL_STREAK="99")
    import requests

    from insurance_chunker import stages

    n = {"i": 0}

    def _post(url, json=None, timeout=None):
        n["i"] += 1
        if n["i"] == 1:
            raise requests.ReadTimeout("read timed out")
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", _post)
    _, results = stages.parse(mixed_pdf)
    outcomes = results[2].metrics["outcomes"]
    assert outcomes["timeout"] == 1
    assert outcomes["error"] == n["i"] - 1


def test_vlm_stats_reset_per_document(ex, fake_vlm, mixed_pdf):
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    _, first = stages.parse(mixed_pdf)
    _, second = stages.parse(mixed_pdf)
    assert len(second[2].metrics["latencies"]) == len(first[2].metrics["latencies"])


def test_table_fingerprint_has_source_and_hash_but_no_text(ex, fake_vlm, mixed_pdf):
    """병렬화 같은 변경은 이 해시가 전부 같아야 한다. 원문은 싣지 않는다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    from insurance_chunker import stages

    pages, _ = stages.parse(mixed_pdf)
    fp = stages.table_fingerprint(pages)
    assert set(fp) == {str(p.page_num) for p in pages if p.tables}
    for page in pages:
        if page.tables:
            entry = fp[str(page.page_num)]
            assert entry["source"] == page.tables[0]["source"]
            assert len(entry["sha256"]) == 16 and "markdown" not in entry


def test_dry_run_file_carries_the_bench_contract(ex, fake_vlm, mixed_pdf, tmp_path):
    """벤치마크가 읽는 형식 — 단계 기록과 페이지별 표 해시가 있어야 변형끼리 비교한다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m")
    import json

    import ingest_many

    args = argparse.Namespace(
        dry_run=True, no_ocr=True, no_vision=False, no_embed=True,
        target_tokens=500, hard_max_tokens=1000, db_url=None,
        ollama_url=None, embed_model=None, overwrite=False, no_init_schema=True)
    doc = {"pdf": mixed_pdf, "doc_type": "plain", "insurer": "t", "product_name": "p"}
    ingest_many._run_one(doc, args, tmp_path)
    out = json.loads((tmp_path / "mixed_dry_run.json").read_text())
    assert out["contract"] == ingest_many.DRY_RUN_CONTRACT == 2
    assert {"doc_hash", "summary", "stages", "tables_by_page", "chunks"} <= out.keys()
    assert "latencies" in out["stages"]["vlm"]
    assert out["tables_by_page"]


def test_truncated_responses_are_counted(ex, monkeypatch, mixed_pdf):
    """max_tokens에서 잘린 응답(finish_reason=length)은 시간을 상한까지 쓰고 표 끝을 잃는다."""
    ex(VLM_BACKEND="local", VLM_MODEL="m", VLM_TARGET="both")
    import requests

    from insurance_chunker import stages

    n = {"i": 0}

    class _Cut(_Resp):
        def json(self):
            body = super().json()
            body["choices"][0]["finish_reason"] = "length" if n["i"] == 1 else "stop"
            return body

    def _post(url, json=None, timeout=None):
        n["i"] += 1
        return _Cut("| a |\n|---|\n| 1 |")

    monkeypatch.setattr(requests, "post", _post)
    _, results = stages.parse(mixed_pdf)
    assert results[2].metrics["truncated"] == 1
