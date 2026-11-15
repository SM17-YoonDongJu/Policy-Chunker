"""eval/bench/vlm_replay.py — 실행 비교와 요청 본문."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "vlm_replay", REPO / "eval" / "bench" / "vlm_replay.py")
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def _run(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return str(path)


def test_compare_counts_mismatches_and_skips_failed_calls(tmp_path, capsys):
    a = _run(tmp_path / "a.jsonl", [
        {"id": "x", "image_px": 10, "content": "| a | b |", "completion_tokens": 5},
        {"id": "y", "image_px": 20, "content": "| c | d |", "completion_tokens": 5},
        {"id": "z", "image_px": 30, "content": "| e |", "completion_tokens": 3},
    ])
    b = _run(tmp_path / "b.jsonl", [
        {"id": "x", "image_px": 10, "content": "| a | b |", "completion_tokens": 5},
        {"id": "y", "image_px": 20, "content": "| c | D |", "completion_tokens": 5},
        {"id": "z", "image_px": 30, "error": "ReadTimeout"},  # 실패한 호출은 비교에서 뺀다
    ])
    vr.cmd_compare(argparse.Namespace(runs=[a, b], show=5))
    out = capsys.readouterr().out
    assert "비교 2건, 출력이 다른 건 1 (50.0%)" in out
    assert "y (px 20" in out and "6자에서 갈림" in out


def test_request_body_matches_production_call(monkeypatch):
    sent = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 1}}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    rec = vr._call("http://v", "M", "P", 4096, 5, {"id": "i", "image_px": 1, "img_b64": "QUJD"})
    assert rec["content"] == "ok" and rec["finish_reason"] == "stop"
    body = sent["body"]
    assert sent["url"] == "http://v/v1/chat/completions"
    assert body["temperature"] == 0 and body["max_tokens"] == 4096 and body["model"] == "M"
    img, text = body["messages"][0]["content"]
    assert img["image_url"]["url"] == "data:image/png;base64,QUJD" and text["text"] == "P"
