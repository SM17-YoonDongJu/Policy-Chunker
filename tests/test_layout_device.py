"""레이아웃 장치(LAYOUT_DEVICE) — GPU를 못 열면 같은 모델을 CPU로 다시 연다.

운영은 gpu:0이다. 드라이버·장치가 없는 노드나 CPU용 paddle 휠에서 GPU 로드가 실패할 때 레이아웃을
꺼 버리면 페이지 전체를 VLM에 보내는 경로로 가고, 그 경로는 VLM_TIMEOUT 60초로 끝나지 않는다.
paddleocr는 테스트 환경에 없으므로 LayoutDetection을 가짜 모듈로 끼운다.
"""
from __future__ import annotations

import logging
import sys
import types

import pytest


@pytest.fixture
def fake_paddle(monkeypatch):
    from insurance_chunker import layout

    def _install(gpu_ok: bool, cpu_ok: bool = True, device: str = "gpu:0", no_driver=False):
        made: list[tuple[str, str]] = []

        class LayoutDetection:
            def __init__(self, model_name, device):
                made.append((model_name, device))
                if no_driver:  # GPU 휠 + 드라이버 없음: 장치와 무관하게 paddle import가 깨진다
                    raise ImportError("libcuda.so.1: cannot open shared object file")
                if device.startswith("gpu") and not gpu_ok:
                    raise RuntimeError("CUDA driver not found")
                if device == "cpu" and not cpu_ok:
                    raise RuntimeError("broken")

        mod = types.ModuleType("paddleocr")
        mod.LayoutDetection = LayoutDetection
        monkeypatch.setitem(sys.modules, "paddleocr", mod)
        monkeypatch.setattr(layout, "LAYOUT_DEVICE", device)
        layout.reset_for_tests()
        return made
    yield _install
    layout.reset_for_tests()


def test_gpu_opens_on_gpu(fake_paddle, caplog):
    from insurance_chunker import layout
    made = fake_paddle(gpu_ok=True)
    with caplog.at_level(logging.WARNING, logger="insurance_chunker.layout"):
        assert layout._load_model() is not None
    assert made == [("PP-DocLayoutV3", "gpu:0")]
    assert layout.device() == "gpu:0"
    assert not [r for r in caplog.records if getattr(r, "event", "") == "layout_device_fallback"]


def test_gpu_failure_reopens_the_same_model_on_cpu_and_says_so(fake_paddle, caplog):
    from insurance_chunker import layout
    made = fake_paddle(gpu_ok=False)
    with caplog.at_level(logging.WARNING, logger="insurance_chunker.layout"):
        assert layout._load_model() is not None
    assert made == [("PP-DocLayoutV3", "gpu:0"), ("PP-DocLayoutV3", "cpu")], \
        "다른 모델로 내려가기 전에 같은 모델을 CPU로 연다"
    assert layout.device() == "cpu" and layout.available()
    fell = [r for r in caplog.records if getattr(r, "event", "") == "layout_device_fallback"]
    assert len(fell) == 1 and fell[0].levelno == logging.ERROR
    assert fell[0].requested == "gpu:0" and fell[0].device == "cpu"


def test_layout_turns_off_only_when_cpu_also_fails(fake_paddle):
    from insurance_chunker import layout
    fake_paddle(gpu_ok=False, cpu_ok=False)
    assert layout._load_model() is None
    assert not layout.available()


def test_cpu_setting_does_not_try_gpu(fake_paddle):
    from insurance_chunker import layout
    made = fake_paddle(gpu_ok=False, device="cpu")
    layout._load_model()
    assert made == [("PP-DocLayoutV3", "cpu")]


def test_fallen_back_layout_is_counted_as_a_cpu_stage(fake_paddle):
    """GPU를 못 열어 CPU로 돌면 하한 계산에서도 CPU 단계다."""
    from insurance_chunker import layout, stages
    fake_paddle(gpu_ok=False)
    assert stages._layout_resource() == "gpu", "열기 전에는 요청값을 따른다"
    layout._load_model()
    assert stages._layout_resource() == "cpu"


def test_gpu_wheel_without_driver_turns_layout_off_and_says_why(fake_paddle, caplog):
    """GPU용 paddle은 드라이버 없이 CPU로도 못 연다 — 꺼진 이유를 로그에 남긴다."""
    from insurance_chunker import layout
    made = fake_paddle(gpu_ok=True, no_driver=True)
    with caplog.at_level(logging.ERROR, logger="insurance_chunker.layout"):
        assert layout._load_model() is None
    assert made == [("PP-DocLayoutV3", "gpu:0")], "폴백 모델을 받으러 가지 않는다"
    off = [r for r in caplog.records if getattr(r, "event", "") == "layout_disabled"]
    assert len(off) == 1 and "NVIDIA_VISIBLE_DEVICES=none" in off[0].getMessage()
