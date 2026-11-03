"""테스트 공통 설정.

VLM 동시도 자동 조정(VLM_ADAPTIVE, 기본 켜짐)은 테스트에서 끈다. 켜 두면 가짜 서버만 있는 테스트도
조정기가 VLM_URL의 /metrics를 실제로 두드린다. 조정기를 보는 테스트는 직접 켠다(test_vlm_limiter).
"""
import os

os.environ.setdefault("VLM_ADAPTIVE", "0")
