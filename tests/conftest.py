"""테스트 공통 설정.

VLM 동시도 자동 조정(VLM_ADAPTIVE, 기본 켜짐)은 테스트에서 끈다. 켜 두면 가짜 서버만 있는 테스트도
조정기가 VLM_URL의 /metrics를 실제로 두드린다. 조정기를 보는 테스트는 직접 켠다(test_vlm_limiter).
"""
import os

os.environ.setdefault("VLM_ADAPTIVE", "0")

# PyMuPDF 표 작업 프로세스(TABLES_CPU_WORKERS, 기본 2)도 끈다. 켜 두면 파싱마다 spawn 프로세스가 떠
# 스위트가 느려지고, 테스트가 이 프로세스에서 바꿔 끼운 함수는 작업 프로세스에 닿지 않는다.
# 작업 프로세스 경로는 test_tables_workers가 직접 켜서 본다.
os.environ.setdefault("TABLES_CPU_WORKERS", "0")
