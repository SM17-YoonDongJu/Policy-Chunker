# Colab 실험 노트북 (2026-10)

표 추출 VLM을 qwen3-vl 8B에서 바꾸기 위해 Colab T4(운영 노드와 같은 GPU)에서 돌린 노트북이다.
결론과 수치는 GitHub 이슈 "PaddleOCR-VL 전환"과 Notion "모델 비교"(ETL 개선 하위)에 있다.

| 파일 | 내용 | 결과 |
|---|---|---|
| `colab_vlm_trial.ipynb` | 10쪽으로 qwen3-vl 8B · GLM-OCR · PaddleOCR-VL 첫 비교 | Paddle 유력, GLM 출력 0건 |
| `colab_glm_ocr_trial.ipynb` | GLM-OCR 단독 진단(경로·해상도·표 영역) | 한국어 오독(약관→악관)으로 탈락 |
| `colab_paddle_insurers_trial.ipynb` | 보험사별 6쪽 × qwen vs Paddle, 운영 모드(gap) 전체 문서 | 입력 단위 실험으로 이어짐 |
| `colab_paddle_accuracy.ipynb` | **입력 단위 A/B/C/D · 해상도 · 1.5/1.6 비교, HTML 리포트** | 레이아웃 crop(C) 채택 |

결과 파일은 Drive `MyDrive/bench/`에 있다: `paddle-accuracy/results.json`(방법·문서·쪽별 원문과
파싱 결과), `paddle-accuracy/report.html`(쪽 그림 + 레이아웃 박스 + 방법별 표), `insurers-trial/`
(쪽 선택과 PNG). CI ruff 범위(`eval/bench`) 밖에 둔 이유는 노트북이 셸 매직과 한 줄 다중문을 써서다.
