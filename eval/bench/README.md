# 벤치마크 (eval/bench)

고정 문서셋을 변형(코드 버전 + 환경변수)별로 돌리고, **같은 문서끼리 짝지어** 속도와 결과를 비교한다.

운영 이력(runlog)은 이미 적재된 문서를 건너뛰어서 주기마다 처리한 문서가 다르다. 그걸로 전후를
비교하면 문서 구성이 바뀐 것이 개선 효과처럼 보인다. 여기서는 문서를 고정한다. 운영 이력은
회귀 감시용(`metrics.py`)으로 따로 본다.

## 흐름

```bash
# 0. 문서셋에 sha256을 채우고 PDF를 받는다 (DATABASE_URL·S3_BUCKET·AWS 자격증명)
python eval/bench/bench.py fetch --docset core

# 1. 기준선 — 지금 운영 구성(PaddleOCR-VL + 레이아웃 crop)
python eval/bench/bench.py run --docset core --variant paddle-crop

# 2. 변형을 정의하고 돌린다 (variants/<slug>.yaml)
python eval/bench/bench.py run --docset core --variant vlm-c2

# 3. 짝지어 비교 — 마크다운을 PR 본문에 붙인다
python eval/bench/bench.py compare --docset core --base paddle-crop --variant vlm-c2 --out r.md
```

`baseline`을 비롯해 `LAYOUT_BACKEND: "off"`로 VLM을 쓰는 변형들은 qwen3-vl + 페이지 전체 입력
시절의 측정이라
`VLM_MODEL: qwen3-vl:8b-instruct`로 고정돼 있다. `run`은 VLM을 부르는 변형이면 먼저 VLM 서버의
`/v1/models`에 그 모델이 있는지 보고, 없으면 돌기 전에 멈춘다 — 모르는 모델로 돌면 페이지마다
거절당해 표가 빠진 결과가 그 변형 이름으로 덮어써지기 때문이다.

`run`은 기본으로 `chunk`까지 돈다. 임베딩까지 재려면 `--to embed`(Ollama 필요).

## 변형 (variants/*.yaml)

| 필드 | 뜻 |
|---|---|
| `ref` | git 커밋·브랜치·태그, 또는 `WORKTREE`(커밋 안 된 변경 포함). ref면 `.worktrees/`에 따로 체크아웃해 그 코드로 돈다 |
| `env` | 이 변형에서 바꾸는 환경변수 (`VLM_DPI`, `VLM_TARGET`, `VLM_CONCURRENCY` …) |
| `expect_identical_to` | 결과가 바뀌면 안 되는 변경(병렬화·단계 겹치기)이면 기준 slug. `compare`가 표 해시·청크 수·조번호가 **전부** 같은지 판정해 PASS/FAIL을 낸다(FAIL이면 종료 코드 1) |
| `guardrail` | 결과가 바뀌는 변형의 품질 가드레일. 기본값(초안)으로 판정하고, 항목별로 덮어쓰거나 `false`로 끈다. 기본: 조번호 손실 0, 경계 신뢰도 하락 금지, 표 페이지 −5% 이내, `[?]` 셀 증가 금지, CPU 소스가 읽던 페이지의 표 변경 금지 |
| `status`·`applied_at`·`note` | 운영 반영 여부와 시각, 메모. 운영 이력 쪽 전후 비교의 구간 경계로 쓴다 |

## 캐시

단계 산출물은 `.artifacts/`에 변형끼리 공유한다. 키에 문서 해시·단계 설정·코드 지문이 들어가므로,
`VLM_DPI`만 바꾼 변형은 텍스트·CPU 표를 재사용하고 VLM부터 다시 돈다. 같은 설정을 반복 측정하려면
`run --from vlm`처럼 다시 돌 단계를 지정한다.

캐시에서 읽은 단계의 시간은 그 산출물을 만들 때 잰 값이다. 바뀐 단계만 다시 재므로 변형 간
시간 비교는 그대로 성립한다.

## 읽는 법

- **실제 시간 / 하한 / 차이**: 하한은 `max(CPU 단계 합 / 작업자, GPU 단계 합 / 동시도)`.
  차이가 크면 병렬화·단계 겹치기로 줄일 몫이 크고, 하한 자체를 내리려면 단계의 단가(표 탐지 범위,
  DPI, 엔진)를 바꿔야 한다
- **유형별 n**: 값에는 항상 표본 수가 붙는다. 유형(scan/table/plain)별로 나눠 보지 않으면 구성비가
  바뀐 것이 효과처럼 보인다
- **표 해시 일치**: 페이지별 최종 채택 표의 해시. 결과를 바꾸는 변경(DPI 등)은 몇 페이지가 바뀌었는지,
  `[?]` 셀이 늘었는지를 속도 이득과 함께 본다

결과 파일에는 약관 본문이 없다 — 수치와 해시만 남긴다.

## 서버 설정만 바꿀 때 (vLLM 플래그·버전)

CUDA 그래프, 어텐션 백엔드, vLLM 버전처럼 **VLM 서버 쪽만** 바꾸는 실험은 문서 단위 bench보다 아래
둘이 싸다. 레이아웃·CPU 표를 다시 돌지 않고, 운영 chunker와 노드를 나눠 쓰는 채로 잴 수 있다.

- `vllm_metrics.py` — vLLM `/metrics` 스냅샷 두 개를 빼서 그 사이 요청의 대기·prefill·decode·토큰
  간격 분포를 낸다. 운영 트래픽에서 설정 전후를 비교할 때 쓴다(출력 토큰 수가 비슷한지 함께 본다).
  클라이언트가 먼저 끊은 요청은 vLLM 집계에 남지 않으니 타임아웃은 chunker 로그로 센다
- `vlm_replay.py` — 고정된 표 crop 묶음을 운영과 같은 요청으로 다시 보내 출력이 같은지 본다.
  같은 설정끼리도 배치 구성에 따라 출력이 갈리므로(eager끼리 98개 중 4개, 2026-11-14), 같은 설정을
  두 번 돌린 불일치율을 기준선으로 두고 판정 기준을 실행 전에 정한다. crop과 결과에는 약관 본문이
  들어가므로 커밋하지 않는다
