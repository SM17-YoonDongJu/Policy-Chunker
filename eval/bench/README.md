# 벤치마크 (eval/bench)

고정 문서셋을 변형(코드 버전 + 환경변수)별로 돌리고, **같은 문서끼리 짝지어** 속도와 결과를 비교한다.

운영 이력(runlog)은 이미 적재된 문서를 건너뛰어서 주기마다 처리한 문서가 다르다. 그걸로 전후를
비교하면 문서 구성이 바뀐 것이 개선 효과처럼 보인다. 여기서는 문서를 고정한다. 운영 이력은
회귀 감시용(`metrics.py`)으로 따로 본다.

## 흐름

```bash
# 0. 문서셋에 sha256을 채우고 PDF를 받는다 (DATABASE_URL·S3_BUCKET·AWS 자격증명)
python eval/bench/bench.py fetch --docset core

# 1. 기준선
python eval/bench/bench.py run --docset core --variant baseline

# 2. 변형을 정의하고 돌린다 (variants/<slug>.yaml)
python eval/bench/bench.py run --docset core --variant vlm-c2

# 3. 짝지어 비교 — 마크다운을 PR 본문에 붙인다
python eval/bench/bench.py compare --docset core --base baseline --variant vlm-c2 --out r.md
```

`run`은 기본으로 `chunk`까지 돈다. 임베딩까지 재려면 `--to embed`(Ollama 필요).

## 변형 (variants/*.yaml)

| 필드 | 뜻 |
|---|---|
| `ref` | git 커밋·브랜치·태그, 또는 `WORKTREE`(커밋 안 된 변경 포함). ref면 `.worktrees/`에 따로 체크아웃해 그 코드로 돈다 |
| `env` | 이 변형에서 바꾸는 환경변수 (`VLM_DPI`, `VLM_TARGET`, `VLM_CONCURRENCY` …) |
| `expect_identical_to` | 결과가 바뀌면 안 되는 변경(병렬화·단계 겹치기)이면 기준 slug. `compare`가 표 해시·청크 수·조번호가 **전부** 같은지 판정해 PASS/FAIL을 낸다(FAIL이면 종료 코드 1) |
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
