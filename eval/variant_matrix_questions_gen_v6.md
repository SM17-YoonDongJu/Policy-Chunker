# 랭그래프 질의 형태별 검색 품질 — `questions_gen_v6.jsonl` (141문항)

- 판정: lenient / 청크셋: v6 / rerank: on
- utterance = 챗봇 원문 발화(기존 평가가 재던 유일한 모양)
- rewritten의 rerank는 원본 발화로 수행 (검색만 리라이팅 질의 — 검증된 구성)

## v6

| 질의 모양 | 방법 | R@1 | R@5 | MRR@10 | ΔR@5 vs utterance |
|---|---|---|---|---|---|
| utterance | bm25 | 0.539 | 0.730 | 0.625 | +0.000 |
| utterance | embed | 0.539 | 0.816 | 0.653 | +0.000 |
| utterance | rrf | 0.603 | 0.801 | 0.695 | +0.000 |
| utterance | rerank | 0.809 | 0.915 | 0.859 | +0.000 |
| rewritten | bm25 | 0.411 | 0.667 | 0.521 | -0.064 |
| rewritten | embed | 0.418 | 0.645 | 0.504 | -0.170 |
| rewritten | rrf | 0.454 | 0.723 | 0.547 | -0.078 |
| rewritten | rerank | 0.723 | 0.837 | 0.769 | -0.078 |
| followup | bm25 | 0.312 | 0.518 | 0.398 | -0.213 |
| followup | embed | 0.234 | 0.489 | 0.348 | -0.326 |
| followup | rrf | 0.326 | 0.532 | 0.421 | -0.270 |
| followup | rerank | 0.433 | 0.638 | 0.522 | -0.277 |
| report | bm25 | 0.291 | 0.433 | 0.350 | -0.298 |
| report | embed | 0.277 | 0.468 | 0.368 | -0.348 |
| report | rrf | 0.262 | 0.440 | 0.351 | -0.362 |
| report | rerank | 0.369 | 0.489 | 0.424 | -0.426 |

### 유형별 R@5 (rerank)

| 질의 모양 | cancel_refund | coverage | definition | duty | exclusion | procedure | table |
|---|---|---|---|---|---|---|---|
| utterance | 0.89 | 0.91 | 0.89 | 1.00 | 0.95 | 0.95 | 0.87 |
| rewritten | 0.86 | 0.78 | 0.79 | 1.00 | 0.90 | 0.89 | 0.73 |
| followup | 0.68 | 0.69 | 0.47 | 0.38 | 0.70 | 0.68 | 0.67 |
| report | 0.25 | 0.66 | 0.63 | 0.38 | 0.85 | 0.32 | 0.20 |
