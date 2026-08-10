# 랭그래프 질의 형태별 검색 품질 — `questions_gen_pilot.jsonl` (10문항)

- 판정: lenient / 청크셋: v6 / rerank: off
- utterance = 챗봇 원문 발화(기존 평가가 재던 유일한 모양)

## v6

| 질의 모양 | 방법 | R@1 | R@5 | MRR@10 | ΔR@5 vs utterance |
|---|---|---|---|---|---|
| utterance | bm25 | 0.300 | 0.600 | 0.460 | +0.000 |
| utterance | embed | 0.400 | 0.700 | 0.542 | +0.000 |
| utterance | rrf | 0.600 | 0.800 | 0.710 | +0.000 |
| rewritten | bm25 | 0.600 | 0.800 | 0.645 | +0.200 |
| rewritten | embed | 0.400 | 0.700 | 0.500 | +0.000 |
| rewritten | rrf | 0.500 | 0.800 | 0.603 | +0.000 |
| followup | bm25 | 0.000 | 0.500 | 0.208 | -0.100 |
| followup | embed | 0.200 | 0.400 | 0.298 | -0.300 |
| followup | rrf | 0.300 | 0.500 | 0.398 | -0.300 |
| report | bm25 | 0.400 | 0.700 | 0.525 | +0.100 |
| report | embed | 0.400 | 0.700 | 0.475 | +0.000 |
| report | rrf | 0.500 | 0.700 | 0.600 | -0.100 |

### 유형별 R@5 (rrf)

| 질의 모양 | coverage | exclusion | procedure | table |
|---|---|---|---|---|
| utterance | 0.50 | 1.00 | 0.50 | 1.00 |
| rewritten | 1.00 | 1.00 | 0.50 | 0.50 |
| followup | 0.50 | 0.50 | 0.00 | 1.00 |
| report | 1.00 | 1.00 | 0.50 | 0.00 |
