# 검색 품질 평가: v1 vs v2 청킹

- 문서: 메리츠 단체안심생활보험(30327) / 질문 10개 (`questions_gen_pilot.jsonl`, 질의 모양: rewritten)
- 임베딩: qwen3-embedding:0.6b (코사인), BM25: Kiwi 형태소 + Okapi, RRF k=60
- strict: (section, article) 라벨 일치 / lenient: section 일치 + 내용(키워드) 일치

## strict

| 방법 | 버전 | R@1 | R@5 | MRR@10 |
|---|---|---|---|---|
| bm25 | v6 | 0.600 | 0.800 | 0.645 |
| embed | v6 | 0.400 | 0.700 | 0.500 |
| rrf | v6 | 0.500 | 0.800 | 0.603 |

## lenient

| 방법 | 버전 | R@1 | R@5 | MRR@10 |
|---|---|---|---|---|
| bm25 | v6 | 0.600 | 0.800 | 0.645 |
| embed | v6 | 0.400 | 0.700 | 0.500 |
| rrf | v6 | 0.500 | 0.800 | 0.603 |

## 질문 유형별 R@5 (lenient)

| 방법 | 버전 | coverage | exclusion | procedure | table |
|---|---|---|---|---|---|
| bm25 | v6 | 1.00 | 1.00 | 0.50 | 0.50 |
| embed | v6 | 1.00 | 1.00 | 0.50 | 0.00 |
| rrf | v6 | 1.00 | 1.00 | 0.50 | 0.50 |
