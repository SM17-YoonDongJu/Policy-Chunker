# examples

저작권 때문에 이 폴더에는 **실제 약관이나 청크를 두지 않는다.**
직접 가진 약관 PDF로 아래처럼 돌려보면 된다.

## 1. 경계 자동 탐지 확인

```bash
python -m policy_chunker /path/to/약관.pdf --detect-only
```

제목 폰트·보통약관 시작·상품명이 합리적으로 잡히는지 먼저 본다.
이상하면(제목 폰트가 본문과 같게 잡히는 등) 그 문서는 폰트 신호가 약한 경우다.

## 2. 전체 청킹

```bash
# chunks.json = 본문+표가 들어있는 결합 청크 (페이지 태그 #pN#NNNN 필요)
python -m policy_chunker /path/to/약관.pdf /path/to/chunks.json -o out.json
```

리포트에서 확인할 것:
- `제47조누수 0` — 보통약관 조가 특약으로 새지 않았는지
- `푸터잔존 0` — 페이지 번호 제거됐는지
- `<50자` 가 작은지 — 과청킹 잔존
- `고유약관` 수가 실제 특약 수와 비슷한지

## 3. 측정

```bash
# 평가셋(eval_set.jsonl)을 실제 구어체 질문 20~30개로 작성한 뒤
python eval/eval_bm25.py --chunks out.json --eval eval/eval_set.jsonl --show-misses
```

## 공개 가능한 샘플을 쓰려면

생명보험협회/손해보험협회 공시실의 **표준약관**이나 직접 만든 합성 약관처럼
저작권 문제가 없는 문서를 쓰라. 상용 보험사 약관은 공개 저장소에 올리지 말 것.
