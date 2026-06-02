# Policy-Chunker

한국 **보험약관(약관) PDF**를 RAG 임베딩용 청크로 자르는 파이프라인.

> 범용 "아무 PDF나" 청커가 아니다. 문서 종류마다 구조 경계 신호가 다르기 때문에,
> 이 도구는 **한국 보험약관**(텍스트 PDF, 조·항·호·별표 구조)에 특화돼 있다.
> 그 안에서는 코드 수정 거의 없이 다른 약관에도 돌아간다.

## 핵심 아이디어

> **구조 경계는 본문 텍스트에서 추정하지 말고, PDF의 시각적 신호(제목 폰트 크기)에서 직접 가져온다.**

약관은 보통약관 + 수십~수백 개 특별약관 + 별표(분류표)로 이루어진다. 어디서 한 약관이
끝나고 다음이 시작하는지를 본문 텍스트로 추정하면, 제목이 줄바꿈·병합된 구간에서 무너진다
(이 프로젝트의 v3가 그렇게 실패했다 — 가짜 약관이 26%의 청크를 잘못 흡수). 해결책은
PDF 조판에서 **특약 제목이 본문보다 큰 폰트**로 찍힌다는 시각적 사실을 권위 경계로 쓰는 것.

## 파이프라인

```
약관.pdf ──┬─▶ [extract]  본문 텍스트 + 표 추출 (페이지 태그 #pN#NNNN)   *외부 도구
           │
           ├─▶ [combine]  표 베스트오브 교체 (combine.py)
           │              VLM/pymupdf 중 더블스페이스 적은 쪽 채택
           │
           └─▶ [boundaries] 제목 폰트 자동탐지 → 약관/별표 경계 (boundaries.py)
                    │
                    ▼
                 [rechunk]  약관 라벨 + 조번호 재추출(인용 가드) +
                            푸터/목차 제거 + 호→항→조 병합 +
                            헤더 prepend + 중복 제거 (rechunk.py)
                    │
                    ▼
                 chunks.json  ─▶ 임베딩 / 벡터DB
```

`extract`(PDF→원시 청크)는 외부 추출기 산출물을 입력으로 받는다. `combine`·`boundaries`·
`rechunk`가 이 저장소의 본체다.

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 또는: pip install -e .
```

## 사용

```bash
# 경계 자동 탐지값만 확인
python -m policy_chunker 약관.pdf --detect-only

# 전체 실행 (PDF=경계탐지용, chunks.json=본문+표 결합청크)
python -m policy_chunker 약관.pdf chunks.json -o out.json --target 700 --hard-max 1400
```

`--target` / `--hard-max`는 하드코딩이 아니라 **임베딩 모델 토큰 한도에 맞춰 조절하는 노브**다.
한국어 700자 ≈ 300토큰이라 512토큰 모델에 안전하다. 8192 모델(OpenAI·Voyage 등)이면 키워도 된다.

## 자동 탐지 (하드코딩 → 측정)

초기 버전(v4.2)은 이 문서에 맞춰 제목 폰트·보통약관 시작 페이지를 하드코딩했다.
현재는 **문서마다 자동 측정**한다 — 그래서 다른 약관에도 그대로 동작한다.

| 항목 | 측정 방법 |
|---|---|
| 제목 폰트 | 본문 최빈 크기보다 크면서 "특약/약관"으로 끝나는 줄에 가장 많이 쓰인 크기 |
| 보통약관 시작 | 목차가 아닌 `제1조(...)`가 처음 나오는 페이지 |
| 표지/목차 끝 | 보통약관 시작 직전 |
| 상품명 | 표지에서 `...보험`으로 끝나는 첫 제목 |

검증: 레퍼런스 문서에서 자동 탐지가 옛 하드코딩값을 그대로 재현했다 — 제목폰트 **12.9**,
보통약관 시작 **p16**, 경계 **약관 138 / 별표 17**, 제47조 누수 **0**, 푸터 잔존 **0**.

## 청크 스키마

```jsonc
{
  "chunk_id": "...#rc0123",
  "header": "[익사사고 사망 특별약관 > 제2조(준용규정)]",  // 본문 앞에 prepend됨
  "body":   "...",
  "text":   "header + \n + body",                          // 임베딩 대상
  "yakwan": "익사사고 사망 특별약관",   // 보통약관/별표는 null
  "section_kind": "yak",               // front | base | yak | byeolpyo
  "article_no": 2, "article_title": "준용규정",
  "chunk_type": "payment",             // general|payment|exclusion|definition|coverage
  "is_table": false, "table_source": null,
  "page_start": 17, "page_end": 18, "char_len": 612,
  "member_ids": ["...#p17#0007", ...]
}
```

헤더를 본문에 prepend하는 이유: 임베딩 벡터에 "어느 약관 어느 조인지"를 같이 담아야
동일한 준용규정도 특약별로 구분되고 검색 정확도가 오른다.

## 평가 (측정으로 판단)

청크를 눈으로 보는 대신 **Recall@k로 측정**한다. `eval/`에 베이스라인과 덴스 러너가 있다.

```bash
# 키워드(BM25) 베이스라인 — 의존성 없이 바로 동작
python eval/eval_bm25.py --chunks out.json --eval eval/eval_set.example.jsonl --k 1 3 5 10

# 덴스 임베딩 (모델 정해지면)
pip install numpy sentence-transformers
python eval/eval_rag.py --chunks out.json --eval eval/eval_set.example.jsonl \
    --backend st --model BAAI/bge-m3 --k 1 3 5 10
```

**중요**: 평가 질문은 실제 사용자 구어체·동의어로 써야 한다. 약관과 같은 단어를 쓰면 BM25가
거저 맞혀 점수가 부풀고(무의미), 말을 바꾸면 진짜 약점이 드러난다. 이 프로젝트의 측정에서:

- 단어를 그대로 쓴 질문 → BM25 Recall@5 ≈ **1.00** (의미 없음)
- 실제 구어체 질문 → BM25 Recall@5 ≈ **0.58**, 특히 **표 조회 0.00**

즉 표·법률용어를 일상어로 물으면 키워드 검색은 못 잡는다 → **덴스/하이브리드가 필요한 지점**.
이게 청크 결함이 아니라 검색 방법의 문제라는 점이 핵심이다(청크 본문엔 정답이 들어 있다).

## 한계 (정직하게)

- **텍스트 PDF 전제.** 스캔본은 OCR 단계가 따로 필요하고 폰트 신호가 없어 다른 접근이 필요하다.
- **조·항·별표 구조의 한국 약관**에 맞춤. 2단 편집·구조가 전혀 다른 타사 약관은 첫 문서 한 번은 사람이 경계를 확인하는 게 안전하다.
- `extract`(PDF→원시 청크)는 외부 도구 산출물을 받는다. 이 저장소는 그 뒤 단계가 본체다.

## ⚠️ 저작권

이 저장소는 **코드만** 담는다. 보험약관 원문(PDF)과 그로부터 만든 **청크·뷰어·표 추출물은
절대 커밋하지 않는다** (각 보험사 저작물). `.gitignore`가 `*.pdf`, `*chunks*.json`,
`data/` 등을 전부 제외하도록 설정돼 있다. 예제는 공개 가능한 표준약관이나 합성 샘플로 대체하라.

## 라이선스

코드: [MIT](LICENSE). 데이터: 라이선스 대상 아님(위 저작권 고지 참고).
