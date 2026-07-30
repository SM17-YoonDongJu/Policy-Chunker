# 검색 품질 평가(Recall@5/MRR) 작업 컨텍스트

> 2026-07-30 세션에서 넘기는 인수인계 문서. 목표: v1/v2 청킹을 같은 질문셋으로
> 검색 평가해서 "조 단위 세분화가 검색에 도움이 됐나"를 숫자로 답한다.

## 1. 프로젝트 상태 (이 문서 작성 시점)

- 저장소: `/Users/jang-gwon/Policy-Chunker`, venv: `.venv` (Python 3.12, `uv pip install -e .`로 설치됨)
- 평가 대상 문서: `in/상해보험_단체안심생활보험_30327.pdf` (메리츠 단체안심생활보험, 337p,
  SHA256 58c3c307…d7cb46, 노션 "[메리츠] 단체안심생활보험 약관" 페이지의 원본과 동일)

### 청킹 결과 파일 (둘 다 같은 PDF, 같은 파서, 청킹 로직만 다름)
| 버전 | 파일 | 청크 수 | 설명 |
|---|---|---|---|
| v1 | `eval/chunks_30327_full.jsonl` | 368 | 조 경계 버그 수정 **전** (여러 조가 한 청크에 병합되던 시절) |
| v2 | `eval/chunks_30327_final.jsonl` | 687 | 무결성 수정 4건 반영 + LLM 분류 병기 (`chunk_type`=LLM, `chunk_type_keyword`=키워드) |

JSONL 필드: `chunk_index, page, section, article("제N조"), article_title, chunk_type,
token_count, table_id, content` (v2는 `chunk_type_keyword`, 일부 `chunk_type_qwen` 추가).
`content`는 "prefix줄(보험사|상품|특약|조) + [헤더] + 본문" 구조.
주의: `content_tokens`(Kiwi 형태소)는 포함 안 됨 — BM25용으로는
`insurance_chunker.tokenizer.tokenize_korean(content)`로 재계산할 것.

## 2. 로컬 인프라 (모두 무료·로컬)

- **ollama** (`~/ollama-bin/ollama`, 서버 `localhost:11434`) 설치 모델:
  - `hf.co/mykor/A.X-4.0-Light-gguf:Q4_K_M` — 분류/질문생성 등 텍스트 태스크 1순위 (0.7s/건)
  - `qwen3.6:35b-a3b`, `gemma4:26b-a4b-it-qat` — 예비/심판용.
    **주의: 둘 다 thinking 모델 — API 호출 시 `"think": false` 필수** (안 끄면 content가 빈 문자열)
- **임베딩**: `insurance_chunker/embedder.py`가 기대하는 기본값은
  `EMBED_BACKEND=ollama`, `EMBED_MODEL=qwen3:embedding`(1024d) —
  **이 임베딩 모델은 아직 ollama에 설치 안 돼 있음.** 선택지:
  1. `~/ollama-bin/ollama pull <임베딩 모델>` (qwen3 embedding 계열 태그 확인 필요)
  2. `EMBED_BACKEND=sentence_transformers`로 BGE-M3 사용 (`uv pip install -e ".[st]"` 필요, 다운로드 ~2GB)
- ollama serve가 죽어 있으면: `nohup ~/ollama-bin/ollama serve > /tmp/ollama-serve.log 2>&1 &`

## 3. 작업 스펙

1. **질문셋 30~50개 생성** — 실사용자 말투로. 유형 분산:
   보장 여부("계단에서 넘어져 골절됐는데 보험금 나와?"), 면책("음주운전 사고도 보상돼?"),
   절차("청구 서류 뭐 내야 해?"), 정의("'상해'가 정확히 뭐야?"), 해지/환급, 표 조회("골이식술 수술종류 몇 종?").
   - 초안은 A.X로 생성 가능하나, **정답 라벨은 사람이(또는 이 세션에서 원문 대조로) 확정**할 것.
2. **정답 라벨링 — chunk_id가 아니라 `(section, article)` 쌍으로** 라벨링할 것.
   v1/v2의 chunk_index는 서로 다르므로, 정답 판정은 "검색된 청크의 (section, article)이
   정답 쌍과 일치하면 hit"로 해야 두 버전을 공정 비교 가능.
   표 질문은 `(section, page)` 또는 정답 문자열 포함 여부로 판정.
   주의: v1은 조 병합 버그 때문에 정답 조항이 이웃 조 article 라벨을 달고 있을 수 있음 —
   이게 바로 측정하려는 차이이므로 그대로 두고 측정 (v1에 불리한 게 아니라 v1의 실제 품질).
3. **검색 2트랙**: ① 임베딩 코사인 top-k ② BM25(Kiwi 토큰, rank_bm25 등) top-k.
   여유가 되면 ③ RRF 하이브리드도.
4. **지표**: Recall@1/@5, MRR. v1 vs v2 × (임베딩/BM25) 매트릭스로 보고.

## 4. 함정/팁 (이번 세션에서 배운 것)

- **백그라운드 작업은 세션 재시작 때 죽는다** — 오늘 다운로드가 6번 끊겼음.
  긴 작업(임베딩 687청크 등)은 반드시 **이어받기(resume) 가능하게** 짜고
  (완료분 jsonl에 append + 시작 시 done set 스킵 — `eval/llm_reclassify.py` 패턴 참조),
  중간 결과를 파일로 flush할 것.
- 임베딩 입력은 `content` 그대로 (prefix가 컨텍스트 앵커 역할 — 이미 설계 의도).
- `_tok = 글자수×0.6`은 Kiwi 근사 — LLM 토큰과 다름. 임베딩 컨텍스트(8K)에는 여유 있음.
- 표 청크(v2에서 `table_id` 있는 것)는 row 분할돼 있고 헤더가 각 조각에 반복됨.
- 평가 스크립트는 `eval/`에, 결과는 `eval/retrieval_eval_results.md`로 남기면
  노션 업로드는 이 세션 패턴(gist 경유 notion-create-attachment) 재사용 가능.

## 5. 참고 파일

- `eval/integrity_check.py` — 무결성 검사 (커버리지/조 연속성/중복)
- `eval/classify_ab_test.py` — LLM 분류 A/B 패턴 (ollama API 호출 + structured output 예시)
- `eval/qwen_cross.jsonl` — A.X vs Qwen 불일치 87건 (골든셋 후보, 별도 작업)
- `insurance_chunker/tokenizer.py` — `tokenize_korean` (BM25용)
- `insurance_chunker/embedder.py` — 임베딩 백엔드 (ollama/BGE-M3)
