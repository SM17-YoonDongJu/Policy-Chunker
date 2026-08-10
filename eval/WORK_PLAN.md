# 작업 계획 — 아토믹 커밋 단위

> 근거: [`GAP_ANALYSIS.md`](GAP_ANALYSIS.md) 백로그 (P0~P6). 2026-08-10 수립.
>
> **원칙**
> - 한 커밋 = 한 관심사. 코드 변경과 평가 산출물(jsonl/md)은 커밋 분리.
> - **한 청크셋 버전 = 한 변경** (v5=호 분할만, v6=본문 하한만) — eval 귀속을 깨끗하게.
> - 각 Phase 끝에 검증 게이트. 게이트 실패 시 다음 Phase 진입 금지.

---

## Phase 0 — 워킹트리 정리 + 프로덕션 버그 (P0) `오늘`

| # | 커밋 | 내용 | 검증 |
|---|---|---|---|
| C0 | `feat(eval): 문서 무결성 배치 측정 추가` | 미커밋 WIP 정리 — `batch_integrity.py` 신규 + `make_chunks_jsonl.py`/`retrieval_eval.py`/`results.md` 수정분 | diff 확인 후 관심사 섞였으면 2커밋으로 분리 |
| C1 | `docs(eval): 연구 레포트·갭 분석 추가` | `GAP_ANALYSIS.md`, `RESEARCH_REPORT.md`, `WORK_PLAN.md` | — |
| C2 | `fix(embedder): 질의 전용 임베딩 함수 추가` | `embedder.py`에 `QUERY_INSTRUCT` 상수 + `embed_query()` 헬퍼 신설, `db/search.py:160`이 이를 사용 (버그 A) | 질의 1건 임베딩해 프리픽스 적용·1024d 확인 |
| C3 | `refactor(eval): QUERY_INSTRUCT를 embedder에서 import` | `retrieval_eval.py:43` 중복 상수 제거 — 단일 출처화 | `--check-gold` 재실행 결과 동일 |
| C4 | `fix(embedder): 기본 태그 qwen3-embedding:0.6b 정렬` | `embedder.py:23` + `README.md:241` + `RETRIEVAL_EVAL_CONTEXT.md` (버그 B) | `ollama list`로 태그 존재 확인 + 임베딩 스모크 |

**게이트 0**: 프로덕션 경로(`db/search.py`)가 eval과 동일 조건(모델·프리픽스)으로 임베딩함을 확인.
⚠️ C2·C4는 **질의 벡터에만** 영향 — 문서 벡터는 원래 프리픽스 없이 색인되어 있으므로 재인덱싱 불필요. 단 기존 DB가 잘못된 태그로 색인된 경우엔 재인덱싱 필요 여부 먼저 확인.

---

## Phase 1 — exclusion 호(號) 단위 분할 (P1, 핵심) `이번 주`

| # | 커밋 | 내용 | 검증 |
|---|---|---|---|
| C5 | `feat(rechunk): 다항목 나열조 호 단위 선택 분할` | `rechunk.py` — 조 토큰 > ~400 **그리고** 호/목 항목(`1.` `가.` 등) ≥ 4 인 조만 호별 청크로 분할. 조 메타(article_number/title) 유지, prefix에 호 번호 표기. env 플래그로 on/off | 30327 보통약관 제5조가 5+ 청크로 분할되는지 단건 확인, 짧은 조는 비분할 확인 |
| C6 | `chore(eval): v5 청크셋 생성·등록` | `make_chunks_jsonl.py`로 v5 jsonl 생성, `VERSIONS`에 `v5` 추가 | `batch_integrity`로 coverage/gaps/dup 회귀 없음 |
| C7 | `docs(eval): v5 평가 결과 기록` | `--embed` 캐시 빌드 → `RERANK=1 --run` → `retrieval_eval_results.md` 갱신 | **합격선: exclusion lenient R@5 ≥ 0.71 && 전체 R@5 ≥ v3(0.851)** |

**게이트 1**: 합격선 통과 시 프로덕션 재인덱싱(운영 절차, 커밋 아님). 미달 시 small2big 방식(자식 색인+부모 반환, `small2big_eval.py` 재활용)으로 전환 검토.

---

## Phase 2 — 검색단 반영 (P2) `v5 확정 후`

| # | 커밋 | 내용 | 검증 |
|---|---|---|---|
| C8 | `feat(search): rerank 토글 환경변수` | `db/search.py` — `SEARCH_RERANK` env로 기본값 제어 (계약 시그니처 불변, 호출부 무수정 배포 가능) | rerank on/off 지연 실측 비교 |

**랭그래프 레포(SM17-YoonDongJu/AI) 몫 — 이 레포 범위 밖, 별도 브랜치**:
- rewrite 노드 (검증된 `query_rewrite_eval.py`의 Gemma4 프롬프트 이식, R@1 +4.9pp)
- grade 노드 + conditional edge 반복검색 (최대 1~2턴, CRAG 패턴)
- Q12형 잔여 케이스: 면책 질의 시 보통약관 총칙 부스트 규칙
- `search(rerank=True)` 호출 전환

---

## Phase 3 — 파싱 품질 지표 확장 (P5) `병렬 가능`

| # | 커밋 | 내용 | 검증 |
|---|---|---|---|
| C9 | `feat(eval): 표 구조 충실도 지표` | `batch_integrity.py` — 행별 열 개수 분산(`table_ragged_rate`), 표 탐지 재현율 | 표 많은 문서 1건으로 스팟체크 |
| C10 | `feat(eval): 조 헤딩 오인율 지표` | 조번호 단조성 위반(`article_nonmono`) + "인용 조문" 잔존(`cite_leak`) | 최근 회귀 수정 커밋의 대상 문서로 확인 |
| C11 | `feat(eval): VLM 표 소스 승률 지표` | `combine.best_table` 결과 집계(`vlm_win_rate`, double-space 개선폭) | use_vision=True 1문서 실측 |
| C12 | `chore(eval): batch_integrity 17문서 완주` | 신규 지표 포함해 전체 재실행, jsonl 갱신 (현재 13/17) | 17행 완성 + 신규 지표 채워짐 |

---

## Phase 4 — 보강·실험 (P4, P6) `선택`

| # | 커밋 | 내용 | 검증 |
|---|---|---|---|
| C13 | `feat(validator): 본문 기준 정보밀도 하한` | body 기준 최소 길이(한글 ~30자), exclusion/definition/표 청크는 화이트리스트. **v6로 별도 평가** | v6 vs v5 R@k 회귀 없음 |
| C14 | `docs(eval): 리라이팅·small2big 47문항 재평가` | kcd_entity 6문항 포함 재실행 (stale 41→47) | 결과 기록 |
| C15 | `feat(eval): HyDE 평가 스크립트` | `query_rewrite_eval.py` 패턴 복제 — 가상 약관문단 생성→벡터검색 | exclusion·kcd_entity 유형 개선 여부 |
| C16 | `feat(eval): 시맨틱 청킹 비교 실험` | semantic splitter로 1벌 생성 → 조 단위와 직접 비교 | "구조 청킹 유지" 근거 확정 |

---

## 순서 의존성 요약

```
C0,C1 (정리)
  → C2→C3, C4 (버그: v5 임베딩 전에 반드시 완료 — 캐시가 올바른 조건으로 생성되도록)
  → C5→C6→C7 (호 분할→v5→평가) ─게이트1─→ 재인덱싱, C8
C9~C11 (지표, 병렬 가능) → C12 (완주는 지표 추가 후 한 번만)
C13~C16 (선택, 게이트1 이후)
```

- 커밋 메시지는 기존 컨벤션(`fix(rechunk): …` 한국어 요약) 유지.
- 각 커밋은 독립적으로 빌드/실행 가능해야 하며, 평가 수치가 바뀌는 커밋은 반드시 `results.md` 갱신을 동반 커밋으로.
