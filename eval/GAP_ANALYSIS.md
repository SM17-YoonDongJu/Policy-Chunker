# 보험약관 청킹·검색 갭 분석 (자료 대조 + 코드 실측)

## Context
보험약관 PDF 청킹/검색 파이프라인(Policy-Chunker)의 다음 개선 방향을 정하기 위해, 외부 연구·실무 자료 6건을 정밀 분석하고 우리 코드/평가 결과와 1:1로 대조했다. 목적은 "감"이 아니라 **데이터·코드 근거로 다음에 뭘 고쳐야 검색 품질이 오를지**를 확정하는 것.

분석 자료:
- InsQABench (arXiv 2501.10943) — 중국 보험약관 QA 벤치마크·파싱
- PolicyBot (arXiv 2511.13489) — 정책문서 QA, 시맨틱 청킹·HyDE·리랭킹
- Chunking/Retrieval/Re-ranking 실증평가 (arXiv 2601.15457)
- Systematic Analysis of Chunking (arXiv 2601.14123)
- OCR Hinders RAG / OHRBench (arXiv 2412.02592) — 파싱 오류의 하류 영향
- AutoRAG 한국어 청킹 실측 (velog), LoyJoy 보험약관 실패 분석

---

## 0. 핵심 결론 (TL;DR)
1. **exclusion(면책) 약점의 지배적 원인은 "청킹 입도"로 데이터 확정.** 긴 면책조(보통약관 제5조)를 ~700토큰 통짜로 묶어 희소어(스카이다이빙·오토바이·고의) 신호가 희석됨. claude 청킹이 exclusion에서 강한 유일한 이유는 **호(號) 단위 세분할**이며, LoyJoy가 지목한 "상호참조 문제"는 **우리 도메인에선 원인이 아니었다(0건)** — 면책조항이 자기완결적 리스트라서.
2. **프로덕션 검색에 실측 버그 2건 확정** (코드 근거 아래). eval이 낸 R@5 0.85는 프로덕션에서 재현되지 않을 가능성이 높다.
3. **파싱 품질 지표에 구조적 맹점**: OHRBench가 최우선 위험으로 지목한 "표 셀/구조 오류"를 우리 3지표(coverage/gaps/dup)가 못 잡는다.
4. 우리가 이미 앞선 것(목차·부록·푸터·중복 제거, 조 단위 병합, 표 다중소스+Surya)은 굳이 논문을 따라갈 필요 없음. small2big(부모-자식)은 전체 평균 이득 없음 — 단, exclusion 유형에는 강함(아래 참고).

---

## 1. ⚠️ 프로덕션 검색 버그 2건 (코드 실측 확정)

### 버그 A — 쿼리 임베딩 instruct 프리픽스 불일치
- **eval (검증된 조건)**: 질의는 프리픽스 포함해 임베딩, 문서는 프리픽스 없이 — qwen3-embedding 권장 **비대칭** 사용법.
  - `eval/retrieval_eval.py:178` → `QUERY_INSTRUCT + q["question"]` (질의)
  - `eval/retrieval_eval.py:182` → 원문 `c["content"]` (문서, 프리픽스 없음)
  - `eval/retrieval_eval.py:43` → `QUERY_INSTRUCT = "Instruct: Given a Korean insurance policy question, retrieve relevant policy clauses...\nQuery: "`
- **프로덕션 (실제 동작)**: `db/search.py:160` → `embed_texts([query])[0]` — **프리픽스 없이** 질의 임베딩.
- **영향**: 프로덕션은 질의/문서 모두 프리픽스 없는 **대칭** 임베딩 → qwen3-embedding이 학습된 비대칭 사용법과 다르고, eval이 측정한 조건과도 다름. **프로덕션 벡터 recall이 eval 수치(R@5 0.851)보다 낮게 나올 개연성.** 저비용 수정(질의에 동일 프리픽스 부착)으로 eval 조건 재현 가능.

### 버그 B — 임베딩 모델 태그 불일치
- eval 기본값: `eval/retrieval_eval.py:42` → `qwen3-embedding:0.6b` (메모리상 **실제 존재하는** 태그)
- 프로덕션 기본값: `insurance_chunker/embedder.py:23` → `qwen3:embedding`, `README.md:241`도 동일 (메모리상 **없는** 태그)
- **영향**: `EMBED_MODEL` 환경변수를 명시하지 않으면 프로덕션 임베딩이 실패하거나 eval과 다른 모델을 사용. 인제스트/검색이 같은 embedder를 타므로 벡터공간 자체는 자기일관적이나, **eval이 측정한 모델과 다를 위험 + 기본값이 없는 태그라는 잠재 오설정**. 기본값을 검증된 태그로 정렬 필요.

---

## 2. exclusion(면책) 약점 — 근본원인 데이터 규명

**대상**: exclusion 유형 = Q10~Q16 (7문항). 5개(Q10~13,16) 정답조가 동일한 `보통약관 제5조`(면책 총칙).

| 쿼리 | 정답조 | v3+rerank | claude+rerank | 원인 |
|---|---|---|---|---|
| Q10 스카이다이빙 | 보통 제5조 | HIT@2 | HIT@1 | (c) 희석→회복 |
| Q11 임신·출산 | 보통 제5조 | HIT@4 | HIT@3 | OK |
| Q12 전쟁·폭동 | 보통 제5조 | **MISS** | **MISS** | (a) 중복 boilerplate |
| Q13 고의 자해 | 보통 제5조 | **MISS** | HIT@5 | (c) 청킹 입도 |
| Q14 도주(벌금) | 자전거벌금 제2조 | HIT@1 | HIT@1 | OK |
| Q15 하역작업 | 대중교통 제2조 | HIT@1 | HIT@1 | OK |
| Q16 오토바이 | 보통 제5조 | **MISS** | HIT@2 | (c) 청킹 입도 (결정적) |

**원인 분포 (생산 스택 v3+rerank 실패 3건)**:
- **(c) 청킹 입도/신호 희석 — 2건 (Q13, Q16), 지배적**
- (a) 중복 boilerplate 비변별 — 1건 (Q12)
- **(b) 상호참조 문맥부족 — 0건** ← LoyJoy 가설이 우리 도메인엔 해당 없음
- (d) 골드라벨 오류 — 0건

**claude가 강한 이유 (규명 완료)**: v3/v4/v4b는 제5조를 1개 ~700토큰 blob으로 유지(면책 5항목 전부 한 청크). claude는 호 단위 11개 청크(12~112토큰)로 분할 → 희소어가 작은 청크를 지배 → **BM25 term-density 급등 → RRF 후보풀(top20) 진입 → cross-encoder가 top5 승격**.

**교차검증 3가지**:
1. 유형별로 claude 우위는 BM25(0.71 vs 0.29)·rerank(0.86 vs 0.57)에 집중, **embed 단독은 claude=v3=0.43(우위 없음)** → dense가 아니라 "세분할→BM25→rerank"가 원인.
2. 레포의 small2big 실험(`small2big.log`): 부모 464→자식 1551(호 단위)에서 **exclusion 0.71** (v3 embed 0.43 대비 급등).
3. v4/v4b 경계수정은 제5조를 여전히 701토큰 blob 유지 → exclusion 오히려 하락(v4 0.29). **경계규칙 손질로는 안 풀리고, 호 단위 분할만이 해법.**

> ⚠️ small2big의 두 얼굴: 분석 2(전체 47문항 평균)는 "small2big 이득 없음"이지만, exclusion 유형에 한정하면 크게 개선(0.43→0.71). 즉 **전면 도입이 아니라 "다항목 나열형 조문만 선택적 호 단위 분할"이 정답.**

---

## 3. 파싱 품질 지표의 맹점 (OHRBench 대조)

OHRBench는 파싱 오류를 **Semantic Noise**(치명적 — 표 셀/구조 오류가 하류 F1 -50%)와 **Formatting Noise**(경미)로 이분.

| 논문 오류 유형 | 잡는 우리 지표 | 커버 |
|---|---|:---:|
| 본문 유실(truncation) | coverage | O |
| 중복 삽입 | dup | O |
| 읽기순서 스크램블 | coverage(프로브 순서민감) | O |
| **표 셀 내용 오류** | 없음 | **X** |
| **표 구조 오류(행·열 붕괴)** | 없음 | **X** ← 논문 최우선 위험 |
| 조 헤딩 오인(타법령 인용→조) | gaps(부분) | △ |

**맹점의 원인**: `batch_integrity.py`의 `norm()`이 파이프(`|`)·공백·구두점을 전부 제거 → **표가 깨져도 coverage는 안 떨어진다**(Formatting Noise를 지우는 것과 동일 동작). 최근 커밋 5건이 전부 "타법령 인용을 조로 오인" 회귀 수정인데, 이걸 batch에서 수치로 감시할 지표가 없음.

**현재 실측 (17문서 중 13개 기록)**: coverage 평균 88.3%(11~15% 유실, 완전유실 페이지 존재), gaps 거의 0, dup 0, 조번호 복원율 ~0.72(편차 큼). `batch_integrity.jsonl` 13/17만 완료 → 재실행 필요.

**추가 지표 후보**: ① 표 구조 충실도(행별 열 개수 분산), ② 조 헤딩 오인율(단조성 위반 + "인용 조문" 잔존), ③ VLM/Surya vs PyMuPDF 표 소스 승률.

---

## 4. 자료별 대조 요약 (그들이 한 것 → 우리 상태 → 갭)

| 자료 | 그들이 한 것 | 우리 상태 | 갭/판단 |
|---|---|---|---|
| InsQABench | bbox로 element→문단 재조립, 본문 50자 하한, 목차/부록 제외, RAG-ReAct 반복검색 | 조 단위 재구성(rechunk), 목차/부록/푸터 제거 우위, `_MIN_TOKEN=10`(prefix 포함이라 사실상 무력), single-shot 검색 | **본문 길이 하한 무력화**(갭), 반복검색 없음(갭), 노이즈 제거는 우위 |
| PolicyBot / 실증평가 | 시맨틱 청킹, HyDE, cross-encoder 리랭킹(faithfulness +28%), 멀티스텝이 병목 | 조 단위 구조 청킹, rerank는 eval서 +10.6pp인데 **프로덕션 기본 off**, HyDE 미평가 | rerank 프로덕션 반영(갭), HyDE 실험(후보), 시맨틱 vs 구조 청킹 직접비교(후보) |
| AutoRAG(한국어) | semantic > recursive/token (정성 0.794 vs 0.471) | 조 단위 구조 청킹(가설: 도메인상 semantic보다 우위) | 우리 도메인서 직접 비교로 확정 필요 |
| OHRBench | 표 오류가 최치명, 파싱지표만 보지 말고 end-to-end | coverage/gaps/dup + retrieval_eval 병행 | 표 품질 지표 부재(갭) |
| LoyJoy | 상호참조가 보험약관 RAG의 핵심 실패원인 | — | **우리 데이터선 exclusion 실패 원인 아님(0건)** — 추격 불필요 |

---

## 5. 개선 백로그 (우선순위)

| # | 항목 | 근거·기대효과 | 비용 | 리스크 | 검증 |
|---|---|---|---|---|---|
| **P0-a** | 버그 A 수정: 프로덕션 질의에 instruct 프리픽스 부착 | eval 조건 재현, 벡터 recall 회복 | 매우 낮음 | 낮음 | 프로덕션 검색 스팟체크 |
| **P0-b** | 버그 B 수정: `embedder.py`/README 기본 태그를 `qwen3-embedding:0.6b`로 정렬 | 오설정 제거 | 매우 낮음 | 낮음 | 인제스트 재확인 |
| **P1** | 다항목 나열형 조(제5조 등)만 **호 단위 선택 분할** (또는 small2big 자식색인+부모반환 채택) | exclusion 0.57→0.71+ 데이터 검증됨 | 중간 | 중간(과분할·회귀) | retrieval_eval v5 vs v3, batch_integrity |
| **P2** | 프로덕션 rerank 기본 on(또는 상위후보 리랭킹) | R@1 +10.6pp(eval) | 낮음 | 지연/비용 | 프로덕션 지연 측정 |
| **P3** | 쿼리 리라이팅 프로덕션 반영 (Gemma4 1콜) | R@1 +4.9pp(eval, 41문항) | 낮음 | 지연 | 47문항 재평가 후 |
| **P4** | 본문 정보밀도 하한(body 기준) `validator.py`에 추가 | 저정보 청크 색인 방지 | 낮음 | 짧은 면책·정의조 오제거 → 화이트리스트 | batch_integrity, R@k 회귀 |
| **P5** | 파싱 품질 지표 확장(표 구조·조헤딩 오인·VLM 승률) + batch 17문서 완주 | 표 오류 사각 해소, 회귀 감시 | 중간 | 낮음 | 지표 자체가 검증 |
| **P6(실험)** | HyDE / 시맨틱 청킹 vs 조 단위 직접비교 | 미평가 유망기법 | 중간 | — | eval 신규 스크립트 |

**보류(근거 있음)**: small2big 전면 도입(전체 평균 이득 없음 — P1의 선택적 분할로 대체), 경계규칙 추가 손질로 exclusion 해결(v4/v4b서 무효 확인), 상호참조 메타데이터(우리 도메인 원인 아님).
