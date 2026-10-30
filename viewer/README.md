# 청크 뷰어

처리한 약관을 원본 페이지와 청크로 나란히 보고, 운영과 같은 검색을 재현하는 읽기 전용 도구다
(#75). 적재 집계가 OK여도 본문이 조용히 빠지는 일(#73 테두리 안 본문, #74 텍스트 레이어 없는 쪽)은
원본과 나란히 놓아야만 보인다.

| 화면 | 보는 것 | 데이터 |
|---|---|---|
| 처리 현황 | 지금 처리 중인 문서·진행률(N/M)·끝을 못 본 시도·최근 결과(실패 포함), 적재된 문서 최신순 | 청커 `/status`, DB |
| 문서 보기 | 쪽마다 원본 이미지 ↔ 그 쪽에서 나온 청크(유형·조항·토큰·표 원문), 청크 0개인 쪽 목록 | DB, S3 |
| 검색 | 운영 검색 재현(AI 레포 `rag.search` 그대로) · 문구 찾기(본문 부분 일치) | DB, 임베딩 서버 |

어디에도 쓰지 않는다. DB 연결은 세션 단위로 읽기 전용(`default_transaction_read_only`)으로 연다.

## 클러스터에서 열기

ClusterIP Service 뒤에 있어 밖으로 노출되지 않는다. kubectl이 붙는 상태(SSO 로그인 + 6443 터널)에서:

```bash
kubectl port-forward svc/chunk-viewer 8501:8501
# → http://localhost:8501
```

배포 구성은 K8s-manifests `charts/chunk-viewer`다. 이미지는 청커와 같은 커밋에서
`Dockerfile.viewer`로 따로 빌드해 같은 ECR 레포에 `viewer-<sha7>` 태그로 올린다(deploy.yml).

## 무엇이 어디서 오나

- **문서 목록·청크**: `corpus.policy_chunks`. 문서는 `doc_hash`(원본 파일 sha256) 하나이고, 최신순은
  `max(ingested_at)`이다. `--overwrite`로 다시 적재하면 그 시각으로 올라온다.
- **원본 PDF**: `ai.corpus_file_part`의 `sha256 = doc_hash`로 S3 키를 찾아 받는다(내용 주소 키라 다시
  검증하지 않는다). 카탈로그를 읽을 권한이 없으면 파일명·원본 없이 청크만 보인다.
- **표 원문**: S3 `policy-tables/{table_id}.md`.
- **처리 중·최근 결과**: 청커의 `GET /status`(메트릭 포트 9101, `exporter.status_snapshot`). 실행 이력은
  청커 파드의 PVC에만 있어서 뷰어가 직접 읽지 못한다. 청커가 없으면 그 영역만 비고 나머지는 돈다.
- **운영 검색**: AI 레포 `rag.search`를 pyproject의 `viewer` extra가 고정한 커밋으로 import한다.
  라우팅·오타 교정·RRF·임베딩 폴백까지 리포트 워커와 같다. 화면에 보이는 교정 결과와 벡터 사용
  여부는 운영 함수를 감싸 반환값만 엿본 것이다(다시 부르지 않는다).

쪽은 1부터 센다. 청크에는 쪽이 하나만 기록되므로 여러 쪽에 걸친 청크는 시작한 쪽에만 보인다.
글자 단위 위치(bbox)는 파서가 남기지 않아 범위 밖이다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `DATABASE_URL` | — | 읽기 전용 롤 권장(아래). 필수 |
| `RDS_CA_PATH` | — | 있으면 `verify-full`로 RDS TLS 검증. 클러스터는 `rds-ca-bundle` ConfigMap |
| `S3_BUCKET`, `AWS_REGION` | — | 원본 PDF·표 원문. 자격증명은 노드 롤(`aws-metadata: allow`) |
| `CHUNKER_STATUS_URL` | `http://insurance-chunker-metrics:9101/status` | 청커 `/status` |
| `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM` | AI 기본값 | 운영 검색의 쿼리 임베딩(AI `core.config`가 읽는다) |
| `AI_TIMEOUT_SECONDS` | 60 | 임베딩 서버가 죽었을 때 키워드 검색으로 떨어지기까지 기다리는 시간 |
| `VIEWER_PDF_DIR` | — | S3 대신 로컬 디렉터리에서 원본을 해시로 찾는다(개발용) |
| `VIEWER_TABLE_DIR` | `.table_cache` | S3 없이 적재했을 때 표 원문 위치(ingest.py가 쓰는 곳) |
| `VIEWER_CACHE_DIR` | `/tmp/chunk-viewer` | 받은 PDF 디스크 캐시(최근 `VIEWER_CACHE_MAX_PDFS`=40개) |
| `VIEWER_RENDER_DPI` | 144 | 쪽 이미지 해상도 |
| `VIEWER_STATUS_REFRESH_SECONDS` | 10 | 처리 현황 자동 새로고침 주기 |

## 읽기 전용 DB 롤

뷰어 전용 롤을 둔다. 롤·스키마 정본은 AI 레포 `deploy/schema_split.sql`이므로 거기에 같이 넣는다.

```sql
CREATE ROLE chunk_viewer LOGIN PASSWORD '...';
GRANT USAGE ON SCHEMA corpus, ai TO chunk_viewer;
-- 청크 보기 + 운영 검색(라우터가 약관·판례·장해분류표, 오타 교정이 search_terms를 읽는다)
GRANT SELECT ON corpus.policy_chunks, corpus.case_chunks, corpus.schedule_chunks,
                corpus.search_terms TO chunk_viewer;
-- 원본 파일(S3 키·파일명)
GRANT SELECT ON ai.corpus_file, ai.corpus_file_part TO chunk_viewer;
ALTER ROLE chunk_viewer SET default_transaction_read_only = on;
ALTER ROLE chunk_viewer SET search_path = corpus, public;
```

접속 문자열은 Secrets Manager `app/chunk-viewer-dev`의 `DATABASE_URL`에 넣는다(ESO가 내린다).
AI 코드는 `policy_chunks`·`search_terms`를 스키마 없이 부르므로, 뷰어는 운영 검색 연결에
`search_path=corpus,public`을 직접 붙인다 — 롤에 설정이 빠져 있어도 같은 테이블을 본다.

## 운영 검색 커밋 맞추기

뷰어의 검색이 운영과 같으려면 `ai-engine` 커밋이 `rag.search`를 쓰는 리포트 워커의 이미지 태그와
같아야 한다.

1. K8s-manifests `envs/dev/report.values.yaml`의 `image.tag`(sha7)를 본다.
2. 그 커밋의 전체 해시로 `pyproject.toml`의 `viewer` extra URL을 바꾼다.
3. 화면 사이드바의 `ai-engine@xxxxxxx`가 그 태그와 같은지 확인한다.

## 로컬에서 돌리기

운영과 같은 스키마를 로컬 Postgres에 만들고(AI 레포 마이그레이션), 이 레포 코드로 약관을 적재해
본다. 원본은 `VIEWER_PDF_DIR`로 주면 S3가 필요 없다.

```bash
pip install -e ".[dev,viewer]"

docker run -d --name chunk-viewer-pg -e POSTGRES_PASSWORD=pw -p 127.0.0.1:55432:5432 pgvector/pgvector:pg16
# AI 레포 체크아웃에서: ai/000, ai/002, corpus/* 순서대로
for f in migrations/ai/000_extensions.sql migrations/ai/002_corpus_catalog.sql migrations/corpus/*.sql; do
  psql postgresql://postgres:pw@127.0.0.1:55432/postgres -v ON_ERROR_STOP=1 -f "$f"; done

# 적재 — corpus 스키마에 들어가도록 search_path를 준다
python ingest.py --pdf ~/약관/어느약관.pdf --insurer 메리츠화재 --product "..." \
  --no-embed --no-init-schema \
  --db-url "postgresql://postgres:pw@127.0.0.1:55432/postgres?options=-csearch_path%3Dcorpus,public"

DATABASE_URL=postgresql://postgres:pw@127.0.0.1:55432/postgres VIEWER_PDF_DIR=~/약관 \
  streamlit run viewer/app.py
```

`--no-embed`로 적재하면 운영 검색은 키워드만으로 돈다(벡터 후보가 없다). 청커 `/status`는 데몬을
로컬에서 띄우거나(`python worker.py`) 클러스터 청커를 포트포워딩해(`kubectl port-forward
svc/insurance-chunker-metrics 9101`) `CHUNKER_STATUS_URL=http://localhost:9101/status`로 준다.

## 테스트

`tests/test_viewer.py`가 데이터 계층을 가짜로 바꾸고 Streamlit `AppTest`로 세 화면을 실제로 돌린다
(DB·S3·청커 없이). `/status` 계약은 `tests/test_status.py`가 고정한다.
