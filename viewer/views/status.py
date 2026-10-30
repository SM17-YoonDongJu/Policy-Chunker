"""처리 현황 — 청커가 지금 무엇을 처리 중인지(/status)와 적재된 문서 목록(DB, 최신순)."""
from __future__ import annotations

from datetime import timedelta

import streamlit as st

from viewer import chunker_status, data, env
from viewer import widgets as w

RUN_STATE = {"running": "처리 중", "finished": "완료", "stopped": "정지 신호로 중단",
             "interrupted": "중간에 끊김", "none": "기록 없음"}
RESULT = {"OK": "적재", "EMPTY": "0청크", "ERROR": "오류"}

st.title("처리 현황")


@st.fragment(run_every=env.STATUS_REFRESH_S)
def chunker_panel() -> None:
    try:
        s = chunker_status.fetch(env.CHUNKER_STATUS_URL, limit=30)
    except Exception as e:  # noqa: BLE001 - 청커가 없어도 아래 문서 목록은 보여야 한다
        st.info(f"청커 상태를 가져오지 못했어요 — {e}\n\n"
                "청커 파드가 떠 있지 않으면 이 영역은 비어요. "
                "적재된 문서는 아래 목록에서 볼 수 있어요.")
        return

    daemon, run = s.get("daemon") or {}, s.get("run") or {}
    state = run.get("state", "none")
    pending, done = run.get("pending") or 0, run.get("done") or 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("이번 실행", RUN_STATE.get(state, state),
              help=f"시작 {w.kst(run.get('started_at'))}" if run else None)
    c2.metric("처리", f"{done} / {pending}건" if run else "-",
              help=f"카탈로그 {run.get('total')}건 중 생략 {run.get('skipped')} · "
                   f"격리 {run.get('quarantined')}" if run else None)
    last_ok = daemon.get("last_run_ok")
    c3.metric("마지막 사이클", "-" if last_ok is None else ("성공" if last_ok else "실패"),
              help=daemon.get("last_run_detail"))
    c4.metric("마지막 성공", w.ago(daemon.get("last_success_at")))

    if state == "running" and pending:
        st.progress(min(done / pending, 1.0),
                    text=f"{done}/{pending}건 · 시작 {w.ago(run.get('started_at'))}")
    elif daemon.get("last_run_at") and s.get("interval_s"):
        nxt = w.to_dt(daemon["last_run_at"]) + timedelta(seconds=int(s["interval_s"]))
        st.caption(f"다음 사이클 예정: {w.kst(nxt)} 무렵 (주기 {w.duration(s['interval_s'])})")

    processing = s.get("processing") or []
    st.markdown(f"**처리 중인 문서 {len(processing)}건**")
    if processing:
        st.dataframe([{"문서": p["name"], "시작": w.kst(p["started_at"], "%m-%d %H:%M:%S"),
                       "경과": w.ago(p["started_at"]).replace(" 전", ""),
                       "시도": p["attempts"]} for p in processing], hide_index=True)
    else:
        st.caption("지금 처리 중인 문서가 없어요.")

    unfinished = s.get("unfinished") or []
    if unfinished:
        with st.expander(f"끝을 못 본 시도 {len(unfinished)}건"):
            st.caption("처리 도중 프로세스가 죽은 문서예요(OOM·강제 종료). "
                       "다음 사이클에 다시 시도하고, 연속 실패가 상한에 닿으면 격리돼요.")
            st.dataframe([{"문서": u["name"], "마지막 시도": w.kst(u["started_at"]),
                           "시도": u["attempts"]} for u in unfinished], hide_index=True)

    recent = s.get("recent") or []
    st.markdown("**최근 처리 결과**")
    if not recent:
        st.caption("처리 기록이 없어요.")
        return
    st.dataframe(
        [{"시각": w.kst(r.get("at"), "%m-%d %H:%M:%S"),
          "결과": RESULT.get(r.get("status"), r.get("status")),
          "문서": r.get("name"), "보험사": r.get("insurer"), "청크": r.get("chunks"),
          "소요": w.duration(r.get("elapsed_s")), "오류": r.get("error") or "",
          "열기": w.doc_url(r["sha256"]) if r.get("status") == "OK" and r.get("sha256")
          else ""} for r in recent],
        hide_index=True,
        column_config={"열기": st.column_config.LinkColumn("열기", display_text="열기")})


chunker_panel()

st.divider()
st.subheader("적재된 문서 (최신순)")
try:
    insurer_options = ["전체", *data.insurers()]
    catalog = data.catalog_access()
except Exception as e:  # noqa: BLE001
    st.error(f"DB를 읽지 못했어요 — {e}")
    st.stop()

f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
insurer = f1.selectbox("보험사", insurer_options)
product = f2.text_input("상품명 포함", placeholder="예: 실손")
name = f3.text_input("파일명 포함", disabled=not catalog,
                     help=None if catalog else
                     "원본 파일 카탈로그(ai.corpus_file*) 읽기 권한이 없어요")
limit = f4.selectbox("개수", [50, 100, 300], index=1)
if not catalog:
    st.caption("원본 파일 카탈로그를 읽을 권한이 없어 파일명과 원본 PDF 없이 보여요.")

try:
    docs = data.recent_documents(limit, None if insurer == "전체" else insurer,
                                 product.strip() or None, name.strip() or None)
except Exception as e:  # noqa: BLE001
    st.error(f"문서 목록을 읽지 못했어요 — {e}")
    st.stop()

if not docs:
    st.info("조건에 맞는 적재 문서가 없어요.")
else:
    st.dataframe(
        [{"적재 시각": w.kst(d["ingested_at"], "%Y-%m-%d %H:%M"), "보험사": d["insurer"],
          "상품": d["product_name"], "파일": d.get("file_name") or "",
          # part_order는 0부터 센다(ai.corpus_file_part)
          "파트": f"{d['part_order'] + 1}/{d['part_total']}" if d.get("part_total") else "",
          "청크": d["chunks"], "청크 있는 쪽": d["pages_with_chunks"],
          "열기": w.doc_url(d["doc_hash"])} for d in docs],
        hide_index=True,
        column_config={"열기": st.column_config.LinkColumn("열기", display_text="열기")})
