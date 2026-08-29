"""인덱싱 실행 이력 — 사이클·문서 결과와 phase 타이밍을 파일로 남긴다.

여태 이 파이프라인은 운영 수치를 남기지 않았다. ingest_catalog가 OK/0청크/SKIPPED/ERROR를
집계하지만 stdout으로만 뱉고, 로그는 json-file 링버퍼(10m x 3)가 전부라 지난 주기 기록이
사라진다. 그래서 "얼마나 걸렸나 / 왜 실패했나 / 언제 마지막으로 성공했나"에 답할 수 없었다.

왜 DB 테이블이 아니라 파일인가:
  corpus.* 의 DDL 진실원은 AI 레포 migrations/corpus 하나다(.env.example의 SKIP_INIT_SCHEMA
  주석 참고). 우리가 운영 DB에 테이블을 새로 만들면 그 원칙이 깨진다. 이력은 우리 쪽 운영
  데이터일 뿐이므로 호스트 볼륨(/data)에 둔다 — 컨테이너를 다시 만들어도 남는다.
  호스트가 여러 대로 늘면 그때 테이블로 옮긴다(그땐 AI 레포 마이그레이션으로).

파일:
  runs.jsonl     적재 CLI 1회 실행 = 1줄 (append-only, 지표 산출용)
  items.jsonl    문서 1건 처리 = 1줄 (append-only, phase 타이밍 포함)
  attempts.json  sha256 -> 최근 시도 요약. 0청크/실패 문서의 무한 재처리를 막는 데 쓴다.
  daemon.json    데몬 기동 시각과 마지막 사이클 성패. healthcheck가 읽는다.

환경변수:
  INGEST_STATE_DIR  상태 디렉터리. 기본 /data/state (쓰기 불가면 ./.state로 폴백).
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DIR = "/data/state"
_FALLBACK_DIR = ".state"

# 시도가 아니라 '생략'인 상태들 — attempts.json 갱신 대상에서 제외한다.
_NON_ATTEMPT = frozenset({"SKIPPED", "QUARANTINED"})

_RUNS = "runs.jsonl"
_ITEMS = "items.jsonl"
_ATTEMPTS = "attempts.json"
_DAEMON = "daemon.json"

_dir_cache: Optional[Path] = None


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def state_dir() -> Path:
    """상태 디렉터리를 만들어 돌려준다. 쓸 수 없으면 ./.state로 폴백한다.

    폴백은 로컬 개발·CI를 위한 것이다. 운영에선 /data가 호스트 볼륨으로 마운트돼 있다
    (deploy/docker-compose.prod.yml). 폴백이 뜨면 이력이 컨테이너와 함께 사라지므로 경고한다.
    """
    global _dir_cache
    if _dir_cache is not None:
        return _dir_cache
    want = Path(os.environ.get("INGEST_STATE_DIR", _DEFAULT_DIR))
    for cand in (want, Path(_FALLBACK_DIR)):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".w"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError:
            continue
        if cand != want:
            logger.warning(f"상태 디렉터리 {want} 쓰기 불가 → {cand} 사용"
                           "(컨테이너와 함께 사라진다)")
        _dir_cache = cand
        return cand
    raise OSError(f"상태 디렉터리를 쓸 수 없다: {want}, {_FALLBACK_DIR}")


def _append(fname: str, record: dict[str, Any]) -> None:
    """JSONL 한 줄 추가. 이력 기록 실패가 인덱싱을 죽이지 않게 예외를 삼킨다."""
    try:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        # O_APPEND 단일 write는 POSIX에서 원자적이다 — 데몬과 일회성 컨테이너가
        # 동시에 써도 줄이 섞이지 않는다.
        with open(state_dir() / fname, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:  # noqa: BLE001 - 이력 때문에 파이프라인이 죽으면 안 된다
        logger.warning(f"이력 기록 실패({fname}): {e}")


def _read_json(fname: str, default: Any) -> Any:
    try:
        path = state_dir() / fname
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"이력 읽기 실패({fname}): {e} — 기본값 사용")
        return default


def _write_json(fname: str, data: Any) -> None:
    """임시 파일에 쓰고 os.replace로 원자 교체한다(중간에 죽어도 파일이 깨지지 않게)."""
    try:
        path = state_dir() / fname
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"이력 기록 실패({fname}): {e}")


# ── phase 타이밍 ──────────────────────────────────────────────────────────────

@contextmanager
def phase(timings: dict[str, float], name: str) -> Iterator[None]:
    """구간 소요 시간을 timings[name]에 초 단위로 적는다.

    예외가 나도 그때까지 걸린 시간은 남긴다 — 실패한 문서가 어느 단계에서 얼마나 쓰고
    죽었는지가 병목 판단에 필요하다.
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round(time.perf_counter() - t0, 2)


# ── 문서 단위 이력 ────────────────────────────────────────────────────────────

def record_item(*, sha256: Optional[str], name: str, status: str,
                chunks: int = 0, warnings: int = 0, elapsed_s: float = 0.0,
                phases: Optional[dict[str, float]] = None,
                error: Optional[str] = None,
                boundary_confidence: Optional[str] = None, **extra: Any) -> None:
    """문서 1건 처리 결과를 items.jsonl에 남기고 attempts.json을 갱신한다.

    status: OK | EMPTY(0청크) | SKIPPED | QUARANTINED | ERROR
    """
    rec = {"at": now_iso(), "sha256": sha256, "name": name, "status": status,
           "chunks": chunks, "warnings": warnings, "elapsed_s": elapsed_s,
           "phases": phases or {}, "error": error,
           # 'weak'이면 섹션이 안 갈려 조번호가 어긋난다 — 적재는 됐지만 검색 품질은
           # 신뢰할 수 없다는 뜻이라 상태(status)와 따로 들고 간다.
           "boundary_confidence": boundary_confidence, **extra}
    _append(_ITEMS, rec)

    # 생략은 시도가 아니다 — 카운터도 상태도 건드리지 않는다.
    #   SKIPPED     이미 적재돼 다운로드조차 안 했다
    #   QUARANTINED 재시도 상한에 걸려 이번 주기엔 손대지 않았다. 여기서 카운터를 올리면
    #               주기마다 늘어나기만 하고, status가 QUARANTINED로 덮여 원래 실패 사유
    #               (EMPTY인지 ERROR인지)를 잃는다.
    if not sha256 or status in _NON_ATTEMPT:
        return
    attempts = _read_json(_ATTEMPTS, {})
    prev = attempts.get(sha256, {})
    # 성공하면 카운터를 0으로 되돌린다 — 다음에 파일이 바뀌어 다시 실패해도 재시도 기회를 준다.
    n = 0 if status == "OK" else int(prev.get("attempts", 0)) + 1
    attempts[sha256] = {"status": status, "attempts": n, "last_at": rec["at"], "name": name}
    _write_json(_ATTEMPTS, attempts)


def attempt(sha256: str) -> Optional[dict[str, Any]]:
    """이 문서의 최근 시도 요약. 없으면 None."""
    if not sha256:
        return None
    return _read_json(_ATTEMPTS, {}).get(sha256)


# ── 사이클 단위 이력 ──────────────────────────────────────────────────────────

def record_run(summary: dict[str, Any]) -> dict[str, Any]:
    """적재 CLI 1회 실행 결과를 runs.jsonl에 남긴다(지표 산출용 append-only 이력).

    데몬 건강 판정은 여기서 하지 않는다 — 한 사이클은 적재 + search_terms 재구성까지이고
    그 전체 성패를 아는 건 worker뿐이라 record_cycle이 따로 맡는다. 이 CLI는 일회성
    (docker compose run)으로도 돌므로 그때는 사이클 기록 없이 이 줄만 남는다.
    """
    rec = {"at": now_iso(), **summary}
    _append(_RUNS, rec)
    return rec


def last_run() -> Optional[dict[str, Any]]:
    """runs.jsonl의 마지막 줄. worker가 subprocess로 돌린 적재 결과를 읽어올 때 쓴다."""
    try:
        path = state_dir() / _RUNS
        if not path.exists():
            return None
        last = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last) if last else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"이력 읽기 실패({_RUNS}): {e}")
        return None


def record_cycle(ok: bool, detail: str = "") -> None:
    """데몬 사이클 1회의 성패. healthcheck가 보는 마지막 성공 시각을 여기서 갱신한다."""
    state = _read_json(_DAEMON, {})
    state["last_run_at"] = now_iso()
    state["last_run_ok"] = ok
    state["last_run_detail"] = detail
    if ok:
        state["last_success_at"] = state["last_run_at"]
    _write_json(_DAEMON, state)


def record_start() -> None:
    """데몬 기동 시각. healthcheck가 '첫 사이클 유예'를 판단하는 기준이 된다."""
    state = _read_json(_DAEMON, {})
    state["started_at"] = now_iso()
    _write_json(_DAEMON, state)


def daemon_state() -> dict[str, Any]:
    return _read_json(_DAEMON, {})
