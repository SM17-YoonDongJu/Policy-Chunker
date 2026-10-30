"""운영 검색 재현 — AI 레포 rag.search를 운영과 같은 커밋·같은 설정으로 부른다.

리포트 워커(report_worker)가 실제로 부르는 함수 그대로라 라우팅·오타 교정·RRF·임베딩 폴백까지 같다.
설정(EMBEDDING_BASE_URL·EMBEDDING_MODEL·AI_TIMEOUT_SECONDS 등)은 AI의 core.config가 환경변수에서
읽고, 차트가 리포트 워커와 같은 모델을 넣는다. 커밋은 pyproject의 viewer extra가 고정한다.

AI 코드는 asyncpg 풀을 모듈 전역으로 들고 있고 풀은 만든 이벤트 루프에 묶인다. Streamlit은
세션마다 스레드가 달라서 호출마다 asyncio.run을 쓰면 풀이 다른 루프에 붙는다. 그래서 루프
하나를 백그라운드 스레드에 띄우고 모든 호출을 그 루프로 보낸다.

화면에는 결과만이 아니라 "운영 코드가 실제로 무엇을 했나"(오타 교정 결과, 벡터 검색 여부)도
보여 준다. 같은 함수를 한 번 더 부르면 비용이 두 배라, 운영 함수를 감싸 반환값만 엿본다.
엿본 값은 요청(태스크)별 ContextVar에 담겨 동시 요청끼리 섞이지 않는다.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Optional

# 요청 하나 동안 운영 함수가 낸 값을 모은다. search()가 create_task로 띄운 임베딩 태스크도
# 컨텍스트를 복사해 가므로 같은 dict를 본다.
_observed: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "prod_search_observed", default=None)


@dataclass
class ProdSearchResult:
    chunks: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    namespaces: list[str]
    in_scope: bool
    reason: Optional[str]
    elapsed_ms: int
    keyword_query: Optional[str] = None
    embed_text: Optional[str] = None
    corrections: list[dict[str, str]] = field(default_factory=list)
    vector_used: Optional[bool] = None  # None = 확인하지 못했다


def prod_dsn(dsn: str) -> str:
    """운영 검색용 DSN. asyncpg는 DSN의 모르는 쿼리 파라미터를 서버 설정으로 넘긴다.

    - 읽기 전용으로 연다.
    - AI 코드는 policy_chunks·search_terms를 스키마 없이 부른다(운영은 롤의 search_path에
      기댄다). 롤에 설정이 없어도 같은 테이블을 보도록 corpus를 앞에 둔다. DSN에 이미 있으면 그대로.
    """
    if not dsn:
        return dsn
    extra = [kv for kv in ("default_transaction_read_only=on", "search_path=corpus,public")
             if kv.split("=")[0] not in dsn]
    if not extra:
        return dsn
    return f"{dsn}{'&' if '?' in dsn else '?'}{'&'.join(extra)}"


def ai_engine_ref() -> str:
    """설치된 ai-engine의 커밋(7자리).

    운영(리포트 워커 이미지 태그)과 같은지 화면에서 확인하려는 것.
    """
    try:
        raw = metadata.distribution("ai-engine").read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return "설치 안 됨"
    try:
        url = json.loads(raw or "{}").get("url", "")
    except ValueError:
        url = ""
    m = re.search(r"[0-9a-f]{40}", url)
    return m.group(0)[:7] if m else "알 수 없음"


def _install_observers(search_mod: Any) -> None:
    """rag.search 모듈의 두 함수를 감싸 반환값을 _observed에 남긴다. 동작은 바꾸지 않는다."""
    if getattr(search_mod, "_viewer_observed", False):
        return
    correct = getattr(search_mod, "correct_query", None)
    embed = getattr(search_mod, "_embed_query", None)

    if correct is not None:
        async def observed_correct(pool: Any, query: str) -> Any:
            out = await correct(pool, query)
            box = _observed.get()
            if box is not None:
                box["corrected"] = out
            return out

        search_mod.correct_query = observed_correct

    if embed is not None:
        async def observed_embed(text: str) -> Any:
            vec = await embed(text)
            box = _observed.get()
            if box is not None:
                box["vector_used"] = vec is not None
            return vec

        search_mod._embed_query = observed_embed

    search_mod._viewer_observed = True


class ProdSearch:
    def __init__(self, dsn: str) -> None:
        self._dsn = prod_dsn(dsn)
        self._loop = asyncio.new_event_loop()
        self._pool_lock: Optional[asyncio.Lock] = None
        threading.Thread(target=self._loop.run_forever, name="prod-search", daemon=True).start()

    def search(self, query: str, *, insurer: Optional[str] = None,
               product: Optional[str] = None, insurance_type: Optional[str] = None,
               namespaces: Optional[list[str]] = None, top_k: int = 8,
               timeout: float = 120.0) -> ProdSearchResult:
        coro = self._run(query, insurer, product, insurance_type, namespaces, top_k)
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _ensure_pool(self) -> None:
        from core import db as core_db
        from core.config import get_settings

        if self._pool_lock is None:
            self._pool_lock = asyncio.Lock()
        async with self._pool_lock:
            try:
                core_db.get_pool()
                return
            except core_db.PoolNotInitializedError:
                pass
            settings = get_settings().model_copy(update={"database_url": self._dsn})
            await core_db.init_pool(settings)

    async def _run(self, query: str, insurer: Optional[str], product: Optional[str],
                   insurance_type: Optional[str], namespaces: Optional[list[str]],
                   top_k: int) -> ProdSearchResult:
        await self._ensure_pool()
        from rag import search as rag_search  # rag/__init__가 같은 이름의 함수로 내보낸다
        from rag.router import route

        _install_observers(sys.modules["rag.search"])
        box: dict[str, Any] = {}
        _observed.set(box)

        decision = route(query, insurance_type, namespaces)
        t0 = time.perf_counter()
        result = await rag_search(query, insurance_type=insurance_type, namespaces=namespaces,
                                  top_k=top_k, insurer=insurer, product=product)
        out = ProdSearchResult(
            chunks=[c.model_dump() for c in result.ranked_chunks],
            citations=[c.model_dump() for c in result.citations],
            namespaces=list(decision.namespaces), in_scope=decision.in_scope,
            reason=decision.reason, elapsed_ms=int((time.perf_counter() - t0) * 1000),
            vector_used=box.get("vector_used"))
        corrected = box.get("corrected")
        if corrected is not None:
            out.keyword_query = corrected.keyword_query
            out.embed_text = corrected.embed_text
            out.corrections = [{"original": c.original, "canonical": c.canonical}
                               for c in corrected.corrections]
        return out
