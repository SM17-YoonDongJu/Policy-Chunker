"""VLM 서버 설정을 바꾸기 전후로, 고정된 표 crop 묶음을 같은 요청으로 다시 보내 출력이 같은지 본다.

bench는 문서 단위로 전 단계를 돌린다. 서버 쪽 설정(CUDA 그래프, 어텐션 백엔드, vLLM 버전)만
바꿀 때는 레이아웃·CPU 표까지 다시 돌 이유가 없고, 운영 chunker와 노드를 나눠 쓰는 클러스터에서는 그
몫의 메모리도 아깝다. 여기서는 crop을 한 번 만들어 두고 VLM 호출만 되풀이한다.

  # 1) crop 묶음 — 표가 있는 쪽의 find_tables 박스를 운영과 같은 함수(render_page, VLM_DPI)로 자른다
  python eval/bench/vlm_replay.py build --pdf a.pdf --pdf b.pdf --n 120 --out set.jsonl
  # 2) 같은 묶음을 서버에 보낸다(요청 본문은 extractor.call_vlm과 같다)
  python eval/bench/vlm_replay.py run --set set.jsonl --url http://vllm:8000 --out eager-1.jsonl
  # 3) 두 실행을 짝지어 비교
  python eval/bench/vlm_replay.py compare eager-1.jsonl eager-2.jsonl graph-1.jsonl

run은 클러스터 안(chunker 이미지 파드, PYTHONPATH=/app)에서 돌린다. 로컬에서 SSM 터널 위의
kubectl port-forward로 보내면 수십 초 걸리는 요청 몇 개가 응답 없이 끊겨, 100건 묶음이 6분이면 끝날
것을 30분 넘게 붙잡고 있었다(2026-11-14).

같은 설정끼리도 출력이 다를 수 있다 — vLLM은 배치 구성(동시에 돌던 다른 요청)에 따라 커널과 합산
순서가 바뀌어 fp16 결과가 달라질 수 있고, temperature 0의 탐욕 디코딩은 토큰 하나만 갈려도 그
뒤가 통째로 바뀐다. 그래서 같은 설정을 두 번 돌린 불일치율을 기준선으로 두고, 설정을 바꾼 실행의
불일치율을 그것과 견준다.

crop 묶음과 실행 결과에는 약관 본문이 들어가므로 레포에 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _extractor():
    # 프롬프트는 VLM_MODEL로 정해진다(paddleocr면 태스크 토큰). 운영 모델을 기본으로 둔다.
    os.environ.setdefault("VLM_MODEL", "PaddleOCR-VL-1.6-0.9B")
    from insurance_chunker import extractor
    return extractor


def cmd_build(a: argparse.Namespace) -> None:
    import pymupdf

    ex = _extractor()
    # 운영 이미지 크기 분포(3일치 109,677건)를 따라 고른다: 0.2M 미만 29%, 0.2~0.5M 30%,
    # 0.5~1M 27%, 1M 이상 14%. 큰 crop일수록 출력이 길어 불일치가 드러나기 쉽다.
    quota = {(0, 2e5): 0.29, (2e5, 5e5): 0.30, (5e5, 1e6): 0.27, (1e6, 1e12): 0.14}
    want = {k: max(1, round(v * a.n)) for k, v in quota.items()}
    got: dict = {k: 0 for k in quota}
    out = open(a.out, "w")
    total = 0
    for pdf in a.pdf:
        doc = pymupdf.open(pdf)
        sha = hashlib.sha256(Path(pdf).read_bytes()).hexdigest()[:12]
        for pno in range(len(doc)):
            if all(got[k] >= want[k] for k in quota):
                break
            page = doc[pno]
            try:
                tabs = page.find_tables().tables
            except Exception:  # noqa: BLE001 — 표 탐지 실패 쪽은 건너뛴다
                continue
            for t in tabs:
                r = ex.render_page(page, pno + 1, clip=list(t.bbox))
                if not r:
                    continue
                img_b64, px = r
                k = next(k for k in quota if k[0] <= px < k[1])
                if got[k] >= want[k]:
                    continue
                got[k] += 1
                total += 1
                # id는 묶음 전체에서 하나여야 한다 — 구간별 번호로 지으면 같은 쪽의 다른 구간
                # crop끼리 겹쳐 비교에서 하나가 덮인다
                out.write(json.dumps({"id": f"{sha}-p{pno + 1}-{total}", "pdf": sha,
                                      "page": pno + 1, "bbox": list(t.bbox),
                                      "image_px": px, "img_b64": img_b64}) + "\n")
    out.close()
    by_bin = {f"{k[0] / 1e6:.1f}M": v for k, v in got.items()}
    print(f"{total}개 crop → {a.out}  구간별 {by_bin}")


def _call(url: str, model: str, prompt: str, max_tokens: int, timeout: float, item: dict) -> dict:
    import requests

    payload = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{item['img_b64']}"}},
            {"type": "text", "text": prompt}]}],
        "temperature": 0, "max_tokens": max_tokens, "model": model,
    }
    rec = {"id": item["id"], "image_px": item["image_px"]}
    t0 = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            body = r.json()
            ch = body["choices"][0]
            rec.update(content=ch["message"]["content"], finish_reason=ch.get("finish_reason"),
                       completion_tokens=(body.get("usage") or {}).get("completion_tokens"),
                       seconds=round(time.perf_counter() - t0, 2), attempts=attempt + 1)
            return rec
        except Exception as e:  # noqa: BLE001 — 포트포워드가 끊기면 다시 보낸다
            rec["error"] = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def cmd_run(a: argparse.Namespace) -> None:
    ex = _extractor()
    items = [json.loads(line) for line in open(a.set)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as pool:
        recs = list(pool.map(lambda it: _call(a.url, ex.VLM_MODEL, ex.VLM_PROMPT,
                                              ex.VLM_MAX_TOKENS, a.timeout, it), items))
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errs = sum(1 for r in recs if "content" not in r)
    toks = sum(r.get("completion_tokens") or 0 for r in recs)
    wall = time.perf_counter() - t0
    print(f"{len(recs)}건, 실패 {errs}, 출력 {toks:,}토큰, {wall:.0f}초 "
          f"({toks / wall:.0f} tok/s) → {a.out}")


def cmd_compare(a: argparse.Namespace) -> None:
    runs = [(p, {r["id"]: r for r in map(json.loads, open(p))}) for p in a.runs]
    base_name, base = runs[0]
    print(f"기준: {base_name}")
    for name, run in runs[1:]:
        ids = [i for i in base if "content" in base[i] and "content" in run.get(i, {})]
        diff = [i for i in ids if base[i]["content"] != run[i]["content"]]
        sims = sorted(difflib.SequenceMatcher(None, base[i]["content"], run[i]["content"]).ratio()
                      for i in diff)
        rate = len(diff) / max(1, len(ids))
        print(f"- {name}: 비교 {len(ids)}건, 출력이 다른 건 {len(diff)} ({rate:.1%})"
              + (f", 다른 건의 문자 유사도 최소 {sims[0]:.3f} · 중앙 {sims[len(sims) // 2]:.3f}"
                 if sims else ""))
        for i in diff[: a.show]:
            b, r = base[i]["content"], run[i]["content"]
            k = next((j for j, (x, y) in enumerate(zip(b, r)) if x != y), min(len(b), len(r)))
            print(f"    {i} (px {base[i]['image_px']:,}, 토큰 {base[i].get('completion_tokens')}→"
                  f"{run[i].get('completion_tokens')}): {k}자에서 갈림 "
                  f"{b[max(0, k - 20):k + 20]!r} / {r[max(0, k - 20):k + 20]!r}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pdf", action="append", required=True)
    b.add_argument("--n", type=int, default=120)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--set", required=True)
    r.add_argument("--url", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--timeout", type=float, default=300)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+")
    c.add_argument("--show", type=int, default=5)
    a = p.parse_args(argv)
    {"build": cmd_build, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
