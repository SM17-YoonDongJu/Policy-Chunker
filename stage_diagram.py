#!/usr/bin/env python3
"""단계 DAG를 코드에서 뽑아 Mermaid로 그린다.

손으로 그린 단계 그림은 단계를 하나 끼워 넣는 순간 조용히 틀린다. 실제로 틀렸다 — parse를
`layout → text → tables_cpu → vlm → combine` 일직선으로 그렸는데, `UPSTREAM`을 보면
`tables_cpu`는 text와 layout을 둘 다 받고 `combine`은 넷을 받는다. 일직선이 아니라 fan-in이
있는 DAG다. 그림이 코드를 따라오게 하려면 코드에서 뽑는 수밖에 없다.

진실원은 두 곳이다.

  stage_runner.ORDER / UPSTREAM          단계 순서와 의존
  stages.StageResult(...)의 4번째 인자    그 단계가 쓰는 자원 (cpu/gpu/io)

임포트가 아니라 `ast`로 읽는다 — `insurance_chunker/__init__.py`가 PyMuPDF까지 끌어오므로
임포트하면 의존 패키지 없이는 못 돈다. CI 게이트로 쓰려면 바닥 파이썬으로 돌아야 한다.

`ORDER`는 embed에서 끝난다. store는 DB 커넥션과 S3 업로드가 CLI 쪽에 있어
`ingest_many._run_one`이 맡기 때문이다 — 그래서 이 그림에도 안 나온다. 그 바깥의 맥락
(Notion → S3 → 파드 → RDS → RAG)은 코드에 선언돼 있지 않으니 손으로 그린 쪽을 따로 쓴다.

쓰임:
  python stage_diagram.py                       # stdout
  python stage_diagram.py -o stage_dag.md       # 파일로
  python stage_diagram.py --check stage_dag.md  # 코드와 다르면 1로 끝난다 (CI 게이트)
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent / "insurance_chunker"
RUNNER, STAGES = PKG / "stage_runner.py", PKG / "stages.py"

# 자원별 표시. 값은 stages.py의 CPU/GPU/IO 상수다.
STYLE = {
    "cpu": ("CPU", "fill:#fef3c7,stroke:#d97706,color:#78350f"),
    "gpu": ("GPU", "fill:#ede9fe,stroke:#7c3aed,color:#4c1d95"),
    "io": ("IO", "fill:#dcfce7,stroke:#16a34a,color:#14532d"),
}


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assigned(tree: ast.Module) -> dict:
    """모듈 최상위의 리터럴 대입을 이름 → 값으로 모은다.

    `CPU, GPU, IO = "cpu", "gpu", "io"` 같은 튜플 대입도 풀어야 자원 상수를 값으로 바꿀 수 있다.
    """
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                for el, v in zip(target.elts, value):
                    if isinstance(el, ast.Name):
                        out[el.id] = v
    return out


def read_dag() -> tuple[list[str], dict[str, list[str]]]:
    """stage_runner에서 단계 순서와 의존을 읽는다."""
    found = _assigned(_module(RUNNER))
    missing = [n for n in ("ORDER", "UPSTREAM") if n not in found]
    if missing:
        raise SystemExit(f"stage_runner.py에서 {', '.join(missing)}를 못 찾았다 — 이름이 바뀌었나?")
    return found["ORDER"], found["UPSTREAM"]


def read_resources() -> dict[str, str]:
    """stages.py의 StageResult(...) 호출에서 단계별 자원을 읽는다.

    한 단계가 여러 번 나오기도 한다(layout·vlm은 건너뛸 때 쓰는 호출이 따로 있다). 같은 자원을
    쓰므로 먼저 본 것을 쓰되, 서로 다르면 조용히 덮지 않고 알린다.
    """
    tree = _module(STAGES)
    const = _assigned(tree)
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "StageResult" or len(node.args) < 4:
            continue
        name, res = node.args[0], node.args[3]
        if not (isinstance(name, ast.Constant) and isinstance(res, ast.Name)):
            continue
        value = const.get(res.id)
        if value is None:
            continue
        if out.setdefault(name.value, value) != value:
            raise SystemExit(f"단계 {name.value}의 자원이 엇갈린다: {out[name.value]} vs {value}")
    return out


def render(order: list[str], upstream: dict[str, list[str]], resources: dict[str, str],
           direction: str) -> str:
    """Mermaid 블록이 든 마크다운 한 장."""
    unknown = [s for s in order if s not in resources]
    lines = [
        "<!-- stage_diagram.py가 생성한다. 손으로 고치지 말 것. -->",
        "# 단계 DAG",
        "",
        "> `insurance_chunker/stage_runner.py`의 `ORDER`·`UPSTREAM`과",
        "> `stages.py`의 `StageResult` 자원 태그에서 생성한다.",
        "",
        "```mermaid",
        f"flowchart {direction}",
    ]

    for stage in order:
        res = resources.get(stage)
        tag = STYLE[res][0] if res in STYLE else "?"
        lines.append(f'  {stage}["{stage}<br/>{tag}"]')

    lines.append("")
    for stage in order:
        for parent in upstream.get(stage, []):
            lines.append(f"  {parent} --> {stage}")

    lines.append("")
    for res, (_, style) in STYLE.items():
        members = [s for s in order if resources.get(s) == res]
        if members:
            lines.append(f"  classDef {res} {style}")
            lines.append(f"  class {','.join(members)} {res}")
    lines.append("```")

    if unknown:
        lines += ["", f"> 자원 태그를 못 찾은 단계: {', '.join(unknown)}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="단계 DAG를 코드에서 뽑아 Mermaid로 그린다")
    p.add_argument("-o", "--out", help="출력 파일 (없으면 stdout)")
    p.add_argument("--check", metavar="FILE",
                   help="FILE이 코드와 어긋나면 1로 끝난다 — CI 게이트용")
    p.add_argument("-d", "--direction", default="LR", choices=["LR", "TB", "RL", "BT"])
    args = p.parse_args()

    order, upstream = read_dag()
    text = render(order, upstream, read_resources(), args.direction)

    if args.check:
        path = Path(args.check)
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current == text:
            print(f"{path}: 코드와 일치한다")
            return 0
        print(f"{path}: 코드와 어긋난다 — `python stage_diagram.py -o {path}`로 다시 뽑을 것",
              file=sys.stderr)
        return 1

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.out} ({len(order)}단계)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
