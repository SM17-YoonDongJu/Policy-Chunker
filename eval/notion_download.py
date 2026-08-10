"""노션 '보험 약관 파일' DB에서 약관 PDF 다운로드.

페이지의 '파일' 속성 서명 URL로 받아 in/에 저장. 분할본(partN)은 PyMuPDF로 병합.
단일 파일은 SHA256을 페이지 기록과 대조.

실행:
  .venv/bin/python eval/notion_download.py                    # PAGES 일괄(클로드 청킹 보유분)
  .venv/bin/python eval/notion_download.py <page_id> <slug>   # 임의 페이지 단건
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, ".")

TOKEN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("NOTION_TOKEN")][0]
H = {"Authorization": f"Bearer {TOKEN}", "Notion-Version": "2022-06-28"}

# 클로드 청킹(VLM 병합) 결과가 있는 약관 페이지들
PAGES = {
    "37230798f08f816a9266e9e06ce0a287": "단체안심상해보험_30324",
    "37230798f08f8167a50dc604454c63c9": "치아보험_이목구비2601",
    "37230798f08f814ab7e0c344a7f675bc": "프리미엄건강보험2604",
    "37230798f08f814d9216e247c641ec4e": "프리미엄간편보험2604",
    "37230798f08f8170b193ff02c4ed4675": "올바른치아보험2601",
    "37230798f08f81f98bbae0253be0a167": "상해안심보험2601",
    "37230798f08f816f9c6cdf019471ae95": "실손의료비보험2605",
    "37230798f08f81279c92e536e1e628cc": "상해보험_30273",
    "37230798f08f8167af3ce0ef75e566ea": "올바른정기보험2601",
    "37230798f08f8178bca8cbce14d3edb5": "다이렉트실손_계약전환2605",
    "37230798f08f818fb63ec417cf10a952": "다이렉트암보험_재가입2601",
    "37230798f08f817ba0ebed90562d029c": "간편한정기보험2601",
    "37230798f08f81a6b6d0fe41522a5d28": "함께하는단체보험2601",
    "37230798f08f81d38773e43d38d4df95": "간편31건강보험2604",
    "37230798f08f815aaf5df38647e2b495": "통합간편건강보험2604",
    "37230798f08f81d4a50ac32211ebddc2": "임원단체안심상해보험_30167",
    "37230798f08f8126a5c3cb3ac29dfc80": "전국민생활체육단체보험_29926",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()


def main(pages: dict[str, str] | None = None) -> None:
    out_dir = Path("in")
    for pid, slug in (pages if pages is not None else PAGES).items():
        dest = out_dir / f"{slug}.pdf"
        if dest.exists():
            print(f"[skip] {slug} — 이미 존재")
            continue
        r = requests.get(f"https://api.notion.com/v1/pages/{pid}", headers=H, timeout=30)
        if not r.ok:
            print(f"[fail] {slug}: 페이지 조회 {r.status_code}")
            continue
        props = r.json()["properties"]
        files = props.get("파일", {}).get("files", [])
        rec_sha = "".join(t["plain_text"] for t in props.get("SHA256", {}).get("rich_text", []))
        if not files:
            print(f"[fail] {slug}: 첨부 없음")
            continue

        parts: list[Path] = []
        for i, f in enumerate(sorted(files, key=lambda x: x.get("name", ""))):
            url = f.get("file", {}).get("url")
            if not url:
                continue
            p = out_dir / f".{slug}.part{i}.pdf"
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with p.open("wb") as fh:
                    for chunk in resp.iter_content(1 << 16):
                        fh.write(chunk)
            parts.append(p)

        if len(parts) == 1:
            parts[0].rename(dest)
            got = sha256(dest)
            ok = "SHA일치" if got == rec_sha else ("SHA기록없음" if not rec_sha else "SHA불일치!")
            print(f"[ok] {slug}: {dest.stat().st_size//1024}KB, {ok}")
        else:
            import fitz
            doc = fitz.open()
            for p in parts:
                with fitz.open(p) as part:
                    doc.insert_pdf(part)
            doc.save(dest)
            doc.close()
            for p in parts:
                p.unlink()
            print(f"[ok] {slug}: {len(parts)}개 파트 병합 → {dest.stat().st_size//1024}KB "
                  f"(분할본이라 SHA 대조 생략)")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main({sys.argv[1].replace("-", ""): sys.argv[2]})
    else:
        main()
