"""청크 품질 검증.
출처: rag/pipeline/validator.py
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from .models import InsuranceChunk

logger = logging.getLogger(__name__)

VALID_CHUNK_TYPES = frozenset([
    "coverage", "exclusion", "definition", "special_clause",
    "duty", "claim", "termination", "schedule", "general",
])

_MIN_TOKEN = 10
_MAX_TOKEN = 600
_MAX_SINGLE_TYPE_RATIO = 0.95


@dataclass
class ValidationResult:
    valid_chunks: list[InsuranceChunk]
    removed: int
    warnings: list[str]

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def log(self) -> None:
        if self.removed:
            logger.warning(f"[validator] {self.removed}개 청크 제거")
        for w in self.warnings:
            logger.warning(f"[validator] {w}")
        if not self.warnings and not self.removed:
            logger.info("[validator] 품질 검증 통과")


def validate_chunks(chunks: list[InsuranceChunk]) -> ValidationResult:
    warnings: list[str] = []
    valid: list[InsuranceChunk] = []
    removed = 0

    for c in chunks:
        if not c.content or not c.content.strip():
            removed += 1
            continue

        issues = []
        for field in ("source_pdf", "doc_type", "insurer", "product_name"):
            if not getattr(c, field, None):
                issues.append(f"{field} 누락")
        if c.page_number is None:
            issues.append("page_number 누락")
        if c.chunk_type not in VALID_CHUNK_TYPES:
            issues.append(f"알 수 없는 chunk_type='{c.chunk_type}'")

        if c.token_count < _MIN_TOKEN:
            removed += 1
            continue
        if c.token_count > _MAX_TOKEN:
            warnings.append(f"p{c.page_number}: {c.token_count}tok 초과 (id={c.chunk_id[:8]})")

        if issues:
            warnings.append(f"id={c.chunk_id[:8]} (p{c.page_number}): {', '.join(issues)}")

        valid.append(c)

    if valid:
        type_counts = Counter(c.chunk_type for c in valid)
        for t, n in type_counts.most_common(1):
            if n / len(valid) > _MAX_SINGLE_TYPE_RATIO:
                warnings.append(f"chunk_type='{t}' 쏠림 {n}/{len(valid)} — 분류 로직 확인")

    return ValidationResult(valid_chunks=valid, removed=removed, warnings=warnings)
