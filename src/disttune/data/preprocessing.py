"""Tokenization and data transforms."""

from __future__ import annotations
from typing import Any


def tokenize_fn(examples: dict[str, Any], tokenizer: Any, max_length: int = 512, text_col: str = "text", text_pair_col: str | None = None) -> dict[str, Any]:
    texts = examples[text_col]
    text_pairs = examples.get(text_pair_col) if text_pair_col else None
    return tokenizer(texts, text_pairs, truncation=True, padding="max_length", max_length=max_length)
