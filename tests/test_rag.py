"""Tests for the chunking logic (no embedding model needed)."""
from __future__ import annotations

from app.rag import _chunk


def test_chunk_empty():
    assert _chunk("") == []
    assert _chunk("   \n  ") == []


def test_chunk_overlap_and_size():
    text = "x" * 2000
    chunks = _chunk(text, size=900, overlap=150)
    assert len(chunks) >= 2
    assert all(len(c) <= 900 for c in chunks)
    # consecutive chunks overlap by `overlap` chars
    assert chunks[0][-150:] == chunks[1][:150]


def test_chunk_collapses_whitespace():
    assert _chunk("a\n\n  b\t c")[0] == "a b c"
