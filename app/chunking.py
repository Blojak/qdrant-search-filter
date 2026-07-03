"""Simple, deterministic chunker.

Splits text into fixed-size, overlapping segments (values from the config).
Size is measured in characters – deliberately kept simple for the PoC.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class Chunk:
    """A text segment with its 0-based position in the document."""

    index: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of fixed character length.

    Empty / whitespace-only chunks are dropped. If the parameters are omitted,
    the values from the configuration are used.
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    ovl = overlap if overlap is not None else settings.chunk_overlap

    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if ovl < 0 or ovl >= size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    step = size - ovl
    chunks: list[Chunk] = []
    idx = 0
    for start in range(0, len(normalized), step):
        piece = normalized[start : start + size].strip()
        if piece:
            chunks.append(Chunk(index=idx, text=piece))
            idx += 1
        if start + size >= len(normalized):
            break
    return chunks
