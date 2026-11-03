"""Split documents into overlapping chunks with stable IDs.

Why chunk at all: embedding models and LLM context windows have size limits, and a small focused passage gives a
sharper embedding (and a more precise citation) than a whole document. Overlap keeps a sentence that straddles a
boundary readable in at least one chunk.

The splitter is recursive: it first tries to break on paragraphs, then lines, then sentences, then words, and only
cuts mid-word when a single word is longer than the chunk size (the same idea as the GenAI Cookbook's recursive
character splitter, re-implemented here without LangChain).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from src.ingestion.documents import Document
from src.processing.cleaning import clean_text

SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_name: str
    source_path: str
    doc_type: str
    groups: list[str]
    start_char: int

    def to_dict(self) -> dict:
        return asdict(self)


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    parts = text.split(sep)
    pieces: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        pieces.extend(_split(piece, size, rest) if len(piece) > size else [piece])
    return pieces


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int]]:
    """Return ``(chunk_text, start_offset)`` pairs. Neighbouring chunks share up to ``chunk_overlap`` characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    pieces = _split(text, chunk_size, SEPARATORS)
    chunks: list[tuple[str, int]] = []
    current, start, offset = "", 0, 0
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append((current.strip(), start))
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            # start the overlap at a word boundary when possible
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]
            start = offset - len(tail)
            current = tail
        if not current:
            start = offset
        current += piece
        offset += len(piece)
    if current.strip():
        chunks.append((current.strip(), start))
    return chunks


def stable_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Same document, position and text give the same ID on every run, so re-indexing is idempotent."""
    digest = hashlib.sha1(f"{doc_id}|{index}|{text}".encode()).hexdigest()[:16]
    return f"{doc_id}_c{index:04d}_{digest}"


def chunk_document(doc: Document, chunk_size: int, chunk_overlap: int, min_chunk_chars: int = 1) -> list[Chunk]:
    text = clean_text(doc.text)
    chunks = []
    for text_part, start in split_text(text, chunk_size, chunk_overlap):
        if len(text_part) < min_chunk_chars:
            continue
        index = len(chunks)
        chunks.append(Chunk(chunk_id=stable_chunk_id(doc.doc_id, index, text_part), doc_id=doc.doc_id,
                            chunk_index=index, text=text_part, source_name=doc.source_name,
                            source_path=doc.source_path, doc_type=doc.doc_type, groups=list(doc.groups),
                            start_char=start))
    return chunks
