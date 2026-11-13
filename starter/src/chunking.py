"""Split pages into traceable chunks."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _make_chunk(
    page: DocumentPage,
    text: str,
    start: int,
    end: int,
    strategy: str,
) -> Chunk:
    identity = f"{strategy}|{page.document_id}|{page.page}|{start}|{end}|{text}"
    chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()

    return Chunk(
        chunk_id=chunk_id,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            **page.metadata,
            "char_start": start,
            "char_end": end,
        },
    )


def fixed_size(
    page: DocumentPage,
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Split a page into fixed character windows with overlap."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size phải > 0; overlap phải từ 0 đến size - 1.")

    text = page.text.strip()
    if not text:
        return []

    chunks = []
    step = size - overlap

    for start in range(0, len(text), step):
        end = min(start + size, len(text))
        value = text[start:end].strip()

        if value:
            chunks.append(_make_chunk(page, value, start, end, "fixed"))

        if end == len(text):
            break

    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Pack paragraphs up to max_size; split oversized paragraphs."""
    if max_size < 1:
        raise ValueError("max_size phải lớn hơn 0.")

    text = page.text.strip()
    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = []
    start = 0
    cursor = 0

    def flush(end: int) -> None:
        nonlocal current
        value = "\n\n".join(current).strip()
        if value:
            chunks.append(_make_chunk(page, value, start, end, "paragraph"))
        current = []

    for paragraph in paragraphs:
        paragraph_start = text.find(paragraph, cursor)
        paragraph_end = paragraph_start + len(paragraph)
        cursor = paragraph_end

        if len(paragraph) > max_size:
            flush(paragraph_start)

            for offset in range(0, len(paragraph), max_size):
                part = paragraph[offset : offset + max_size]
                part_start = paragraph_start + offset
                chunks.append(
                    _make_chunk(
                        page,
                        part,
                        part_start,
                        part_start + len(part),
                        "paragraph",
                    )
                )

            start = paragraph_end
            continue

        proposed = "\n\n".join([*current, paragraph])

        if current and len(proposed) > max_size:
            flush(paragraph_start)
            start = paragraph_start
        elif not current:
            start = paragraph_start

        current.append(paragraph)

    flush(len(text))
    return chunks