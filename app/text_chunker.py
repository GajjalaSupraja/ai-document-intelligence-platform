
def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    Split text into word-based chunks with overlap.

    chunk_size: maximum words per chunk
    overlap: words shared between neighbouring chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


def chunk_sections(
    sections: list[dict],
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[dict]:
    """Chunk extracted document sections while preserving source metadata."""

    all_chunks = []

    for section_index, section in enumerate(sections):
        text = section.get("text", "").strip()

        if not text:
            continue

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append({
                "section_index": section_index,
                "chunk_index": chunk_index,
                "source": section.get("source", "unknown"),
                "text": chunk,
            })

    return all_chunks
