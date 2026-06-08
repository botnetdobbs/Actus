from pathlib import Path

import structlog

log = structlog.get_logger()

SUPPORTED = {".pdf", ".docx"}
MAX_CHUNKS = 500


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {SUPPORTED}")

    if path.suffix.lower() == ".pdf":
        text = _extract_pdf(path)
    else:
        text = _extract_docx(path)

    if not text.strip():
        raise ValueError("No extractable text found in document (may be scanned or image-only)")
    return text


def _extract_pdf(path: Path) -> str:
    import pdfplumber

    try:
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(p for p in pages if p.strip())
    except Exception as e:
        msg = str(e).lower()
        if "password" in msg or "encrypted" in msg:
            raise ValueError("PDF is password-protected") from e
        raise ValueError(f"PDF parse error: {e}") from e


def _extract_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _chunk_text(text: str, chunk_chars: int = 1200, overlap_chars: int = 200) -> list[str]:
    import re

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        split = end
        for i in range(end, max(start, end - 200), -1):
            if text[i].isspace():
                split = i
                break

        chunks.append(text[start:split].strip())
        start = max(start + 1, split - overlap_chars)

        if len(chunks) >= MAX_CHUNKS:
            log.warning("chunk_cap_reached", max_chunks=MAX_CHUNKS, text_len=len(text))
            break

    return [c for c in chunks if c]
