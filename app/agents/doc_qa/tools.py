import os
import uuid
from pathlib import Path

import structlog

from app.agents.tools import tool
from app.agents.doc_qa.parser import MAX_CHUNKS, _chunk_text, extract_text
from app.rag.indexer import delete_by_type, index_text
from app.rag.retriever import retrieve

log = structlog.get_logger()

# Must match router.UPLOAD_DIR — kept local to avoid circular import.
_UPLOAD_DIR = Path("/tmp/actus-uploads").resolve()


def _assert_in_upload_dir(file_path: str) -> Path:
    """Resolve and verify file_path is inside _UPLOAD_DIR. Raises ValueError on violation."""
    try:
        resolved = Path(file_path).resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    if not str(resolved).startswith(str(_UPLOAD_DIR) + os.sep):
        raise ValueError(
            f"file_path must be inside the upload directory ({_UPLOAD_DIR})"
        )
    return resolved


@tool(
    "chunk_and_index_document",
    "Parse a PDF or DOCX file, split into chunks, and index for semantic search. "
    "Returns session_id, chunks_indexed, char_count. Use session_id in all subsequent calls.",
)
def chunk_and_index_document(file_path: str, chunk_chars: int = 1200) -> dict:
    resolved = _assert_in_upload_dir(file_path)
    session_id = str(uuid.uuid4())
    text = extract_text(str(resolved))
    chunks = _chunk_text(text, chunk_chars)[:MAX_CHUNKS]
    if not chunks:
        raise ValueError("Document produced no chunks after parsing")

    indexed = 0
    failed = 0
    for i, chunk in enumerate(chunks):
        try:
            index_text(f"doc:{session_id}", i, chunk)
            indexed += 1
        except Exception as e:
            log.warning("chunk_index_failed", chunk_index=i, error=str(e))
            failed += 1

    if indexed == 0:
        raise RuntimeError(f"Indexing failed: 0/{len(chunks)} chunks indexed")

    return {
        "session_id": session_id,
        "chunks_indexed": indexed,
        "char_count": len(text),
        "warning": f"{failed} chunks failed to index" if failed else None,
    }


@tool(
    "search_document",
    "Search indexed document chunks by semantic similarity. "
    "Requires session_id from chunk_and_index_document.",
)
def search_document(session_id: str, query: str, top_k: int = 5) -> list[dict]:
    results = retrieve(query, type_name=f"doc:{session_id}", top_k=top_k)
    return [
        {
            "text": r["document"],
            "chunk_index": r["metadata"]["object_id"],
            "score": r["score"],
        }
        for r in results
    ]


@tool(
    "cleanup_document",
    "Remove all indexed chunks for a document session. Always call before done.",
)
def cleanup_document(session_id: str) -> dict:
    deleted = delete_by_type(f"doc:{session_id}")
    return {"deleted_chunks": deleted}
