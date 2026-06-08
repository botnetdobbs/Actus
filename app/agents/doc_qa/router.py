import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.auth.models import User
from app.auth.jwt import require_role

router = APIRouter(prefix="/doc-qa", tags=["Doc Q&A"])

UPLOAD_DIR = Path("/tmp/actus-uploads")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
_CHUNK = 64 * 1024  # 64 KB read buffer


@router.post("/upload", status_code=201)
async def upload_document(
    file: UploadFile,
    _: User = Depends(require_role("analyst")),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=422, detail=f"Unsupported file type '{suffix}'. Allowed: .pdf, .docx")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    total = 0
    loop = asyncio.get_running_loop()
    try:
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    await loop.run_in_executor(None, dest.unlink, True)
                    raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit")
                await loop.run_in_executor(None, fh.write, chunk)
    except HTTPException:
        raise
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e

    return {"file_path": str(dest), "filename": file.filename, "size_bytes": total}
