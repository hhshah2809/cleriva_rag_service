import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from rag_service import ingest_docx_file

router = APIRouter()
LOG = logging.getLogger(__name__)


@router.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...), rag_document_id: str = Form(None)):
    """Upload a DOCX file and index it into Supabase vector store.

    Returns the number of chunks indexed as `chunks_indexed`.
    """
    try:
        LOG.info("/upload called user_id=%s rag_document_id=%s filename=%s", user_id, rag_document_id, getattr(file, "filename", None))

        if not file or not file.file:
            raise HTTPException(status_code=400, detail="file is required")

        count = ingest_docx_file(file.file, user_id, rag_document_id)
        return {"success": True, "chunks_indexed": count}
    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))