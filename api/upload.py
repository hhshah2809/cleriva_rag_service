import logging
import os
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from inngest.client import send_event
from services.status_service import set_status

router = APIRouter()
LOG = logging.getLogger(__name__)


@router.post("/upload")
async def upload(file: UploadFile = File(...), user_id: str = Form(...), rag_document_id: str = Form(None)):
    """Upload a DOCX file: save, create a document id, emit event, return immediately.

    This endpoint no longer runs ingestion synchronously.
    """
    try:
        LOG.info("/upload called user_id=%s rag_document_id=%s filename=%s", user_id, rag_document_id, getattr(file, "filename", None))

        if not file or not file.file:
            raise HTTPException(status_code=400, detail="file is required")

        # Persist upload to local `uploads/` directory
        uploads_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        doc_id = str(uuid.uuid4())
        filename = f"{doc_id}_{getattr(file, 'filename', 'upload.docx')}"
        path = os.path.join(uploads_dir, filename)

        # Write file to disk
        with open(path, "wb") as out_f:
            contents = await file.read()
            out_f.write(contents)

        # Create initial status
        set_status(doc_id, "uploaded")

        # Emit an event for background processing
        payload = {
            "document_id": doc_id,
            "file_path": path,
            "user_id": user_id,
            "rag_document_id": rag_document_id,
        }

        send_event("document/uploaded", payload)

        return {"status": "processing", "document_id": doc_id}

    except HTTPException:
        raise
    except Exception as exc:
        LOG.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))