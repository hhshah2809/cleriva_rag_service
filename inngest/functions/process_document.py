import logging
import traceback
from typing import Dict

from rag_service import ingest_docx_file
from services.status_service import set_status

LOG = logging.getLogger(__name__)


def handle_event(payload: Dict):
    """Entry point for document processing events.

    Expected payload keys: document_id, file_path, user_id, rag_document_id (optional)
    """
    document_id = payload.get("document_id")
    file_path = payload.get("file_path")
    user_id = payload.get("user_id")
    rag_document_id = payload.get("rag_document_id")

    LOG.info("Workflow started for document_id=%s file=%s", document_id, file_path)

    # Mark processing
    try:
        set_status(document_id, "processing")
    except Exception:
        LOG.exception("Failed to set processing status for %s", document_id)

    try:
        # Open file and call existing ingestion pipeline
        with open(file_path, "rb") as f:
            count = ingest_docx_file(f, user_id, rag_document_id)

        LOG.info("Workflow completed for document_id=%s indexed_chunks=%s", document_id, count)
        set_status(document_id, "ready")
    except Exception as exc:
        LOG.exception("Workflow failed for document_id=%s error=%s", document_id, exc)
        try:
            set_status(document_id, "failed", error=str(exc))
        except Exception:
            LOG.exception("Failed to set failed status for %s", document_id)
        # Re-raise if necessary or just return
        return
