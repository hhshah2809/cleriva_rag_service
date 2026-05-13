import logging
from fastapi import APIRouter, HTTPException

from services.status_service import get_status

LOG = logging.getLogger(__name__)
router = APIRouter()


@router.get("/document/status/{document_id}")
async def document_status(document_id: str):
    try:
        LOG.info("/document/status called document_id=%s", document_id)
        status = get_status(document_id)
        return {"document_id": document_id, "status": status}
    except Exception as exc:
        LOG.exception("Status check failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
