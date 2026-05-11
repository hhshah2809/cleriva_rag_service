import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_service import retrieve as rag_retrieve

LOG = logging.getLogger(__name__)
router = APIRouter()


class RetrieveRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    k: int = 5


@router.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    try:
        LOG.info("/retrieve called user_id=%s query=%s k=%s", req.user_id, req.query, req.k)
        results = rag_retrieve(query=req.query, user_id=req.user_id, k=req.k)
        return {"results": results}
    except Exception as exc:
        LOG.exception("Retrieve failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))