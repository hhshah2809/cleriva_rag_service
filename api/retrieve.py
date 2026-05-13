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
    retrieval_k: Optional[int] = None


@router.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    try:
        LOG.info("/retrieve called user_id=%s query=%s k=%s", req.user_id, req.query, req.k)
        out = rag_retrieve(query=req.query, user_id=req.user_id, k=req.k, retrieval_k=req.retrieval_k)

        # `rag_retrieve` now returns a dict with results and timings
        if isinstance(out, dict) and "results" in out:
            return {"results": out["results"], "timings": out.get("timings")}

        return {"results": out}
    except Exception as exc:
        LOG.exception("Retrieve failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))