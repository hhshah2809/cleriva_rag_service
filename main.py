import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils.logging_config import configure_logging

load_dotenv()
configure_logging(os.getenv("LOG_LEVEL", "INFO"))

from api.upload import router as upload_router
from api.retrieve import router as retrieve_router


app = FastAPI(title="RAG Service")


@app.on_event("startup")
async def startup_event():
    # simple startup validation
    from core.config import (
        SUPABASE_URL,
        SUPABASE_KEY,
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION,
    )

    missing = []
    for name, val in [
        ("SUPABASE_URL", SUPABASE_URL),
        ("SUPABASE_KEY", SUPABASE_KEY),
        ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
        ("AZURE_OPENAI_ENDPOINT", AZURE_OPENAI_ENDPOINT),
        ("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", AZURE_OPENAI_EMBEDDING_DEPLOYMENT),
        ("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION),
    ]:
        if not val:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


# CORS: allow from localhost/dev by default; update as needed in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(upload_router, prefix="/api")
app.include_router(retrieve_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)