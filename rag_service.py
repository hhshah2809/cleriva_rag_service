import logging
from typing import IO, List, Optional, Dict

from supabase import create_client, Client

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_QUERY_NAME,
)

from core.embeddings import embeddings

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------------
# Text splitter
# -----------------------------------

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# -----------------------------------
# Supabase client
# -----------------------------------


def _get_supabase_client() -> Client:
    """
    Create and cache Supabase client.
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError("Supabase configuration missing")

    if not hasattr(_get_supabase_client, "_client"):

        LOG.info("Creating Supabase client")

        _get_supabase_client._client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

    return _get_supabase_client._client


# -----------------------------------
# Ingest DOCX
# -----------------------------------


def ingest_docx_file(
    file: IO,
    user_id: str,
    rag_document_id: Optional[str] = None
) -> int:
    """
    Ingest DOCX into vector database.
    """

    try:

        from docx import Document as DocxDocument

        if file is None:
            raise ValueError("file is required")

        if not user_id:
            raise ValueError("user_id is required")

        # Read DOCX
        doc = DocxDocument(file)

        text = "\n".join([
            p.text
            for p in doc.paragraphs
            if p.text and p.text.strip()
        ])

        if not text:
            LOG.warning("Uploaded DOCX contains no text")
            return 0

        # Split text into chunks
        chunks = _SPLITTER.split_text(text)

        documents = []

        for chunk in chunks:

            metadata = {
                "source": "docx",
                "user_id": user_id,
            }

            if rag_document_id:
                metadata["rag_document_id"] = rag_document_id

            documents.append({
                "content": chunk,
                "metadata": metadata
            })

        # Generate embeddings
        texts = [doc["content"] for doc in documents]

        LOG.info(
            "Generating embeddings for %d chunks",
            len(texts)
        )

        embeddings_list = embeddings.embed_documents(texts)

        rows = []

        for i, doc in enumerate(documents):

            rows.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "embedding": embeddings_list[i]
            })

        supabase = _get_supabase_client()

        LOG.info(
            "Indexing %d chunks for user_id=%s",
            len(rows),
            user_id
        )

        response = supabase.table("documents").insert(rows).execute()

        LOG.info(
            "Successfully indexed %d chunks",
            len(rows)
        )

        return len(rows)

    except Exception as exc:

        LOG.exception(
            "Failed to ingest DOCX: %s",
            exc
        )

        raise


# -----------------------------------
# Retrieve similar chunks
# -----------------------------------


def retrieve(
    query: str,
    user_id: Optional[str] = None,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k similar chunks.
    """

    try:

        if not query:
            raise ValueError("query is required")

        supabase = _get_supabase_client()

        # Generate query embedding
        query_embedding = embeddings.embed_query(query)

        # Metadata filter
        filter_data = {}

        if user_id:
            filter_data["user_id"] = user_id

        LOG.info(
            "Running vector similarity search for user_id=%s",
            user_id
        )

        response = supabase.rpc(
            SUPABASE_QUERY_NAME,
            {
                "query_embedding": query_embedding,
                "match_count": k,
                "filter": filter_data
            }
        ).execute()

        matches = response.data or []

        results = []

        for match in matches:

            results.append({
                "text": match["content"],
                "metadata": match["metadata"],
                "similarity": match["similarity"]
            })

        LOG.info(
            "Retrieved %d chunks",
            len(results)
        )

        return results

    except Exception as exc:

        LOG.exception(
            "Retrieval failed: %s",
            exc
        )

        raise


# -----------------------------------
# CLI testing
# -----------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 3:

        print("Usage:")
        print("python rag_service.py <docx_path> <user_id>")

        raise SystemExit(1)

    path = sys.argv[1]
    uid = sys.argv[2]

    with open(path, "rb") as f:

        count = ingest_docx_file(
            f,
            uid
        )

    print(f"Indexed {count} chunks")