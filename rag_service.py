import logging
from typing import IO, List, Optional, Dict

import numpy as np
from nltk.tokenize import sent_tokenize
from core.reranker import rerank_documents
from supabase import create_client, Client

from core.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_QUERY_NAME,
)

from core.embeddings import embeddings

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
# -----------------------------------
# Semantic Chunking Configuration
# -----------------------------------

SIMILARITY_THRESHOLD = 0.75

MAX_CHUNK_SENTENCES = 8


# -----------------------------------
# Supabase client
# -----------------------------------

def _get_supabase_client() -> Client:
    """
    Create and cache Supabase client.
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "Supabase configuration missing"
        )

    if not hasattr(_get_supabase_client, "_client"):

        LOG.info("Creating Supabase client")

        _get_supabase_client._client = create_client(
            SUPABASE_URL,
            SUPABASE_KEY
        )

    return _get_supabase_client._client


# -----------------------------------
# Cosine similarity
# -----------------------------------

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# -----------------------------------
# Semantic Chunking
# -----------------------------------

def semantic_chunk_text(
    text: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_chunk_sentences: int = MAX_CHUNK_SENTENCES
) -> List[str]:
    """
    Split text semantically using sentence embeddings.
    """

    # Paragraph-aware splitting first
    paragraphs = [
        p.strip()
        for p in text.split("\n")
        if p.strip()
    ]

    all_chunks = []

    for paragraph in paragraphs:

        sentences = sent_tokenize(paragraph)

        if not sentences:
            continue

        # Small paragraph → keep directly
        if len(sentences) <= 2:

            all_chunks.append(paragraph)

            continue

        LOG.info(
            "Generating sentence embeddings for %d sentences",
            len(sentences)
        )

        sentence_embeddings = embeddings.embed_documents(
            sentences
        )

        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):

            prev_embedding = sentence_embeddings[i - 1]

            curr_embedding = sentence_embeddings[i]

            similarity = cosine_similarity(
                prev_embedding,
                curr_embedding
            )

            # Topic changed OR chunk too large
            if (
                similarity < similarity_threshold
                or len(current_chunk)
                >= max_chunk_sentences
            ):

                chunk_text = " ".join(current_chunk)

                all_chunks.append(chunk_text)

                current_chunk = [sentences[i]]

            else:

                current_chunk.append(sentences[i])

        # Final chunk
        if current_chunk:

            all_chunks.append(
                " ".join(current_chunk)
            )

    return all_chunks


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

            LOG.warning(
                "Uploaded DOCX contains no text"
            )

            return 0

        # -----------------------------------
        # SEMANTIC CHUNKING
        # -----------------------------------

        chunks = semantic_chunk_text(text)

        LOG.info(
            "Created %d semantic chunks",
            len(chunks)
        )

        documents = []

        for idx, chunk in enumerate(chunks):

            metadata = {
                "source": "docx",
                "user_id": user_id,
                "chunk_index": idx,
            }

            if rag_document_id:

                metadata[
                    "rag_document_id"
                ] = rag_document_id

            documents.append({
                "content": chunk,
                "metadata": metadata
            })

        # -----------------------------------
        # Generate embeddings
        # -----------------------------------

        texts = [
            doc["content"]
            for doc in documents
        ]

        LOG.info(
            "Generating embeddings for %d chunks",
            len(texts)
        )

        embeddings_list = embeddings.embed_documents(
            texts
        )

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

        supabase.table(
            "documents"
        ).insert(rows).execute()

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
    Retrieve and rerank relevant chunks.
    """

    try:

        if not query:
            raise ValueError("query is required")

        supabase = _get_supabase_client()

        # -----------------------------------
        # Generate query embedding
        # -----------------------------------

        query_embedding = embeddings.embed_query(query)

        filter_data = {}

        if user_id:
            filter_data["user_id"] = user_id

        LOG.info(
            "Running hybrid retrieval search for user_id=%s",
            user_id
        )

        # -----------------------------------
        # HYBRID SEARCH
        # Retrieve MORE chunks for reranking
        # -----------------------------------

        retrieval_k = 20

        response = supabase.rpc(
            SUPABASE_QUERY_NAME,
            {
                "query_embedding": query_embedding,
                "query_text": query,
                "match_count": retrieval_k,
                "filter": filter_data
            }
        ).execute()

        matches = response.data or []

        LOG.info(
            "Retrieved %d chunks before reranking",
            len(matches)
        )

        results = []

        for idx, match in enumerate(matches):

            LOG.info(
                "Chunk %d | similarity=%.4f | keyword_rank=%.4f | final_score=%.4f",
                idx,
                match["similarity"],
                match["keyword_rank"],
                match["final_score"]
            )

            results.append({
                "text": match["content"],
                "metadata": match["metadata"],
                "similarity": match["similarity"],
                "keyword_rank": match["keyword_rank"],
                "final_score": match["final_score"]
            })

        # -----------------------------------
        # RERANKING
        # -----------------------------------

        LOG.info(
            "Running reranker on %d chunks",
            len(results)
        )

        reranked_results = rerank_documents(
            query=query,
            documents=results,
            top_k=k
        )

        # -----------------------------------
        # Log reranked results
        # -----------------------------------

        for idx, doc in enumerate(reranked_results):

            LOG.info(
                "RERANKED %d | rerank_score=%.4f",
                idx,
                doc["rerank_score"]
            )

        LOG.info(
            "Returning %d reranked chunks",
            len(reranked_results)
        )

        return reranked_results

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
        print(
            "python rag_service.py <docx_path> <user_id>"
        )

        raise SystemExit(1)

    path = sys.argv[1]

    uid = sys.argv[2]

    with open(path, "rb") as f:

        count = ingest_docx_file(
            f,
            uid
        )

    print(f"Indexed {count} chunks")