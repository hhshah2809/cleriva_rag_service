from sentence_transformers import CrossEncoder

# Load reranker model once globally
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = 5
):
    """
    Rerank retrieved chunks using BGE reranker.
    """

    if not documents:
        return []

    # Create query-document pairs
    pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    # Generate rerank scores
    scores = reranker.predict(pairs)

    # Attach scores
    for i, score in enumerate(scores):

        documents[i]["rerank_score"] = float(score)

    # Sort by rerank score
    reranked = sorted(
        documents,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    # Return top-k
    return reranked[:top_k]