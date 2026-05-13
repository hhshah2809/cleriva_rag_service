import logging

from langchain_openai import AzureOpenAIEmbeddings

from core.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)

LOG = logging.getLogger(__name__)


def get_azure_embeddings():
    """
    Create Azure OpenAI embeddings client.
    """

    if not all([
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    ]):
        LOG.error("Azure OpenAI embedding configuration is incomplete")
        raise EnvironmentError("Azure OpenAI configuration missing")

    LOG.info("Initializing Azure OpenAI embeddings")

    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )


embeddings = get_azure_embeddings()


def _make_batches(items, batch_size: int):
    """Yield successive batches from items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def embed_documents_batched(
    texts,
    batch_size: int = 20,
    concurrency: int = 4,
):
    """
    Embed a list of texts using batching and limited concurrency.

    This function keeps the external (synchronous) API but runs
    embedding calls concurrently in threads using asyncio.
    """

    import asyncio
    import time

    LOG.info(
        "Embedding %d texts using batch_size=%d concurrency=%d",
        len(texts),
        batch_size,
        concurrency,
    )

    batches = list(_make_batches(texts, batch_size))

    async def _embed_batch(batch):
        # Run the (potentially blocking) embedding call in a thread
        start = time.perf_counter()
        vectors = await asyncio.to_thread(embeddings.embed_documents, batch)
        duration = time.perf_counter() - start
        LOG.info("Embedded batch size=%d in %.2fs", len(batch), duration)
        return vectors

    async def _run():
        sem = asyncio.Semaphore(concurrency)

        async def _worker(batch):
            await sem.acquire()
            try:
                return await _embed_batch(batch)
            finally:
                sem.release()

        tasks = [_worker(batch) for batch in batches]

        start_total = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_dur = time.perf_counter() - start_total
        LOG.info(
            "Completed %d batches (total texts=%d) in %.2fs",
            len(batches),
            len(texts),
            total_dur,
        )

        # flatten results preserving order
        flattened = [vec for batch_result in results for vec in batch_result]
        return flattened

    # Run the async runner
    return asyncio.run(_run())