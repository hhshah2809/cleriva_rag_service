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