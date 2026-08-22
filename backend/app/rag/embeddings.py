"""
Embedding service for the Aster & Row knowledge base.

This module provides a single embedding-model instance for the application.
The same model must be used when building the FAISS index and when embedding
user queries at retrieval time.

Keeping model creation in one module avoids repeatedly loading the
SentenceTransformer model, which is an expensive operation.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the shared Hugging Face embedding model.

    The function is cached so that the Sentence Transformer model is loaded
    only once per application process.

    Returns:
        Configured LangChain HuggingFaceEmbeddings instance.
    """

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )