"""
prompted.types.openai.embeddings
"""

from typing import List, Literal
from pydantic import BaseModel

__all__ = [
    "OpenAIEmbedding",
    "OpenAIEmbeddingUsage",
]


class OpenAIEmbeddingUsage(BaseModel):
    """
    Represents the usage information for an embedding request in the
    OpenAI specification.
    """
    prompt_tokens: int
    """Number of tokens in the input prompt."""
    total_tokens: int
    """Total number of tokens used in the request (prompt + embedding)."""


class OpenAIEmbedding(BaseModel):
    """
    Represents an embedding vector returned by the embedding endpoint.
    """
    embedding: List[float]
    """
    The embedding vector, which is a list of floats. The length of vector depends
    on the model.
    """
    index: int
    """
    The index of the embedding in the list of embeddings.
    """
    object: Literal["embedding"]
    """
    The object type, which is always `embedding`.
    """