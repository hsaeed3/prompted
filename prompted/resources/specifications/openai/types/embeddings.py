"""
prompted.resources.specifications.openai.types.embeddings
"""

from typing import List, Literal
from typing_extensions import Required, TypedDict

__all__ = ["OpenAIEmbeddingsUsage", "OpenAIEmbedding"]


class OpenAIEmbeddingsUsage(TypedDict):
    """
    Usage response from the OpenAI Embeddings API.
    """

    prompt_tokens: Required[int]
    """
    The number of tokens in the prompt.
    """
    total_tokens: Required[int]
    """
    The total number of tokens processed.
    """


class OpenAIEmbedding(TypedDict):
    """
    Represents an embedding vector returned by the embedding endpoint.
    """

    embedding: Required[List[float]]
    """
    The embedding vector, which is a list of floats. The length of vector depends
    on the model.
    """
    index: Required[int]
    """
    The index of the embedding in the list of embeddings.
    """
    object: Required[Literal["embedding"]]
    """
    The object type, which is always `embedding`.
    """
