
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_model():
    """Load the pretrained embedding model only once."""
    return TextEmbedding(model_name=MODEL_NAME)


@lru_cache(maxsize=2048)
def _cached_embedding(text: str) -> tuple:
    """Cache embeddings for repeated text inputs."""
    normalized_text = " ".join(text.strip().split())

    if not normalized_text:
        return tuple(
            np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
        )

    embedding = next(
        get_model().embed([normalized_text])
    )

    vector = np.asarray(embedding, dtype=np.float32)

    if vector.ndim != 1 or vector.size != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSION}-dimensional embedding, "
            f"got shape {vector.shape}"
        )

    return tuple(vector.tolist())


def embed_text(text: str) -> np.ndarray:
    """Convert text into a 384-dimensional embedding vector."""
    if not isinstance(text, str):
        text = ""

    return np.asarray(
        _cached_embedding(text),
        dtype=np.float32
    )


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Calculate cosine similarity between two text inputs."""
    vector_a = embed_text(text_a)
    vector_b = embed_text(text_b)

    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )

    # Keep the mathematical cosine value within [-1, 1].
    return max(-1.0, min(1.0, similarity))