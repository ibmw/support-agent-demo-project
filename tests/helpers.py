"""Shared test utilities and helpers."""

import hashlib

import numpy as np


def generate_fake_embedding(text: str, dim: int = 1536) -> list[float]:
    """
    Generate a deterministic fake embedding vector for testing.

    Uses MD5 hash for cross-session reproducibility, numpy RNG for
    realistic normal distribution, and L2 normalization to mimic
    real embedding behavior.

    Args:
        text: Input text to generate embedding for
        dim: Embedding dimension (default 1536 for text-embedding-3-small)

    Returns:
        Normalized embedding vector as list of floats
    """
    # MD5 hash for deterministic seed across sessions
    hash_digest = hashlib.md5(text.encode()).digest()
    seed = int.from_bytes(hash_digest, "big") % (2**32)

    # Local RNG to avoid affecting global random state
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim)

    # L2 normalize to mimic real embeddings (unit vectors)
    normalized = vector / np.linalg.norm(vector)
    return normalized.tolist()
