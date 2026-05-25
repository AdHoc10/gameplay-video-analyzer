"""
Backwards-compatibility shim.

The embedders moved to `utils.embedders` so other features (e.g., the
team-learning module) can share them. Imports from this path keep working.
"""

from utils.embedders import (
    Embedder,
    OSNetEmbedder,
    ResNet50Embedder,
    TORCHREID_AVAILABLE,
    build_default_embedder,
)

__all__ = [
    "Embedder",
    "OSNetEmbedder",
    "ResNet50Embedder",
    "TORCHREID_AVAILABLE",
    "build_default_embedder",
]
