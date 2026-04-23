"""OneMem V1: file-backed AI memory with rebuildable SQLite sidecar."""

import sys

if sys.version_info < (3, 12):
    raise RuntimeError("OneMem requires Python 3.12 or newer.")

from .chatbot import MemoryChatbot
from .embedding_providers import GeminiEmbeddingProvider, HashEmbeddingProvider
from .env import load_default_env, load_env_file
from .models import MemoryNode, Relation
from .retrieval import RankedMemory, RetrievalResult
from .runtime import MemoryRuntime
from .write_policy import MemoryWriteDecision, MemoryWritePolicy

__all__ = [
    "__version__",
    "MemoryChatbot",
    "GeminiEmbeddingProvider",
    "HashEmbeddingProvider",
    "load_default_env",
    "load_env_file",
    "MemoryNode",
    "MemoryRuntime",
    "MemoryWriteDecision",
    "MemoryWritePolicy",
    "RankedMemory",
    "Relation",
    "RetrievalResult",
]

__version__ = "0.1.0"
