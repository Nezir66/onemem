"""OneMem V1: file-backed AI memory with rebuildable SQLite sidecar."""

import sys

if sys.version_info < (3, 12):
    raise RuntimeError("OneMem requires Python 3.12 or newer.")

from .chatbot import MemoryChatbot
from .models import MemoryNode, Relation
from .retrieval import RankedMemory, RetrievalResult
from .runtime import MemoryRuntime

__all__ = [
    "__version__",
    "MemoryChatbot",
    "MemoryNode",
    "MemoryRuntime",
    "RankedMemory",
    "Relation",
    "RetrievalResult",
]

__version__ = "0.1.0"
