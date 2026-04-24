"""OneMem V1: file-backed AI memory with rebuildable SQLite sidecar."""

import sys

if sys.version_info < (3, 12):
    raise RuntimeError("OneMem requires Python 3.12 or newer.")

from .dedupe import DedupeEngine, MergeCandidate
from .embedding_providers import (
    EmbeddingProvider,
    GeminiEmbeddingProvider,
    HashEmbeddingProvider,
    provider_from_env,
)
from .env import load_default_env, load_env_file
from .models import AppliedOperation, MemoryNode, MemoryOperation, Relation
from .operations import MemoryOperationManager
from .reader import EvidenceReader, ReaderAnswer
from .retrieval import RankedMemory, RetrievalResult
from .runtime import MemoryRuntime
from .summaries import LayeredSummaryBuilder
from .write_policy import MemoryWriteDecision, MemoryWritePolicy

__all__ = [
    "__version__",
    "AppliedOperation",
    "DedupeEngine",
    "EmbeddingProvider",
    "EvidenceReader",
    "GeminiEmbeddingProvider",
    "HashEmbeddingProvider",
    "LayeredSummaryBuilder",
    "MemoryNode",
    "MemoryOperation",
    "MemoryOperationManager",
    "MemoryRuntime",
    "MemoryWriteDecision",
    "MemoryWritePolicy",
    "MergeCandidate",
    "RankedMemory",
    "ReaderAnswer",
    "Relation",
    "RetrievalResult",
    "load_default_env",
    "load_env_file",
    "provider_from_env",
]

__version__ = "0.1.0"
