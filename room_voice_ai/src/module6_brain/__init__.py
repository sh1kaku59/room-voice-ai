"""
src/module6_brain/__init__.py
=============================
Export các thành phần chính của Module 6: Core Reasoning AI Brain & RAG Engine.
"""

from src.module6_brain.brain import (
    TOOL_SCHEMAS,
    BrainCore,
    ConstrainedJSONGrammar,
    Module6AIBrainRAG,
)
from src.module6_brain.rag_engine import (
    DEFAULT_PROJECT_KNOWLEDGE_DOCS,
    DenseSparseHybridRAG,
    RAGEngine,
)

__all__ = [
    "DEFAULT_PROJECT_KNOWLEDGE_DOCS",
    "TOOL_SCHEMAS",
    "BrainCore",
    "ConstrainedJSONGrammar",
    "DenseSparseHybridRAG",
    "Module6AIBrainRAG",
    "RAGEngine",
]
