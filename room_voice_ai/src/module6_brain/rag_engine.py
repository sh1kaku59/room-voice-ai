"""
src/module6_brain/rag_engine.py
=================================
MODULE 6 – TẦNG 2: DENSE-SPARSE HYBRID RAG ENGINE
Đặc tả kỹ thuật: Doc/module6_specification.md, Mục 2.2 & Mục 3.1

Chiến lược Hybrid RAG 3 tầng:
  1. Dense Semantic Search: Vector Embeddings (BGE-M3 / FAISS / Normalized CPU Cosine)
  2. Sparse Lexical Search: Okapi BM25 Keyword Matching (Mã lỗi, API, thuật ngữ IT)
  3. Reciprocal Rank Fusion (RRF): score(d) = 1/(k + r_dense) + 1/(k + r_sparse) (với k=60)
"""

from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import numpy as np

from configs import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tài liệu Tri thức Mặc định của Hệ thống RooM (Default Meeting & Tech Knowledge)
# ---------------------------------------------------------------------------
DEFAULT_PROJECT_KNOWLEDGE_DOCS = [
    {
        "id": "KB-01",
        "title": "Kiến trúc Hệ thống RooM V2.0",
        "content": (
            "Hệ thống RooM (Voice Call Meeting AI) gồm 7 module vận hành theo mô hình Hybrid Edge: "
            "Module 1 (Audio Signal Preprocessing), Module 2 (OpenWakeWord Call Name), "
            "Module 3 (Vietnamese Streaming ASR), Module 4 (PhoBERT-Joint NLU), "
            "Module 5 (MemGPT Context Manager), Module 6 (Core Reasoning AI Brain & RAG), "
            "và Module 7 (Multimodal Output Generator)."
        ),
        "metadata": {"category": "architecture", "tags": ["room", "hybrid_edge", "7_modules"]},
    },
    {
        "id": "KB-02",
        "title": "Quy chuẩn Giao thức Microservices & gRPC",
        "content": (
            "Dự án sử dụng kiến trúc Microservices kết nối các dịch vụ nội bộ qua giao thức gRPC "
            "và Protobuf để giảm thiểu độ trễ mạng dưới 10ms. Mọi API công khai giao tiếp qua REST/WebSocket."
        ),
        "metadata": {"category": "networking", "tags": ["grpc", "protobuf", "microservices"]},
    },
    {
        "id": "KB-03",
        "title": "Quy trình Phân công Công việc & Quản lý Task (Jira Workflow)",
        "content": (
            "Quy trình giao việc trong cuộc họp: Mọi Action Item mới phải xác định rõ Người phụ trách (Assignee), "
            "Mô tả công việc (Task Description) và Hạn chót hoàn thành (Deadline) trước 17:00 chiều. "
            "Sau cuộc họp, thư ký hoặc AI Trợ lý sẽ tự động tạo Ticket lên Jira và thông báo qua Email."
        ),
        "metadata": {"category": "procedure", "tags": ["jira", "action_item", "deadline", "task"]},
    },
    {
        "id": "KB-04",
        "title": "Phân bổ Ngân sách Tài nguyên Phần cứng & VRAM",
        "content": (
            "Phần cứng thực nghiệm laptop: Intel Core i7-12650H, NVIDIA RTX 3050 4GB VRAM. "
            "Ngân sách VRAM: Module 3 PhoWhisper-small ASR chiếm ~900MB, Module 6 Qwen2.5-3B LLM GGUF Q4_K_M "
            "chiếm ~2.2GB - 2.5GB. Các Module 1, 2, 4, 5, 7 chạy thuần CPU AVX2 với 0MB VRAM GPU."
        ),
        "metadata": {"category": "hardware", "tags": ["vram", "rtx3050", "gpu", "cpu"]},
    },
    {
        "id": "KB-05",
        "title": "Bộ lọc Âm học & Triệt tiêu Tiếng vọng (WebRTC APM + DeepFilterNet3)",
        "content": (
            "Module 1 sử dụng WebRTC APM (AEC3, HPF 80Hz, AGC -18dBFS) kết hợp DeepFilterNet3 32 dải ERB "
            "và Silero VAD v5 với máy trạng thái Hangover 256ms (8 frames) để bảo toàn F0 tiếng Việt và khử ồn."
        ),
        "metadata": {"category": "audio", "tags": ["webrtc", "deepfilternet", "vad", "aec3"]},
    },
]


# ---------------------------------------------------------------------------
# Trình Vector Hóa Nhẹ & Tìm kiếm Lexical BM25 Thuần CPU
# ---------------------------------------------------------------------------
class LightweightOkapiBM25:
    """
    Thuật toán BM25 Okapi chuẩn mực:
      score(D, Q) = Σ IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * (|D| / avgdl)))
    Hỗ trợ cả khi không có thư viện rank_bm25.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(self.corpus_size, 1)
        self.doc_freqs: dict[str, int] = {}
        self.corpus_tokens = corpus_tokens

        for doc in corpus_tokens:
            seen: set[str] = set()
            for token in doc:
                if token not in seen:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
                    seen.add(token)

        # Tính IDF
        self.idf: dict[str, float] = {}
        for token, freq in self.doc_freqs.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        """Tính điểm BM25 cho tất cả tài liệu trong corpus."""
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for i, doc in enumerate(self.corpus_tokens):
                f = doc.count(q)
                if f > 0:
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / max(self.avgdl, 1e-6)))
                    scores[i] += q_idf * (numerator / denominator)
        return scores


# ---------------------------------------------------------------------------
# Lớp Hybrid RAG Engine Chính
# ---------------------------------------------------------------------------
class DenseSparseHybridRAG:
    """
    TẦNG 2: Dense-Sparse Hybrid RAG Engine
    Kết hợp BGE-M3 Dense Embeddings và BM25 Sparse Lexical Search qua RRF.
    """

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self._dense_model = None
        self._faiss_index = None
        self._bm25: LightweightOkapiBM25 | Any | None = None
        self._documents: list[dict[str, Any]] = []

        # Tự động nạp bộ tri thức mặc định
        self.index_documents(DEFAULT_PROJECT_KNOWLEDGE_DOCS)

    # ------------------------------------------------------------------
    # Lifecycle & Initialization
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Tải embedding model BGE-M3 và khôi phục index nếu có."""
        self._load_dense_model()
        self._load_faiss_or_fallback()
        logger.info("[Module6/RAG] Hybrid RAG Engine fully initialized.")

    def _load_dense_model(self) -> None:
        model_name_or_path = getattr(settings, "RAG_DENSE_MODEL", "BAAI/bge-m3")
        # Chỉ tải trực tiếp nếu là đường dẫn local hoặc model đã cache local
        if os.path.isdir(model_name_or_path) or os.path.exists(model_name_or_path):
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._dense_model = SentenceTransformer(model_name_or_path, local_files_only=True)
                logger.info(f"[Module6/RAG] Dense model loaded locally from '{model_name_or_path}'.")
                return
            except Exception as e:
                logger.warning(f"[Module6/RAG] Failed to load local SentenceTransformer: {e}")

        # Thử nạp từ local cache HF nếu có sẵn mà không gửi request mạng
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._dense_model = SentenceTransformer(model_name_or_path, local_files_only=True)
            logger.info(f"[Module6/RAG] Dense model '{model_name_or_path}' loaded from local cache.")
        except Exception:
            logger.info(
                f"[Module6/RAG] Dense model '{model_name_or_path}' not cached locally. "
                f"Using Fast CPU Normalized Vectorizer (0 MB VRAM, 100% Offline Mode)."
            )
            self._dense_model = None

    def _load_faiss_or_fallback(self) -> None:
        try:
            import faiss  # type: ignore

            index_path = f"{settings.RAG_VECTOR_DB_PATH}/index.faiss"
            if os.path.exists(index_path):
                self._faiss_index = faiss.read_index(index_path)
                logger.info(f"[Module6/RAG] Loaded FAISS index from disk ({self._faiss_index.ntotal} docs).")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Indexing Documents
    # ------------------------------------------------------------------
    def index_documents(self, docs: list[dict[str, Any]]) -> None:
        """
        Đánh chỉ mục tập tài liệu cho cả Dense và Sparse Search.
        Args:
            docs: list[{"id": str, "title": str, "content": str, "metadata": dict}]
        """
        if not docs:
            return

        self._documents = list(docs)
        texts = [f"{d.get('title', '')} {d.get('content', '')}".strip() for d in self._documents]

        # 1. Build Dense Index
        if self._dense_model is not None:
            try:
                import faiss  # type: ignore

                embeddings = self._dense_model.encode(texts, normalize_embeddings=True)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings.astype(np.float32))
                self._faiss_index = index
                logger.info(f"[Module6/RAG] FAISS Dense Index built ({len(texts)} docs, dim={dim}).")
            except Exception as e:
                logger.warning(f"[Module6/RAG] FAISS build fallback: {e}")

        # 2. Build Sparse BM25 Index
        tokenized_corpus = [self._tokenize_vietnamese(t) for t in texts]
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            self._bm25 = BM25Okapi(tokenized_corpus)
        except Exception:
            self._bm25 = LightweightOkapiBM25(tokenized_corpus)
        logger.info(f"[Module6/RAG] BM25 Sparse Index built ({len(texts)} docs).")

    def _tokenize_vietnamese(self, text: str) -> list[str]:
        """Tách từ và chuẩn hóa cho BM25."""
        text_clean = text.lower()
        words = re.findall(r"\w+", text_clean)
        return words

    # ------------------------------------------------------------------
    # Hybrid Retrieval (Dense + Sparse + RRF)
    # ------------------------------------------------------------------
    async def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Thực hiện tìm kiếm lai Dense-Sparse và hợp nhất bằng Reciprocal Rank Fusion.
        Args:
            query (str): Câu hỏi/từ khóa truy vấn.
            top_k (int): Số tài liệu liên quan nhất trả về (mặc định K=3).
        Returns:
            list các tài liệu có điểm RRF cao nhất.
        """
        if not self._documents or not query.strip():
            return []

        search_k = max(top_k * 2, 5)
        dense_ranked_indices = self._dense_search(query, search_k)
        sparse_ranked_indices = self._sparse_search(query, search_k)

        # Hợp nhất bằng Reciprocal Rank Fusion (k=60)
        fused_indices = self._reciprocal_rank_fusion(
            dense_ranked_indices, sparse_ranked_indices, k=self.rrf_k
        )

        results = []
        for rank, idx in enumerate(fused_indices[:top_k]):
            doc_copy = dict(self._documents[idx])
            doc_copy["rrf_rank"] = rank + 1
            results.append(doc_copy)

        return results

    def _dense_search(self, query: str, k: int) -> list[int]:
        """Tìm kiếm tương đồng Dense Vector."""
        if not self._documents:
            return []

        if self._dense_model is not None and self._faiss_index is not None:
            try:
                q_emb = self._dense_model.encode([query], normalize_embeddings=True).astype(np.float32)
                _, indices = self._faiss_index.search(q_emb, min(k, len(self._documents)))
                return [int(i) for i in indices[0] if i >= 0]
            except Exception:
                pass

        # CPU Fallback Dense Search (Normalized Hashing Vector Cosine)
        q_tokens = set(self._tokenize_vietnamese(query))
        scores = []
        for idx, doc in enumerate(self._documents):
            d_text = f"{doc.get('title', '')} {doc.get('content', '')}"
            d_tokens = set(self._tokenize_vietnamese(d_text))
            intersection = q_tokens.intersection(d_tokens)
            score = len(intersection) / math.sqrt(max(len(q_tokens) * len(d_tokens), 1))
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [idx for _, idx in scores[:k]]

    def _sparse_search(self, query: str, k: int) -> list[int]:
        """Tìm kiếm từ khóa chính xác BM25."""
        if not self._documents or self._bm25 is None:
            return []

        query_tokens = self._tokenize_vietnamese(query)
        if not query_tokens:
            return list(range(min(k, len(self._documents))))

        try:
            scores = self._bm25.get_scores(query_tokens)
            top_indices = np.argsort(scores)[::-1][: min(k, len(self._documents))].tolist()
            return [int(i) for i in top_indices]
        except Exception as e:
            logger.warning(f"[Module6/RAG] Sparse search fallback: {e}")
            return list(range(min(k, len(self._documents))))

    def _reciprocal_rank_fusion(
        self, list1: list[int], list2: list[int], k: int = 60
    ) -> list[int]:
        """
        Thuật toán Reciprocal Rank Fusion:
          RRF_Score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_sparse(d))
        """
        scores: dict[int, float] = {}
        for rank, idx in enumerate(list1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, idx in enumerate(list2):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)

        # Sắp xếp giảm dần theo điểm RRF (Chuẩn hóa Type Inference cho Pyright / Linter)
        return sorted(scores.keys(), key=lambda idx: scores.get(idx, 0.0), reverse=True)


# ---------------------------------------------------------------------------
# Alias tương thích ngược cho các module khác
# ---------------------------------------------------------------------------
class RAGEngine(DenseSparseHybridRAG):
    """Alias tương thích hoàn toàn cho RAGEngine."""

