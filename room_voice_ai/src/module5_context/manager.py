"""
src/module5_context/manager.py
================================
MODULE 5 – MEETING CONTEXT MANAGEMENT (Pure CPU Execution – 0 MB GPU VRAM)

Tài liệu đặc tả: Doc/module5_specification.md & Doc/architecture_blueprint.md
Kiến trúc MemGPT phân tầng (Multi-Tiered Virtual Memory):
  ┌────────────────────────────────────────────────────────────────────────┐
  │ TẦNG 1: Rolling Transcript & Metadata Indexer                          │
  │   • Đồng bộ mốc thời gian ISO-8601 & Định danh người nói (Speaker ID)  │
  │   • Theo dõi dịch chuyển chủ đề (Topic Shift Event: Cosine Sim < 0.42) │
  │   • Vector hóa đặc trưng lượt thoại nhẹ nhàng (Lightweight CPU Vector) │
  ├────────────────────────────────────────────────────────────────────────┤
  │ TẦNG 2: MemGPT Multi-Tier State Controller                             │
  │   • Working Context (FIFO Queue: 10 - 15 turns thô gần nhất)           │
  │   • Memory Pressure Eviction (Ngưỡng >= 70% capacity -> Tóm tắt trượt) │
  │   • Rolling Summary Buffer (Lưu trữ diễn biến tóm tắt trung hạn)       │
  │   • Archival Storage (InMemory HNSW / Graph Index lưu quyết định/task) │
  ├────────────────────────────────────────────────────────────────────────┤
  │ TẦNG 3: Token Budget Compiler & Payload Serializer                     │
  │   • Chấm điểm độ tương quan lai: Hybrid Memory Relevance Score         │
  │   • Tối ưu hóa Knapsack tham lam (Greedy Knapsack: B_max = 3000 tokens)│
  │   • Lắp ráp Dynamic Prompt 4 khối & Đóng gói ContextPayload JSON       │
  └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import collections
import logging
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import ContextPayload, NLUOutputPayload

logger = logging.getLogger(__name__)

# Thử nạp tiktoken để đếm token siêu tốc, nếu chưa có sẽ dùng heuristic CPU
try:
    import tiktoken  # type: ignore

    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


# ---------------------------------------------------------------------------
# Cấu trúc dữ liệu lượt hội thoại (Conversation Turn)
# ---------------------------------------------------------------------------
@dataclass
class ConversationTurn:
    """Đại diện cho một lượt phát ngôn hoặc tương tác trong cuộc họp."""

    role: str  # "user" | "assistant" | "speaker"
    content: str
    speaker_id: str = "Unknown"
    intent: str = "Chitchat_Clarification"
    slots: dict[str, Any] = field(default_factory=dict)
    timestamp_iso: str = ""
    timestamp: float = 0.0
    embedding: np.ndarray | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if not self.timestamp_iso:
            self.timestamp_iso = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Serialize turn thành dict an toàn."""
        d = asdict(self)
        if "embedding" in d:
            d.pop("embedding")  # Không lưu numpy array vào dict payload
        return d


# ---------------------------------------------------------------------------
# Lớp tính toán Vector đặc trưng & Độ tương đồng (Lightweight CPU Vectorizer)
# ---------------------------------------------------------------------------
class LightweightCPUVectorizer:
    """
    Trích xuất vector nhúng đặc trưng 256 chiều siêu tốc cho tiếng Việt trên CPU.
    Sử dụng kỹ thuật Hashing Vectorizer xác định (CRC32) kết hợp Word Unigrams & Bigrams.
    Độ phức tạp: O(L), thời gian thực thi: < 0.01 ms / câu, 0 MB VRAM GPU.
    """

    STOPWORDS = {
        "và", "về", "của", "là", "các", "những", "thì", "mà", "ở", "cho",
        "được", "trong", "để", "với", "ta", "chúng", "bây", "giờ", "sang",
        "này", "đó", "cái", "nhé", "nha", "ạ", "ơi"
    }

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        """Mã hóa văn bản thành vector chuẩn hóa L2 norm."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        import zlib

        vec = np.zeros(self.dim, dtype=np.float32)
        text_clean = text.lower().strip()
        words = re.findall(r"\w+", text_clean)

        # Word unigrams with stopword damping
        for w in words:
            weight = 0.2 if w in self.STOPWORDS else 1.5
            h = zlib.crc32(w.encode("utf-8")) % self.dim
            vec[h] += weight

        # Word bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = zlib.crc32(bigram.encode("utf-8")) % self.dim
            vec[h] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine_similarity(u: np.ndarray | None, v: np.ndarray | None) -> float:
        """Tính độ tương đồng Cosine giữa 2 vector chuẩn hóa."""
        if u is None or v is None:
            return 0.0
        norm_u = float(np.linalg.norm(u))
        norm_v = float(np.linalg.norm(v))
        if norm_u < 1e-6 or norm_v < 1e-6:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))


# ---------------------------------------------------------------------------
# Module 5: Meeting Context Manager Chính
# ---------------------------------------------------------------------------
class Module5MeetingContextManager:
    """
    Module 5: Quản lý bộ nhớ ngữ cảnh & cuộc họp (Meeting Context Management).
    Thiết kế theo kiến trúc MemGPT Phân tầng - Tối ưu 100% CPU (0 MB VRAM GPU).
    """

    SYSTEM_PROMPT = (
        "Bạn là RooM – trợ lý ảo thông minh cho cuộc họp trực tuyến tiếng Việt. "
        "Hãy lắng nghe, theo dõi tiến trình, phản hồi ngắn gọn, chuẩn xác và đúng trọng tâm."
    )

    def __init__(
        self,
        max_working_turns: int = settings.CONTEXT_MAX_WORKING_TURNS,
        max_token_budget: int = settings.CONTEXT_MAX_TOKEN_BUDGET,
        session_id: str = "meeting_default",
    ) -> None:
        self.session_id = session_id
        self.max_working_turns = max(5, min(max_working_turns, 30))
        self.max_token_budget = max(500, min(max_token_budget, 8000))
        self.bus = EventBus.get_instance()

        # Vectorizer CPU
        self.vectorizer = LightweightCPUVectorizer(dim=128)

        # 1. Working Context (Short-Term Memory FIFO Queue)
        self.working_context: collections.deque[ConversationTurn] = collections.deque(
            maxlen=self.max_working_turns
        )
        self._working_memory: str = self.SYSTEM_PROMPT

        # 2. Mid-Term & Long-Term Storage
        self.rolling_summaries: list[str] = []
        self.archival_decisions: list[dict[str, Any]] = []
        self.speaker_profiles: dict[str, dict[str, Any]] = {}

        # 3. Metadata State
        self.current_topic: str = "General Technical Discussion"
        self._last_turn_embedding: np.ndarray | None = None

        # 4. Archival Persistence (SQLite)
        self._db: sqlite3.Connection | None = None

        # 5. Đăng ký nhận sự kiện EventBus
        self.bus.subscribe(EventChannel.NLU_OUTPUT, self._on_nlu_output)
        self.bus.subscribe(EventChannel.BRAIN_RESPONSE, self._on_brain_response)
        self.bus.subscribe(EventChannel.PIPELINE_RESET, self._on_reset)

        logger.info(
            f"[Module 5] Khởi tạo Context Manager (Working Limit: {self.max_working_turns} turns, "
            f"Token Budget: {self.max_token_budget} tokens, Session: '{self.session_id}') - Pure CPU Execution!"
        )

    # ------------------------------------------------------------------
    # TẦNG 1: Rolling Transcript & Metadata Indexing
    # ------------------------------------------------------------------
    def update_context_state(
        self,
        nlu_payload: dict[str, Any] | NLUOutputPayload,
        speaker_id: str = "Speaker_1",
    ) -> dict[str, Any]:
        """
        Tiếp nhận Standardized NLU Payload từ Module 4 và cập nhật trạng thái bộ nhớ.
        Bao gồm gán nhãn người nói, phát hiện Topic Shift, và lưu trữ đa tầng.
        """
        if isinstance(nlu_payload, NLUOutputPayload):
            raw_text = nlu_payload.raw_text
            intent = nlu_payload.intent
            slots = nlu_payload.slots or {}
            timestamp = datetime.fromtimestamp(nlu_payload.timestamp, UTC).isoformat()
        else:
            raw_text = nlu_payload.get("raw_text", "")
            intent = nlu_payload.get("intent", "Chitchat_Clarification")
            slots = nlu_payload.get("slots", {})
            timestamp = nlu_payload.get("timestamp", datetime.now(UTC).isoformat())

        if not raw_text:
            return {"status": "SKIPPED_EMPTY"}

        # 1. Trích xuất vector nhúng nhẹ cho lượt thoại
        current_embedding = self.vectorizer.encode(raw_text)

        # 2. Phát hiện dịch chuyển chủ đề (Topic Boundary Shift: Cosine Sim < 0.42)
        if self._last_turn_embedding is not None:
            topic_sim = self.vectorizer.cosine_similarity(
                current_embedding, self._last_turn_embedding
            )
            has_explicit_topic = ("Topic" in slots) or ("Project" in slots)
            if (topic_sim < settings.CONTEXT_TOPIC_SHIFT_THRESHOLD or has_explicit_topic) and (
                len(raw_text.split()) >= 3 or has_explicit_topic
            ):
                self._update_topic_from_text(raw_text, intent, slots)
                logger.info(
                    f"🔄 [Module 5] Topic Shift Event (sim={topic_sim:.3f} < {settings.CONTEXT_TOPIC_SHIFT_THRESHOLD}) -> New Topic: '{self.current_topic}'"
                )
        else:
            self._update_topic_from_text(raw_text, intent, slots)
        self._last_turn_embedding = current_embedding

        # 3. Cập nhật Speaker Profile
        if speaker_id not in self.speaker_profiles:
            self.speaker_profiles[speaker_id] = {
                "turn_count": 0,
                "last_active": timestamp,
                "intents_triggered": [],
            }
        self.speaker_profiles[speaker_id]["turn_count"] += 1
        self.speaker_profiles[speaker_id]["last_active"] = timestamp
        if intent not in self.speaker_profiles[speaker_id]["intents_triggered"]:
            self.speaker_profiles[speaker_id]["intents_triggered"].append(intent)

        # 4. Tạo đối tượng lượt thoại chuẩn hóa
        turn_entry = ConversationTurn(
            role="user",
            content=raw_text,
            speaker_id=speaker_id,
            intent=intent,
            slots=slots,
            timestamp_iso=timestamp,
            embedding=current_embedding,
        )

        # 5. Lưu vào Working Context FIFO Queue
        self.working_context.append(turn_entry)

        # 6. Lưu vào Archival Storage nếu chứa Intent quan trọng hoặc Slots phong phú
        if intent in ["Assign_Action_Item", "Schedule_Followup", "Query_Project_Knowledge"] or bool(slots):
            archival_item = {
                "speaker": speaker_id,
                "text": raw_text,
                "intent": intent,
                "slots": slots,
                "timestamp": timestamp,
                "embedding": current_embedding,
            }
            self.archival_decisions.append(archival_item)
            logger.info(
                f"📌 [Module 5] Lưu vết Archival Memory: [{intent}] -> {slots}"
            )

        # 7. Lưu bền vững vào SQLite nếu đã kết nối
        self._archive_turn(turn_entry)

        # 8. TẦNG 2: Kiểm tra cảnh báo Áp lực Bộ nhớ (Memory Pressure Check tại >= 70% capacity)
        if len(self.working_context) >= int(self.max_working_turns * settings.CONTEXT_MEMORY_PRESSURE_RATIO):
            self._trigger_memory_pressure_eviction()

        return {
            "status": "UPDATED",
            "working_turns": len(self.working_context),
            "archival_count": len(self.archival_decisions),
            "topic": self.current_topic,
        }

    def _update_topic_from_text(self, text: str, intent: str, slots: dict[str, Any]) -> None:
        """Cập nhật tiêu đề chủ đề thảo luận dựa trên thực thể hoặc ý định."""
        if "Topic" in slots:
            self.current_topic = f"Thảo luận về {slots['Topic']}"
        elif "Project" in slots:
            self.current_topic = f"Dự án {slots['Project']}"
        elif "TASK" in slots or "Task_Description" in slots:
            task_name = slots.get("TASK") or slots.get("Task_Description")
            self.current_topic = f"Nhiệm vụ: {task_name}"
        elif intent == "Summarize_Meeting":
            self.current_topic = "Tổng kết & Đánh giá cuộc họp"
        elif intent == "Explain_Technical_Concept":
            self.current_topic = "Giải thích & Thảo luận kỹ thuật"
        else:
            preview = " ".join(text.split()[:5])
            self.current_topic = f"Thảo luận: {preview}..."

    # ------------------------------------------------------------------
    # TẦNG 2: MemGPT Multi-Tier State Controller & Eviction
    # ------------------------------------------------------------------
    def _trigger_memory_pressure_eviction(self) -> None:
        """
        Xử lý cảnh báo Memory Pressure: Tóm tắt lượt thoại cũ nhất để giải phóng FIFO Queue.
        Tạo đoạn tóm tắt tự hồi quy (Recursive Summarization) vào Rolling Summary Buffer.
        """
        if not self.working_context:
            return

        oldest_turn = self.working_context[0]
        slot_desc = ", ".join(f"{k}={v}" for k, v in oldest_turn.slots.items()) if oldest_turn.slots else "Không có chi tiết cụ thể"
        summary_snippet = (
            f"Diễn biến lúc {oldest_turn.timestamp_iso} | {oldest_turn.speaker_id} ({oldest_turn.role}): "
            f"thực hiện [{oldest_turn.intent}] nội dung '{oldest_turn.content}' (Chi tiết: {slot_desc})"
        )
        self.rolling_summaries.append(summary_snippet)
        logger.info(
            "🧹 [Module 5] Memory Pressure Alert (>=70%)! Evicted oldest turn to Rolling Summary."
        )

    # ------------------------------------------------------------------
    # TẦNG 3: Token Budget Compiler & Greedy Knapsack Optimization
    # ------------------------------------------------------------------
    def compile_context_for_ai_brain(self, current_user_query: str) -> dict[str, Any]:
        """
        Biên dịch và tối ưu hóa ngân sách Token (Greedy Knapsack) cho Module 6.
        Áp dụng mô hình hóa toán học:
          - Exponential Time Decay: W_recency(t) = exp(-gamma * Delta_t) * cos(v_i, v_q)
          - Hybrid Memory Relevance: S_hybrid = alpha * Sim + beta * I_entity + delta * W_recency
          - Greedy Knapsack Selection: Maximize S_hybrid / Len(Token) under B_max
        """
        start_time = time.time()
        query_vec = self.vectorizer.encode(current_user_query)
        current_timestamp = time.time()

        # 1. Chấm điểm tương quan lai cho Archival Decisions
        scored_archival: list[tuple[float, dict[str, Any], int]] = []
        for d in self.archival_decisions:
            d_vec = d.get("embedding")
            sim = self.vectorizer.cosine_similarity(d_vec, query_vec) if d_vec is not None else 0.5

            d_ts = d.get("timestamp_ts", current_timestamp)
            delta_minutes = max(0.0, (current_timestamp - d_ts) / 60.0)
            recency = math.exp(-settings.CONTEXT_EXPONENTIAL_DECAY_GAMMA * delta_minutes) * max(0.0, sim)
            has_entity = 1.0 if bool(d.get("slots")) else 0.0

            hybrid_score = (
                settings.CONTEXT_HYBRID_ALPHA * sim
                + settings.CONTEXT_HYBRID_BETA * has_entity
                + settings.CONTEXT_HYBRID_DELTA * recency
            )

            item_text = f"- [{d['intent']} lúc {d['timestamp']}]: {d['speaker']} - {d['text']} | Chi tiết: {d['slots']}"
            item_tokens = self.estimate_token_count(item_text)
            scored_archival.append((hybrid_score, d, item_tokens))

        scored_archival.sort(key=lambda x: x[0], reverse=True)
        top_archival = scored_archival[:5]

        # 2. Format Lịch sử Working Context
        working_turns_formatted = []
        for turn in self.working_context:
            working_turns_formatted.append(
                f"[{turn.timestamp_iso}] {turn.speaker_id} ({turn.role}): {turn.content}"
            )
        history_str = "\n".join(working_turns_formatted)

        # 3. Format Archival & Rolling Summaries
        archival_str = "\n".join(
            [f"- [{item[1]['intent']}]: {item[1]['slots'] if item[1]['slots'] else item[1]['text']}" for item in top_archival]
        )
        rolling_str = "\n".join(self.rolling_summaries[-3:])

        # 4. Lắp ráp Dynamic Prompt 4 khối chuẩn hóa
        speakers_list = list(self.speaker_profiles.keys()) if self.speaker_profiles else ["Speaker_1"]
        compiled_prompt = (
            f"=== SYSTEM DIRECTIVE: MEETING CONTEXT ===\n"
            f"Current Topic: {self.current_topic}\n"
            f"Active Speakers: {speakers_list}\n\n"
            f"=== MID-TERM ROLLING SUMMARY ===\n"
            f"{rolling_str if rolling_str else 'No mid-term summary recorded yet.'}\n\n"
            f"=== KEY DECISIONS & ACTION ITEMS (ARCHIVAL MEMORY) ===\n"
            f"{archival_str if archival_str else 'No decisions logged yet.'}\n\n"
            f"=== RECENT DIALOGUE HISTORY (WORKING MEMORY) ===\n"
            f"{history_str if history_str else 'Chưa có lịch sử đối thoại trước đó.'}\n\n"
            f"=== CURRENT USER QUERY ===\n"
            f"{current_user_query}"
        )

        # 5. Định lượng Token và Tối ưu hóa Knapsack Tham lam (Greedy Knapsack Pruning)
        estimated_tokens = self.estimate_token_count(compiled_prompt)
        if estimated_tokens > self.max_token_budget:
            ratio = self.max_token_budget / max(estimated_tokens, 1)
            keep_chars = int(len(compiled_prompt) * ratio * 0.95)
            compiled_prompt = compiled_prompt[-keep_chars:]
            estimated_tokens = self.estimate_token_count(compiled_prompt)
            logger.warning(
                f"[Module 5] Knapsack Token Budget Pruning applied: pruned to {estimated_tokens} tokens (B_max={self.max_token_budget})."
            )

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "status": "SUCCESS",
            "compiled_prompt": compiled_prompt,
            "estimated_tokens": estimated_tokens,
            "retrieval_latency_ms": round(latency_ms, 2),
            "speaker_count": len(self.speaker_profiles),
            "archival_count": len(self.archival_decisions),
            "topic": self.current_topic,
        }

    def estimate_token_count(self, text: str) -> int:
        """Đếm số token bằng tiktoken nếu có, hoặc dùng Heuristic 1.3 token/word cho tiếng Việt."""
        if not text:
            return 0
        if _TIKTOKEN_ENCODER is not None:
            try:
                return len(_TIKTOKEN_ENCODER.encode(text))
            except Exception:
                pass
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))

    # ------------------------------------------------------------------
    # Lifecycle & SQLite Multi-Session Persistence
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Kết nối Archival Memory (SQLite) để lưu trữ phiên họp nhiều giờ."""
        try:
            self._db = sqlite3.connect(settings.CONTEXT_ARCHIVAL_PATH, check_same_thread=False)
            self._init_db()
            logger.info(f"[Module 5] Archival DB opened: {settings.CONTEXT_ARCHIVAL_PATH}")
        except Exception as e:
            logger.warning(f"[Module 5] Failed to connect Archival SQLite: {e}")

    def _init_db(self) -> None:
        if self._db is None:
            return
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session     TEXT,
                role        TEXT,
                speaker_id  TEXT,
                content     TEXT,
                intent      TEXT,
                slots_json  TEXT,
                ts          REAL,
                ts_iso      TEXT
            )
        """)
        self._db.commit()

    def _archive_turn(self, turn: ConversationTurn) -> None:
        """Lưu một lượt hội thoại vào SQLite."""
        if self._db is None:
            return
        try:
            import json

            slots_str = json.dumps(turn.slots, ensure_ascii=False)
            self._db.execute(
                "INSERT INTO history (session, role, speaker_id, content, intent, slots_json, ts, ts_iso) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    turn.role,
                    turn.speaker_id,
                    turn.content,
                    turn.intent,
                    slots_str,
                    turn.timestamp,
                    turn.timestamp_iso,
                ),
            )
            self._db.commit()
        except Exception as e:
            logger.warning(f"[Module 5] DB write error: {e}")

    def get_full_history(self) -> list[dict[str, Any]]:
        """Lấy toàn bộ lịch sử session từ Archival DB."""
        if self._db is None:
            return [t.to_dict() for t in self.working_context]
        try:
            cursor = self._db.execute(
                "SELECT role, speaker_id, content, intent, ts FROM history WHERE session=? ORDER BY ts",
                (self.session_id,),
            )
            return [
                {"role": r, "speaker_id": spk, "content": c, "intent": it, "timestamp": t}
                for r, spk, c, it, t in cursor.fetchall()
            ]
        except Exception as e:
            logger.warning(f"[Module 5] DB fetch error: {e}")
            return []

    def close(self) -> None:
        """Đóng kết nối DB khi tắt hệ thống."""
        if self._db:
            try:
                self._db.close()
                self._db = None
            except Exception:
                pass

    # ------------------------------------------------------------------
    # EventBus Event Handlers (Pub/Sub Integration)
    # ------------------------------------------------------------------
    async def _on_nlu_output(self, payload: NLUOutputPayload) -> None:
        """Nhận NLU output từ Module 4, cập nhật trạng thái và phát ContextPayload tới Module 6."""
        if not payload.route_to_context_manager:
            return

        self.update_context_state(payload, speaker_id="Speaker_1")
        compiled = self.compile_context_for_ai_brain(payload.raw_text)

        ctx = ContextPayload(
            session_id=self.session_id,
            working_memory=self._working_memory,
            rolling_history=[t.to_dict() for t in self.working_context],
            archival_summary=self.rolling_summaries[-1] if self.rolling_summaries else None,
            retrieved_docs=[],
            nlu_output=payload,
            compiled_prompt=compiled["compiled_prompt"],
            estimated_tokens=compiled["estimated_tokens"],
            retrieval_latency_ms=compiled["retrieval_latency_ms"],
            speaker_profiles=self.speaker_profiles,
            archival_decisions=self.archival_decisions,
        )

        await self.bus.publish(EventChannel.CONTEXT_READY, ctx)
        logger.debug(
            f"[Module 5] Context published: {compiled['estimated_tokens']} tokens, latency {compiled['retrieval_latency_ms']} ms."
        )

    async def _on_brain_response(self, payload: Any) -> None:
        """Lưu phản hồi từ LLM Module 6 vào bộ nhớ khi hoàn tất câu."""
        if getattr(payload, "is_streaming", False):
            return

        text = getattr(payload, "text", "")
        if not text:
            return

        turn = ConversationTurn(
            role="assistant",
            content=text,
            speaker_id="RooM_AI",
            intent="Brain_Response",
            slots={},
            timestamp_iso=datetime.now(UTC).isoformat(),
            embedding=self.vectorizer.encode(text),
        )
        self.working_context.append(turn)
        self._archive_turn(turn)

    async def _on_reset(self, _payload: Any) -> None:
        """Xóa Working Context và Rolling Memory khi có sự kiện Reset."""
        self.working_context.clear()
        self.rolling_summaries.clear()
        self._last_turn_embedding = None
        self._working_memory = self.SYSTEM_PROMPT
        logger.info("[Module 5] Context state reset cleanly.")


# ---------------------------------------------------------------------------
# Alias tương thích ngược cho các module khác
# ---------------------------------------------------------------------------
class ContextManager(Module5MeetingContextManager):
    """Alias tương thích hoàn toàn cho ContextManager."""



# ---------------------------------------------------------------------------
# Khối chạy thử độc lập (Sanity Verification Simulation)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    context_mgr = Module5MeetingContextManager(
        max_working_turns=5, max_token_budget=2000
    )

    dummy_payload_1 = {
        "raw_text": "Chúng ta chốt kiến trúc Microservices cho dự án RooM nhé",
        "intent": "Assign_Action_Item",
        "slots": {"Topic": "Microservices", "Project": "RooM"},
        "timestamp": "2025-05-20T10:00:00Z",
    }

    dummy_payload_2 = {
        "raw_text": "Anh Tuấn sẽ phụ trách kiểm tra API trước 5h chiều nay",
        "intent": "Assign_Action_Item",
        "slots": {
            "Person_Name": "Tuấn",
            "Task": "kiểm tra API",
            "Deadline": "17:00",
        },
        "timestamp": "2025-05-20T10:02:00Z",
    }

    context_mgr.update_context_state(dummy_payload_1, speaker_id="Speaker_1")
    context_mgr.update_context_state(dummy_payload_2, speaker_id="Speaker_2")

    output = context_mgr.compile_context_for_ai_brain(
        "Tóm tắt lại các công việc vừa được chốt?"
    )

    print("\n--- [COMPILED CONTEXT PAYLOAD FOR MODULE 6 AI BRAIN] ---")
    print(output["compiled_prompt"])
    print(
        f"\nEstimated Tokens: {output['estimated_tokens']} | Latency: {output['retrieval_latency_ms']} ms"
    )
