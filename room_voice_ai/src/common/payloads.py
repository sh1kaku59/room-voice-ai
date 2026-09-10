"""
src/common/payloads.py
======================
Đặc tả JSON Payload dùng chung giữa các module.
Sử dụng dataclass để đảm bảo type-safety và dễ serialize/deserialize.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base Payload
# ---------------------------------------------------------------------------
@dataclass
class BasePayload:
    """Payload gốc chứa metadata chung cho mọi sự kiện."""
    timestamp: float = field(default_factory=time.time)
    session_id: str  = ""

    def to_dict(self) -> dict[str, Any]:
        """Chuyển dataclass thành dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize payload thành chuỗi JSON UTF-8."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Module 1 → Module 2/3: Audio Chunk
# ---------------------------------------------------------------------------
@dataclass
class AudioChunkPayload(BasePayload):
    """
    Payload âm thanh đã qua xử lý (enhanced).
    Phát ra bởi Module 1, tiêu thụ bởi Module 2 và Module 3.
    """
    audio_bytes: bytes = b""     # PCM 16-bit mono 16kHz
    is_speech: bool    = False   # Kết quả VAD
    rms_db: float      = -60.0   # Mức âm lượng (dBFS)


# ---------------------------------------------------------------------------
# Module 2 → Pipeline: Wake-word Event
# ---------------------------------------------------------------------------
@dataclass
class WakeWordPayload(BasePayload):
    """
    Payload sự kiện wake-word.
    Phát ra bởi Module 2 khi phát hiện từ khoá kích hoạt.
    """
    keyword: str       = ""
    confidence: float  = 0.0


# ---------------------------------------------------------------------------
# Module 3 → Module 4: ASR Transcript
# ---------------------------------------------------------------------------
@dataclass
class TranscriptPayload(BasePayload):
    """
    Payload kết quả chuyển giọng nói thành văn bản.
    Phát ra bởi Module 3 (ASR), tiêu thụ bởi Module 4 (NLU).
    """
    text: str          = ""
    is_final: bool     = False   # True = đoạn hoàn chỉnh, False = partial streaming
    language: str      = "vi"
    avg_logprob: float = 0.0     # Độ tin cậy trung bình từ Whisper


# ---------------------------------------------------------------------------
# Module 4 → Module 5/6: NLU Output
# ---------------------------------------------------------------------------
@dataclass
class EntitySlot:
    """Một thực thể (entity) được trích xuất từ câu người dùng."""
    name: str
    value: str
    start: int = 0
    end: int   = 0
    score: float = 0.0


@dataclass
class NLUOutputPayload(BasePayload):
    """
    Payload phân tích ngôn ngữ tự nhiên.
    Phát ra bởi Module 4 (NLU), tiêu thụ bởi Module 5 và Module 6.
    """
    raw_text: str                 = ""
    intent: str                   = ""
    intent_confidence: float      = 0.0
    slots: dict[str, str]         = field(default_factory=dict)
    entities: list[EntitySlot]    = field(default_factory=list)
    sentiment: str                = "neutral"   # positive | negative | neutral
    route_to_context_manager: bool = True
    route_to_ai_brain: bool       = True


# ---------------------------------------------------------------------------
# Module 5 → Module 6: Enriched Context
# ---------------------------------------------------------------------------
@dataclass
class ContextPayload(BasePayload):
    """
    Payload ngữ cảnh hội thoại đã được làm giàu.
    Phát ra bởi Module 5 (Context Manager), tiêu thụ bởi Module 6 (Brain).
    """
    working_memory: str               = ""   # Nội dung Working Memory (prompt snippet)
    rolling_history: list[dict[str, Any]] = field(default_factory=list)  # Lịch sử gần đây
    archival_summary: str | None      = None  # Tóm tắt từ Archival Memory
    retrieved_docs: list[dict[str, Any]] = field(default_factory=list)  # RAG results
    nlu_output: NLUOutputPayload | None = None
    compiled_prompt: str              = ""   # Chuỗi Prompt ngữ cảnh đã tối ưu Knapsack
    estimated_tokens: int             = 0    # Ước tính số token tiêu thụ
    retrieval_latency_ms: float       = 0.0  # Thời gian biên dịch ngữ cảnh (ms)
    speaker_profiles: dict[str, Any]  = field(default_factory=dict)  # Bản đồ thông tin người nói
    archival_decisions: list[dict[str, Any]] = field(default_factory=list)  # Danh sách quyết định/hành động đã chốt


# ---------------------------------------------------------------------------
# Module 6 → Module 7: Brain Response
# ---------------------------------------------------------------------------
@dataclass
class BrainResponsePayload(BasePayload):
    """
    Payload phản hồi từ LLM Brain.
    Phát ra bởi Module 6, tiêu thụ bởi Module 7 (Output Generator).
    """
    text: str           = ""
    is_streaming: bool  = True    # True = token đang stream, False = hoàn chỉnh
    markdown: str       = ""      # Phiên bản Markdown để hiển thị
    action: str         = ""      # Hành động cần thực thi (nếu có)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module 7 → Client: Final Output
# ---------------------------------------------------------------------------
@dataclass
class OutputPayload(BasePayload):
    """
    Payload đầu ra cuối cùng gửi đến client qua WebSocket.
    """
    text_response: str  = ""
    audio_bytes: bytes  = b""    # TTS audio PCM
    tts_done: bool      = False
    display_markdown: str = ""
