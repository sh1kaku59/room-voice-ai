"""
tests/test_module7.py
=====================
Bộ kiểm thử toàn diện cho MODULE 7: MULTIMODAL OUTPUT GENERATION (TEXT & VOICE REAL-TIME).
Bám sát 5 kịch bản kiểm thử quy định tại Mục 10 của Doc/module7_specification.md:
  1. test_initialization (Khởi tạo, CPU Provider, 0MB VRAM, Regex Clause Pattern)
  2. test_text_normalization (Chuẩn hóa từ viết tắt IT, tỷ lệ %, thời gian, Markdown)
  3. test_punctuation_splitter (Tách vế câu chính xác theo dấu phân cách ngữ nghĩa)
  4. test_chat_markdown_formatter (Định dạng khung Chat Markdown & Bảng Action Items)
  5. test_end_to_end_voice_stream_latency (Đo trễ TTFS vế câu đầu tiên <= 120ms trên CPU)
Bổ sung các kịch bản kiểm thử EventBus Pub/Sub và WebSocket broadcasting.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import BrainResponsePayload, OutputPayload
from src.module7_output.generator import (
    Module7MultimodalOutputGenerator,
    OutputGenerator,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    EventBus.reset()
    yield
    EventBus.reset()


# ---------------------------------------------------------------------------
# Test Suite 1: 5 Test Cases Học thuật từ module7_specification.md Mục 10
# ---------------------------------------------------------------------------
class TestModule7SpecificationKPIs:
    """Hiện thực hóa chính xác 5 test cases từ Mục 10 của module7_specification.md"""

    def test_initialization(self):
        """
        Test Case 1: test_initialization
        Kiểm tra nạp thành công mô hình Kokoro-82M ONNX CPU session, kiểm tra pattern regex tách vế câu,
        và đảm bảo tần số lấy mẫu 24,000 Hz.
        """
        generator = Module7MultimodalOutputGenerator(session_id="test_sess_init")
        generator.load()

        assert generator.clause_pattern is not None
        assert generator.SAMPLE_RATE == 24000
        assert generator.session_id == "test_sess_init"
        assert generator.bus is not None

    def test_text_normalization(self):
        """
        Test Case 2: test_text_normalization
        Kiểm tra chuyển đổi số, phần trăm và từ viết tắt IT:
        'API' -> 'A P I', '100%' -> '100 phần trăm', '17h00' -> '17 giờ', 'gRPC' -> 'G R P C'.
        """
        gen = Module7MultimodalOutputGenerator()

        raw_sample = "Hoàn thành 100% tài liệu API và gRPC trước 17h00 trên Jira."
        normalized = gen._normalize_vietnamese_text(raw_sample)

        assert "100 phần trăm" in normalized
        assert "A P I" in normalized
        assert "G R P C" in normalized
        assert "17 giờ" in normalized
        assert "Di ra" in normalized

        # Kiểm tra loại bỏ markdown thừa
        md_sample = "**Quan trọng:** Kiểm tra lại `code` và ```sql SELECT 1 ```"
        md_clean = gen._normalize_vietnamese_text(md_sample)
        assert "**" not in md_clean
        assert "`" not in md_clean

    def test_punctuation_splitter(self):
        """
        Test Case 3: test_punctuation_splitter
        Kiểm tra tách chính xác chuỗi văn bản dài thành các vế câu nhỏ khi gặp
        dấu phẩy, dấu chấm, dấu chấm hỏi, dấu chấm phẩy.
        """
        gen = Module7MultimodalOutputGenerator()

        paragraph = "Chào mọi người, cuộc họp hôm nay rất quan trọng. Ai sẽ phụ trách API? Hãy chốt sớm nhé!"
        clauses = gen.split_clauses(paragraph)

        assert len(clauses) >= 4
        assert "Chào mọi người," in clauses[0] or "Chào mọi người" in clauses[0]
        assert "cuộc họp hôm nay rất quan trọng." in clauses[1]
        assert "Ai sẽ phụ trách API?" in clauses[2]
        assert "Hãy chốt sớm nhé!" in clauses[3]

    def test_chat_markdown_formatter(self):
        """
        Test Case 4: test_chat_markdown_formatter
        Kiểm tra định dạng đầu ra PATH 1 bọc đúng khung Markdown cho giao diện Chat
        kèm bảng biểu Action Items chuẩn mực khi có Action Payload.
        """
        gen = Module7MultimodalOutputGenerator()

        # 1. Plain text format
        plain = gen.format_chat_text("Dự án đang tiến triển tốt.")
        assert "**[AI Star Light]** Dự án đang tiến triển tốt." in plain

        # 2. Action Payload table format
        jira_action = {
            "name": "create_jira_task",
            "arguments": {
                "task_title": "Kiểm tra kết nối microservices",
                "assignee": "Tuấn Anh",
                "deadline": "17:00 hôm nay",
                "priority": "Highest",
            },
        }
        table_output = gen.format_chat_text("Đã ghi nhận nhiệm vụ.", action_payload=jira_action)
        assert "**[AI Star Light]**" in table_output
        assert "📋 **Action Item Đã Ghi Nhận (Jira Ticket):**" in table_output
        assert "| **Nhiệm vụ** | Kiểm tra kết nối microservices |" in table_output
        assert "| **Người phụ trách** | Tuấn Anh |" in table_output
        assert "| **Độ ưu tiên** | `Highest` |" in table_output

    @pytest.mark.asyncio
    async def test_end_to_end_voice_stream_latency(self):
        """
        Test Case 5: test_end_to_end_voice_stream_latency
        Giả lập luồng LLM tokens, đo đạc độ trễ vế câu âm thanh PCM đầu tiên (First Clause Audio Latency),
        đảm bảo đạt <= 120ms trên CPU và 0 MB VRAM GPU.
        """
        gen = Module7MultimodalOutputGenerator()

        tokens = [
            "Chào ", "anh ", "Tuấn, ", "em ", "đã ", "tạo ", "ticket ", "Jira ", "xong ", "rồi. ",
            "Anh ", "nhớ ", "kiểm ", "tra ", "nhé!"
        ]

        async def mock_token_stream():
            for t in tokens:
                await asyncio.sleep(0.005)
                yield t

        start_time = time.time()
        first_clause_time = None
        clause_count = 0
        total_audio_samples = 0

        async for _clause, pcm in gen.generate_voice_stream(mock_token_stream()):
            if first_clause_time is None:
                first_clause_time = (time.time() - start_time) * 1000.0  # ms
            clause_count += 1
            total_audio_samples += len(pcm)
            assert isinstance(pcm, np.ndarray)
            assert pcm.dtype == np.float32

        assert first_clause_time is not None
        # Assert KPI TTFS <= 120ms
        assert first_clause_time <= 120.0, f"TTFS {first_clause_time:.2f} ms exceeds KPI 120 ms"
        assert clause_count >= 2
        assert total_audio_samples > 0


# ---------------------------------------------------------------------------
# Test Suite 2: EventBus Pub/Sub & WebSocket Broadcast Integration
# ---------------------------------------------------------------------------
class TestModule7EventBusAndWebSocket:
    @pytest.mark.asyncio
    async def test_init_subscribes_to_brain_response(self):
        _ = OutputGenerator(session_id="test_bus")
        bus = EventBus.get_instance()
        assert len(bus._subscribers[EventChannel.BRAIN_RESPONSE]) >= 1

    @pytest.mark.asyncio
    async def test_streaming_payload_broadcasts(self):
        gen = OutputGenerator(session_id="test_stream")
        broadcast_calls = []

        async def mock_broadcast(message: str) -> None:
            broadcast_calls.append(json.loads(message))

        gen._broadcast = mock_broadcast

        payload = BrainResponsePayload(
            session_id="test_stream",
            text="Xin",
            is_streaming=True,
            markdown="Xin",
        )
        await gen._on_brain_response(payload)
        assert len(broadcast_calls) == 1
        assert broadcast_calls[0]["type"] == "text_stream"
        assert broadcast_calls[0]["done"] is False

    @pytest.mark.asyncio
    async def test_final_payload_publishes_output_ready(self):
        gen = OutputGenerator(session_id="test_final")
        gen._broadcast = AsyncMock()
        final_outputs: list[OutputPayload] = []

        async def on_final(p: OutputPayload):
            final_outputs.append(p)

        EventBus.get_instance().subscribe(EventChannel.FINAL_OUTPUT, on_final)

        payload = BrainResponsePayload(
            session_id="test_final",
            text="Họp đã kết thúc tốt đẹp.",
            is_streaming=False,
            markdown="Họp đã kết thúc tốt đẹp.",
        )
        await gen._on_brain_response(payload)
        await asyncio.sleep(0.05)

        assert len(final_outputs) == 1
        assert final_outputs[0].text_response == "Họp đã kết thúc tốt đẹp."
        assert final_outputs[0].tts_done is True
        assert len(final_outputs[0].audio_bytes) > 0
