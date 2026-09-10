"""
tests/test_module4.py
======================
KỊCH BẢN KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CHO MODULE 4 NLU
Được xây dựng chuẩn xác theo mục 9 của Doc/module4_specification.md.
Để chạy kiểm thử: pytest -v tests/test_module4.py
"""

import asyncio
import time
from datetime import datetime

import numpy as np
import pytest

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import NLUOutputPayload, TranscriptPayload
from src.module4_nlu.interpreter import Module4MeetingNLU, NLUInterpreter


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus trước và sau mỗi test case."""
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def nlu_engine():
    """Khởi tạo đối tượng NLU cho toàn bộ test cases."""
    return Module4MeetingNLU(onnx_model_path="models/phobert_joint_meeting_nlu.onnx")


# =====================================================================
# 5 Bài kiểm thử chuyên sâu theo đặc tả Doc/module4_specification.md
# =====================================================================

def test_nlu_initialization(nlu_engine):
    """
    Test 1: Kiểm thử khởi tạo đối tượng, cấu hình CPU ONNX và nạp đầy đủ không gian nhãn.
    """
    assert nlu_engine is not None
    assert len(nlu_engine.intent_labels) == 6
    assert "Assign_Action_Item" in nlu_engine.intent_labels
    assert "Summarize_Meeting" in nlu_engine.intent_labels
    assert "Query_Project_Knowledge" in nlu_engine.intent_labels
    assert "Schedule_Followup" in nlu_engine.intent_labels
    assert "Explain_Technical_Concept" in nlu_engine.intent_labels
    assert "Chitchat_Clarification" in nlu_engine.intent_labels
    assert "CPUExecutionProvider" in nlu_engine.session.get_providers()


def test_linguistic_preprocessing_word_segmentation(nlu_engine):
    """
    Test 2: Kiểm thử logic phân đoạn từ ghép và tiền xử lý tiếng Việt.
    """
    text = "Giao cho anh Tuấn kiểm tra tiến độ"
    segmented = nlu_engine._vietnamese_word_segmentation(text)

    # Kiểm tra các từ ghép cuộc họp được nối chuẩn bằng dấu gạch dưới
    assert "kiểm_tra" in segmented
    assert "tiến_độ" in segmented
    assert "giao" in segmented


def test_joint_intent_and_slot_inference(nlu_engine):
    """
    Test 3: Kiểm thử luồng suy luận Joint Learning, đảm bảo đầu ra chứa dự đoán song hành.
    """
    text = "Giao cho anh Tuấn kiểm tra lỗi API trước 5h chiều"
    payload = nlu_engine.process_transcript(text)

    assert payload["intent"] == "Assign_Action_Item"
    assert payload["intent_confidence"] >= 0.90
    assert "PER" in payload["slots"]
    assert "TASK" in payload["slots"]
    assert payload["slots"]["PER"] == "tuấn"


def test_slot_canonicalization_and_normalization(nlu_engine):
    """
    Test 4: Kiểm thử Tầng 3: Ghép nhãn BIO và chuẩn hóa thời gian tương đối về tuyệt đối ISO-8601.
    """
    # Test chuẩn hóa mốc thời gian tương đối
    normalized_time_1 = nlu_engine._normalize_temporal_slot("5h chiều nay")
    normalized_time_2 = nlu_engine._normalize_temporal_slot("15 phút nữa")

    assert "T" in normalized_time_1  # Chứa ký tự phân tách ngày giờ ISO-8601
    assert "T" in normalized_time_2

    # Kiểm tra logic cộng thời gian phút
    dt1 = datetime.fromisoformat(normalized_time_2)
    dt_now = datetime.now()
    assert (dt1 - dt_now).total_seconds() > 0


def test_nlu_payload_schema_and_performance_kpi(nlu_engine):
    """
    Test 5: Kiểm tra tính hợp lệ của cấu trúc JSON Payload đầu ra và đo lường thời gian xử lý (KPI <= 35ms).
    """
    text = "Chúng ta cần họp tóm tắt tiến độ dự án vào ngày mai"

    # Đo đạc thời gian suy luận trên CPU
    t_start = time.perf_counter()
    payload = nlu_engine.process_transcript(text)
    t_end = time.perf_counter()

    latency_ms = (t_end - t_start) * 1000
    print(f"\n[KPI Measure] Latency: {latency_ms:.3f} ms")

    # Kiểm chứng KPI thời gian
    assert latency_ms <= 35.0  # Phải đạt chỉ tiêu dưới 35ms trên CPU

    # Kiểm chứng độ hoàn thiện của cấu trúc Schema
    required_keys = {
        "raw_text", "intent", "intent_confidence",
        "slots", "route_to_context_manager", "route_to_ai_brain"
    }
    assert required_keys.issubset(payload.keys())
    assert isinstance(payload["slots"], dict)
    assert isinstance(payload["route_to_context_manager"], bool)
    assert isinstance(payload["route_to_ai_brain"], bool)


# =====================================================================
# Kiểm thử tích hợp EventBus Pipeline
# =====================================================================

class TestNLUInterpreterEventBus:
    def test_fallback_returns_unknown(self):
        nlu = NLUInterpreter(session_id="test")
        result = nlu._fallback_result("xin chào")
        assert isinstance(result, NLUOutputPayload)
        assert result.intent == "unknown"
        assert result.raw_text == "xin chào"

    @pytest.mark.asyncio
    async def test_ignores_partial_transcript(self):
        nlu = NLUInterpreter(session_id="test")
        outputs = []

        async def on_nlu(p):
            outputs.append(p)

        EventBus.get_instance().subscribe(EventChannel.NLU_OUTPUT, on_nlu)

        partial = TranscriptPayload(session_id="test", text="xin", is_final=False)
        await nlu._on_transcript(partial)
        assert len(outputs) == 0

    @pytest.mark.asyncio
    async def test_processes_final_transcript(self):
        nlu = NLUInterpreter(session_id="test")
        outputs = []

        async def on_nlu(p):
            outputs.append(p)

        EventBus.get_instance().subscribe(EventChannel.NLU_OUTPUT, on_nlu)

        final = TranscriptPayload(
            session_id="test",
            text="Giao cho anh Tuấn kiểm tra lỗi API trước 5h chiều",
            is_final=True,
        )
        await nlu._on_transcript(final)

        # Chờ executor hoàn tất
        await asyncio.sleep(0.3)
        assert len(outputs) == 1
        assert outputs[0].intent == "Assign_Action_Item"
        assert "PER" in outputs[0].slots

    def test_softmax_sums_to_one(self):
        nlu = NLUInterpreter(session_id="test")
        logits = np.array([1.0, 2.0, 0.5, -1.0])
        probs = nlu._softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert all(p >= 0 for p in probs)
