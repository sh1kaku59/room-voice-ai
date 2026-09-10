"""
tests/test_module5.py
=====================
Bộ kiểm thử toàn diện cho MODULE 5: MEETING CONTEXT MANAGEMENT.
Bám sát 5 kịch bản kiểm thử quy định tại Mục 10 của Doc/module5_specification.md:
  1. test_working_memory_fifo_limit (FIFO Queue 10-15 turns)
  2. test_memory_pressure_eviction_threshold (Cảnh báo >= 70% capacity)
  3. test_archival_decision_logging (Lưu trữ Action Items/Decisions)
  4. test_token_budget_knapsack_pruning (Greedy Knapsack B_max <= 3000)
  5. test_end_to_end_context_compilation_latency (KPI <= 15.0 ms trên CPU)
Bổ sung các kịch bản kiểm thử tích hợp EventBus, Topic Shift và Decay Scoring.
"""

import asyncio
import time

import pytest

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import ContextPayload, NLUOutputPayload
from src.module5_context.manager import (
    ConversationTurn,
    LightweightCPUVectorizer,
    Module5MeetingContextManager,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def temp_db(tmp_path):
    """Tạo file SQLite tạm cho test."""
    return str(tmp_path / "test_archival.db")


# ---------------------------------------------------------------------------
# Test Case Suite 1: ConversationTurn & Vectorizer
# ---------------------------------------------------------------------------
class TestConversationTurnAndVectorizer:
    def test_timestamp_auto_set(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.timestamp > 0
        assert "T" in turn.timestamp_iso

    def test_to_dict_serialization(self):
        turn = ConversationTurn(
            role="assistant",
            content="hi",
            timestamp=1000.0,
            speaker_id="Speaker_2",
            intent="Assign_Action_Item",
            slots={"TASK": "deploy"},
        )
        d = turn.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "hi"
        assert d["speaker_id"] == "Speaker_2"
        assert d["intent"] == "Assign_Action_Item"
        assert d["slots"] == {"TASK": "deploy"}
        assert "embedding" not in d  # Không rò rỉ numpy array vào payload dict

    def test_cpu_vectorizer_properties(self):
        vec = LightweightCPUVectorizer(dim=128)
        v1 = vec.encode("Chúng ta cần họp bàn về kiến trúc Microservices")
        v2 = vec.encode("Chúng ta họp thảo luận về Microservices")
        v3 = vec.encode("Hôm nay thời tiết bên ngoài rất đẹp và mát mẻ")

        assert len(v1) == 128
        sim_related = vec.cosine_similarity(v1, v2)
        sim_unrelated = vec.cosine_similarity(v1, v3)

        assert sim_related > sim_unrelated
        assert 0.0 <= sim_related <= 1.0


# ---------------------------------------------------------------------------
# Test Case Suite 2: 5 Kịch bản Học thuật từ module5_specification.md Mục 10
# ---------------------------------------------------------------------------
class TestModule5SpecificationKPIs:
    """Hiện thực hóa chính xác 5 test cases từ Mục 10 của module5_specification.md"""

    def test_working_memory_fifo_limit(self):
        """
        Test Case 1: test_working_memory_fifo_limit
        Kiểm tra hàng đợi Working Context giữ đúng giới hạn max_working_turns (12 turns)
        và đẩy lượt thoại cũ ra ngoài theo đúng cơ chế FIFO.
        """
        mgr = Module5MeetingContextManager(max_working_turns=12)
        for i in range(20):
            payload = {
                "raw_text": f"Lượt thoại số {i}: thảo luận nội dung {i}",
                "intent": "Chitchat_Clarification",
                "slots": {},
                "timestamp": f"2026-09-09T10:{i:02d}:00Z",
            }
            mgr.update_context_state(payload, speaker_id=f"Speaker_{i % 3}")

        # Assert: Working context chỉ giữ đúng max_working_turns gần nhất
        assert len(mgr.working_context) == 12
        # Assert: Lượt thoại đầu tiên trong hàng đợi là turn số 8 (20 - 12)
        assert "Lượt thoại số 8" in mgr.working_context[0].content
        # Assert: Lượt thoại cuối cùng là turn số 19
        assert "Lượt thoại số 19" in mgr.working_context[-1].content

    def test_memory_pressure_eviction_threshold(self):
        """
        Test Case 2: test_memory_pressure_eviction_threshold
        Giả lập nạp liên tục 10 lượt thoại, kiểm chứng cờ cảnh báo Memory Pressure
        tự động kích hoạt tại ngưỡng >= 70% capacity và tạo tóm tắt vào Rolling Summary Buffer.
        """
        mgr = Module5MeetingContextManager(max_working_turns=10)
        # 70% của 10 = 7 turns. Từ turn 7 trở đi phải có eviction
        for i in range(10):
            payload = {
                "raw_text": f"Nội dung thảo luận số {i}",
                "intent": "Assign_Action_Item" if i % 2 == 0 else "Chitchat_Clarification",
                "slots": {"TASK": f"task_{i}"} if i % 2 == 0 else {},
                "timestamp": f"2026-09-09T11:{i:02d}:00Z",
            }
            mgr.update_context_state(payload, speaker_id="Speaker_1")

        # Assert: Rolling summaries đã nhận ít nhất 3 tóm tắt từ các lần trigger
        assert len(mgr.rolling_summaries) >= 3
        # Assert: Nội dung tóm tắt có chứa từ khóa
        assert any("Diễn biến lúc" in s for s in mgr.rolling_summaries)

    def test_archival_decision_logging(self):
        """
        Test Case 3: test_archival_decision_logging
        Kiểm tra các NLU Payloads chứa Intent Assign_Action_Item và Schedule_Followup
        được lưu trữ chính xác vào Archival Storage cùng đầy đủ Slots.
        """
        mgr = Module5MeetingContextManager(max_working_turns=12)

        p1 = {
            "raw_text": "Giao cho anh Tuấn kiểm tra API trước 5h chiều",
            "intent": "Assign_Action_Item",
            "slots": {"PER": "Tuấn", "TASK": "kiểm tra API", "TIME": "17:00"},
            "timestamp": "2026-09-09T14:00:00Z",
        }
        p2 = {
            "raw_text": "Lên lịch họp tiếp theo vào sáng mai lúc 9h",
            "intent": "Schedule_Followup",
            "slots": {"TIME": "sáng mai 9h"},
            "timestamp": "2026-09-09T14:05:00Z",
        }
        p3 = {
            "raw_text": "Xin chào mọi người hôm nay trời đẹp quá",
            "intent": "Chitchat_Clarification",
            "slots": {},
            "timestamp": "2026-09-09T14:10:00Z",
        }

        mgr.update_context_state(p1, speaker_id="Speaker_Leader")
        mgr.update_context_state(p2, speaker_id="Speaker_Secretary")
        mgr.update_context_state(p3, speaker_id="Speaker_Member")

        # Assert: Chỉ có 2 turns chứa Intent quan trọng được lưu vào Archival Decisions
        assert len(mgr.archival_decisions) == 2
        assert mgr.archival_decisions[0]["intent"] == "Assign_Action_Item"
        assert mgr.archival_decisions[0]["slots"]["PER"] == "Tuấn"
        assert mgr.archival_decisions[1]["intent"] == "Schedule_Followup"

    def test_token_budget_knapsack_pruning(self):
        """
        Test Case 4: test_token_budget_knapsack_pruning
        Tạo prompt giả lập dài > 5,000 tokens, kiểm chứng thuật toán cắt gọt
        khống chế thành công dung lượng đầu ra trong khoảng <= 3,000 tokens.
        """
        mgr = Module5MeetingContextManager(max_working_turns=20, max_token_budget=1500)

        # Sinh 20 lượt thoại dài
        long_paragraph = "Đây là đoạn văn bản thảo luận kỹ thuật rất dài chứa nhiều thông tin chi tiết về hệ thống. " * 30
        for i in range(20):
            payload = {
                "raw_text": f"Lượt {i}: {long_paragraph}",
                "intent": "Explain_Technical_Concept",
                "slots": {"Concept": f"Concept_{i}"},
                "timestamp": f"2026-09-09T15:{i:02d}:00Z",
            }
            mgr.update_context_state(payload, speaker_id=f"Speaker_{i % 4}")

        # Biên dịch Context với câu truy vấn người dùng
        res = mgr.compile_context_for_ai_brain("Tóm tắt toàn bộ các khái niệm kỹ thuật?")

        # Assert: Token sau khi cắt gọt phải <= max_token_budget
        assert res["status"] == "SUCCESS"
        assert res["estimated_tokens"] <= 1500
        assert len(res["compiled_prompt"]) > 0

    def test_end_to_end_context_compilation_latency(self):
        """
        Test Case 5: test_end_to_end_context_compilation_latency
        Đo lường tổng thời gian biên dịch context cho 50 lượt thoại liên tiếp,
        yêu cầu đạt KPI độ trễ <= 15.0 ms trên CPU.
        """
        mgr = Module5MeetingContextManager(max_working_turns=15, max_token_budget=3000)

        # Nạp sẵn 30 lượt thoại
        for i in range(30):
            payload = {
                "raw_text": f"Thảo luận về module {i % 7}: kiểm tra hiệu năng hệ thống và giải pháp mở rộng quy mô",
                "intent": "Assign_Action_Item" if i % 3 == 0 else "Chitchat_Clarification",
                "slots": {"Module": f"Mod_{i % 7}"} if i % 3 == 0 else {},
                "timestamp": f"2026-09-09T16:{i:02d}:00Z",
            }
            mgr.update_context_state(payload, speaker_id=f"Speaker_{i % 5}")

        # Đo đạc 50 lần biên dịch liên tiếp
        latencies = []
        for j in range(50):
            t0 = time.perf_counter()
            _out = mgr.compile_context_for_ai_brain(f"Truy vấn {j}: Ai đang phụ trách task?")
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Assert KPI: Latency trung bình <= 15.0 ms (thực tế thuần CPU < 1.0 ms)
        assert avg_latency <= 15.0, f"Avg latency {avg_latency:.2f} ms exceeds KPI 15.0 ms"
        assert max_latency <= 30.0, f"Max latency {max_latency:.2f} ms is too high"


# ---------------------------------------------------------------------------
# Test Case Suite 3: Topic Shift & EventBus Integration
# ---------------------------------------------------------------------------
class TestTopicShiftAndEventBus:
    def test_topic_shift_detection(self):
        mgr = Module5MeetingContextManager(max_working_turns=10)

        p1 = {
            "raw_text": "Chúng ta bắt đầu thảo luận về kiến trúc Microservices và gRPC backend",
            "intent": "Explain_Technical_Concept",
            "slots": {"Topic": "Microservices"},
            "timestamp": "2026-09-09T10:00:00Z",
        }
        p2 = {
            "raw_text": "Bây giờ chuyển sang bàn về kế hoạch tài chính và ngân sách marketing quý 4",
            "intent": "Assign_Action_Item",
            "slots": {"Topic": "Ngân sách Marketing"},
            "timestamp": "2026-09-09T10:15:00Z",
        }

        mgr.update_context_state(p1, speaker_id="Speaker_1")
        assert "Microservices" in mgr.current_topic

        mgr.update_context_state(p2, speaker_id="Speaker_2")
        assert "Marketing" in mgr.current_topic or "Ngân sách" in mgr.current_topic

    @pytest.mark.asyncio
    async def test_publishes_context_on_nlu(self, temp_db, monkeypatch):
        monkeypatch.setattr(settings, "CONTEXT_ARCHIVAL_PATH", temp_db)
        mgr = Module5MeetingContextManager(session_id="test_session")
        mgr.load()

        received = []
        async def on_ctx(p: ContextPayload):
            received.append(p)

        EventBus.get_instance().subscribe(EventChannel.CONTEXT_READY, on_ctx)

        nlu = NLUOutputPayload(
            session_id="test_session",
            raw_text="Giao cho anh Tuấn kiểm tra API chiều nay",
            intent="Assign_Action_Item",
            intent_confidence=0.95,
            slots={"PER": "Tuấn", "TASK": "kiểm tra API"},
        )

        await mgr._on_nlu_output(nlu)
        await asyncio.sleep(0.05)

        assert len(received) == 1
        ctx = received[0]
        assert isinstance(ctx, ContextPayload)
        assert len(ctx.compiled_prompt) > 0
        assert ctx.estimated_tokens > 0
        assert "SYSTEM DIRECTIVE" in ctx.compiled_prompt
        assert "RECENT DIALOGUE HISTORY" in ctx.compiled_prompt
        mgr.close()

    @pytest.mark.asyncio
    async def test_archival_persistence_sqlite(self, temp_db, monkeypatch):
        monkeypatch.setattr(settings, "CONTEXT_ARCHIVAL_PATH", temp_db)
        mgr = Module5MeetingContextManager(session_id="session_persist")
        mgr.load()

        turn = ConversationTurn(
            role="user",
            content="Quyết định chọn React làm frontend",
            speaker_id="Speaker_1",
            intent="Assign_Action_Item",
            slots={"Framework": "React"},
        )
        mgr._archive_turn(turn)

        history = mgr.get_full_history()
        assert len(history) >= 1
        assert history[0]["content"] == "Quyết định chọn React làm frontend"
        assert history[0]["speaker_id"] == "Speaker_1"
        mgr.close()

    @pytest.mark.asyncio
    async def test_reset_event_handling(self):
        mgr = Module5MeetingContextManager(max_working_turns=10)
        p = {
            "raw_text": "Nội dung cần xóa",
            "intent": "Chitchat_Clarification",
            "slots": {},
        }
        mgr.update_context_state(p)
        assert len(mgr.working_context) == 1

        await mgr._on_reset(None)
        assert len(mgr.working_context) == 0
        assert len(mgr.rolling_summaries) == 0
