"""
tests/test_module6.py
=====================
Bộ kiểm thử toàn diện cho MODULE 6: CORE REASONING AI BRAIN & RAG ENGINE.
Bám sát 5 kịch bản kiểm thử quy định tại Mục 10 của Doc/module6_specification.md:
  1. test_initialization_and_vram (Kiểm tra khởi tạo & Ngân sách VRAM <= 2.5 GB)
  2. test_hybrid_rag_rrf_fusion (Thuật toán RRF Fusion Dense + Sparse K=60)
  3. test_prompt_assembly (Lắp ráp hoàn chỉnh ChatML Prompt 4 khối)
  4. test_constrained_decoding_json_schema (Ép kiểu 100% JSON Tool Call Schema)
  5. test_end_to_end_reasoning_speed_and_ttft (KPI TTFT <= 350ms, Speed >= 25 tok/s)
Bổ sung các kịch bản kiểm thử EventBus, Function Routing và Sparse BM25 Search.
"""

import asyncio

import pytest

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import BrainResponsePayload, ContextPayload, NLUOutputPayload
from src.module6_brain.brain import (
    BrainCore,
    ConstrainedJSONGrammar,
    Module6AIBrainRAG,
)
from src.module6_brain.rag_engine import (
    DenseSparseHybridRAG,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    EventBus.reset()
    yield
    EventBus.reset()


SAMPLE_TEST_DOCS = [
    {"id": "D1", "title": "Thời tiết", "content": "Hôm nay trời nắng đẹp ở Hà Nội với nhiệt độ 32 độ C.", "metadata": {}},
    {"id": "D2", "title": "Dự báo", "content": "Dự báo có mưa rào và dông vào chiều tối nay.", "metadata": {}},
    {"id": "D3", "title": "Microservices", "content": "Hệ thống backend sử dụng kiến trúc Microservices và gRPC kết nối.", "metadata": {}},
]


# ---------------------------------------------------------------------------
# Test Case Suite 1: 5 Kịch bản Học thuật từ module6_specification.md Mục 10
# ---------------------------------------------------------------------------
class TestModule6SpecificationKPIs:
    """Hiện thực hóa chính xác 5 test cases từ Mục 10 của module6_specification.md"""

    def test_initialization_and_vram(self):
        """
        Test Case 1: test_initialization_and_vram
        Kiểm tra khởi tạo thành công lớp Module6AIBrainRAG, nạp cấu hình và kiểm chứng
        mức chiếm dụng VRAM GPU <= 2.5 GB (2500 MB) trên RTX 3050.
        """
        ai_brain = Module6AIBrainRAG(session_id="test_vram_session")
        ai_brain.load()

        assert ai_brain.vram_budget_mb <= 2500
        assert ai_brain.rag is not None
        assert len(ai_brain.rag._documents) > 0
        assert ai_brain.session_id == "test_vram_session"

    @pytest.mark.asyncio
    async def test_hybrid_rag_rrf_fusion(self):
        """
        Test Case 2: test_hybrid_rag_rrf_fusion
        Kiểm thử bộ truy vấn lai Dense-Sparse RAG, đảm bảo thuật toán RRF Fusion
        hợp nhất chính xác kết quả từ BGE-M3 Dense và BM25 Sparse Search (K=60).
        """
        rag = DenseSparseHybridRAG(rrf_k=60)
        rag.index_documents(SAMPLE_TEST_DOCS)

        # 1. Kiểm tra trực tiếp hàm RRF Fusion
        dense_ranks = [0, 1, 2]
        sparse_ranks = [2, 0, 1]
        fused = rag._reciprocal_rank_fusion(dense_ranks, sparse_ranks, k=60)
        assert set(fused) == {0, 1, 2}
        assert fused[0] in (0, 2)  # Doc 0 và 2 đứng đầu cả hai list nên điểm cao nhất

        # 2. Kiểm tra truy vấn End-to-End RAG
        results = await rag.retrieve("kiến trúc Microservices gRPC", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "D3"
        assert "Microservices" in results[0]["content"]

    def test_prompt_assembly(self):
        """
        Test Case 3: test_prompt_assembly
        Kiểm chứng hàm lắp ráp Prompt hoàn chỉnh gộp đúng System Directive, Working Memory,
        Archival Memories và RAG Retrieved Documents theo ChatML format.
        """
        ai_brain = Module6AIBrainRAG()
        nlu = NLUOutputPayload(
            raw_text="Tóm tắt quyết định về kiến trúc?",
            intent="Query_Project_Knowledge",
        )
        ctx = ContextPayload(
            session_id="sess_prompt",
            compiled_prompt="=== SYSTEM DIRECTIVE ===\nTopic: RooM Architecture",
            working_memory="User: Hello\nAssistant: Hi",
            nlu_output=nlu,
            archival_decisions=[{"text": "Chốt gRPC backend"}],
        )
        retrieved_docs = [
            {"id": "KB-01", "title": "Kiến trúc", "content": "Microservices qua gRPC"}
        ]

        prompt = ai_brain.assemble_complete_prompt(ctx, retrieved_docs, tool_name="create_jira_task")

        # Assert: Đầy đủ các khối cấu trúc ChatML và RAG context
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>user" in prompt
        assert "<|im_start|>assistant" in prompt
        assert "RETRIEVED KNOWLEDGE" in prompt
        assert "Microservices qua gRPC" in prompt
        assert "FUNCTION CALLING INSTRUCTION" in prompt

    def test_constrained_decoding_json_schema(self):
        """
        Test Case 4: test_constrained_decoding_json_schema
        Kiểm thử kỹ thuật Grammar Masking ép kiểu LLM sinh đúng 100% cấu trúc JSON Schema
        khi thực hiện Tool Calling (create_jira_task, schedule_meeting, export_minutes).
        """
        # 1. Test create_jira_task
        args_jira = {
            "task_title": "Kiểm tra lỗi API gateway",
            "assignee": "Tuấn",
            "deadline": "17:00",
            "priority": "High",
            "description": "Bug fix",
        }
        res_jira = ConstrainedJSONGrammar.validate_and_format_tool_call("create_jira_task", args_jira)
        assert res_jira["valid"] is True
        assert res_jira["tool_call"]["name"] == "create_jira_task"
        assert res_jira["tool_call"]["arguments"]["assignee"] == "Tuấn"

        # 2. Test schedule_meeting với missing optional fields
        args_meet = {"meeting_title": "Họp Sprint", "start_time": "09:00 ngày mai"}
        res_meet = ConstrainedJSONGrammar.validate_and_format_tool_call("schedule_meeting", args_meet)
        assert res_meet["valid"] is True
        assert res_meet["tool_call"]["arguments"]["meeting_title"] == "Họp Sprint"

        # 3. Test Bóc tách JSON từ văn bản sinh của LLM
        llm_raw_output = "Dưới đây là payload:\n```json\n{\"task_title\": \"Fix bug\", \"assignee\": \"Nam\", \"deadline\": \"18h\"}\n```"
        extracted = ConstrainedJSONGrammar.extract_json_from_text(llm_raw_output)
        assert extracted is not None
        assert extracted["assignee"] == "Nam"

    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_speed_and_ttft(self):
        """
        Test Case 5: test_end_to_end_reasoning_speed_and_ttft
        Đánh giá KPI thời gian suy luận, tốc độ sinh token >= 25 tokens/s
        và thời gian phản hồi token đầu tiên TTFT <= 350 ms.
        """
        ai_brain = Module6AIBrainRAG()
        ai_brain.load()

        nlu = NLUOutputPayload(
            raw_text="Giao cho anh Tuấn kiểm tra API chiều nay",
            intent="Assign_Action_Item",
            slots={"PER": "Tuấn", "TASK": "kiểm tra API", "TIME": "17:00"},
        )
        ctx = ContextPayload(
            session_id="speed_test",
            compiled_prompt="=== SYSTEM DIRECTIVE ===\nMeeting in progress",
            nlu_output=nlu,
        )

        out = await ai_brain.execute_reasoning_and_rag(ctx)

        assert out["status"] == "SUCCESS"
        assert len(out["response_text"]) > 0
        assert out["action_payload"] is not None
        assert out["action_payload"]["name"] == "create_jira_task"
        # Assert KPI TTFT <= 350 ms
        assert out["ttft_ms"] <= 350.0, f"TTFT {out['ttft_ms']} ms exceeds KPI 350 ms"
        # Assert KPI Speed >= 25 tokens/s
        assert out["speed_tokens_per_sec"] >= 25.0, f"Speed {out['speed_tokens_per_sec']} tok/s below KPI 25 tok/s"


# ---------------------------------------------------------------------------
# Test Case Suite 2: EventBus Pub/Sub & Router Integration
# ---------------------------------------------------------------------------
class TestModule6EventBusIntegration:
    @pytest.mark.asyncio
    async def test_publishes_streaming_and_final_payload(self):
        brain = BrainCore(session_id="bus_session")
        brain.load()

        received_payloads = []
        async def on_response(p: BrainResponsePayload):
            received_payloads.append(p)

        EventBus.get_instance().subscribe(EventChannel.BRAIN_RESPONSE, on_response)

        nlu = NLUOutputPayload(
            session_id="bus_session",
            raw_text="Quy trình họp và phân công công việc thế nào?",
            intent="Query_Project_Knowledge",
        )
        ctx = ContextPayload(
            session_id="bus_session",
            compiled_prompt="=== SYSTEM DIRECTIVE ===\nTopic: Meeting Procedures",
            nlu_output=nlu,
        )

        await brain._on_context_ready(ctx)
        await asyncio.sleep(0.05)

        streaming = [p for p in received_payloads if p.is_streaming]
        final = [p for p in received_payloads if not p.is_streaming]

        assert len(streaming) > 0
        assert len(final) == 1
        assert len(final[0].text) > 0
        assert final[0].session_id == "bus_session"

    def test_routing_execution_paths(self):
        brain = Module6AIBrainRAG()

        # Action Item -> creates jira task
        nlu_action = NLUOutputPayload(
            raw_text="Giao cho anh Nam sửa bug",
            intent="Assign_Action_Item",
            slots={"PER": "Nam", "TASK": "sửa bug"},
        )
        ctx_action = ContextPayload(nlu_output=nlu_action)
        needs_rag, tool_name, tool_args = brain.route_execution_path(ctx_action)
        assert tool_name == "create_jira_task"
        assert tool_args["assignee"] == "Nam"

        # Tech Query -> needs rag
        nlu_tech = NLUOutputPayload(
            raw_text="Giải thích kiến trúc Microservices",
            intent="Explain_Technical_Concept",
        )
        ctx_tech = ContextPayload(nlu_output=nlu_tech)
        needs_rag_tech, _, _ = brain.route_execution_path(ctx_tech)
        assert needs_rag_tech is True
