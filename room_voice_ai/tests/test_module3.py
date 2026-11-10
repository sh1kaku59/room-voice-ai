"""
tests/test_module3.py
======================
KỊCH BẢN KIỂM THỬ ĐƠN VỊ (UNIT TESTS) CHO MODULE 3 ASR
Được xây dựng chuẩn xác theo mục 7 của Doc/module3_specification.md.
Để chạy kiểm thử: pytest -v tests/test_module3.py
"""

import time

import numpy as np
import pytest

from src.common.event_bus import EventBus
from src.common.payloads import AudioChunkPayload, WakeWordPayload
from src.module3_asr.transcriber import (
    Module3VietnameseStreamingASR,
    StreamingTranscriber,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus trước và sau mỗi test case."""
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def asr_engine():
    """Fixture khởi tạo mô hình ASR ở chế độ kiểm thử (Mock mode / CPU)."""
    return Module3VietnameseStreamingASR(model_path="", device="cpu", compute_type="int8")


# =====================================================================
# 7 Bài kiểm thử chuyên sâu theo đặc tả Doc/module3_specification.md
# =====================================================================

def test_asr_initialization(asr_engine):
    """Test 1: Kiểm thử khởi tạo thành công class và các bộ đệm rỗng."""
    assert asr_engine.device == "cpu"
    assert asr_engine.compute_type == "int8"
    assert len(asr_engine.context_buffer) == 0
    assert asr_engine.committed_text == ""
    assert asr_engine.unconfirmed_text == ""


def test_context_overlap_buffer(asr_engine):
    """Test 2: Kiểm thử gối đầu bộ đệm 150ms (2400 samples) cho chunk sau."""
    dummy_audio_large = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)  # 1.0 giây
    asr_engine.transcribe_streaming_chunk(dummy_audio_large, is_speech_final=False)

    # Kích thước bộ đệm sau khi trượt phải bằng đúng overlap_samples (2400 samples)
    assert len(asr_engine.context_buffer) == asr_engine.overlap_samples


def test_longest_common_prefix():
    """Test 3: Kiểm thử thuật toán tìm Word-level LCP giữa các giả thuyết."""
    # Khớp hoàn toàn phần đầu
    s1 = "Tôi muốn deploy hệ thống"
    s2 = "Tôi muốn deploy sản phẩm mới"
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s2) == "Tôi muốn deploy"

    # Không khớp từ đầu tiên
    s3 = "Anh muốn deploy"
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s3) == ""

    # Khớp hoàn toàn cả hai chuỗi
    assert Module3VietnameseStreamingASR.longest_common_prefix(s1, s1) == s1.strip()


def test_local_agreement_committing(asr_engine):
    """Test 4: Kiểm thử cơ chế khóa tiền tố LCP qua các bước nhận dạng."""
    chunk1 = np.random.uniform(-0.01, 0.01, 8000).astype(np.float32)

    # Giả thuyết đầu tiên được đưa ra
    asr_engine.unconfirmed_text = "Tôi muốn deploy"

    # chunk tiếp theo giải mã khớp 2 từ đầu, thay thế từ sau
    stable, _hyp = asr_engine.transcribe_streaming_chunk(chunk1, is_speech_final=False)

    # Tiền tố chung "Tôi muốn" phải được commit
    assert "Tôi muốn" in stable


def test_code_switching_normalization(asr_engine):
    """Test 5: Kiểm thử bộ chuyển đổi phiên âm IT Việt-Anh cực đại."""
    input_text = "Hôm nay tôi sẽ đì ploi cái sụt áp bi ai lên bét en để sửa bấc"
    expected_output = "Hôm nay tôi sẽ deploy cái source code API lên backend để sửa bug"

    normalized = asr_engine.apply_code_switching(input_text)
    assert normalized == expected_output


def test_vietnamese_punctuation_and_capitalization(asr_engine):
    """Test 6: Kiểm thử bộ chuẩn hóa chữ cái đầu câu và khoảng trắng."""
    dirty_text = "   hôm nay   phòng họp chúng ta bắt đầu bàn về gRPC.  "
    expected = "Hôm nay phòng họp chúng ta bắt đầu bàn về gRPC."

    output = asr_engine.normalize_vietnamese_text(dirty_text)
    assert output == expected


def test_end_to_end_streaming_kpi(asr_engine):
    """Test 7: Đánh giá KPI thời gian xử lý một khung (Latency < 150ms)."""
    dummy_chunk = np.random.uniform(-0.05, 0.05, 8000).astype(np.float32)  # 0.5s audio

    start_time = time.perf_counter()
    _committed, _hyp = asr_engine.transcribe_streaming_chunk(dummy_chunk, is_speech_final=False)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n[ASR KPI] Thời gian xử lý chunk 500ms: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 150.0  # Thử nghiệm trên CPU phải đạt < 150ms


# =====================================================================
# Kiểm thử tích hợp EventBus & Async Pipeline
# =====================================================================

class TestStreamingTranscriberEventBus:
    def test_not_listening_by_default(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        assert t._listening is False

    @pytest.mark.asyncio
    async def test_starts_listening_on_wake_word(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        await t._on_wake_word(WakeWordPayload(session_id="test", keyword="hey_room"))
        assert t._listening is True

    @pytest.mark.asyncio
    async def test_ignores_chunks_when_not_listening(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        chunk = AudioChunkPayload(session_id="test", audio_bytes=b"\x00" * 960, is_speech=True)
        await t._on_audio_chunk(chunk)
        assert len(t._audio_buffer) == 0

    @pytest.mark.asyncio
    async def test_buffers_speech_chunks_when_listening(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        await t._on_wake_word(WakeWordPayload(session_id="test", keyword="hey_room"))

        pcm = (np.random.randn(480) * 5000).astype(np.int16).tobytes()
        chunk = AudioChunkPayload(session_id="test", audio_bytes=pcm, is_speech=True)
        await t._on_audio_chunk(chunk)
        assert len(t._audio_buffer) >= 1

    @pytest.mark.asyncio
    async def test_reset_clears_buffer(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        t._listening = True
        t._audio_buffer = [np.zeros(480, dtype=np.float32)]
        await t._on_reset(None)
        assert not t._listening
        assert len(t._audio_buffer) == 0

    def test_prefix_alignment_matching(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        t._confirmed_prefix = "xin chào"
        result = t._align_prefix("xin chào bạn")
        assert result == "xin chào bạn"

    def test_prefix_alignment_mismatch_resets(self):
        t = StreamingTranscriber(model_path="", device="cpu", session_id="test")
        t._confirmed_prefix = "xin chào"
        result = t._align_prefix("hôm nay trời đẹp")
        assert result == "hôm nay trời đẹp"
        assert t._confirmed_prefix == ""
