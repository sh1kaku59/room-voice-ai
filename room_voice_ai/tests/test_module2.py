"""
tests/test_module2.py
======================
Kịch bản kiểm thử độc lập (Unit Test) cho Module 2: Wake-word Detection (Call Name).

Tuân thủ 100% kịch bản 5 Test Cases chuyên sâu tại Mục 7 – module2_specification.md:
  1. test_initialization: Kiểm thử Khởi tạo & Cấu hình bộ đệm trượt 1.5s.
  2. test_circular_buffer_fifo: Kiểm thử Cơ chế trượt FIFO của RingBuffer.
  3. test_temporal_smoothing_spike_rejection: Kiểm thử Kháng gai nhiễu đột biến (Spike rejection).
  4. test_hysteresis_and_cooldown_lockout: Khóa thời gian trơ 2.0s chống kích đúp.
  5. test_end_to_end_dummy_stream_performance: Kiểm thử Hiệu năng luồng & Giới hạn CPU (<= 3.5ms).
  6. test_event_bus_integration: Kiểm thử Tích hợp EventBus & Payloads.

Để chạy:
    cd room_voice_ai
    pytest tests/test_module2.py -v
"""

# pylint: disable=redefined-outer-name,protected-access,unused-argument,duplicate-code

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import AudioChunkPayload, WakeWordPayload
from src.module2_wakeword.detector import (
    BUFFER_SIZE,
    COOLDOWN_FRAMES,
    FRAME_SIZE,
    SAMPLE_RATE,
    WAKEWORD_THRESHOLD,
    WINDOW_DURATION,
    Module2WakeWordDetector,
    WakeWordDetector,
)


@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus singleton trước và sau mỗi test case."""
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def detector() -> Module2WakeWordDetector:
    """Tạo instance Module2WakeWordDetector cho test."""
    det = Module2WakeWordDetector(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        window_duration=WINDOW_DURATION,
        threshold=WAKEWORD_THRESHOLD,
        cooldown_seconds=2.0,
        session_id="pytest_mod2",
    )
    return det


class TestWakeWordDetector:
    """Bộ kiểm thử 5 test cases chuẩn hóa theo Mục 7 đặc tả Module 2."""

    # ── Test Case 1: test_initialization ─────────────────────────────────
    def test_initialization(self, detector: Module2WakeWordDetector):
        """
        [Test Case 1] Kiểm thử Khởi tạo & Cấu hình bộ đệm:
        - Nạp detector thành công.
        - Kích thước audio_buffer đạt chính xác 24,000 samples (1.5s @ 16kHz).
        - Cooldown counter ban đầu bằng 0.
        - Ngưỡng threshold mặc định bằng 0.75.
        """
        assert len(detector.audio_buffer) == BUFFER_SIZE
        assert len(detector.audio_buffer) == 24_000
        assert detector.cooldown_counter == 0
        assert detector.threshold == 0.75
        assert detector.cooldown_frames_limit == COOLDOWN_FRAMES
        assert detector.cooldown_frames_limit == 62
        assert detector.smoothing_window_size >= 1

    # ── Test Case 2: test_circular_buffer_fifo ───────────────────────────
    def test_circular_buffer_fifo(self, detector: Module2WakeWordDetector):
        """
        [Test Case 2] Kiểm thử Cơ chế trượt FIFO của RingBuffer:
        - Ban đầu toàn bộ buffer chứa 0.0.
        - Đẩy 512 mẫu giá trị 1.0 vào.
        - 512 phần tử cuối cùng phải bằng chính xác 1.0, các phần tử đầu vẫn là 0.0.
        """
        # Đảm bảo ban đầu toàn 0.0
        detector.reset_audio_buffer()
        assert all(s == 0.0 for s in detector.audio_buffer)

        # Đẩy khung 512 samples chứa toàn số 1.0
        ones_frame = np.ones(FRAME_SIZE, dtype=np.float32)
        detector.process_speech_frame(ones_frame)

        buffer_list = list(detector.audio_buffer)
        assert len(buffer_list) == BUFFER_SIZE

        # 512 mẫu cuối cùng phải là 1.0
        assert all(s == 1.0 for s in buffer_list[-FRAME_SIZE:])
        # 512 mẫu đầu tiên vẫn phải là 0.0
        assert all(s == 0.0 for s in buffer_list[:FRAME_SIZE])

    # ── Test Case 3: test_temporal_smoothing_spike_rejection ─────────────
    def test_temporal_smoothing_spike_rejection(self, detector: Module2WakeWordDetector):
        """
        [Test Case 3] Kiểm thử Kháng gai nhiễu đột biến (Spike rejection):
        - Giả lập 100 frames có xác suất thấp 0.05.
        - Duy nhất frame thứ 50 xuất hiện gai xác suất đột biến 1.0.
        - Bộ lọc trung bình trượt phải san phẳng gai này xuống dưới ngưỡng 0.75.
        - Toàn bộ 100 frames đều trả về triggered == False.
        """
        triggered_results: list[bool] = []
        scores: list[float] = []

        # Mock phương thức _run_onnx_inference
        # Frame 50 -> 1.0, các frame khác -> 0.05
        frame_idx = 0

        def mock_inference(_feature_tensor: Any, _audio_array: Any) -> float:
            nonlocal frame_idx
            prob = 1.0 if frame_idx == 50 else 0.05
            frame_idx += 1
            return prob

        detector._run_onnx_inference = mock_inference  # type: ignore

        dummy_frame = np.random.uniform(-0.01, 0.01, FRAME_SIZE).astype(np.float32)
        for _ in range(100):
            trig, sc = detector.process_speech_frame(dummy_frame)
            triggered_results.append(trig)
            scores.append(sc)

        # Gai xác suất 1.0 tại frame 50 bị chia trung bình cho smoothing_window_size
        # -> điểm số làm mịn = (0.05 * (N-1) + 1.0) / N < 0.75
        assert not any(triggered_results), (
            f"Spike bị kích hoạt nhầm! Max score = {max(scores):.4f}"
        )

    # ── Test Case 4: test_hysteresis_and_cooldown_lockout ─────────────────
    def test_hysteresis_and_cooldown_lockout(self, detector: Module2WakeWordDetector):
        """
        [Test Case 4] Kiểm thử Kích hoạt & Khóa thời gian trơ chống kích đúp:
        - Giả lập chuỗi xác suất cao 0.95 liên tiếp.
        - Frame đầu tiên đạt ngưỡng làm mịn phải trả về triggered == True.
        - Frame ngay sau đó phải trả về triggered == False do cooldown lockout.
        - Cooldown counter phải được thiết lập đúng 62 frames (2.0s).
        """
        # Mock phương thức inference luôn trả về 0.95
        detector._run_onnx_inference = MagicMock(return_value=0.95)  # type: ignore

        dummy_frame = np.random.uniform(-0.01, 0.01, FRAME_SIZE).astype(np.float32)

        # Chạy cho đến khi cửa sổ làm mịn tích lũy đủ để vượt ngưỡng
        activated_frame = -1
        for f in range(20):
            trig, _ = detector.process_speech_frame(dummy_frame)
            if trig:
                activated_frame = f
                break

        assert activated_frame >= 0, "Không kích hoạt được với chuỗi xác suất 0.95!"

        # Kiểm tra trạng thái ngay tại thời điểm kích hoạt
        assert detector.cooldown_counter == detector.cooldown_frames_limit
        assert detector.cooldown_counter == 62

        # Frame tiếp theo ngay sau đó PHẢI là False mặc dù inference vẫn là 0.95
        trig_next, score_next = detector.process_speech_frame(dummy_frame)
        assert trig_next is False, "Frame tiếp theo trong cooldown không được phép kích hoạt!"
        assert score_next == 0.0, "Score trong thời gian cooldown phải là 0.0"
        assert detector.cooldown_counter == detector.cooldown_frames_limit - 1

    # ── Test Case 5: test_end_to_end_dummy_stream_performance ────────────
    def test_end_to_end_dummy_stream_performance(self, detector: Module2WakeWordDetector):
        """
        [Test Case 5] Kiểm thử Hiệu năng luồng & Giới hạn CPU:
        - Chạy stream 5 giây (150 frames 32ms).
        - Đo đạc thời gian suy luận từng khung hình.
        - Yêu cầu KPI: mean_latency <= 3.5 ms trên CPU.
        """
        latencies: list[float] = []
        dummy_frame = np.random.uniform(-0.02, 0.02, FRAME_SIZE).astype(np.float32)

        for _ in range(150):
            t0 = time.perf_counter()
            detector.process_speech_frame(dummy_frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

        mean_latency = float(np.mean(latencies))
        assert mean_latency <= 3.5, (
            f"KPI FAIL: Độ trễ trung bình {mean_latency:.3f} ms vượt quá ngưỡng 3.5 ms!"
        )

    # ── Test Case 6: test_event_bus_integration ──────────────────────────
    @pytest.mark.asyncio
    async def test_event_bus_integration(self, detector: Module2WakeWordDetector):
        """
        Kiểm thử Tích hợp EventBus:
        - Nhận AudioChunkPayload (is_speech=True).
        - Khi kích hoạt thành công, phát WakeWordPayload lên EventChannel.WAKE_WORD.
        """
        bus = EventBus.get_instance()
        received_payloads: list[WakeWordPayload] = []

        async def on_wake(p: WakeWordPayload):
            received_payloads.append(p)

        bus.subscribe(EventChannel.WAKE_WORD, on_wake)

        # Mock inference trả về xác suất cao 0.99
        detector._run_onnx_inference = MagicMock(return_value=0.99)  # type: ignore

        # Gửi liên tiếp các AudioChunkPayload để làm đầy cửa sổ làm mịn
        pcm_bytes = (np.ones(FRAME_SIZE, dtype=np.int16) * 1000).tobytes()
        chunk = AudioChunkPayload(
            session_id="bus_test_sess",
            audio_bytes=pcm_bytes,
            is_speech=True,
            rms_db=-18.0,
        )

        for _ in range(detector.smoothing_window_size + 1):
            await detector._on_audio_chunk(chunk)

        assert len(received_payloads) == 1
        assert received_payloads[0].session_id == "pytest_mod2"
        assert received_payloads[0].keyword == "AI Star Light"
        assert received_payloads[0].confidence >= 0.75

    def test_alias_compatibility(self):
        """Kiểm tra alias WakeWordDetector = Module2WakeWordDetector."""
        assert WakeWordDetector is Module2WakeWordDetector
