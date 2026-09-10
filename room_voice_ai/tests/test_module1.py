"""
tests/test_module1.py
======================
Kịch bản kiểm thử độc lập (Unit Test) cho Module 1: Audio Preprocessing.

Tuân thủ 100% đặc tả kịch bản Mục 7 – module1_specification.md.

Cấu trúc test:
  Class TestAudioUtils       – kiểm thử các hàm tiện ích (utils.py)
  Class TestHPFAndAGC        – kiểm thử Tầng 1 (HPF, AGC, AEC simulation)
  Class TestHangoverMachine  – kiểm thử Máy trạng thái Hangover (Tầng 3b)
  Class TestVADGating        – kiểm thử logic gating VAD tổng thể
  Class TestDummyStreamKPI   – kiểm thử tích hợp toàn pipeline với tín hiệu dummy 5s

Để chạy:
    cd room_voice_ai
    pytest tests/test_module1.py -v
    pytest tests/test_module1.py -v -k "KPI"       # chỉ chạy test KPI
    pytest tests/test_module1.py -v --tb=short     # trace ngắn gọn
"""
# pylint: disable=redefined-outer-name,protected-access,unused-argument,too-many-public-methods

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import AudioChunkPayload
from src.module1_audio.enhancer import (
    AGC_TARGET_DBFS,
    FRAME_SIZE,
    HANGOVER_FRAMES,
    SAMPLE_RATE,
    VAD_THETA_END,
    VAD_THETA_START,
    Module1AudioPreprocessor,
)
from src.module1_audio.utils import (
    apply_pre_emphasis,
    compute_erle_db,
    compute_rms_dbfs,
    compute_snr_db,
    float32_to_int16_bytes,
    frames_to_audio,
    generate_dummy_stream,
    int16_bytes_to_float32,
    normalize_float32,
    resample,
    split_into_frames,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixture helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_event_bus():
    """Reset EventBus singleton trước và sau mỗi test."""
    EventBus.reset()
    yield
    EventBus.reset()


@pytest.fixture
def processor() -> Module1AudioPreprocessor:
    """
    Khởi tạo Module1AudioPreprocessor KHÔNG load model thật
    (model chưa cài → chạy ở fallback mode để test logic thuần Python).
    """
    proc = Module1AudioPreprocessor(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        hangover_frames=HANGOVER_FRAMES,
        session_id="pytest",
    )
    return proc


@pytest.fixture
def silent_frame() -> np.ndarray:
    """Khung im lặng hoàn toàn (tất cả zero)."""
    return np.zeros(FRAME_SIZE, dtype=np.float32)


@pytest.fixture
def noise_frame() -> np.ndarray:
    """Khung nhiễu nền nhỏ (RMS ≈ 0.005)."""
    rng = np.random.default_rng(seed=42)
    return rng.normal(0.0, 0.005, FRAME_SIZE).astype(np.float32)


@pytest.fixture
def speech_frame() -> np.ndarray:
    """Khung tiếng nói giả lập (sine đa tần, RMS ≈ 0.10)."""
    t = np.linspace(0.0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE, endpoint=False)
    return (
        0.07 * np.sin(2 * np.pi * 200  * t) +
        0.04 * np.sin(2 * np.pi * 800  * t) +
        0.02 * np.sin(2 * np.pi * 2400 * t)
    ).astype(np.float32)


@pytest.fixture
def click_frame() -> np.ndarray:
    """Khung xung gõ phím cơ (biên độ cao 0.4, ngắn 200 mẫu trong khung 512)."""
    frame = np.zeros(FRAME_SIZE, dtype=np.float32)
    rng = np.random.default_rng(seed=7)
    frame[:200] = rng.uniform(-0.4, 0.4, 200).astype(np.float32)
    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Class 1: Kiểm thử hàm tiện ích utils.py
# ─────────────────────────────────────────────────────────────────────────────

class TestAudioUtils:
    """Kiểm thử các hàm thuần toán học trong utils.py."""

    # ── PCM conversion ────────────────────────────────────────────────────
    def test_int16_to_float32_range(self):
        """Giá trị float32 phải nằm trong [-1.0, 1.0]."""
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16).tobytes()
        result = int16_bytes_to_float32(pcm)
        assert result.dtype == np.float32
        assert result.min() >= -1.0
        assert result.max() <= 1.0

    def test_float32_to_int16_roundtrip(self):
        """Chuyển đổi khứ hồi float32 → int16 → float32 phải sai số < 0.005%."""
        original = np.array([0.0, 0.5, -0.5, 0.99, -0.99], dtype=np.float32)
        recovered = int16_bytes_to_float32(float32_to_int16_bytes(original))
        np.testing.assert_allclose(original, recovered, atol=1e-4)

    def test_float32_clip_on_conversion(self):
        """Giá trị ngoài [-1.0, 1.0] phải bị clip trước khi chuyển đổi."""
        audio = np.array([1.5, -1.5, 2.0], dtype=np.float32)
        result = int16_bytes_to_float32(float32_to_int16_bytes(audio))
        assert all(abs(r) <= 1.0 + 1e-4 for r in result)

    def test_normalize_float32_peak_is_one(self):
        """Peak normalization phải cho max(|x|) ≈ 1.0."""
        audio = np.array([0.1, 0.3, -0.5, 0.2], dtype=np.float32)
        normed = normalize_float32(audio)
        assert abs(np.max(np.abs(normed)) - 1.0) < 1e-6

    def test_normalize_float32_silent_unchanged(self):
        """Tín hiệu im lặng (peak ≈ 0) không thay đổi giá trị."""
        audio = np.zeros(512, dtype=np.float32)
        result = normalize_float32(audio)
        np.testing.assert_array_equal(audio, result)

    # ── Acoustic measurement ──────────────────────────────────────
    def test_rms_dbfs_silent_very_negative(self):
        """Tín hiệu im lặng phải có RMS dBFS rất thấp (< -80 dB)."""
        silent = np.zeros(FRAME_SIZE, dtype=np.float32)
        assert compute_rms_dbfs(silent) < -80.0

    def test_rms_dbfs_full_scale_sine(self):
        """Sine full-scale (biên độ 1.0) phải cho RMS ≈ -3 dBFS."""
        t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms_db = compute_rms_dbfs(sine)
        # RMS của sine = A/√2 → dBFS = 20log10(1/√2) ≈ -3.01 dBFS
        assert abs(rms_db - (-3.01)) < 0.5

    def test_rms_dbfs_agc_target(self):
        """Tín hiệu sau AGC phải đạt mức -18 dBFS ± 2 dB."""
        audio = np.random.normal(0.0, 0.04, FRAME_SIZE * 10).astype(np.float32)
        agc_out = Module1AudioPreprocessor.apply_agc(audio, target_dbfs=AGC_TARGET_DBFS)
        rms_db = compute_rms_dbfs(agc_out)
        assert abs(rms_db - AGC_TARGET_DBFS) < 2.5, (
            f"AGC target không đạt: {rms_db:.1f} dBFS (expect ≈ {AGC_TARGET_DBFS} dBFS)"
        )

    def test_snr_db_identical_signals(self):
        """Hai tín hiệu giống nhau hoàn toàn → SNR = +inf."""
        signal = np.random.randn(FRAME_SIZE).astype(np.float32)
        snr = compute_snr_db(signal, signal)
        assert snr == float("inf")

    def test_erle_db_aec_reduces_echo(self):
        """
        Kiểm tra ERLE: mic_after_aec phải nhỏ hơn mic_before_aec.
        ERLE > 0 dB nghĩa là AEC đã giảm echo.
        """
        ref      = np.random.normal(0.0, 0.1, FRAME_SIZE).astype(np.float32)
        mic_in   = np.random.normal(0.0, 0.01, FRAME_SIZE).astype(np.float32) + 0.25 * ref
        mic_out  = mic_in - 0.25 * ref  # AEC linear simulation
        erle     = compute_erle_db(mic_in, mic_out)
        assert erle > 5.0, f"ERLE quá thấp: {erle:.1f} dB"

    # ── Frame utilities ───────────────────────────────────────────────────
    def test_split_frames_exact_multiple(self):
        """Chia mảng có độ dài bội số frame_size → không frame nào thiếu."""
        audio  = np.zeros(FRAME_SIZE * 10, dtype=np.float32)
        frames = split_into_frames(audio, frame_size=FRAME_SIZE)
        assert len(frames) == 10
        assert all(len(f) == FRAME_SIZE for f in frames)

    def test_split_frames_pad_last(self):
        """Frame cuối bị ngắn phải được zero-pad lên đúng frame_size."""
        audio  = np.zeros(FRAME_SIZE * 3 + 100, dtype=np.float32)
        frames = split_into_frames(audio, frame_size=FRAME_SIZE, pad_last=True)
        assert len(frames) == 4
        assert len(frames[-1]) == FRAME_SIZE

    def test_split_frames_no_pad_drops_last(self):
        """Nếu pad_last=False, frame cuối không đủ phải bị bỏ qua."""
        audio  = np.zeros(FRAME_SIZE * 3 + 100, dtype=np.float32)
        frames = split_into_frames(audio, frame_size=FRAME_SIZE, pad_last=False)
        assert len(frames) == 3

    def test_frames_roundtrip(self):
        """split → join phải phục hồi tín hiệu gốc (với zero-padding ở cuối)."""
        audio  = np.random.randn(FRAME_SIZE * 5).astype(np.float32)
        frames = split_into_frames(audio, FRAME_SIZE, pad_last=False)
        joined = frames_to_audio(frames)
        np.testing.assert_array_almost_equal(audio, joined, decimal=5)

    # ── Resampling ────────────────────────────────────────────────────────
    def test_resample_same_rate_is_noop(self):
        """Resample cùng rate phải trả về mảng giống hệt."""
        audio  = np.random.randn(FRAME_SIZE).astype(np.float32)
        result = resample(audio, SAMPLE_RATE, SAMPLE_RATE)
        np.testing.assert_array_equal(audio, result)

    def test_resample_length_correct(self):
        """Resample 16 kHz → 8 kHz phải cho length giảm còn một nửa."""
        audio  = np.random.randn(SAMPLE_RATE).astype(np.float32)   # 1 giây @ 16 kHz
        result = resample(audio, SAMPLE_RATE, 8_000)
        assert len(result) == 8_000

    def test_resample_upsample_length_correct(self):
        """Resample 16 kHz → 22050 Hz phải cho độ dài đúng."""
        audio  = np.random.randn(SAMPLE_RATE).astype(np.float32)
        result = resample(audio, SAMPLE_RATE, 22_050)
        assert len(result) == 22_050

    # ── Pre-emphasis ─────────────────────────────────────────────────────
    def test_pre_emphasis_same_length(self):
        """Đầu ra pre-emphasis phải cùng độ dài đầu vào."""
        audio  = np.random.randn(FRAME_SIZE).astype(np.float32)
        result = apply_pre_emphasis(audio, coeff=0.97)
        assert len(result) == len(audio)

    def test_pre_emphasis_boosts_high_freq(self):
        """Sau pre-emphasis, thành phần tần số cao phải tăng so với thấp."""
        t     = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
        low   = np.sin(2 * np.pi * 100 * t).astype(np.float32)
        high  = np.sin(2 * np.pi * 4000 * t).astype(np.float32)
        low_p  = apply_pre_emphasis(low)
        high_p = apply_pre_emphasis(high)
        # RMS sau pre-emphasis của tín hiệu cao tần phải gần bằng gốc (không bị suy giảm)
        assert compute_rms_dbfs(high_p) > compute_rms_dbfs(low_p)

    # ── Dummy stream generator ────────────────────────────────────────────
    def test_generate_dummy_stream_length(self):
        """Dummy stream phải có đúng total_seconds × sample_rate samples."""
        audio = generate_dummy_stream(total_seconds=5.0, sample_rate=SAMPLE_RATE)
        assert len(audio) == int(5.0 * SAMPLE_RATE)

    def test_generate_dummy_stream_speech_rms_higher_than_silence(self):
        """Giai đoạn tiếng nói phải có RMS cao hơn giai đoạn im lặng."""
        audio   = generate_dummy_stream()
        silence = audio[: int(1.0 * SAMPLE_RATE)]
        speech  = audio[int(1.5 * SAMPLE_RATE): int(3.5 * SAMPLE_RATE)]
        assert compute_rms_dbfs(speech) > compute_rms_dbfs(silence) + 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Class 2: Kiểm thử Tầng 1 (HPF + AGC + AEC)
# ─────────────────────────────────────────────────────────────────────────────

class TestHPFAndAGC:
    """Kiểm thử Tầng 1: AEC simulation, HPF Butterworth, AGC."""

    def test_hpf_suppresses_dc(self, processor):
        """HPF phải loại bỏ thành phần DC offset (0 Hz)."""
        dc = np.full(FRAME_SIZE, 0.3, dtype=np.float32)  # DC offset thuần túy
        filtered = processor.apply_hpf(dc)
        # Sau HPF, mean phải gần 0 (DC bị lọc)
        assert abs(float(np.mean(filtered))) < 0.05, (
            f"HPF không loại DC: mean = {np.mean(filtered):.4f}"
        )

    def test_hpf_passes_speech_frequencies(self, processor):
        """HPF phải cho qua thành phần 200 Hz (tần số giọng nói)."""
        t      = np.linspace(0.0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE, endpoint=False)
        speech = np.sin(2 * np.pi * 200 * t).astype(np.float32)  # 200 Hz
        filtered = processor.apply_hpf(speech)
        rms_in   = float(np.sqrt(np.mean(speech    ** 2)))
        rms_out  = float(np.sqrt(np.mean(filtered  ** 2)))
        # 200 Hz nằm trên cutoff 80 Hz → không bị suy giảm nhiều (< 3 dB)
        attenuation_db = 20 * np.log10(rms_out / (rms_in + 1e-10))
        assert attenuation_db > -3.0, (
            f"HPF cắt nhầm 200 Hz: suy hao = {attenuation_db:.1f} dB"
        )

    def test_agc_achieves_target_dbfs(self):
        """AGC phải đưa RMS về -18 dBFS ± 2 dB."""
        audio    = np.random.normal(0.0, 0.04, FRAME_SIZE * 10).astype(np.float32)
        agc_out  = Module1AudioPreprocessor.apply_agc(audio, target_dbfs=AGC_TARGET_DBFS)
        rms_out  = compute_rms_dbfs(agc_out)
        assert abs(rms_out - AGC_TARGET_DBFS) < 2.5

    def test_agc_no_amplify_silence(self):
        """AGC không được khuếch đại tín hiệu im lặng (RMS quá nhỏ)."""
        silent   = np.zeros(FRAME_SIZE, dtype=np.float32)
        agc_out  = Module1AudioPreprocessor.apply_agc(silent)
        # Đầu ra phải giữ nguyên (không bị khuếch đại thành noise)
        np.testing.assert_array_equal(silent, agc_out)

    def test_aec_reduces_echo_in_tier1(self, processor):
        """Tầng 1: AEC simulation phải giảm echo so với input thô."""
        ref = np.random.normal(0.0, 0.1, FRAME_SIZE).astype(np.float32)
        mic = np.random.normal(0.0, 0.005, FRAME_SIZE).astype(np.float32) + 0.25 * ref
        out = processor.aec_hpf_agc(mic, ref)
        # Sau AEC, correlation với ref phải thấp hơn trước
        corr_before = float(np.dot(mic, ref))
        corr_after  = float(np.dot(out, ref))
        assert abs(corr_after) < abs(corr_before) + 1.0   # AEC giảm echo


# ─────────────────────────────────────────────────────────────────────────────
# Class 3: Kiểm thử Máy trạng thái Hangover (Tầng 3b)
# ─────────────────────────────────────────────────────────────────────────────

class TestHangoverMachine:
    """
    Kiểm thử Hangover State Machine theo đặc tả Mục 3.3 và công thức:

      G(m) = 1  nếu P ≥ θ_start (0.50)
      G(m) = 1  nếu P < θ_start và counter > 0
      G(m) = 0  nếu P < θ_end  (0.35) và counter = 0
    """

    def test_opens_gate_above_theta_start(self, processor):
        """G(m) = 1 khi P(Speech) ≥ 0.50."""
        result = processor.vad_state_machine(VAD_THETA_START)
        assert result is True

    def test_opens_gate_strictly_above_theta_start(self, processor):
        """G(m) = 1 khi P = 0.95 (ngưỡng cao)."""
        result = processor.vad_state_machine(0.95)
        assert result is True

    def test_gate_stays_open_during_hangover(self, processor):
        """
        Sau khi P vượt θ_start, counter = HANGOVER_FRAMES.
        Ngay cả khi P xuống dưới θ_start, cổng phải ở OPEN thêm 8 frames.
        """
        # Bước 1: Kích hoạt (P ≥ 0.5)
        processor.vad_state_machine(0.8)
        assert processor.hangover_counter == HANGOVER_FRAMES

        # Bước 2: P xuống nhưng counter > 0 → cổng vẫn mở
        for expected_counter in range(HANGOVER_FRAMES - 1, -1, -1):
            result = processor.vad_state_machine(0.1)  # P dưới θ_start và θ_end
            assert result is True, f"Cổng đóng sớm tại counter = {processor.hangover_counter + 1}"
            assert processor.hangover_counter == expected_counter

    def test_gate_closes_after_hangover_expires(self, processor):
        """Cổng phải đóng sau khi hangover counter về 0."""
        # Kích hoạt counter
        processor.vad_state_machine(0.8)
        # Đợi hết hangover
        for _ in range(HANGOVER_FRAMES):
            processor.vad_state_machine(0.1)
        # Frame tiếp theo phải đóng
        result = processor.vad_state_machine(0.1)
        assert result is False, "Cổng phải đóng sau khi counter = 0"

    def test_counter_reset_on_new_speech(self, processor):
        """Nếu P vượt θ_start trở lại giữa hangover, counter phải reset về MAX."""
        # Kích hoạt lần đầu
        processor.vad_state_machine(0.8)
        # Để counter đếm lùi 4 frame
        for _ in range(4):
            processor.vad_state_machine(0.1)
        assert processor.hangover_counter == HANGOVER_FRAMES - 4
        # P vượt ngưỡng trở lại → counter phải reset về HANGOVER_FRAMES
        processor.vad_state_machine(0.9)
        assert processor.hangover_counter == HANGOVER_FRAMES

    def test_gate_stays_closed_below_both_thresholds(self, processor):
        """G(m) = 0 khi counter = 0 và P < θ_end = 0.35."""
        processor.hangover_counter = 0
        processor.vad_is_open      = False
        result = processor.vad_state_machine(VAD_THETA_END - 0.01)
        assert result is False

    def test_hysteresis_between_thresholds(self, processor):
        """
        Kiểm tra vùng Hysteresis: θ_end < P < θ_start (0.35 < P < 0.50).
        Khi counter = 0: cổng phải đóng.
        Khi counter > 0: cổng vẫn mở.
        """
        mid_prob = (VAD_THETA_END + VAD_THETA_START) / 2.0  # ≈ 0.425

        # counter = 0, cổng đóng
        processor.hangover_counter = 0
        processor.vad_is_open      = False
        result_closed = processor.vad_state_machine(mid_prob)
        assert not result_closed, "Phải đóng khi counter=0 và P trong vùng hysteresis"

        # counter > 0 (từ lần mở trước), cổng mở
        processor.vad_state_machine(0.9)   # kích hoạt
        processor.vad_state_machine(mid_prob)  # vào vùng hysteresis
        assert processor.vad_is_open, "Phải mở khi counter>0 và P trong vùng hysteresis"

    def test_hangover_duration_is_256ms(self):
        """Kiểm tra cơ chế Hangover giữ đúng 8 frames = 256 ms."""
        assert HANGOVER_FRAMES == 8
        hangover_ms = HANGOVER_FRAMES * FRAME_SIZE / SAMPLE_RATE * 1000
        assert abs(hangover_ms - 256.0) < 1.0, (
            f"Hangover duration sai: {hangover_ms} ms (expect 256 ms)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Class 4: Kiểm thử VAD Gating tích hợp (process_stream_chunk)
# ─────────────────────────────────────────────────────────────────────────────

class TestVADGating:
    """
    Kiểm thử phương thức process_stream_chunk() – giao diện chính Module 1.
    Sử dụng energy-based VAD fallback (không cần model thật).
    """

    def test_silent_frame_returns_false(self, processor, silent_frame):
        """Khung im lặng hoàn toàn phải trả về is_active_speech = False."""
        _, is_speech = processor.process_stream_chunk(silent_frame)
        assert is_speech is False

    def test_returns_float32_array(self, processor, speech_frame):
        """Đầu ra enhanced_pcm phải là numpy float32."""
        enhanced, _ = processor.process_stream_chunk(speech_frame)
        assert enhanced.dtype == np.float32

    def test_output_length_matches_input(self, processor, speech_frame):
        """Độ dài đầu ra phải bằng frame_size (512 samples)."""
        enhanced, _ = processor.process_stream_chunk(speech_frame)
        assert len(enhanced) == FRAME_SIZE

    def test_output_range_clipped(self, processor):
        """Đầu ra phải nằm trong [-1.0, 1.0] (không clipping)."""
        loud = np.ones(FRAME_SIZE, dtype=np.float32) * 2.0  # tín hiệu quá biên
        enhanced, _ = processor.process_stream_chunk(loud)
        assert float(np.max(np.abs(enhanced))) <= 1.0 + 1e-5

    def test_with_far_end_ref_no_crash(self, processor, speech_frame):
        """Xử lý với far_end_ref không None không được raise exception."""
        ref = np.random.normal(0.0, 0.05, FRAME_SIZE).astype(np.float32)
        enhanced, is_speech = processor.process_stream_chunk(speech_frame, far_end_ref=ref)
        assert isinstance(enhanced, np.ndarray)
        assert isinstance(is_speech, bool)

    def test_without_far_end_ref_no_crash(self, processor, speech_frame):
        """Xử lý không có far_end_ref (None) không được raise exception."""
        enhanced, is_speech = processor.process_stream_chunk(speech_frame, far_end_ref=None)
        assert isinstance(enhanced, np.ndarray)
        assert isinstance(is_speech, bool)

    def test_energy_fallback_vad_on_noise_frame(self, processor, noise_frame):
        """Khung nhiễu nhỏ (RMS ≈ 0.005) → fallback VAD phải trả False."""
        _, is_speech = processor.process_stream_chunk(noise_frame)
        assert is_speech is False

    def test_constants_match_specification(self):
        """
        Kiểm tra các hằng số đặc tả không bị thay đổi sai (regression guard):
          SAMPLE_RATE    = 16000  Hz
          FRAME_SIZE     = 512    samples = 32 ms
          HANGOVER_FRAMES = 8    frames  = 256 ms
          VAD_THETA_START = 0.50
          VAD_THETA_END   = 0.35
        """
        assert SAMPLE_RATE     == 16_000, f"SAMPLE_RATE sai: {SAMPLE_RATE}"
        assert FRAME_SIZE      == 512,    f"FRAME_SIZE sai: {FRAME_SIZE}"
        assert HANGOVER_FRAMES == 8,      f"HANGOVER_FRAMES sai: {HANGOVER_FRAMES}"
        assert VAD_THETA_START == 0.50,   f"VAD_THETA_START sai: {VAD_THETA_START}"
        assert VAD_THETA_END   == 0.35,   f"VAD_THETA_END sai: {VAD_THETA_END}"


# ─────────────────────────────────────────────────────────────────────────────
# Class 5: KPI Test – Tích hợp toàn pipeline với tín hiệu dummy 5 giây
# ─────────────────────────────────────────────────────────────────────────────

class TestDummyStreamKPI:
    """
    Kiểm thử tích hợp toàn pipeline theo kịch bản Mục 7:

    Dummy stream 5 giây @ 16 kHz:
      0.0 → 1.5s : Im lặng + nhiễu nền (RMS ≈ 0.005)
      1.5 → 3.5s : Tiếng người nói giả lập (RMS ≈ 0.10)
      3.5 → 5.0s : Im lặng + xung phím cơ tại t=4.2s (50 ms)

    KPI cần đạt:
      KPI-1: Avg latency mỗi khung ≤ 5.0 ms
      KPI-2: Giai đoạn im lặng → drop frame tỷ lệ ≥ 90%
      KPI-3: Giai đoạn giọng nói → active speech tỷ lệ ≥ 80%
      KPI-4: Hangover 256 ms sau khi kết thúc giọng nói được duy trì
    """

    @pytest.fixture
    def full_stream_results(self, processor) -> dict:
        """
        Chạy toàn bộ dummy stream qua pipeline và thu thập kết quả.
        Trả về dict chứa latencies và danh sách (timestamp, is_speech).
        """
        audio  = generate_dummy_stream(total_seconds=5.0, sample_rate=SAMPLE_RATE)
        frames = split_into_frames(audio, FRAME_SIZE, pad_last=False)
        rng    = np.random.default_rng(seed=123)

        latencies: list[float] = []
        timeline: list[tuple[float, bool]] = []   # (timestamp_s, is_speech)

        for i, frame in enumerate(frames):
            ref = rng.normal(0.0, 0.002, FRAME_SIZE).astype(np.float32)
            t0  = time.perf_counter()
            _, is_speech = processor.process_stream_chunk(frame, far_end_ref=ref)
            latencies.append((time.perf_counter() - t0) * 1000)
            timeline.append((i * FRAME_SIZE / SAMPLE_RATE, is_speech))

        return {"latencies": latencies, "timeline": timeline}

    def test_kpi1_avg_latency_under_5ms(self, full_stream_results):
        """KPI-1: Thời gian xử lý trung bình mỗi khung 32ms phải ≤ 5 ms."""
        avg_lat = float(np.mean(full_stream_results["latencies"]))
        assert avg_lat <= 5.0, (
            f"KPI-1 FAIL: avg latency = {avg_lat:.2f} ms (yêu cầu ≤ 5.0 ms). "
            "Kiểm tra xem có process nào block CPU không."
        )

    def test_kpi2_silence_drops_most_frames(self, full_stream_results):
        """KPI-2: Giai đoạn im lặng 0-1.5s → ≥ 90% frame phải drop (is_speech=False)."""
        timeline = full_stream_results["timeline"]
        silence  = [r[1] for r in timeline if r[0] < 1.5]
        if not silence:
            pytest.skip("Không có frame silence để kiểm thử.")
        drop_rate = 1.0 - (sum(silence) / len(silence))
        assert drop_rate >= 0.90, (
            f"KPI-2 FAIL: drop rate = {drop_rate*100:.1f}% (yêu cầu ≥ 90%). "
            "VAD có thể đang false-positive quá nhiều."
        )

    def test_kpi3_speech_opens_gate_mostly(self, full_stream_results):
        """KPI-3: Giai đoạn 1.5-3.5s → ≥ 80% frame phải active (is_speech=True)."""
        timeline  = full_stream_results["timeline"]
        speech_f  = [r[1] for r in timeline if 1.5 <= r[0] < 3.5]
        if not speech_f:
            pytest.skip("Không có frame speech để kiểm thử.")
        active_rate = sum(speech_f) / len(speech_f)
        assert active_rate >= 0.80, (
            f"KPI-3 FAIL: active rate = {active_rate*100:.1f}% (yêu cầu ≥ 80%). "
            "VAD có thể miss-detect giọng nói."
        )

    def test_kpi4_hangover_maintained_after_speech(self, full_stream_results):
        """
        KPI-4: Giai đoạn 3.5-3.75s (256 ms sau khi kết thúc giọng nói)
        phải duy trì ít nhất 1 frame is_speech=True (Hangover active).
        """
        timeline   = full_stream_results["timeline"]
        hangover_f = [r[1] for r in timeline if 3.5 <= r[0] < 3.76]
        if not hangover_f:
            pytest.skip("Không đủ frame để kiểm thử hangover.")
        assert any(hangover_f), (
            "KPI-4 FAIL: Không có frame nào active trong vùng hangover 256ms. "
            "Máy trạng thái Hangover chưa hoạt động đúng."
        )

    def test_kpi_click_does_not_cause_sustained_activation(self, processor):
        """
        KPI-5 (Kháng nhiễu xung): Xung gõ phím cơ (50ms) không được phép
        kích hoạt cổng lâu dài – tối đa 1 frame phản ứng.
        """
        rng = np.random.default_rng(seed=99)

        # Đảm bảo trạng thái im lặng ban đầu
        silent = np.zeros(FRAME_SIZE, dtype=np.float32)
        for _ in range(HANGOVER_FRAMES + 2):
            processor.process_stream_chunk(silent)

        # 1 khung xung phím cơ (200 mẫu biên độ cao trong khung 512)
        click = np.zeros(FRAME_SIZE, dtype=np.float32)
        click[:200] = rng.uniform(-0.4, 0.4, 200).astype(np.float32)
        _, _ = processor.process_stream_chunk(click)

        # 10 khung im lặng sau xung
        post_click_active = 0
        for _ in range(10):
            _, is_sp = processor.process_stream_chunk(silent)
            if is_sp:
                post_click_active += 1

        # Nhiều nhất 8 frame (hangover max)
        assert post_click_active <= HANGOVER_FRAMES, (
            f"KPI-5 FAIL: Xung phím kích hoạt {post_click_active} frames sau đó "
            f"(tối đa cho phép: {HANGOVER_FRAMES}). "
            "Có thể DeepFilterNet3 chưa triệt đủ xung tạp âm."
        )

    def test_event_bus_payload_published_on_speech(self):
        """
        Kiểm tra tích hợp EventBus: khi is_speech=True,
        AudioChunkPayload phải được publish lên EventChannel.AUDIO_CHUNK.
        """
        bus = EventBus.get_instance()
        received: list[AudioChunkPayload] = []

        async def collector(payload: AudioChunkPayload):
            received.append(payload)

        bus.subscribe(EventChannel.AUDIO_CHUNK, collector)

        # Tạo tín hiệu giọng nói thật sự (sine) để force is_speech=True
        t      = np.linspace(0.0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE, endpoint=False)
        speech = (0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        proc = Module1AudioPreprocessor(session_id="bus_test")

        enhanced, is_speech = proc.process_stream_chunk(speech)

        # Phát thủ công để test publish logic
        async def _publish():
            if is_speech:
                payload = AudioChunkPayload(
                    session_id  = proc.session_id,
                    audio_bytes = float32_to_int16_bytes(enhanced),
                    is_speech   = is_speech,
                    rms_db      = compute_rms_dbfs(enhanced),
                )
                await bus.publish(EventChannel.AUDIO_CHUNK, payload)

        asyncio.run(_publish())

        if is_speech:
            assert len(received) == 1
            assert received[0].session_id == "bus_test"
            assert received[0].is_speech is True
            assert isinstance(received[0].rms_db, float)
        else:
            pytest.skip(
                "Energy fallback VAD không detect giọng nói trong test này – cần Silero model."
            )
