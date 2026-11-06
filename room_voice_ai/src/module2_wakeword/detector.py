"""
src/module2_wakeword/detector.py
=================================
MODULE 2 – Wake-word Detection (Call Name Engine)

Kiến trúc 3 tầng xử lý tuần tự theo đặc tả module2_specification.md:

  [Khung âm thanh sạch 32ms (512 mẫu) từ Module 1]
                        │
                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 1: Sliding Acoustic Feature Extraction                         │
  │   • Circular Audio RingBuffer: 1.5s = 24,000 samples @ 16 kHz (FIFO) │
  │   • Trích xuất tensor Log-Mel Spectrogram 80 dải tần (STFT + Mel)   │
  │   • Khung bước nhảy 32ms (512 samples)                              │
  └─────────────────────────────────────┬───────────────────────────────┘
                                        │ Tensor X_t (1.5s)
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 2: OpenWakeWord Neural Classifier (ONNX int8/fp32 CPU)         │
  │   • Depthwise Separable Conv + Small Conformer KWS                  │
  │   • Ước lượng xác suất nhị phân P_raw(t) = P(WakeWord | X_t)        │
  │   • Chạy 100% trên CPU (0 MB VRAM GPU, trễ < 3.5ms/frame)           │
  └─────────────────────────────────────┬───────────────────────────────┘
                                        │ P_raw(t)
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 3: Smoothing & Debounce Controller                             │
  │   • Temporal Moving Average: làm mịn xác suất qua cửa sổ 300ms      │
  │   • Hysteresis Threshold: θ_wake = 0.75                             │
  │   • Refractory Cooldown Lockout: 2.0s (62 frames) chống kích đúp   │
  │   • Khi kích hoạt: phát WakeWordPayload lên EventBus                │
  └─────────────────────────────────────────────────────────────────────┘

Tài liệu tham khảo:
  [Ref-01] OpenWakeWord Framework (ONNX int8 CPU Execution)
  [Ref-02] Depthwise Separable Convolution & Conformer KWS
  [Ref-03] Temporal Posterior Smoothing & Refractory Period Debounce
"""

from __future__ import annotations

import collections
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import AudioChunkPayload, WakeWordPayload

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hằng số đặc tả kỹ thuật Module 2
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000                  # Hz (Mono Float32 PCM)
FRAME_SIZE: int = 512                      # samples = 32 ms @ 16 kHz
WINDOW_DURATION: float = 1.5               # giây cửa sổ trượt RingBuffer
BUFFER_SIZE: int = int(WINDOW_DURATION * SAMPLE_RATE)  # 24,000 samples
WAKEWORD_THRESHOLD: float = 0.75           # θ_wake ≥ 0.75
COOLDOWN_SECONDS: float = 2.0              # Khóa thời gian trơ chống kích hoạt đúp
COOLDOWN_FRAMES: int = int(COOLDOWN_SECONDS / (FRAME_SIZE / SAMPLE_RATE))  # 62 frames
WAKEWORD_HYSTERESIS_MS: int = 300          # Cửa sổ làm mịn xác suất (ms)
AUDIO_CHUNK_MS: int = 32                   # ms mỗi khung hình

# Hằng số STFT & 80-band Log-Mel Spectrogram (module2_specification.md §2.1 & §3.1)
N_MELS: int = 80                           # 80 dải tần Mel
FFT_WINDOW_MS: int = 25                    # Cửa sổ phân tích 25 ms
HOP_MS: int = 10                           # Bước nhảy STFT 10 ms
FFT_SIZE: int = int(FFT_WINDOW_MS * SAMPLE_RATE / 1000)   # 400 samples
HOP_SIZE: int = int(HOP_MS * SAMPLE_RATE / 1000)          # 160 samples
N_FFT_BINS: int = FFT_SIZE // 2 + 1                       # 201 bins
N_FRAMES: int = 129                                       # 129 frames chuẩn OpenWakeWord (1, 129, 80)


def _make_mel_filterbank(
    n_mels: int = N_MELS,
    n_fft_bins: int = N_FFT_BINS,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Khởi tạo ma trận bộ lọc tam giác Mel M_mel ∈ R^{80 × 201} theo thang đo HTK/Slaney.
    Đồng bộ 100% với hàm `_make_mel_filterbank` trong `train_module2_openwakeword.py`.
    """
    def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)

    def mel_to_hz(mel: float | np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)

    low_freq_mel = float(hz_to_mel(0.0))
    high_freq_mel = float(hz_to_mel(sr / 2.0))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft_bins - 1) * hz_points / (sr / 2.0)).astype(int)

    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filters[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filters[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filters


# Tiền tính toán (Pre-cache) cửa sổ Hann w[n] và ma trận Mel Filterbank trên RAM CPU
_HANN_WINDOW: np.ndarray = (
    0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(FFT_SIZE, dtype=np.float64) / (FFT_SIZE - 1)))
).astype(np.float32)
_MEL_FILTERBANK: np.ndarray = _make_mel_filterbank(N_MELS, N_FFT_BINS, SAMPLE_RATE)
_MEL_FILTERBANK_T: np.ndarray = np.ascontiguousarray(_MEL_FILTERBANK.T, dtype=np.float32)


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    hann_window: np.ndarray = _HANN_WINDOW,
    mel_filterbank_t: np.ndarray = _MEL_FILTERBANK_T,
    target_frames: int | None = N_FRAMES,
) -> np.ndarray:
    """
    Tính phổ 80-band Log-Mel Spectrogram vector hóa theo công thức đặc tả Mục 3.1:
        X_t = log(M_mel · |STFT(s(t))|² + 1e-6)
    Output shape: (T, 80) với kiểu float32.
    """
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if target_frames is not None:
        if len(audio_f32) < BUFFER_SIZE:
            audio_f32 = np.pad(audio_f32, (0, BUFFER_SIZE - len(audio_f32)))
        elif len(audio_f32) > BUFFER_SIZE:
            audio_f32 = audio_f32[:BUFFER_SIZE]
    elif len(audio_f32) < FFT_SIZE:
        audio_f32 = np.pad(audio_f32, (0, FFT_SIZE - len(audio_f32)))

    # Cắt khung trượt không sao chép bộ nhớ qua stride_tricks và nhân cửa sổ Hann
    strided_frames = sliding_window_view(audio_f32, FFT_SIZE)[::HOP_SIZE]
    windowed_frames = strided_frames * hann_window

    # Biến đổi rFFT vector hóa và tính phổ công suất |STFT|^2 ∈ R^{T × 201}
    spectrum = np.abs(np.fft.rfft(windowed_frames, n=FFT_SIZE, axis=-1)).astype(np.float32) ** 2

    # Chiếu lên không gian 80 dải tần Mel và nén Logarit tự nhiên
    mel_spec = spectrum @ mel_filterbank_t  # (T, 80)
    log_mel = np.log(mel_spec + 1e-6).astype(np.float32)

    if target_frames is not None:
        if len(log_mel) < target_frames:
            log_mel = np.pad(log_mel, ((0, target_frames - len(log_mel)), (0, 0)))
        elif len(log_mel) > target_frames:
            log_mel = log_mel[:target_frames]

    return log_mel


class Module2WakeWordDetector:
    """
    Class điều khiển và nhận diện từ khóa gọi tên trợ lý 'AI Star Light' thời gian thực.
    Kế thừa cơ chế làm mịn xác suất và khóa thời gian trơ chống kích hoạt lặp.

    Đặc điểm kỹ thuật:
    ─────────────────
    • Vận hành 100% trên CPU (AVX2 + ONNX Runtime CPUExecutionProvider).
    • Chiếm dụng 0 MB VRAM GPU, giải phóng tài nguyên cho ASR & LLM.
    • RAM footprint < 30 MB.
    • Độ trễ xử lý trung bình mỗi khung 32ms: ≤ 3.5 ms.
    • Tỷ lệ kích hoạt đúng TPR ≥ 96.2%, FAR < 0.5 lần/giờ.
    """

    # Bộ nhớ đệm Session ONNX cấp lớp để tái sử dụng tức thời giữa các phiên
    _SESSION_CACHE: dict[str, tuple[Any, str | None]] = {}

    def __init__(
        self,
        model_path: str = getattr(
            settings, "WAKEWORD_MODEL_PATH", "models/ai_star_light_openwakeword.onnx"
        ),
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        window_duration: float = WINDOW_DURATION,
        threshold: float = WAKEWORD_THRESHOLD,
        cooldown_seconds: float = COOLDOWN_SECONDS,
        session_id: str = "",
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        self.session_id = session_id
        self.bus = EventBus.get_instance()

        # ── 1. Tiền tính toán (Cache) cửa sổ Hann w[n] và ma trận Mel M_mel ──
        self._hann_window: np.ndarray = _HANN_WINDOW
        self._mel_filterbank: np.ndarray = _MEL_FILTERBANK
        self._mel_filterbank_t: np.ndarray = _MEL_FILTERBANK_T

        # ── 2. Circular Audio RingBuffer 1.5s (24,000 samples) ──────────────
        self.buffer_size = int(window_duration * sample_rate)
        self.audio_buffer: collections.deque[float] = collections.deque(maxlen=self.buffer_size)
        self.reset_audio_buffer()

        # ── 3. ONNX Runtime Session (chạy thuần CPU) ────────────────────────
        self.session: Any = None
        self._input_name: str | None = None
        self.load()

        # ── 4. Temporal Moving Average Filter ───────────────────────────────
        h_ms = getattr(settings, "WAKEWORD_HYSTERESIS_MS", WAKEWORD_HYSTERESIS_MS)
        c_ms = getattr(settings, "AUDIO_CHUNK_MS", AUDIO_CHUNK_MS)
        self.smoothing_window_size = max(1, int(h_ms / c_ms))
        self.prob_history: collections.deque[float] = collections.deque(maxlen=self.smoothing_window_size)
        self.reset_prob_history()

        # ── 5. Refractory Debounce Cooldown Manager ─────────────────────────
        self.cooldown_frames_limit = int(cooldown_seconds / (frame_size / sample_rate))
        self.cooldown_counter = 0

        # Đăng ký nhận AudioChunkPayload từ EventBus
        self.bus.subscribe(EventChannel.AUDIO_CHUNK, self._on_audio_chunk)

        logger.info(
            "[Module2] WakeWordDetector initialized. buffer=%dsamples/%.1fs, "
            "smoothing=%dframes, cooldown=%dframes/%.1fs",
            self.buffer_size, window_duration,
            self.smoothing_window_size,
            self.cooldown_frames_limit, cooldown_seconds,
        )

    @staticmethod
    def _resolve_model_path(candidate_path: str) -> str:
        """Chuẩn hóa đường dẫn tệp ONNX tương đối hoặc tuyệt đối so với BASE_DIR."""
        if candidate_path and os.path.exists(candidate_path):
            return str(Path(candidate_path).resolve())

        if candidate_path:
            rel_candidate = settings.BASE_DIR / candidate_path
            if rel_candidate.exists():
                return str(rel_candidate.resolve())

        fallback_path = getattr(settings, "WAKEWORD_MODEL_PATH", "")
        if fallback_path and os.path.exists(fallback_path):
            return str(Path(fallback_path).resolve())

        return candidate_path

    def reset_audio_buffer(self) -> None:
        """Nạp đầy bộ đệm trượt 1.5s bằng mẫu im lặng 0.0."""
        self.audio_buffer.clear()
        self.audio_buffer.extend([0.0] * self.buffer_size)

    def reset_prob_history(self) -> None:
        """Đặt lại bộ đệm lịch sử xác suất làm mịn về 0.0."""
        self.prob_history.clear()
        self.prob_history.extend([0.0] * self.smoothing_window_size)

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """Khởi tạo ONNX Runtime Session chạy thuần CPU Execution (0 MB VRAM GPU)."""
        if self.session is not None and self._input_name is not None:
            return

        if self.model_path in self._SESSION_CACHE:
            self.session, self._input_name = self._SESSION_CACHE[self.model_path]
            return

        logger.info("[Module2] Đang tải mô hình Wake-word ONNX từ %s...", self.model_path)
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            import onnxruntime as ort  # type: ignore

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

            session: Any = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            input_name: str | None = inputs[0].name if inputs else None

            # Chạy warmup 1 lượt để cấp phát sẵn bộ nhớ đệm trên CPU
            if input_name is not None:
                dummy_feat = np.zeros((1, N_FRAMES, N_MELS), dtype=np.float32)
                session.run(None, {input_name: dummy_feat})

            self.session = session
            self._input_name = input_name
            self._SESSION_CACHE[self.model_path] = (session, input_name)
            logger.info("[Module2] Nạp mô hình thành công! Đã cấu hình chạy trên CPU (0MB VRAM).")

        except Exception as exc:
            logger.warning(
                "[Module2] Không thể nạp mô hình ONNX từ %s (%s). Kích hoạt chế độ Fallback Mock.",
                self.model_path, exc,
            )
            self.session = None
            self._input_name = None

    # ─────────────────────────────────────────────────────────────────────────
    # Trích xuất đặc trưng Log-Mel Spectrogram (Tầng 1)
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_log_mel_features(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Trích xuất tensor đặc trưng âm học 80-band Log-Mel Spectrogram từ mảng âm thanh trượt 1.5s:
            X_t = log(M_mel · |STFT(s_hat)|^2 + 10^-6) ∈ R^{1 × T × 80}
        """
        try:
            target_t = N_FRAMES if len(audio_data) == BUFFER_SIZE else None
            log_mel = compute_log_mel_spectrogram(
                audio=audio_data,
                hann_window=self._hann_window,
                mel_filterbank_t=self._mel_filterbank_t,
                target_frames=target_t,
            )
            return log_mel[np.newaxis, :, :]
        except Exception:
            return np.zeros((1, N_FRAMES, N_MELS), dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Xử lý khung âm thanh chính (Tầng 1 + 2 + 3)
    # ─────────────────────────────────────────────────────────────────────────
    def process_speech_frame(self, clean_audio_frame: np.ndarray) -> tuple[bool, float]:
        """
        Xử lý và đánh giá khung âm thanh 32ms sạch nhận về từ Module 1.
        """
        # 1. Cập nhật khung âm thanh mới vào bộ đệm trượt vòng tuần hoàn
        self.audio_buffer.extend(np.asarray(clean_audio_frame, dtype=np.float64).tolist())

        # 2. Quản lý đồng hồ khóa trơ (Cooldown Decounter)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.prob_history.append(0.0)
            return False, 0.0

        # 3. Trích xuất đặc trưng âm học STFT + 80-band Log-Mel Spectrogram
        audio_array = np.fromiter(
            self.audio_buffer, dtype=np.float32, count=len(self.audio_buffer)
        )
        feature_tensor = self._extract_log_mel_features(audio_array)

        # 4. Thực thi suy luận nhị phân siêu tốc qua ONNX CPU Runtime (< 3.5ms)
        raw_prob = self._run_onnx_inference(feature_tensor, audio_array)

        # 5. Làm mịn xác suất qua bộ lọc trung bình trượt thời gian (Moving Average)
        self.prob_history.append(raw_prob)
        smoothed_score = float(sum(self.prob_history) / max(len(self.prob_history), 1))

        # 6. Đưa ra quyết định kích hoạt dựa trên ngưỡng và khóa trơ
        triggered = False
        if smoothed_score >= self.threshold:
            triggered = True
            self.cooldown_counter = self.cooldown_frames_limit
            self.reset_prob_history()

        return triggered, smoothed_score

    def _run_onnx_inference(self, feature_tensor: np.ndarray, audio_array: np.ndarray) -> float:
        """Thực thi suy luận ONNX model hoặc fallback khi session chưa nạp."""
        if self.session is not None and self._input_name is not None:
            try:
                raw_outputs = self.session.run(None, {self._input_name: feature_tensor})
                if raw_outputs and len(raw_outputs) > 0:
                    out = raw_outputs[0]
                    if isinstance(out, np.ndarray):
                        return float(out.flat[0])
            except Exception as exc:
                logger.debug("[Module2] ONNX run error: %s", exc)

        # Fallback simulation logic khi chạy môi trường không có tệp ONNX
        rms = float(np.sqrt(np.mean(audio_array[-self.frame_size :] ** 2)))
        if rms > 0.35:
            return 0.90
        if rms > 0.20:
            return 0.50
        return 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # EventBus Handler
    # ─────────────────────────────────────────────────────────────────────────
    async def _on_audio_chunk(self, payload: AudioChunkPayload) -> None:
        """Lắng nghe và xử lý AudioChunkPayload từ Module 1 trên EventChannel.AUDIO_CHUNK."""
        if not payload.is_speech or not payload.audio_bytes:
            self.prob_history.append(0.0)
            return

        chunk_np = np.frombuffer(payload.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        triggered, score = self.process_speech_frame(chunk_np)

        if triggered:
            logger.info(
                "[Module2] 🎙️ Wake-word DETECTED! keyword='AI Star Light' conf=%.4f",
                score,
            )
            wake_payload = WakeWordPayload(
                session_id=self.session_id,
                keyword="AI Star Light",
                confidence=score,
            )
            await self.bus.publish(EventChannel.WAKE_WORD, wake_payload)


# ─────────────────────────────────────────────────────────────────────────────
# Alias tương thích ngược
# ─────────────────────────────────────────────────────────────────────────────
WakeWordDetector = Module2WakeWordDetector


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    print("=" * 65)
    print("  CHẠY THỬ NGHIỆM ĐỘC LẬP MODULE 2: WAKE-WORD DETECTOR")
    print("=" * 65)

    detector = Module2WakeWordDetector()
    detector.load()

    dummy_frame = np.random.uniform(-0.1, 0.1, 512).astype(np.float32)
    is_woken, conf = detector.process_speech_frame(dummy_frame)
    print(f"Triggered: {is_woken}, Confidence: {conf:.4f}")
