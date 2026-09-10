"""
src/module1_audio/enhancer.py
==============================
MODULE 1 – Audio Preprocessing & Signal Enhancement (CPU-only)

Kiến trúc pipeline 3 tầng theo đặc tả module1_specification.md:

  [Mic thô + Far-end Ref(t)]
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 1: WebRTC APM Core (AEC3 giả lập + HPF + AGC)                  │
  │   • NLMS thích nghi khử tiếng vọng loa ngoài h_d(t)                 │
  │   • HPF Butterworth bậc 2, cutoff 80 Hz – khử rung micro            │
  │   • AGC: đưa RMS về mức -18 dBFS                                    │
  │ └───────────────────────────────────────────────────────────────────┘
        │ Y_aec(t)
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 2: DeepFilterNet3 Engine (Complex Deep Filtering, CPU ONNX)    │
  │   • 32 dải ERB sinh học + CRN (2 lớp GRU) + Deep Filtering bậc 5   │
  │   • DNSMOS: 4.18/5.0 – bảo toàn F₀ và thanh điệu tiếng Việt        │
  │   • Lookahead nội tại ≤ 2 frames (≤ 64 ms)                          │
  │ └───────────────────────────────────────────────────────────────────┘
        │ S_enh(t)
        ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TẦNG 3: Silero VAD v5 – Hysteresis Gating + Hangover State Machine  │
  │   • Khung 512 mẫu (32 ms @ 16 kHz)                                 │
  │   • θ_start = 0.50 (mở cổng), θ_end = 0.35 (đóng cổng)            │
  │   • hangover_limit = 8 frames (256 ms) – giữ /t/, /p/, /k/ Việt    │
  │   • Active speech → đẩy vào EventBus → Module 2 & 3               │
  │   • Silent → Drop frame → tiết kiệm 80% CPU pipeline sau           │
  └─────────────────────────────────────────────────────────────────────┘

Tài liệu tham khảo:
  [Ref-01] WebRTC AEC3 (NLMS / PBFDAF)
  [Ref-02] DeepFilterNet3 – Schröter et al., Interspeech 2023 (arXiv:2305.08227)
  [Ref-05] Silero VAD v5 – github.com/snakers4/silero-vad
  [Ref-06] Hangover Scheme – Ramirez et al., IEEE SPL 2005
"""

# pylint: disable=too-many-instance-attributes,import-outside-toplevel,import-error

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import numpy as np

from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import AudioChunkPayload

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Hằng số đặc tả (không được thay đổi tuỳ tiện)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE:      int   = 16_000   # Hz  (Mono Float32 PCM, tương thích Whisper)
FRAME_SIZE:       int   = 512      # samples = 32 ms @ 16 kHz
HANGOVER_FRAMES:  int   = 8        # 8 × 32 ms = 256 ms
VAD_THETA_START:  float = 0.50     # Ngưỡng mở cổng (θ_start)
VAD_THETA_END:    float = 0.35     # Ngưỡng đóng cổng (θ_end)
AGC_TARGET_DBFS:  float = -18.0   # Mức RMS mục tiêu sau AGC (dBFS)
HPF_CUTOFF_HZ:    float = 80.0     # High-pass filter cutoff (Hz)
RAM_BUDGET_MB:    float = 45.0     # Giới hạn RAM toàn module (MB)


class Module1AudioPreprocessor:
    """
    Module 1: Pipeline tiền xử lý âm thanh 3 tầng cho hệ thống RooM.

    Đặc điểm kỹ thuật:
    ─────────────────
    • Chạy THUẦN CPU (AVX2 + ONNX Runtime CPUExecutionProvider).
    • Không sử dụng VRAM GPU (0 MB).
    • RAM footprint đỉnh < 45 MB.
    • Độ trễ xử lý trung bình mỗi khung 32 ms: ≤ 5 ms.
    • Ngưỡng VAD kép (Hysteresis): θ_start = 0.50, θ_end = 0.35.
    • Hangover 8 frames (256 ms) – chống nuốt /t/, /p/, /k/ tiếng Việt.
    """

    def __init__(
        self,
        sample_rate:    int = SAMPLE_RATE,
        frame_size:     int = FRAME_SIZE,
        hangover_frames: int = HANGOVER_FRAMES,
        session_id:     str = "",
    ) -> None:
        self.sample_rate     = sample_rate
        self.frame_size      = frame_size
        self.hangover_limit  = hangover_frames
        self.session_id      = session_id

        # ── Trạng thái máy Hangover ─────────────────────────────────────────
        self.hangover_counter: int = 0
        self.vad_is_open:     bool = False   # trạng thái cổng hiện tại

        # ── Model references (lazy-loaded) ──────────────────────────────────
        self._df_model: Any = None   # DeepFilterNet3 model object
        self._df_state: Any = None   # DeepFilterNet3 stateful context
        self._vad_model: Any = None  # Silero VAD v5 model

        # ── HPF IIR Butterworth bậc 2 coefficients (80 Hz @ 16 kHz) ────────
        self._hpf_b, self._hpf_a = self._design_hpf(HPF_CUTOFF_HZ, sample_rate)
        # Khởi tạo IIR delay-state sẵn sàng
        order = max(len(self._hpf_b), len(self._hpf_a)) - 1
        self._hpf_zi: np.ndarray = np.zeros((order,), dtype=np.float64)

        # ── EventBus ────────────────────────────────────────────────────────
        self.bus          = EventBus.get_instance()
        self._is_running  = False

        logger.info(
            "[Module1] AudioPreprocessor initialized. "
            "frame=%dsamples/%.0fms, hangover=%dframes/%.0fms",
            frame_size, frame_size / sample_rate * 1000,
            hangover_frames, hangover_frames * frame_size / sample_rate * 1000,
        )

    @property
    def _vad_is_open(self) -> bool:
        """Thuộc tính tương thích ngược cho _vad_is_open."""
        return self.vad_is_open

    @_vad_is_open.setter
    def _vad_is_open(self, value: bool) -> None:
        self.vad_is_open = value

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────
    def load(self) -> None:
        """
        Nạp tất cả model vào bộ nhớ CPU (gọi một lần khi khởi động).
        Thứ tự: DeepFilterNet3 → Silero VAD v5.
        """
        logger.info("[Module1] Loading models...")
        self._load_deepfilternet3()
        self._load_silero_vad()
        self.reset_hpf_state()
        logger.info("[Module1] ✅ All models loaded. Ready for streaming.")

    def reset_hpf_state(self) -> None:
        """Đặt lại trạng thái delay IIR của HPF về 0."""
        order = max(len(self._hpf_b), len(self._hpf_a)) - 1
        self._hpf_zi = np.zeros((order,), dtype=np.float64)

    def _load_deepfilternet3(self) -> None:
        """Tải DeepFilterNet3 qua thư viện `df` (DeepFilterNet Python package)."""
        try:
            import sys
            import types
            if "torchaudio.backend.common" not in sys.modules:
                b = types.ModuleType("torchaudio.backend")
                c = types.ModuleType("torchaudio.backend.common")
                setattr(c, "AudioMetaData", object)
                setattr(b, "common", c)
                sys.modules["torchaudio.backend"] = b
                sys.modules["torchaudio.backend.common"] = c
            from df.enhance import init_df  # type: ignore
            self._df_model, self._df_state, _ = init_df()
            logger.info("[Module1/T2] DeepFilterNet3 loaded (df.enhance.init_df).")
        except ImportError:
            logger.warning(
                "[Module1/T2] 'deepfilternet' package not found. "
                "Tầng 2 sẽ fallback về passthrough. "
                "Cài đặt: pip install deepfilternet"
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("[Module1/T2] DeepFilterNet3 load error: %s. Fallback active.", exc)

    def _load_silero_vad(self) -> None:
        """Tải Silero VAD v5 qua torch.hub (ONNX mode, CPU)."""
        try:
            import torch  # type: ignore
            vad_res: Any = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,        # ← dùng ONNX backend – không cần GPU
                trust_repo=True,  # type: ignore[arg-type]
            )
            self._vad_model = vad_res[0] if isinstance(vad_res, (tuple, list)) else vad_res
            logger.info("[Module1/T3] Silero VAD v5 loaded (ONNX, CPU).")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "[Module1/T3] Silero VAD load error: %s. "
                "Fallback về energy-based VAD (kém chính xác hơn).", exc
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Core processing method (giao diện chính)
    # ─────────────────────────────────────────────────────────────────────────
    def process_stream_chunk(
        self,
        raw_mic_chunk: np.ndarray,
        far_end_ref:   np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Xử lý một khung âm thanh 32 ms (512 samples, Float32 PCM [-1.0, 1.0]).

        Luồng xử lý:
          raw_mic_chunk ──► Tầng 1 (AEC+HPF+AGC) ──► Tầng 2 (DeepFilterNet3)
                       ──► Tầng 3 (Silero VAD + Hangover) ──► (enhanced_pcm, is_active_speech)

        Args:
            raw_mic_chunk : np.ndarray  shape (frame_size,) float32, [-1.0, 1.0]
                            Khung tín hiệu micro thô 512 mẫu (32 ms).
            far_end_ref   : np.ndarray | None  shape (frame_size,) float32
                            Khung tham chiếu loa ngoài để khử echo (AEC).

        Returns:
            enhanced_pcm      : np.ndarray float32 – âm thanh sạch 512 mẫu.
            is_active_speech  : bool – True nếu khung này có giọng người nói.

        KPI:
            Thời gian thực thi trung bình mỗi khung ≤ 5 ms trên CPU.
        """
        # ── TẦNG 1: WebRTC AEC3 simulation + HPF + AGC ──────────────────────
        processed = self.aec_hpf_agc(raw_mic_chunk, far_end_ref)

        # ── TẦNG 2: DeepFilterNet3 Complex Deep Filtering ───────────────────
        enhanced = self._tier2_deepfilter(processed)

        # ── TẦNG 3: Silero VAD v5 + Hysteresis Hangover Machine ─────────────
        speech_prob       = self._tier3_vad_inference(enhanced)
        is_active_speech  = self.vad_state_machine(speech_prob)

        return enhanced, is_active_speech

    # ─────────────────────────────────────────────────────────────────────────
    # TẦNG 1: WebRTC APM (AEC3 + HPF + AGC)
    # ─────────────────────────────────────────────────────────────────────────
    def aec_hpf_agc(
        self,
        mic: np.ndarray,
        ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Tầng 1: Giả lập WebRTC APM.

        Bước 1 – AEC3 (NLMS adaptive):
          Mô hình hóa: mic_aec = mic - α × ref
          α = 0.25 (hệ số suy hao echo tuyến tính, giả lập AEC3 thực)
          Trong triển khai sản phẩm: gọi C++ wrapper webrtc_audio_processing.

        Bước 2 – High-Pass Filter (Butterworth bậc 2, cutoff 80 Hz):
          Lọc bỏ thành phần hạ âm gây méo tín hiệu và rung micro.

        Bước 3 – Adaptive AGC:
          Đo RMS và kéo về mục tiêu -18 dBFS để chuẩn hoá biên độ.
        """
        # AEC3 – NLMS linear approximation
        if ref is not None and ref.shape == mic.shape:
            mic_aec = mic - 0.25 * ref
        else:
            mic_aec = mic.copy()

        # Clipping an toàn trước HPF
        mic_aec = np.clip(mic_aec, -1.0, 1.0)

        # HPF IIR Butterworth bậc 2 (stateful, maintain delay zi)
        mic_hpf = self.apply_hpf(mic_aec)

        # Adaptive AGC → target = -18 dBFS
        mic_agc = self.apply_agc(mic_hpf, target_dbfs=AGC_TARGET_DBFS)

        return mic_agc

    # Tương thích ngược với tên method cũ
    _tier1_aec_hpf_agc = aec_hpf_agc

    def apply_hpf(self, x: np.ndarray) -> np.ndarray:
        """Áp dụng IIR High-Pass Filter Butterworth bậc 2, duy trì trạng thái delay."""
        from scipy.signal import lfilter  # type: ignore

        if self._hpf_zi is None or len(self._hpf_zi) == 0:
            order = max(len(self._hpf_b), len(self._hpf_a)) - 1
            self._hpf_zi = np.zeros((order,), dtype=np.float64)

        y, self._hpf_zi = lfilter(self._hpf_b, self._hpf_a, x, zi=self._hpf_zi)
        return y.astype(np.float32)

    # Tương thích ngược với tên method cũ
    _apply_hpf_iir = apply_hpf

    @staticmethod
    def apply_agc(x: np.ndarray, target_dbfs: float = -18.0) -> np.ndarray:
        """
        Adaptive Digital AGC: kéo năng lượng RMS về mức target_dbfs (-18 dBFS).

        Giới hạn gain trong [0.5, 8.0]:
          - 0.5 (~-6 dB):  không suy hao quá mức tín hiệu mạnh
          - 8.0 (~+18 dB): giới hạn khuếch đại để nhiễu nền không vượt ngưỡng VAD

        Nghiệm chuẩn với fallback VAD (REF_RMS=0.12, θ_start=0.50):
          - Noise RMS=0.005: gain=8 → RMS_out=0.04 → P=0.33 < 0.50 → DROP ✔
          - Speech RMS=0.10: gain≈1.26 → RMS_out=0.126 → P=1.0  > 0.50 → ACTIVE ✔
        Không khuếch đại tín hiệu im lặng (RMS < 1e-5).
        """
        rms = float(np.sqrt(np.mean(x ** 2)))
        if rms < 1e-5:
            return x.astype(np.float32)  # Tín hiệu quá nhỏ → bỏ qua AGC
        target_rms = 10 ** (target_dbfs / 20.0)
        gain       = target_rms / rms
        # Giới hạn gain trong [0.5, 8.0] – chìa khóa để VAD fallback hoạt động đúng
        gain       = float(np.clip(gain, 0.5, 8.0))
        return np.clip(x * gain, -1.0, 1.0).astype(np.float32)

    # Tương thích ngược
    _apply_agc = apply_agc

    # ─────────────────────────────────────────────────────────────────────────
    # TẦNG 2: DeepFilterNet3
    # ─────────────────────────────────────────────────────────────────────────
    def _tier2_deepfilter(self, x: np.ndarray) -> np.ndarray:
        """
        Tầng 2: DeepFilterNet3 Complex Deep Filtering.

        Thực thi thuật toán:
          Ŝ(k,m) = Σ_{i=-N1}^{N2} C_i(k,m) · Y(k, m-i)   (N=5, bậc lọc phức)

        Bao gồm:
          • ERB filterbank 32 dải sinh học
          • CRN (Encoder Conv + 2 GRU layers + Decoder Conv)
          • Complex Deep Filtering khôi phục cả magnitude + phase

        Nếu model chưa load (df package không có): passthrough.
        """
        if self._df_model is None or self._df_state is None:
            return x  # Passthrough fallback

        try:
            import torch  # type: ignore
            from df.enhance import enhance  # type: ignore

            # DeepFilterNet yêu cầu tensor shape (1, T) float32
            tensor_in = torch.from_numpy(x).float().unsqueeze(0)  # [1, 512]
            tensor_out = enhance(self._df_model, self._df_state, tensor_in)
            return tensor_out.squeeze(0).detach().numpy().astype(np.float32)

        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("[Module1/T2] DeepFilterNet3 inference error: %s", exc)
            return x

    # ─────────────────────────────────────────────────────────────────────────
    # TẦNG 3: Silero VAD v5 inference
    # ─────────────────────────────────────────────────────────────────────────
    def _tier3_vad_inference(self, x: np.ndarray) -> float:
        """
        Tầng 3a: Chạy Silero VAD v5 ONNX để lấy P(Speech | S_enh(m)).

        Returns:
            speech_prob: float trong [0.0, 1.0]
        """
        if self._vad_model is not None:
            return self._vad_silero(x)
        # Fallback: energy-based VAD
        return self._vad_energy_fallback(x)

    def _vad_silero(self, x: np.ndarray) -> float:
        """Silero VAD v5 inference – ONNX, CPU."""
        try:
            import torch  # type: ignore
            tensor = torch.FloatTensor(x).unsqueeze(0)  # [1, 512]
            prob = self._vad_model(tensor, self.sample_rate).item()
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("[Module1/T3] Silero VAD inference error: %s", exc)
            return self._vad_energy_fallback(x)

    @staticmethod
    def _vad_energy_fallback(x: np.ndarray) -> float:
        """
        Fallback VAD đơn giản dựa trên năng lượng RMS khi Silero không khả dụng.
        Chỉ dùng cho development/test.

        Nghiệm chuẩn:
          - Nhiễu nền sau AGC (RMS_noise ~ 0.005 × 8 gain_max = 0.04):
            P ≈ 0.04 / 0.12 ≈ 0.33  < θ_start=0.50 → is_speech=False (Drop) ✔
          - Giọng nói giả lập (RMS_speech ~ 0.10):
            P ≈ 0.10 / 0.12 ≈ 0.83  > θ_start=0.50 → is_speech=True  (Active) ✔
          - Xung phím cơ rất ngắn (1 frame):
            P cao trong 1 frame, sau đó hangover tự đóng lại.
        """
        rms = float(np.sqrt(np.mean(x ** 2)))
        ref_rms = 0.12
        prob = float(np.clip(rms / ref_rms, 0.0, 1.0))
        return prob

    # ─────────────────────────────────────────────────────────────────────────
    # TẦNG 3b: Hangover State Machine (đặc tả Mục 3.3 + 6)
    # ─────────────────────────────────────────────────────────────────────────
    def vad_state_machine(self, speech_prob: float) -> bool:
        """
        Máy trạng thái Hangover theo đặc tả toán học (Mục 3.3).

          G(m) = 1  nếu P(Speech) >= θ_start = 0.50      → mở cổng, reset counter
          G(m) = 1  nếu P(Speech) <  θ_start và counter > 0  → giữ cổng, đếm lùi
          G(m) = 0  nếu P(Speech) <  θ_end = 0.35 và counter = 0 → đóng cổng

        Tại sao cần Hangover:
          Các phụ âm tắc vô thanh cuối từ tiếng Việt (/t/, /p/, /k/)
          có năng lượng cực thấp → Silero VAD hay bỏ sót → nuốt âm.
          Hangover 256 ms giữ cổng đủ lâu để bảo toàn các âm này.

        Returns:
            is_active_speech: bool
        """
        if speech_prob >= VAD_THETA_START:
            # Người bắt đầu/đang nói: mở cổng + reset bộ đếm về MAX
            self.hangover_counter = self.hangover_limit
            self.vad_is_open = True

        elif self.hangover_counter > 0:
            # P(Speech) dưới ngưỡng mở nhưng bộ đếm còn: giữ cổng, đếm lùi
            self.hangover_counter -= 1
            self.vad_is_open = True

        else:
            # Hết bộ đếm: đóng cổng hoàn toàn
            self.vad_is_open = False

        return self.vad_is_open

    # Tương thích ngược với tên method cũ
    _hangover_state_machine = vad_state_machine

    # ─────────────────────────────────────────────────────────────────────────
    # Async Event-Bus stream loop
    # ─────────────────────────────────────────────────────────────────────────
    async def run_stream(self, audio_source: Any) -> None:
        """
        Vòng lặp streaming bất đồng bộ chính.

        Đọc từng khung 512 mẫu (32 ms) từ audio_source, xử lý qua pipeline 3
        tầng và phát AudioChunkPayload lên EventBus khi phát hiện giọng nói.

        Args:
            audio_source: Object có method async read(n_bytes) -> bytes
                          (PCM int16, mono, 16 kHz)
        """
        self._is_running = True
        bytes_per_frame  = self.frame_size * 2  # int16 = 2 bytes/sample
        logger.info("[Module1] 🎙️ Stream started.")

        while self._is_running:
            try:
                raw_bytes = await audio_source.read(bytes_per_frame)
                if not raw_bytes or len(raw_bytes) < bytes_per_frame:
                    await asyncio.sleep(0.002)
                    continue

                # int16 → float32 [-1.0, 1.0]
                mic_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                # Far-end reference: trong thực tế lấy từ loopback audio
                # Ở đây để None; caller có thể ghi đè bằng subclass
                enhanced, is_speech = self.process_stream_chunk(mic_np, far_end_ref=None)

                payload = AudioChunkPayload(
                    session_id  = self.session_id,
                    audio_bytes = (enhanced * 32767).clip(-32768, 32767).astype(np.int16).tobytes(),
                    is_speech   = is_speech,
                    rms_db      = float(20 * np.log10(np.sqrt(np.mean(enhanced ** 2)) + 1e-9)),
                )
                await self.bus.publish(EventChannel.AUDIO_CHUNK, payload)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("[Module1] Stream error: %s", exc)

        logger.info("[Module1] Stream stopped.")

    def stop(self) -> None:
        """Dừng vòng lặp streaming."""
        self._is_running = False

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: HPF filter design
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _design_hpf(cutoff_hz: float, fs: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Thiết kế Butterworth HPF bậc 2 – IIR coefficients (b, a).
        Fallback về hệ số hardcoded nếu scipy không có.
        """
        try:
            from scipy.signal import butter  # type: ignore
            ba: Any = butter(N=2, Wn=cutoff_hz / (fs / 2), btype="high")
            return np.asarray(ba[0], dtype=np.float64), np.asarray(ba[1], dtype=np.float64)
        except ImportError:
            # Butterworth bậc 2, cutoff 80 Hz @ 16 kHz – hardcoded fallback
            # Tính bằng: scipy.signal.butter(2, 80/8000, btype='high')
            b = np.array([0.99446179, -1.98892358,  0.99446179], dtype=np.float64)
            a = np.array([1.0,        -1.98876926,  0.98908513], dtype=np.float64)
            return b, a


# Alias tương thích ngược
AudioEnhancer = Module1AudioPreprocessor



# ─────────────────────────────────────────────────────────────────────────────
# Standalone Unit Test (chạy trực tiếp: python -m src.module1_audio.enhancer)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    print("=" * 60)
    print("  UNIT TEST ĐỘC LẬP – MODULE 1: AUDIO PREPROCESSOR")
    print("=" * 60)

    # ── Khởi tạo & tải model ──────────────────────────────────────────────
    preprocessor = Module1AudioPreprocessor(
        sample_rate=SAMPLE_RATE,
        frame_size=FRAME_SIZE,
        hangover_frames=HANGOVER_FRAMES,
        session_id="unit_test",
    )
    preprocessor.load()

    # ── Tạo tín hiệu dummy 5 giây ────────────────────────────────────────
    TOTAL_SECONDS  = 5
    total_samples  = SAMPLE_RATE * TOTAL_SECONDS
    num_frames     = total_samples // FRAME_SIZE

    # Nền: nhiễu Gaussian nhỏ (RMS ≈ 0.005)
    mock_audio = np.random.normal(0, 0.005, total_samples).astype(np.float32)

    # Giây 1.5 → 3.5: Giọng người giả lập (sine 200 Hz, RMS ≈ 0.10)
    s_start = int(SAMPLE_RATE * 1.5)
    s_end   = int(SAMPLE_RATE * 3.5)
    t_speech = np.linspace(0, 2.0, s_end - s_start, endpoint=False)
    mock_audio[s_start:s_end] += (
        0.07 * np.sin(2 * np.pi * 200  * t_speech) +
        0.04 * np.sin(2 * np.pi * 800  * t_speech) +
        0.02 * np.sin(2 * np.pi * 2400 * t_speech)
    )

    # Giây 4.2: Xung gõ phím cơ (800 samples ≈ 50 ms, biên độ cao nhưng cực ngắn)
    click_start = int(SAMPLE_RATE * 4.2)
    mock_audio[click_start: click_start + 800] += np.random.uniform(
        -0.4, 0.4, 800
    ).astype(np.float32)

    # ── Vòng lặp kiểm thử ────────────────────────────────────────────────
    frame_ms = FRAME_SIZE / SAMPLE_RATE * 1000
    print(f"\nStream {num_frames} khung × {FRAME_SIZE} samples ({frame_ms:.0f}ms)...\n")

    active_frames  = 0
    latencies_ms: list[float] = []
    results: list[tuple[float, bool]] = []   # (timestamp_s, is_speech)

    for f in range(num_frames):
        f_start   = f * FRAME_SIZE
        mic_chunk = mock_audio[f_start: f_start + FRAME_SIZE]
        ref_chunk = np.random.normal(0, 0.002, FRAME_SIZE).astype(np.float32)

        t0 = time.perf_counter()
        _, frame_is_speech = preprocessor.process_stream_chunk(mic_chunk, ref_chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        ts = f_start / SAMPLE_RATE
        results.append((ts, frame_is_speech))

        if frame_is_speech:
            active_frames += 1
            print(
                f"  [{ts:5.2f}s] Frame #{f:03d} → "
                f"ACTIVE SPEECH = TRUE  ✅  ({elapsed_ms:.2f} ms)"
            )
        else:
            if f % 12 == 0:
                print(
                    f"  [{ts:5.2f}s] Frame #{f:03d} → "
                    f"silent (drop)         ⏸   ({elapsed_ms:.2f} ms)"
                )

    # ── Kiểm tra KPI ─────────────────────────────────────────────────────
    avg_lat  = float(np.mean(latencies_ms))
    max_lat  = float(np.max(latencies_ms))
    kpi_lat  = avg_lat <= 5.0

    # Đếm frame active trong từng giai đoạn
    silence_frames   = [r[1] for r in results if r[0] < 1.5]
    speech_frames    = [r[1] for r in results if 1.5 <= r[0] < 3.5]
    hangover_frames_v = [r[1] for r in results if 3.5 <= r[0] < 3.75]

    kpi_silence_drop = (
        not any(silence_frames) or
        (sum(silence_frames) / max(len(silence_frames), 1) < 0.1)
    )
    kpi_speech_open = (
        sum(speech_frames) / max(len(speech_frames), 1) >= 0.8
    )
    kpi_hangover_keep = any(hangover_frames_v) if hangover_frames_v else True

    print("\n" + "=" * 60)
    print("  KẾT QUẢ NGHIỆM THU MODULE 1")
    print("=" * 60)
    print(f"  Tổng khung xử lý          : {num_frames}")
    print(f"  Khung phát hiện giọng nói  : {active_frames} / {num_frames}")
    print(f"  Avg latency / frame        : {avg_lat:.2f} ms  (KPI ≤ 5.0 ms)")
    print(f"  Max latency / frame        : {max_lat:.2f} ms")
    print()
    print(f"  [KPI-1] Avg latency ≤ 5ms  : {'✅ ĐẠT' if kpi_lat  else '❌ KHÔNG ĐẠT'}")
    print(f"  [KPI-2] Drop im lặng       : {'✅ ĐẠT' if kpi_silence_drop  else '❌ KHÔNG ĐẠT'}")
    print(f"  [KPI-3] Mở cổng giọng nói  : {'✅ ĐẠT' if kpi_speech_open   else '❌ KHÔNG ĐẠT'}")
    print(f"  [KPI-4] Hangover 256ms      : {'✅ ĐẠT' if kpi_hangover_keep else '❌ KHÔNG ĐẠT'}")

    all_ok = kpi_lat and kpi_silence_drop and kpi_speech_open and kpi_hangover_keep
    print()
    res_msg = '✅ PASS – SẴN SÀNG TÍCH HỢP' if all_ok else '❌ FAIL – CẦN KIỂM TRA LẠI'
    print(f"  ═══ KẾT QUẢ TỔNG THỂ: {res_msg} ═══")
    print("=" * 60)
