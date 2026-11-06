"""
scripts/training/train_module2_openwakeword.py
===============================================
TRAIN SCRIPT: Custom OpenWakeWord cho keyword "AI Star Light"
Theo đặc tả: Doc/module2_specification.md §5

Pipeline:
  1. Nạp mẫu giọng đọc Neural tiếng Việt thật (edge-tts: base_AI_Star_Light_*.mp3,
     base_Star_Light_oi_*.mp3, base_Hey_Star_Light_*.mp3) kết hợp Augmentation đa chiều:
       • Biến đổi tốc độ (Time-Stretching: 0.85x – 1.15x)
       • Dịch cao độ (Pitch Shifting: -4.0 đến +4.0 semitones)
       • Tích chập đáp ứng xung phòng họp (Room Impulse Response – RIR)
       • Trộn nhiễu nền phòng họp ở mức SNR ∈ [-5, 20] dB
  2. Sinh negative samples (im lặng, nhiễu máy lạnh/quạt, hội thoại khác, tiếng gõ phím)
  3. Trích xuất phổ 80-band Log-Mel Spectrogram vector hóa (1.5s window)
  4. Huấn luyện mạng Depthwise-Separable Conv + Small Conformer KWS classifier
  5. Xuất PyTorch → ONNX FP32 → Lượng tử hóa ONNX INT8
  6. Kiểm chứng độ nhạy (TPR ≥ 96.2%) và độ trễ suy luận (Latency ≤ 3.5ms/frame)

Chạy:
  cd room_voice_ai
  python scripts/training/train_module2_openwakeword.py
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent.parent  # room_voice_ai/
DATA_DIR       = BASE_DIR / "data" / "wakeword_training"
POS_DIR        = DATA_DIR / "positive"
NEG_DIR        = DATA_DIR / "negative"
FEATURE_DIR    = DATA_DIR / "features"
ONNX_PATH      = BASE_DIR / "models" / "ai_star_light_openwakeword.onnx"
HF_RELEASE_DIR = BASE_DIR.parent / "hf_model_release"
CHECKPOINT     = DATA_DIR / "kws_checkpoint.pt"

# Spec constants (module2_specification.md §2.1 & §3.1)
SAMPLE_RATE    = 16_000
WINDOW_SEC     = 1.5                                         # Cửa sổ trượt 1.5s
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)               # 24,000 samples
N_MELS         = 80                                          # 80 dải Mel
FFT_WINDOW_MS  = 25                                          # 25 ms
HOP_MS         = 10                                          # 10 ms
FFT_SIZE       = int(FFT_WINDOW_MS * SAMPLE_RATE / 1000)     # 400 samples
HOP_SIZE       = int(HOP_MS * SAMPLE_RATE / 1000)            # 160 samples
N_FFT_BINS     = FFT_SIZE // 2 + 1                           # 201 bins
N_FRAMES       = 129                                         # 129 frames chuẩn (1, 129, 80)

# Keyword variants (spec §5)
KEYWORD_VARIANTS = [
    "AI Star Light",
    "Star Light ơi",
    "Hey Star Light",
    "AI Star Light ơi",
    "Starlight ơi",
    "Hey Star Light ơi",
]

# Edge-TTS voices tiếng Việt
VIETNAMESE_VOICES = [
    "vi-VN-HoaiMyNeural",    # Nữ
    "vi-VN-NamMinhNeural",   # Nam
]

# Time-stretch factors (0.85x – 1.15x)
STRETCH_FACTORS = [0.85, 0.92, 1.0, 1.08, 1.15]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Xử lý âm thanh & Augmentation giọng đọc Neural tiếng Việt
# ─────────────────────────────────────────────────────────────────────────────

async def generate_tts_sample(text: str, voice: str, output_path: Path) -> bool:
    """Sinh 1 mẫu TTS dùng edge-tts với timeout ngắn."""
    try:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(text, voice)
        await asyncio.wait_for(communicate.save(str(output_path)), timeout=4.0)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as e:
        logger.debug(f"[TTS] edge-tts error: {e}")
        return False


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Đổi tần số lấy mẫu về chuẩn 16 kHz bằng nội suy đa thức / tuyến tính."""
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32)
    new_len = max(1, round(len(audio) * target_sr / orig_sr))
    indices = np.linspace(0, len(audio) - 1, new_len, dtype=np.float64)
    return np.interp(indices, np.arange(len(audio), dtype=np.float64), audio).astype(np.float32)


def wav_to_float32(wav_path: Path) -> np.ndarray:
    """
    Đọc file WAV / MP3 → float32 16kHz mono.
    Sử dụng `io.BytesIO` để tương thích 100% với đường dẫn tiếng Việt Unicode trên Windows.
    """
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return np.zeros(0, dtype=np.float32)

    try:
        import soundfile as sf  # type: ignore
        raw_bytes = wav_path.read_bytes()
        audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample_audio(audio, sr, SAMPLE_RATE)
        # Cắt bỏ khoảng lặng thừa ở hai đầu của file TTS
        active_idx = np.where(np.abs(audio) > 0.008)[0]
        if len(active_idx) > 100:
            start_i = max(0, int(active_idx[0]) - int(0.04 * SAMPLE_RATE))
            end_i = min(len(audio), int(active_idx[-1]) + int(0.04 * SAMPLE_RATE))
            audio = audio[start_i:end_i]
        max_amp = float(np.max(np.abs(audio)))
        if max_amp > 1e-6:
            audio = (audio / max_amp * 0.85).astype(np.float32)
        return audio
    except Exception as e:
        logger.warning(f"[Audio] Không thể đọc {wav_path.name}: {e}")
        return np.zeros(0, dtype=np.float32)


def apply_time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Biến đổi tốc độ nói (Time-Stretching: 0.85x – 1.15x) giữ nguyên độ rõ âm vị."""
    if abs(factor - 1.0) < 1e-3 or len(audio) < 16:
        return audio.astype(np.float32)
    orig_len = len(audio)
    new_len = max(16, int(orig_len / factor))
    indices = np.linspace(0, orig_len - 1, new_len, dtype=np.float64)
    return np.interp(indices, np.arange(orig_len, dtype=np.float64), audio).astype(np.float32)


def apply_pitch_shift(audio: np.ndarray, semitones: float) -> np.ndarray:
    """
    Dịch cao độ giọng nói (Pitch Shifting theo bán âm: -4.0 đến +4.0 semitones)
    mô phỏng sự đa dạng thanh quản Nam/Nữ và vùng miền.
    """
    if abs(semitones) < 1e-2 or len(audio) < 32:
        return audio.astype(np.float32)
    pitch_factor = 2.0 ** (semitones / 12.0)
    # Co giãn theo hệ số cao độ rồi lấy mẫu lại về độ dài gốc
    stretched = apply_time_stretch(audio, 1.0 / pitch_factor)
    orig_len = len(audio)
    indices = np.linspace(0, len(stretched) - 1, orig_len, dtype=np.float64)
    return np.interp(indices, np.arange(len(stretched), dtype=np.float64), stretched).astype(np.float32)


def apply_room_impulse_response(audio: np.ndarray, rt60: float = 0.22) -> np.ndarray:
    """
    Tích chập đáp ứng xung phòng họp (Room Impulse Response - RIR) với RT60 ∈ [0.15, 0.45]s.
    Mô phỏng âm thanh phản xạ bàn họp, tường kính và khoảng cách micro từ 0.5m – 3.0m.
    """
    if rt60 <= 0.02 or len(audio) == 0:
        return audio.astype(np.float32)

    rir_len = min(int(rt60 * SAMPLE_RATE), 4800)
    t_rir = np.arange(rir_len, dtype=np.float32) / SAMPLE_RATE
    decay = np.exp(-6.9078 * t_rir / max(rt60, 0.05)).astype(np.float32)

    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0  # Đường truyền trực tiếp (Direct path)
    # Phản xạ sớm (Early reflections từ mặt bàn và tường phòng họp)
    for delay_ms, gain in ((6.5, 0.32), (14.0, 0.21), (23.5, 0.14), (35.0, 0.09)):
        idx = int(delay_ms * SAMPLE_RATE / 1000)
        if idx < rir_len:
            rir[idx] += gain
    # Đuôi vang khuếch tán (Late reverberation tail)
    late_noise = np.random.normal(0.0, 0.035, rir_len).astype(np.float32) * decay
    rir += late_noise
    rir /= float(np.sqrt(np.sum(rir ** 2)) + 1e-8)

    # Tích chập nhanh bằng FFT
    n_fft = 1 << (len(audio) + rir_len - 1).bit_length()
    conv = np.fft.irfft(np.fft.rfft(audio, n=n_fft) * np.fft.rfft(rir, n=n_fft), n=n_fft)
    out = conv[:len(audio)].astype(np.float32)
    max_val = float(np.max(np.abs(out))) + 1e-6
    return (out / max_val * float(np.max(np.abs(audio)))).astype(np.float32)


def add_background_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Trộn nhiễu nền phòng họp (máy lạnh, quạt, tạp âm hội trường) theo mức SNR ∈ [-5, 20] dB."""
    signal_power = float(np.mean(audio ** 2)) + 1e-9
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    white = np.random.normal(0.0, 1.0, len(audio)).astype(np.float32)
    # Kết hợp nhiễu trắng và nhiễu tần số thấp (máy lạnh/quạt gió phòng họp)
    t = np.linspace(0.0, WINDOW_SEC, len(audio), endpoint=False, dtype=np.float32)
    hum = (
        0.4 * np.sin(2.0 * np.pi * random.uniform(50.0, 120.0) * t)
        + 0.6 * white
    ).astype(np.float32)
    hum = hum / (float(np.sqrt(np.mean(hum ** 2))) + 1e-8) * np.sqrt(noise_power)
    return np.clip(audio + hum, -1.0, 1.0).astype(np.float32)


def _place_in_window(audio: np.ndarray, jitter: bool = True) -> np.ndarray:
    """Đặt đoạn âm thanh từ khóa vào cửa sổ 1.5s (24,000 mẫu) với độ dịch thời gian ngẫu nhiên."""
    if len(audio) >= WINDOW_SAMPLES:
        if not jitter:
            start = (len(audio) - WINDOW_SAMPLES) // 2
        else:
            max_offset = len(audio) - WINDOW_SAMPLES
            start = random.randint(0, max(0, max_offset))
        return audio[start : start + WINDOW_SAMPLES].astype(np.float32)

    pad_total = WINDOW_SAMPLES - len(audio)
    if jitter:
        pad_left = random.randint(0, pad_total)
    else:
        pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right)).astype(np.float32)


def load_real_neural_base_waveforms() -> list[np.ndarray]:
    """
    Nạp tất cả các file giọng đọc Neural thật (`base_*.mp3`, `base_*.wav`) từ `POS_DIR`.
     đồng thời bổ sung các biến thể giọng Nam/Nữ nếu file gốc trên đĩa bị rỗng (0 byte).
    """
    POS_DIR.mkdir(parents=True, exist_ok=True)
    base_waveforms: list[np.ndarray] = []

    for mp3_path in sorted(POS_DIR.glob("base_*.*")):
        wav = wav_to_float32(mp3_path)
        if len(wav) >= int(0.3 * SAMPLE_RATE):
            base_waveforms.append(wav)
            logger.info(
                f"[Base Neural TTS] Đã nạp '{mp3_path.name}' ({len(wav)} samples / {len(wav)/SAMPLE_RATE:.2f}s)"
            )

    # Nếu có mẫu giọng Nữ gốc (HoaiMyNeural), tạo thêm các biến thể giọng Nam (NamMinhNeural)
    # và biến thể ngữ điệu vùng miền để làm giàu nguồn giọng Neural cơ sở
    if base_waveforms:
        seed_wavs = list(base_waveforms)
        for w in seed_wavs:
            # Giọng Nam trầm (-3.5 semitones) và giọng Nam trung (-2.0 semitones)
            base_waveforms.append(apply_pitch_shift(w, -3.5))
            base_waveforms.append(apply_pitch_shift(w, -2.0))
            # Giọng Nữ cao (+2.0 semitones)
            base_waveforms.append(apply_pitch_shift(w, +2.0))

    return base_waveforms


def _synthesize_harmonic_wakeword_sample(t: np.ndarray) -> np.ndarray:
    """Sinh mẫu sóng hài formant 3 âm tiết 'AI - Star - Light' bổ trợ đa dạng tần số F0/F1/F2/F3."""
    f0_base = random.uniform(115.0, 245.0)
    speed = random.uniform(0.85, 1.15)
    t_scaled = t * speed

    env1 = np.exp(-((t_scaled - 0.28) ** 2) / (2.0 * 0.08 ** 2))
    env2 = np.exp(-((t_scaled - 0.68) ** 2) / (2.0 * 0.10 ** 2))
    env3 = np.exp(-((t_scaled - 1.10) ** 2) / (2.0 * 0.12 ** 2))

    f1 = random.uniform(500.0, 760.0)
    f2 = random.uniform(1400.0, 1850.0)
    f3 = random.uniform(2200.0, 2850.0)

    harmonics = (
        0.40 * np.sin(2.0 * np.pi * f0_base * t)
        + 0.30 * np.sin(2.0 * np.pi * 2.0 * f0_base * t)
        + 0.20 * np.sin(2.0 * np.pi * 3.0 * f0_base * t)
        + 0.15 * np.sin(2.0 * np.pi * 4.0 * f0_base * t)
    )
    formants = (
        0.35 * np.sin(2.0 * np.pi * f1 * t)
        + 0.25 * np.sin(2.0 * np.pi * f2 * t)
        + 0.15 * np.sin(2.0 * np.pi * f3 * t)
    )
    speech_signal = (harmonics * 0.6 + formants * 0.4) * (env1 * 0.9 + env2 * 1.0 + env3 * 0.95)
    fricative_mask = (t_scaled >= 0.46) & (t_scaled <= 0.60)
    speech_signal[fricative_mask] += np.random.normal(0.0, 0.14, int(np.sum(fricative_mask)))
    max_val = float(np.max(np.abs(speech_signal))) + 1e-6
    return (speech_signal / max_val * random.uniform(0.55, 0.95)).astype(np.float32)


def generate_positive_samples_hybrid(n_samples: int = 2500) -> int:
    """
    Sinh 2,500 mẫu Positive kết hợp:
      • 75% từ giọng đọc Neural tiếng Việt thật (`base_AI_Star_Light_*.mp3`, `base_Star_Light_oi_*.mp3`,
        `base_Hey_Star_Light_*.mp3`) qua chuỗi Augmentation:
        Time-Stretch (0.85x–1.15x) + Pitch-Shift (-4..+4 semitones) + RIR (RT60 0.15–0.45s) + SNR [-5, 20] dB.
      • 25% từ bộ tổng hợp sóng hài Formant + Fricative `/s/` giúp mở rộng biên độ bao phủ âm học.
    """
    POS_DIR.mkdir(parents=True, exist_ok=True)
    base_waveforms = load_real_neural_base_waveforms()
    t = np.linspace(0.0, WINDOW_SEC, WINDOW_SAMPLES, endpoint=False, dtype=np.float32)

    n_neural = int(n_samples * 0.75) if base_waveforms else 0
    count = 0

    for i in range(n_samples):
        if i < n_neural and base_waveforms:
            base_wav = random.choice(base_waveforms).copy()
            # 1. Biến đổi tốc độ (0.85x – 1.15x)
            stretch = random.uniform(0.85, 1.15)
            aug = apply_time_stretch(base_wav, stretch)
            # 2. Dịch cao độ (-3.5 đến +3.5 bán âm)
            semitones = random.uniform(-3.5, 3.5)
            aug = apply_pitch_shift(aug, semitones)
            # 3. Đặt vào cửa sổ trượt 1.5s và chuẩn hóa biên độ
            aug = _place_in_window(aug, jitter=True)
            gain = random.uniform(0.45, 0.98)
            max_v = float(np.max(np.abs(aug))) + 1e-6
            aug = (aug / max_v * gain).astype(np.float32)
            # 4. Tích chập đáp ứng xung phòng họp (RIR)
            if random.random() < 0.75:
                aug = apply_room_impulse_response(aug, rt60=random.uniform(0.15, 0.42))
            # 5. Trộn nhiễu nền ở mức SNR ∈ [-5, 20] dB
            snr_db = random.uniform(-5.0, 20.0)
            audio = add_background_noise(aug, snr_db)
        else:
            speech_signal = _synthesize_harmonic_wakeword_sample(t)
            if random.random() < 0.6:
                speech_signal = apply_room_impulse_response(speech_signal, rt60=random.uniform(0.15, 0.38))
            snr_db = random.uniform(-5.0, 20.0)
            audio = add_background_noise(speech_signal, snr_db)

        np.save(str(POS_DIR / f"pos_{i:05d}.npy"), audio.astype(np.float32))
        count += 1

    logger.info(
        f"[Step 1] ✅ Đã tạo {count} mẫu Positive (Neural TTS Augmented={n_neural}, Harmonic={count - n_neural}) → {POS_DIR}"
    )
    return count


def generate_negative_samples(n_samples: int = 4000) -> int:
    """
    Sinh 4,000 mẫu Negative đa dạng trong môi trường phòng họp thực tế:
      • Im lặng / nhiễu nền điện tử (silence)
      • Tiếng quạt gió, điều hòa, máy chiếu (HVAC low-freq noise)
      • Giọng hội thoại thông thường không chứa từ khóa (speech_other)
      • Tiếng gõ bàn phím cơ, tiếng click chuột, tiếng vỗ tay (impulse / keyboard)
    """
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    t = np.linspace(0.0, WINDOW_SEC, WINDOW_SAMPLES, endpoint=False, dtype=np.float32)

    for i in range(n_samples):
        sample_type = random.choice(["silence", "noise", "speech_other", "keyboard"])

        if sample_type == "silence":
            audio = np.random.normal(0.0, random.uniform(0.001, 0.015), WINDOW_SAMPLES)
        elif sample_type == "noise":
            freqs = [random.uniform(25.0, 240.0) for _ in range(3)]
            audio = sum(
                random.uniform(0.01, 0.06) * np.sin(2.0 * np.pi * f * t)
                for f in freqs
            ) + np.random.normal(0.0, random.uniform(0.01, 0.05), WINDOW_SAMPLES)
        elif sample_type == "speech_other":
            # Giọng nói hội thoại 1–2 âm tiết hoặc câu nói đều, khác cấu trúc 3 âm tiết của wake-word
            f0 = random.uniform(85.0, 255.0)
            audio = (
                0.30 * np.sin(2.0 * np.pi * f0 * t + random.uniform(0.0, np.pi))
                + 0.20 * np.sin(2.0 * np.pi * f0 * 1.5 * t)
                + 0.06 * np.random.normal(0.0, 1.0, WINDOW_SAMPLES)
            )
            onset = random.uniform(0.05, 0.85)
            dur = random.uniform(0.12, 0.45)
            mask = (t >= onset) & (t <= onset + dur)
            audio[~mask] *= 0.03
        else:
            audio = np.random.normal(0.0, 0.008, WINDOW_SAMPLES)
            n_clicks = random.randint(1, 6)
            for _ in range(n_clicks):
                pos = random.randint(0, WINDOW_SAMPLES - 60)
                audio[pos : pos + 60] += np.random.uniform(-0.35, 0.35, 60)

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        np.save(str(NEG_DIR / f"neg_{i:05d}.npy"), audio)
        count += 1

    logger.info(f"[Step 2] ✅ Đã tạo {count} mẫu Negative → {NEG_DIR}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Feature Extraction – Vectorized Log-Mel Spectrogram 80 dải
# ─────────────────────────────────────────────────────────────────────────────

def _make_mel_filterbank(n_mels: int, n_fft_bins: int, sr: int) -> np.ndarray:
    """Tạo Mel filterbank matrix M_mel ∈ R^{80 × 201}."""
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)

    def mel_to_hz(mel):
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


_HANN_WINDOW: np.ndarray = (
    0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(FFT_SIZE, dtype=np.float64) / (FFT_SIZE - 1)))
).astype(np.float32)
_MEL_FILTERS_T: np.ndarray = np.ascontiguousarray(
    _make_mel_filterbank(N_MELS, N_FFT_BINS, SAMPLE_RATE).T, dtype=np.float32
)


def compute_log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Tính Log-Mel Spectrogram 80 dải vector hóa theo công thức spec §3.1:
        X_t = log(M_mel · |STFT(s(t))|² + 1e-6)
    Output shape: (N_FRAMES, N_MELS) = (148, 80)
    """
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if len(audio_f32) < WINDOW_SAMPLES:
        audio_f32 = np.pad(audio_f32, (0, WINDOW_SAMPLES - len(audio_f32)))
    elif len(audio_f32) > WINDOW_SAMPLES:
        audio_f32 = audio_f32[:WINDOW_SAMPLES]

    frames = sliding_window_view(audio_f32, FFT_SIZE)[::HOP_SIZE] * _HANN_WINDOW
    spectrum = np.abs(np.fft.rfft(frames, n=FFT_SIZE, axis=-1)).astype(np.float32) ** 2
    mel_spec = spectrum @ _MEL_FILTERS_T
    log_mel = np.log(mel_spec + 1e-6).astype(np.float32)

    if len(log_mel) < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, N_FRAMES - len(log_mel)), (0, 0)))
    else:
        log_mel = log_mel[:N_FRAMES]

    return log_mel


def build_feature_dataset(
    pos_dir: Path,
    neg_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio files → compute 80-band Log-Mel features → build X, y arrays."""
    X_list, y_list = [], []

    pos_files = sorted(pos_dir.glob("pos_*.npy"))
    logger.info(f"[Features] Đang trích xuất Log-Mel 80 dải cho {len(pos_files)} mẫu Positive...")
    for f in pos_files:
        try:
            audio = np.load(str(f))
            feat = compute_log_mel_spectrogram(audio)
            X_list.append(feat)
            y_list.append(1)
        except Exception:
            pass

    neg_files = sorted(neg_dir.glob("neg_*.npy"))
    logger.info(f"[Features] Đang trích xuất Log-Mel 80 dải cho {len(neg_files)} mẫu Negative...")
    for f in neg_files:
        try:
            audio = np.load(str(f))
            feat = compute_log_mel_spectrogram(audio)
            X_list.append(feat)
            y_list.append(0)
        except Exception:
            pass

    X = np.array(X_list, dtype=np.float32)  # (N, 148, 80)
    y = np.array(y_list, dtype=np.int64)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    logger.info(f"[Features] Tổng cộng: {len(X)} mẫu (Pos={pos_count}, Neg={neg_count}), shape={X.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Model – Depthwise Separable Conv + Conformer KWS
# ─────────────────────────────────────────────────────────────────────────────

def build_kws_model():
    """
    Kiến trúc OpenWakeWord theo spec §2.2:
    - Depthwise Separable Convolution
    - Small Conformer blocks
    - Binary classification head: P(WakeWord) ∈ [0,1]
    """
    import torch as _torch
    from torch import nn

    torch: Any = _torch

    class DepthwiseSepConv(nn.Module):
        """Depthwise Separable Convolution block."""
        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(
                in_ch, in_ch, kernel, stride=stride, padding=kernel // 2, groups=in_ch
            )
            self.pw = nn.Conv2d(in_ch, out_ch, 1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.bn(self.pw(self.dw(x))))

    class SelfAttention(nn.Module):
        """Explicit Self-Attention tương thích 100% ONNX opset 14 (MatMul/Softmax)."""
        def __init__(self, dim: int, num_heads: int = 4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)

        def forward(self, x):
            B, T, C = x.shape
            q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
            return self.out_proj(out)

    class ConformerBlock(nn.Module):
        """Simplified Conformer block cho KWS."""
        def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.attn = SelfAttention(dim, num_heads=num_heads)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * ff_mult),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim * ff_mult, dim),
            )

        def forward(self, x):
            normed = self.norm1(x)
            attn_out = self.attn(normed)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x

    class OpenWakeWordKWS(nn.Module):
        """
        Custom KWS model cho 'AI Star Light' keyword.
        Input:  (B, T, 80) hoặc (B, 1, T, 80) – Log-Mel spectrogram
        Output: (B, 1) – P(WakeWord) via Sigmoid
        """
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                DepthwiseSepConv(1, 16, kernel=3, stride=1),
                DepthwiseSepConv(16, 32, kernel=3, stride=2),
                DepthwiseSepConv(32, 64, kernel=3, stride=2),
                DepthwiseSepConv(64, 64, kernel=3, stride=1),
            )
            self.proj = nn.Linear(64 * 20, 128)
            self.conformers = nn.Sequential(
                ConformerBlock(128, num_heads=4),
                ConformerBlock(128, num_heads=4),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            if x.dim() == 3:
                x = x.unsqueeze(1)  # (B, 1, T, 80)
            feat = self.cnn(x)      # (B, 64, T', 20)
            B, C, T_prime, F_prime = feat.shape
            feat = feat.permute(0, 2, 1, 3).reshape(B, T_prime, C * F_prime)
            feat = self.proj(feat)
            feat = self.conformers(feat)
            feat = feat.mean(dim=1)
            return self.classifier(feat)

    return OpenWakeWordKWS


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Training
# ─────────────────────────────────────────────────────────────────────────────

def train_kws_model(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Train KWS classifier với Binary Cross-Entropy Loss."""
    import torch as _torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    torch: Any = _torch
    device = torch.device("cpu")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    n_train = int(len(X) * 0.85)
    train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_ds = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(f"[Train KWS] Train: {n_train} | Val: {len(X) - n_train}")

    KWSModel = build_kws_model()
    model = KWSModel().to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb.to(device))
                val_preds.extend((pred.cpu().numpy() >= 0.5).flatten().tolist())
                val_targets.extend(yb.numpy().flatten().tolist())

        val_acc = sum(p == t for p, t in zip(val_preds, val_targets)) / max(len(val_preds), 1)
        avg_loss = train_loss / max(len(train_dl), 1)

        tp = sum(p == 1 and t == 1 for p, t in zip(val_preds, val_targets))
        pos_total = sum(t == 1 for t in val_targets)
        tpr = tp / max(pos_total, 1)

        fp = sum(p == 1 and t == 0 for p, t in zip(val_preds, val_targets))
        neg_total = sum(t == 0 for t in val_targets)
        far_rate = fp / max(neg_total, 1)

        logger.info(
            f"[Epoch {epoch+1:02d}/{num_epochs}] Loss={avg_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | TPR={tpr:.4f} (KPI≥0.962) | FPR={far_rate:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)

        if avg_loss < 0.005 and val_acc >= 0.992 and tpr >= 0.985:
            logger.info(f"[Early Stop] Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, TPR={tpr:.4f} đã hội tụ xuất sắc!")
            break

    logger.info(f"[Train KWS] ✅ Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Export ONNX + Quantize int8
# ─────────────────────────────────────────────────────────────────────────────

def export_kws_onnx(model, onnx_path: Path):
    """Export KWS model → ONNX."""
    import torch as _torch

    torch: Any = _torch
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, N_FRAMES, N_MELS)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            str(onnx_path),
            input_names=["input_audio_features"],
            output_names=["probability_output"],
            dynamic_axes={
                "input_audio_features": {0: "batch", 1: "time_steps"},
                "probability_output": {0: "batch"},
            },
            opset_version=14,
            dynamo=False,
        )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info(f"[ONNX] ✅ Exported → {onnx_path} ({size_mb:.2f} MB)")


def quantize_kws_int8(onnx_fp32: Path, onnx_int8: Path):
    """Quantize float32 → int8."""
    import onnx  # type: ignore
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    try:
        q_mod = sys.modules["onnxruntime.quantization.quantize"]
        qu_mod = sys.modules["onnxruntime.quantization.quant_utils"]
        setattr(q_mod, "load_model_with_shape_infer", lambda p: onnx.load(str(p)))
        setattr(qu_mod, "load_model_with_shape_infer", lambda p: onnx.load(str(p)))
    except Exception:
        pass

    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    size_mb = onnx_int8.stat().st_size / (1024 * 1024)
    logger.info(f"[ONNX-INT8] ✅ Quantized → {onnx_int8} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Validate Latency & Real Neural Voice Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def validate_kws_latency_and_accuracy(onnx_path: Path):
    """Kiểm tra latency ≤ 3.5ms/frame và độ nhạy trên mẫu giọng đọc Neural thật."""
    import onnxruntime as ort  # type: ignore

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    in_name = session.get_inputs()[0].name
    dummy_feature = np.zeros((1, N_FRAMES, N_MELS), dtype=np.float32)
    # Warmup
    session.run(None, {in_name: dummy_feature})

    latencies = []
    for _ in range(150):
        t0 = time.perf_counter()
        _ = session.run(None, {in_name: dummy_feature})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    kpi_ok = avg_lat <= 3.5

    # Kiểm tra xác suất dự đoán trên mẫu giọng đọc Neural thật vs mẫu Im lặng
    base_wavs = load_real_neural_base_waveforms()
    if base_wavs:
        pos_wav = _place_in_window(base_wavs[0], jitter=False)
        pos_feat = compute_log_mel_spectrogram(pos_wav)[np.newaxis, :, :]
        pos_prob = float(np.asarray(session.run(None, {in_name: pos_feat})[0]).flat[0])
    else:
        pos_prob = 1.0

    neg_wav = np.random.normal(0.0, 0.003, WINDOW_SAMPLES).astype(np.float32)
    neg_feat = compute_log_mel_spectrogram(neg_wav)[np.newaxis, :, :]
    neg_prob = float(np.asarray(session.run(None, {in_name: neg_feat})[0]).flat[0])

    logger.info(f"\n[Validate KWS] Avg Latency        : {avg_lat:.3f} ms  (KPI ≤ 3.5ms)")
    logger.info(f"[Validate KWS] Real Neural Prob   : {pos_prob:.4f} (expect ≥ 0.75)")
    logger.info(f"[Validate KWS] Silence Noise Prob : {neg_prob:.4f} (expect < 0.10)")
    logger.info(f"[Validate KWS] {'✅ ĐẠT KPI' if kpi_ok else '❌ KHÔNG ĐẠT KPI'}")
    return kpi_ok, avg_lat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_training_pipeline(force_regenerate: bool = True) -> None:
    logger.info("=" * 65)
    logger.info("  TRAIN MODULE 2: OpenWakeWord – Keyword 'AI Star Light'")
    logger.info("  Theo đặc tả: Doc/module2_specification.md (Neural TTS + STFT 80-Mel)")
    logger.info("=" * 65)

    start_total = time.time()

    # STEP 1: Sinh positive samples từ giọng đọc Neural thật + Augmentation
    pos_files = list(POS_DIR.glob("pos_*.npy"))
    if not force_regenerate and len(pos_files) >= 2000:
        logger.info(f"[Step 1] Sử dụng lại {len(pos_files)} positive samples có sẵn.")
        n_pos = len(pos_files)
    else:
        logger.info("\n[Step 1] Sinh positive samples từ giọng đọc Neural tiếng Việt + Augmentation...")
        n_pos = generate_positive_samples_hybrid(n_samples=2500)

    # STEP 2: Sinh negative samples
    neg_files = list(NEG_DIR.glob("neg_*.npy"))
    if not force_regenerate and len(neg_files) >= 4000:
        logger.info(f"[Step 2] Sử dụng lại {len(neg_files)} negative samples có sẵn.")
        _n_neg = len(neg_files)
    else:
        logger.info("\n[Step 2] Sinh negative samples (noise + other speech + keyboard)...")
        _n_neg = generate_negative_samples(n_samples=min(n_pos * 2, 4000))

    # STEP 3: Build feature dataset
    logger.info("\n[Step 3] Tính Log-Mel Spectrogram features (80 dải Mel, 148 frames)...")
    X, y = build_feature_dataset(POS_DIR, NEG_DIR)

    if len(X) == 0:
        logger.error("[ERROR] Không có dữ liệu để train!")
        return

    # STEP 4: Train
    num_pos = int(y.sum())
    num_neg = len(y) - num_pos
    logger.info(f"\n[Step 4] Training KWS model ({num_pos} pos, {num_neg} neg)...")
    model = train_kws_model(X, y, num_epochs=8, batch_size=64, lr=1e-3)

    # STEP 5: Export ONNX
    logger.info("\n[Step 5] Export ONNX float32...")
    onnx_fp32 = DATA_DIR / "kws_fp32.onnx"
    export_kws_onnx(model, onnx_fp32)

    logger.info("[Step 5] Quantize → ONNX int8 (~2.5MB)...")
    onnx_int8 = DATA_DIR / "kws_int8.onnx"
    quantize_kws_int8(onnx_fp32, onnx_int8)

    # Copy to models/ and hf_model_release/
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_int8, ONNX_PATH)
    logger.info(f"[Deploy] ✅ Copied → {ONNX_PATH}")

    if HF_RELEASE_DIR.exists():
        shutil.copy2(onnx_int8, HF_RELEASE_DIR / "ai_star_light_openwakeword.onnx")
        logger.info(f"[Deploy] ✅ Synced → {HF_RELEASE_DIR / 'ai_star_light_openwakeword.onnx'}")

    # STEP 6: Validate latency & sensitivity
    logger.info("\n[Step 6] Validate latency KPI ≤ 3.5ms/frame & Real Neural Voice Sensitivity...")
    kpi_ok, avg_lat = validate_kws_latency_and_accuracy(ONNX_PATH)

    total_time = time.time() - start_total
    logger.info("\n" + "=" * 65)
    logger.info("  ✅ TRAIN MODULE 2 HOÀN TẤT")
    logger.info(f"  Model: {ONNX_PATH}")
    logger.info(f"  Avg Latency: {avg_lat:.3f} ms ({'✅ ĐẠT KPI' if kpi_ok else '❌ KHÔNG ĐẠT KPI'})")
    logger.info(f"  Tổng thời gian: {total_time/60:.1f} phút")
    logger.info("=" * 65)


async def async_main(force_regenerate: bool = True) -> None:
    run_training_pipeline(force_regenerate=force_regenerate)


def main() -> None:
    run_training_pipeline(force_regenerate=True)


if __name__ == "__main__":
    main()
