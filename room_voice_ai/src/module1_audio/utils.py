"""
src/module1_audio/utils.py
===========================
Các hàm tiện ích âm thanh cho Module 1 – Audio Preprocessing.

Tất cả hàm ở đây đều:
  • Không phụ thuộc GPU (CPU-only).
  • Không có side-effect (pure functions) – dễ unit test độc lập.
  • Tương thích chuẩn: 16 kHz, Mono, Float32 PCM [-1.0, 1.0].

Tài liệu tham khảo:
  [Ref-07] DNSMOS P.835 – Reddy et al., ICASSP 2021
  [Ref-08] PESQ ITU-T P.862
  [Ref-10] VIVOS Vietnamese Speech Corpus
"""

from __future__ import annotations

import io
import wave

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SR: int   = 16_000   # Hz
INT16_MAX:  float = 32768.0


# ─────────────────────────────────────────────────────────────────────────────
# PCM conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

def int16_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """
    Chuyển PCM bytes (int16, little-endian) → numpy float32 [-1.0, 1.0].

    Args:
        pcm_bytes: bytes – raw PCM int16 data (mono, 16 kHz)

    Returns:
        np.ndarray float32, shape (n_samples,), range [-1.0, 1.0]

    Example:
        >>> audio = int16_bytes_to_float32(b"\\x00\\x80")  # one sample: -32768
        >>> audio[0]
        -1.0
    """
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / INT16_MAX


def float32_to_int16_bytes(audio: np.ndarray) -> bytes:
    """
    Chuyển numpy float32 [-1.0, 1.0] → PCM bytes int16.

    Args:
        audio: np.ndarray float32 [-1.0, 1.0]

    Returns:
        bytes – raw PCM int16 data

    Note:
        Clip [-1.0, 1.0] trước khi chuyển đổi để tránh clipping artifacts.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * (INT16_MAX - 1)).astype(np.int16).tobytes()


def normalize_float32(audio: np.ndarray) -> np.ndarray:
    """
    Chuẩn hoá biên độ peak về [-1.0, 1.0] (peak normalization).
    Không thay đổi tín hiệu nếu peak đã ≤ 1.0.

    Args:
        audio: np.ndarray float32

    Returns:
        np.ndarray float32 normalized
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return (audio / peak).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic measurement
# ─────────────────────────────────────────────────────────────────────────────

def compute_rms_dbfs(audio: np.ndarray) -> float:
    """
    Tính mức âm lượng RMS theo thang dBFS (decibels relative to full scale).

    dBFS = 20 × log10(RMS)   với RMS_max_sinewave = 0 dBFS

    Args:
        audio: np.ndarray float32, range [-1.0, 1.0]

    Returns:
        rms_db: float, ví dụ: -18.0 dBFS (mức AGC mục tiêu của hệ thống)
    """
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return 20.0 * np.log10(rms + 1e-10)


def compute_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Ước lượng Signal-to-Noise Ratio (SNR) giữa tín hiệu sạch và nhiễu.

    SNR = 10 × log10(P_signal / P_noise)

    Args:
        clean : np.ndarray – tín hiệu tham chiếu (sạch)
        noisy : np.ndarray – tín hiệu đo lường (nhiễu)

    Returns:
        snr_db: float (dB). Giá trị càng cao → nhiễu càng ít.
    """
    noise     = noisy - clean
    p_signal  = float(np.mean(clean.astype(np.float64) ** 2))
    p_noise   = float(np.mean(noise.astype(np.float64) ** 2))
    if p_noise < 1e-12:
        return float("inf")
    return 10.0 * np.log10(p_signal / p_noise)


def compute_erle_db(mic_before_aec: np.ndarray, mic_after_aec: np.ndarray) -> float:
    """
    Tính Echo Return Loss Enhancement (ERLE) của bộ AEC.

    ERLE = 10 × log10(E[mic_before²] / E[mic_after²])
    KPI thiết kế: ERLE > 35 dB (WebRTC AEC3).

    Args:
        mic_before_aec: tín hiệu micro trước AEC
        mic_after_aec : tín hiệu micro sau AEC

    Returns:
        erle_db: float (dB)
    """
    p_in  = float(np.mean(mic_before_aec.astype(np.float64) ** 2))
    p_out = float(np.mean(mic_after_aec.astype(np.float64) ** 2))
    if p_out < 1e-12:
        return float("inf")
    return 10.0 * np.log10(p_in / p_out)


# ─────────────────────────────────────────────────────────────────────────────
# Frame utilities
# ─────────────────────────────────────────────────────────────────────────────

def split_into_frames(
    audio: np.ndarray,
    frame_size: int = 512,
    pad_last: bool = True,
) -> list[np.ndarray]:
    """
    Chia mảng audio thành các khung (frames) kích thước cố định.

    Args:
        audio     : np.ndarray float32 – tín hiệu đầu vào
        frame_size: int – số samples mỗi khung (default: 512 = 32 ms @ 16 kHz)
        pad_last  : bool – nếu True, khung cuối được zero-pad đến frame_size

    Returns:
        List[np.ndarray] – danh sách khung float32

    Example:
        >>> frames = split_into_frames(audio, frame_size=512)
        >>> assert all(len(f) == 512 for f in frames)
    """
    frames = []
    n = len(audio)
    for start in range(0, n, frame_size):
        chunk = audio[start: start + frame_size]
        if len(chunk) < frame_size:
            if pad_last:
                chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            else:
                break
        frames.append(chunk.astype(np.float32))
    return frames


def frames_to_audio(frames: list[np.ndarray]) -> np.ndarray:
    """
    Ghép danh sách khung lại thành tín hiệu liên tục (concatenation).

    Args:
        frames: List[np.ndarray]

    Returns:
        np.ndarray float32
    """
    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames).astype(np.float32)


def apply_pre_emphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """
    Áp dụng bộ lọc pre-emphasis để tăng cường tần số cao.
    Giúp bù phổ giọng nói trước khi phân tích STFT / FFT.

    y[n] = x[n] - coeff × x[n-1]

    Args:
        audio : np.ndarray float32
        coeff : float hệ số pre-emphasis (thường 0.95–0.99)

    Returns:
        np.ndarray float32 – tín hiệu sau pre-emphasis
    """
    return np.append(audio[0], audio[1:] - coeff * audio[:-1]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Đổi sample rate bằng nội suy tuyến tính (linear interpolation).
    Sử dụng khi librosa/soxr không có sẵn.

    Args:
        audio    : np.ndarray float32 – tín hiệu nguồn
        orig_sr  : int – sample rate gốc (Hz)
        target_sr: int – sample rate đích (Hz)

    Returns:
        np.ndarray float32 – tín hiệu đã resample

    Note:
        Nếu orig_sr == target_sr, trả về ngay không xử lý.
        Để chất lượng tốt hơn trong production, dùng:
          pip install soxr
          import soxr; output = soxr.resample(audio, orig_sr, target_sr)
    """
    if orig_sr == target_sr:
        return audio
    duration    = len(audio) / orig_sr
    target_len  = int(duration * target_sr)
    x_old = np.linspace(0, len(audio) - 1, len(audio))
    x_new = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# WAV I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_wav(  # pylint: disable=no-member
    audio:       np.ndarray,
    path:        str,
    sample_rate: int = DEFAULT_SR,
) -> None:
    """
    Lưu mảng float32 PCM thành file WAV 16-bit mono.

    Args:
        audio      : np.ndarray float32 [-1.0, 1.0]
        path       : str – đường dẫn file WAV đầu ra
        sample_rate: int – sample rate (default: 16000 Hz)
    """
    pcm_bytes = float32_to_int16_bytes(audio)
    with wave.open(path, "wb") as wf:  # pylint: disable=no-member
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Đọc file WAV 16-bit mono → numpy float32 [-1.0, 1.0].

    Args:
        path: str – đường dẫn file WAV

    Returns:
        (audio_float32, sample_rate)

    Raises:
        ValueError nếu file không phải mono hoặc không phải 16-bit.
    """
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr           = wf.getframerate()
        raw_bytes    = wf.readframes(wf.getnframes())

    if n_channels != 1:
        raise ValueError(
            f"load_wav: cần file mono (1 channel), nhưng file có {n_channels} channels."
        )
    if sample_width != 2:
        raise ValueError(
            f"load_wav: cần 16-bit PCM (2 bytes/sample), nhưng file có {sample_width} bytes/sample."
        )

    return int16_bytes_to_float32(raw_bytes), sr


def load_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Đọc WAV từ bytes buffer (không cần file trên disk).

    Args:
        wav_bytes: bytes – nội dung file WAV

    Returns:
        (audio_float32, sample_rate)
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sr        = wf.getframerate()
        raw_bytes = wf.readframes(wf.getnframes())
    return int16_bytes_to_float32(raw_bytes), sr


# ─────────────────────────────────────────────────────────────────────────────
# Dummy audio generators (dùng cho Unit Test theo đặc tả Mục 7)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dummy_stream(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    total_seconds:  float = 5.0,
    sample_rate:    int   = DEFAULT_SR,
    noise_rms:      float = 0.005,
    speech_start_s: float = 1.5,
    speech_end_s:   float = 3.5,
    click_time_s:   float = 4.2,
    click_dur_ms:   float = 50.0,
) -> np.ndarray:
    """
    Tạo tín hiệu âm thanh dummy 5 giây theo kịch bản kiểm thử đặc tả Mục 7:

      0.0 → 1.5s : Im lặng + nhiễu nền nhỏ (RMS ≈ noise_rms)
      1.5 → 3.5s : Tiếng người nói (sine đa tần, RMS ≈ 0.10)
      3.5 → 5.0s : Im lặng + xung gõ phím cơ tại click_time_s (50 ms)

    Args:
        total_seconds  : float – tổng thời lượng (giây)
        sample_rate    : int   – sample rate (Hz)
        noise_rms      : float – biên độ nhiễu nền
        speech_start_s : float – giây bắt đầu tiếng nói
        speech_end_s   : float – giây kết thúc tiếng nói
        click_time_s   : float – giây xuất hiện xung phím cơ
        click_dur_ms   : float – độ dài xung phím cơ (ms)

    Returns:
        np.ndarray float32, shape (total_seconds × sample_rate,)
    """
    n_total  = int(total_seconds * sample_rate)
    audio    = np.random.normal(0.0, noise_rms, n_total).astype(np.float32)

    # Tiếng nói: sine đa tần mô phỏng formant giọng người (200 Hz + harmonics)
    s_start = int(speech_start_s * sample_rate)
    s_end   = int(speech_end_s   * sample_rate)
    n_sp    = s_end - s_start
    t_sp    = np.linspace(0.0, (s_end - s_start) / sample_rate, n_sp, endpoint=False)
    audio[s_start:s_end] += (
        0.07 * np.sin(2 * np.pi * 200  * t_sp) +
        0.04 * np.sin(2 * np.pi * 800  * t_sp) +
        0.02 * np.sin(2 * np.pi * 2400 * t_sp)
    ).astype(np.float32)

    # Xung phím cơ: nhiễu biên độ cao, cực ngắn
    c_start = int(click_time_s * sample_rate)
    c_len   = int(click_dur_ms / 1000.0 * sample_rate)
    audio[c_start: c_start + c_len] += np.random.uniform(-0.40, 0.40, c_len).astype(np.float32)

    return np.clip(audio, -1.0, 1.0)
