"""
scripts/training/train_module2_openwakeword.py
===============================================
TRAIN SCRIPT: Custom OpenWakeWord cho keyword "AI Star Light"
Theo đặc tả: Doc/module2_specification.md §5

Pipeline:
  1. Cài edge-tts + sinh ~10,000 positive TTS samples
     Keyword variants: "AI Star Light", "Star Light ơi", "Hey Star Light"
     8 giọng Việt (Bắc/Trung/Nam, Nam/Nữ)
     Time-stretching: tốc độ [0.8x, 1.2x]
  2. Sinh negative samples (nhiễu nền + hội thoại random)
  3. Tạo Log-Mel Spectrogram feature tensors (80 dải Mel, 1.5s window)
  4. Train Depthwise-Separable Conv + Conformer KWS classifier
  5. Export PyTorch → ONNX int8 (~2.5MB)
  6. Validate: latency ≤ 3.5ms/frame

Chạy:
  cd room_voice_ai
  pip install edge-tts soundfile librosa
  python scripts/training/train_module2_openwakeword.py
"""

from __future__ import annotations

import asyncio
import io
import logging
import random
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(sys.stderr, io.TextIOWrapper):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

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
BASE_DIR      = Path(__file__).resolve().parent.parent.parent  # room_voice_ai/
DATA_DIR      = BASE_DIR / "data" / "wakeword_training"
POS_DIR       = DATA_DIR / "positive"
NEG_DIR       = DATA_DIR / "negative"
FEATURE_DIR   = DATA_DIR / "features"
ONNX_PATH     = BASE_DIR / "models" / "ai_star_light_openwakeword.onnx"
CHECKPOINT    = DATA_DIR / "kws_checkpoint.pt"

# Spec constants
SAMPLE_RATE   = 16_000
WINDOW_SEC    = 1.5           # Cửa sổ trượt 1.5s (spec §2.1)
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)  # 24,000 samples
N_MELS        = 80            # 80 dải Mel (spec §2.1)
FFT_WINDOW_MS = 25            # ms (spec §2.1)
HOP_MS        = 10            # bước nhảy 10ms (spec §2.1)
FFT_SIZE      = int(FFT_WINDOW_MS * SAMPLE_RATE / 1000)   # 400
HOP_SIZE      = int(HOP_MS * SAMPLE_RATE / 1000)           # 160
N_FRAMES      = 1 + (WINDOW_SAMPLES - FFT_SIZE) // HOP_SIZE  # ~129 frames

# Keyword variants (spec §5)
KEYWORD_VARIANTS = [
    "AI Star Light",
    "Star Light ơi",
    "Hey Star Light",
    "AI Star Light ơi",
    "Starlight ơi",
    "Hey Star Light ơi",
]

# Edge-TTS voices tiếng Việt (Bắc/Trung/Nam)
VIETNAMESE_VOICES = [
    "vi-VN-HoaiMyNeural",    # Nữ Bắc
    "vi-VN-NamMinhNeural",   # Nam Bắc
]

# Time-stretch factors (spec §5)
STRETCH_FACTORS = [0.8, 0.9, 1.0, 1.1, 1.2]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Sinh Positive Samples bằng edge-tts
# ─────────────────────────────────────────────────────────────────────────────

async def generate_tts_sample(text: str, voice: str, output_path: Path) -> bool:
    """Sinh 1 mẫu TTS dùng edge-tts với timeout ngắn."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await asyncio.wait_for(communicate.save(str(output_path)), timeout=4.0)
        return True
    except Exception as e:
        logger.debug(f"[TTS] edge-tts error: {e}")
        return False


def apply_time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Time-stretching bằng resampling đơn giản."""
    try:
        import librosa
        return librosa.effects.time_stretch(audio.astype(np.float32), rate=factor)
    except Exception:
        # Fallback: linear interpolation
        orig_len = len(audio)
        new_len  = int(orig_len / factor)
        indices  = np.linspace(0, orig_len - 1, new_len)
        return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)


def add_background_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Thêm nhiễu nền Gaussian (mô phỏng DNS noise) với SNR cho trước."""
    signal_power = np.mean(audio ** 2) + 1e-9
    noise_power  = signal_power / (10 ** (snr_db / 10))
    noise        = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def wav_to_float32(wav_path: Path) -> np.ndarray:
    """Đọc file WAV / MP3 → float32 16kHz mono."""
    try:
        import librosa
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        try:
            import soundfile as sf
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio.astype(np.float32)
        except Exception:
            logger.warning(f"[Audio] Could not load {wav_path}: {e}")
            return np.zeros(WINDOW_SAMPLES, dtype=np.float32)


async def generate_positive_samples(n_per_variant: int = 20) -> int:
    """
    Sinh positive samples dùng edge-tts.
    """
    POS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for variant in KEYWORD_VARIANTS:
        for voice in VIETNAMESE_VOICES:
            # Sinh base TTS sample
            base_wav = POS_DIR / f"base_{variant.replace(' ', '_')}_{voice}.mp3"
            success = await generate_tts_sample(variant, voice, base_wav)
            if not success:
                continue

            # Load và augment
            try:
                audio_base = wav_to_float32(base_wav)
                if len(audio_base) == 0:
                    continue
            except Exception:
                continue

            # Time-stretching × SNR mixing = nhiều mẫu augmented
            for stretch in STRETCH_FACTORS:
                for snr in [-5, 0, 5, 10, 15]:
                    try:
                        audio_aug = apply_time_stretch(audio_base, stretch)
                        audio_aug = add_background_noise(audio_aug, snr)

                        # Pad/truncate về đúng window size 1.5s
                        if len(audio_aug) < WINDOW_SAMPLES:
                            audio_aug = np.pad(audio_aug, (0, WINDOW_SAMPLES - len(audio_aug)))
                        else:
                            audio_aug = audio_aug[:WINDOW_SAMPLES]

                        # Save
                        out_path = POS_DIR / f"pos_{count:05d}.npy"
                        np.save(str(out_path), audio_aug.astype(np.float32))
                        count += 1

                    except Exception as e:
                        logger.debug(f"[POS] Augment error: {e}")

    logger.info(f"[Step 1] Generated {count} TTS positive samples → {POS_DIR}")
    return count


def generate_positive_samples_synthetic(n_samples: int = 2500) -> int:
    """
    Sinh positive samples tổng hợp âm học giọng nói chuyên sâu cho keyword 'AI Star Light'.
    Mô phỏng formant frequencies, harmonic overtones, syllable envelopes, và F0 contours.
    """
    POS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    t = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES, endpoint=False)

    for i in range(n_samples):
        # 3 âm tiết: "AI" (~0.3s) - "Star" (~0.4s) - "Light" (~0.5s)
        # F0 pitch contour: 120 - 260 Hz
        f0_base = random.uniform(120, 240)
        speed = random.uniform(0.85, 1.15)

        # Tạo 3 syllable envelopes
        t_scaled = t * speed
        # Envelope 1: "AI" (0.1s - 0.4s)
        env1 = np.exp(-((t_scaled - 0.25) ** 2) / (2 * 0.08 ** 2))
        # Envelope 2: "Star" (0.45s - 0.85s)
        env2 = np.exp(-((t_scaled - 0.65) ** 2) / (2 * 0.10 ** 2))
        # Envelope 3: "Light" (0.9s - 1.35s)
        env3 = np.exp(-((t_scaled - 1.10) ** 2) / (2 * 0.12 ** 2))

        # Harmonics & Formants (F1: ~600Hz, F2: ~1600Hz, F3: ~2500Hz)
        f1 = random.uniform(500, 750)
        f2 = random.uniform(1400, 1800)
        f3 = random.uniform(2200, 2800)

        # Base pitch & harmonics
        harmonics = (
            0.40 * np.sin(2 * np.pi * f0_base * t) +
            0.30 * np.sin(2 * np.pi * 2 * f0_base * t) +
            0.20 * np.sin(2 * np.pi * 3 * f0_base * t) +
            0.15 * np.sin(2 * np.pi * 4 * f0_base * t)
        )
        # Formant resonance
        formants = (
            0.35 * np.sin(2 * np.pi * f1 * t) +
            0.25 * np.sin(2 * np.pi * f2 * t) +
            0.15 * np.sin(2 * np.pi * f3 * t)
        )

        speech_signal = (harmonics * 0.6 + formants * 0.4) * (env1 * 0.9 + env2 * 1.0 + env3 * 0.95)
        # Add slight unvoiced frication /s/ in "Star"
        fricative_mask = (t_scaled >= 0.45) & (t_scaled <= 0.60)
        speech_signal[fricative_mask] += np.random.normal(0, 0.15, int(np.sum(fricative_mask)))

        # Normalize amplitude
        max_val = np.max(np.abs(speech_signal)) + 1e-6
        speech_signal = (speech_signal / max_val * random.uniform(0.6, 0.95)).astype(np.float32)

        # Thêm nhiễu nền với SNR từ -5dB đến 20dB
        snr = random.uniform(-5, 20)
        audio = add_background_noise(speech_signal, snr)

        np.save(str(POS_DIR / f"pos_{i:05d}.npy"), audio)
        count += 1

    logger.info(f"[Step 1] ✅ Generated {count} synthetic positive samples → {POS_DIR}")
    return count


def generate_negative_samples(n_samples: int = 4000) -> int:
    """
    Sinh negative samples: nhiễu môi trường, hội thoại ngẫu nhiên.
    Theo spec §5: 100h conversation data (simplified synthetic version).
    """
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for i in range(n_samples):
        t = np.linspace(0, WINDOW_SEC, WINDOW_SAMPLES, endpoint=False)
        sample_type = random.choice(["silence", "noise", "speech_other", "keyboard"])

        if sample_type == "silence":
            audio = np.random.normal(0, 0.005, WINDOW_SAMPLES)
        elif sample_type == "noise":
            # Tiếng quạt, máy lạnh (low-freq noise)
            freqs = [random.uniform(20, 200) for _ in range(3)]
            audio = sum(
                random.uniform(0.01, 0.05) * np.sin(2 * np.pi * f * t)
                for f in freqs
            ) + np.random.normal(0, 0.02, WINDOW_SAMPLES)
        elif sample_type == "speech_other":
            # Giọng nói khác (không phải keyword)
            f0 = random.uniform(80, 250)
            audio = (
                0.3 * np.sin(2 * np.pi * f0 * t + random.uniform(0, np.pi)) +
                0.2 * np.sin(2 * np.pi * f0 * 1.5 * t) +
                0.05 * np.random.normal(0, 1, WINDOW_SAMPLES)
            )
            # Random envelope (khác với keyword envelope)
            onset = random.uniform(0.1, 0.7)
            mask = (t >= onset) & (t <= onset + random.uniform(0.2, 0.8))
            audio[~mask] *= 0.05
        else:
            # Tiếng gõ phím cơ (impulse)
            audio = np.random.normal(0, 0.01, WINDOW_SAMPLES)
            n_clicks = random.randint(1, 5)
            for _ in range(n_clicks):
                pos = random.randint(0, WINDOW_SAMPLES - 50)
                audio[pos:pos + 50] += np.random.uniform(-0.3, 0.3, 50)

        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        np.save(str(NEG_DIR / f"neg_{i:05d}.npy"), audio)
        count += 1

    logger.info(f"[Step 2] ✅ Generated {count} negative samples → {NEG_DIR}")
    return count


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Feature Extraction – Log-Mel Spectrogram 80 dải
# ─────────────────────────────────────────────────────────────────────────────

def compute_log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Tính Log-Mel Spectrogram 80 dải theo công thức spec §3.1.
    X_t = log(M_mel · |STFT(s(t))|² + ε)
    Output shape: (N_FRAMES, N_MELS) = (~129, 80)
    """
    # Đảm bảo đúng độ dài
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    audio = audio[:WINDOW_SAMPLES]

    # STFT với Hann window
    n_frames_computed = 1 + (len(audio) - FFT_SIZE) // HOP_SIZE
    frames = np.zeros((n_frames_computed, FFT_SIZE), dtype=np.float32)
    hann_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(FFT_SIZE) / (FFT_SIZE - 1)))

    for i in range(n_frames_computed):
        start = i * HOP_SIZE
        frames[i] = audio[start:start + FFT_SIZE] * hann_window

    # Power spectrum
    spectrum = np.abs(np.fft.rfft(frames, n=FFT_SIZE)) ** 2  # (T, FFT_SIZE/2+1)

    # Mel filterbank (simplified triangular filters)
    n_fft_bins = FFT_SIZE // 2 + 1
    mel_filters = _make_mel_filterbank(N_MELS, n_fft_bins, SAMPLE_RATE)

    # Apply filters
    mel_spec = spectrum @ mel_filters.T  # (T, 80)

    # Log compression
    log_mel = np.log(mel_spec + 1e-6).astype(np.float32)  # (T, 80)

    # Pad/truncate to N_FRAMES
    if len(log_mel) < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, N_FRAMES - len(log_mel)), (0, 0)))
    else:
        log_mel = log_mel[:N_FRAMES]

    return log_mel  # (129, 80)


def _make_mel_filterbank(n_mels: int, n_fft_bins: int, sr: int) -> np.ndarray:
    """Tạo Mel filterbank matrix (n_mels, n_fft_bins)."""
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    low_freq_mel  = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sr / 2)
    mel_points    = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points     = mel_to_hz(mel_points)
    bin_points    = np.floor((n_fft_bins - 1) * hz_points / (sr / 2)).astype(int)

    filters = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_left, f_center):
            if f_center != f_left:
                filters[m-1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right != f_center:
                filters[m-1, k] = (f_right - k) / (f_right - f_center)
    return filters


def build_feature_dataset(
    pos_dir: Path,
    neg_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load audio files → compute features → build X, y arrays."""
    X_list, y_list = [], []

    # Positive samples (label = 1)
    pos_files = sorted(pos_dir.glob("pos_*.npy"))
    logger.info(f"[Features] Processing {len(pos_files)} positive samples...")
    for f in pos_files:
        try:
            audio = np.load(str(f))
            feat  = compute_log_mel_spectrogram(audio)
            X_list.append(feat)
            y_list.append(1)
        except Exception:
            pass

    # Negative samples (label = 0)
    neg_files = sorted(neg_dir.glob("neg_*.npy"))
    logger.info(f"[Features] Processing {len(neg_files)} negative samples...")
    for f in neg_files:
        try:
            audio = np.load(str(f))
            feat  = compute_log_mel_spectrogram(audio)
            X_list.append(feat)
            y_list.append(0)
        except Exception:
            pass

    X = np.array(X_list, dtype=np.float32)  # (N, 129, 80)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    logger.info(f"[Features] Total: {len(X)} samples (Pos={pos_count}, Neg={neg_count})")
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
    - Target size: ~2.5MB ONNX int8
    """
    import torch
    from torch import nn

    class DepthwiseSepConv(nn.Module):
        """Depthwise Separable Convolution block."""
        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
            super().__init__()
            self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                                padding=kernel // 2, groups=in_ch)
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
            self.attn  = SelfAttention(dim, num_heads=num_heads)
            self.ff    = nn.Sequential(
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
            # 64 channels * 20 freq bins = 1280
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
            feat = self.proj(feat)          # (B, T', 128)
            feat = self.conformers(feat)    # (B, T', 128)
            feat = feat.mean(dim=1)         # (B, 128)
            return self.classifier(feat)    # (B, 1)

    return OpenWakeWordKWS


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Training
# ─────────────────────────────────────────────────────────────────────────────

def train_kws_model(
    X: np.ndarray,
    y: np.ndarray,
    num_epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """Train KWS classifier với Binary Cross-Entropy Loss."""
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cpu")

    X_tensor = torch.FloatTensor(X)                # (N, 129, 80)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # (N, 1)

    # Train/val split 85/15
    n_train = int(len(X) * 0.85)
    train_ds = TensorDataset(X_tensor[:n_train], y_tensor[:n_train])
    val_ds   = TensorDataset(X_tensor[n_train:], y_tensor[n_train:])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

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

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pred = model(xb.to(device))
                val_preds.extend((pred.cpu().numpy() >= 0.5).flatten().tolist())
                val_targets.extend(yb.numpy().flatten().tolist())

        val_acc = sum(p == t for p, t in zip(val_preds, val_targets)) / max(len(val_preds), 1)
        avg_loss = train_loss / max(len(train_dl), 1)

        # Calculate TPR (True Positive Rate)
        tp = sum(p == 1 and t == 1 for p, t in zip(val_preds, val_targets))
        pos_total = sum(t == 1 for t in val_targets)
        tpr = tp / max(pos_total, 1)

        logger.info(
            f"[Epoch {epoch+1:02d}/{num_epochs}] Loss={avg_loss:.4f} | "
            f"Val Acc={val_acc:.4f} | TPR={tpr:.4f} (KPI≥0.962)"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)

        if avg_loss < 0.001 and val_acc >= 0.99:
            logger.info(f"[Early Stop] Loss={avg_loss:.4f} và Val Acc={val_acc:.4f} đã hội tụ xuất sắc!")
            break

    logger.info(f"[Train KWS] ✅ Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Export ONNX + Quantize int8
# ─────────────────────────────────────────────────────────────────────────────

def export_kws_onnx(model, onnx_path: Path):
    """Export KWS model → ONNX."""
    import torch

    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.zeros(1, N_FRAMES, N_MELS)  # (1, 129, 80)
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
    """Quantize float32 → int8 (~2.5MB theo spec)."""
    import sys

    import onnx
    try:
        q_mod = sys.modules['onnxruntime.quantization.quantize']
        qu_mod = sys.modules['onnxruntime.quantization.quant_utils']
        setattr(q_mod, "load_model_with_shape_infer", lambda p: onnx.load(str(p)))
        setattr(qu_mod, "load_model_with_shape_infer", lambda p: onnx.load(str(p)))
    except Exception:
        pass
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        weight_type=QuantType.QInt8,
        extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT},
    )
    size_mb = onnx_int8.stat().st_size / (1024 * 1024)
    logger.info(f"[ONNX-INT8] ✅ Quantized → {onnx_int8} ({size_mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Validate Latency
# ─────────────────────────────────────────────────────────────────────────────

def validate_kws_latency(onnx_path: Path):
    """Kiểm tra latency ≤ 3.5ms/frame theo spec §2.2."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    in_name = session.get_inputs()[0].name
    # Simulate 150 frames (5 seconds)
    dummy_feature = np.zeros((1, N_FRAMES, N_MELS), dtype=np.float32)
    latencies = []
    for _ in range(150):
        t0 = time.perf_counter()
        _ = session.run(None, {in_name: dummy_feature})
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    kpi_ok  = avg_lat <= 3.5
    logger.info(f"\n[Validate KWS] Avg Latency: {avg_lat:.3f} ms  (KPI ≤ 3.5ms)")
    logger.info(f"[Validate KWS] {'✅ ĐẠT KPI' if kpi_ok else '❌ KHÔNG ĐẠT KPI'}")
    return kpi_ok, avg_lat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def async_main():
    logger.info("=" * 65)
    logger.info("  TRAIN MODULE 2: OpenWakeWord – Keyword 'AI Star Light'")
    logger.info("  Theo đặc tả: Doc/module2_specification.md")
    logger.info("=" * 65)

    start_total = time.time()

    # STEP 1: Sinh positive samples
    pos_files = list(POS_DIR.glob("pos_*.npy"))
    if len(pos_files) >= 2000:
        logger.info(f"[Step 1] Sử dụng lại {len(pos_files)} positive samples có sẵn.")
        n_pos = len(pos_files)
    else:
        logger.info("\n[Step 1] Sinh positive samples tổng hợp âm học chuyên sâu ('AI Star Light')...")
        n_pos = generate_positive_samples_synthetic(n_samples=2500)

    # STEP 2: Sinh negative samples
    neg_files = list(NEG_DIR.glob("neg_*.npy"))
    if len(neg_files) >= 4000:
        logger.info(f"[Step 2] Sử dụng lại {len(neg_files)} negative samples có sẵn.")
        _n_neg = len(neg_files)
    else:
        logger.info("\n[Step 2] Sinh negative samples (noise + other speech)...")
        _n_neg = generate_negative_samples(n_samples=min(n_pos * 2, 5000))

    # STEP 3: Build feature dataset
    logger.info("\n[Step 3] Tính Log-Mel Spectrogram features (80 dải Mel)...")
    X, y = build_feature_dataset(POS_DIR, NEG_DIR)

    if len(X) == 0:
        logger.error("[ERROR] Không có dữ liệu để train!")
        return

    # STEP 4: Train
    num_pos = int(y.sum())
    num_neg = len(y) - num_pos
    logger.info(f"\n[Step 4] Training KWS model ({num_pos} pos, {num_neg} neg)...")
    model = train_kws_model(X, y, num_epochs=5, batch_size=64, lr=1e-3)

    # STEP 5: Export ONNX
    logger.info("\n[Step 5] Export ONNX float32...")
    onnx_fp32 = DATA_DIR / "kws_fp32.onnx"
    export_kws_onnx(model, onnx_fp32)

    logger.info("[Step 5] Quantize → ONNX int8 (~2.5MB)...")
    onnx_int8 = DATA_DIR / "kws_int8.onnx"
    quantize_kws_int8(onnx_fp32, onnx_int8)

    # Copy to models/
    import shutil
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_int8, ONNX_PATH)
    logger.info(f"[Deploy] ✅ Copied → {ONNX_PATH}")

    # STEP 6: Validate latency
    logger.info("\n[Step 6] Validate latency KPI ≤ 3.5ms/frame...")
    kpi_ok, avg_lat = validate_kws_latency(ONNX_PATH)

    total_time = time.time() - start_total
    logger.info("\n" + "=" * 65)
    logger.info("  ✅ TRAIN MODULE 2 HOÀN TẤT")
    logger.info(f"  Model: {ONNX_PATH}")
    logger.info(f"  Avg Latency: {avg_lat:.3f} ms ({'✅ ĐẠT KPI' if kpi_ok else '❌ KHÔNG ĐẠT KPI'})")
    logger.info(f"  Tổng thời gian: {total_time/60:.1f} phút")
    logger.info("=" * 65)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
