"""
configs/settings.py
===================
Toàn bộ cấu hình hệ thống RooM Voice AI.
Bao gồm: VRAM budget, đường dẫn model, ngưỡng threshold, API keys.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Đường dẫn gốc dự án
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# VRAM / Tài nguyên phần cứng
# ---------------------------------------------------------------------------
# Phần cứng thực nghiệm: Intel Core i7-12650H | RTX 3050 Laptop 4GB | RAM 8GB
# Chiến lược: Hybrid Edge (CPU-first), GPU chỉ dành cho ASR + LLM
VRAM_BUDGET_GB = 4.0          # RTX 3050 Laptop tổng VRAM 4GB
VRAM_ASR_MB    = 950           # Module 3 ASR chiếm tối đa ~950 MB
VRAM_LLM_MB    = 2500          # Module 6 LLM chiếm tối đa ~2.5 GB
CPU_THREADS    = 8             # i7-12650H có 10 cores, để 8 cho pipeline

# ---------------------------------------------------------------------------
# Module 1 – Audio Preprocessing & Signal Enhancement (CPU-only, 0 MB VRAM)
# Đặc tả: module1_specification.md, Mục 6
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE      = 16_000  # Hz (Mono Float32 PCM – tương thích Whisper)
AUDIO_FRAME_SIZE       = 512     # samples = 32 ms @ 16 kHz (chuẩn Silero + DeepFilterNet3)
AUDIO_CHUNK_MS         = 32      # ms mỗi frame (512 / 16000 * 1000)
AUDIO_CHANNELS         = 1       # Mono
DEEPFILTER_MODEL_PATH  = str(BASE_DIR / "models" / "deepfilternet" / "DeepFilterNet3.onnx")
# VAD Hysteresis thresholds (Mục 3.3 – chống nuốt âm /t/, /p/, /k/ tiếng Việt)
VAD_THETA_START        = 0.50    # Ngưỡng mở cổng (P_start)
VAD_THETA_END          = 0.35    # Ngưỡng đóng cổng (P_end)
VAD_HANGOVER_FRAMES    = 8       # 8 × 32ms = 256ms – Hangover Buffer
AGC_TARGET_DBFS        = -18.0   # Mức RMS mục tiêu sau AGC (dBFS)
HPF_CUTOFF_HZ          = 80.0    # High-pass filter cutoff (Hz)

# ---------------------------------------------------------------------------
# Module 2 – Wake-word Detector (CPU, ONNX int8, <3.5ms/frame)
# Đặc tả: blueprint Mục 6, TPR ≥ 96.2%, FAR < 0.5/h
# ---------------------------------------------------------------------------
WAKEWORD_MODEL_PATH    = str(BASE_DIR / "models" / "openwakeword" / "hey_room.onnx")
WAKEWORD_THRESHOLD     = 0.75    # θ_wake ≥ 0.75 theo blueprint Mục 4.1
WAKEWORD_COOLDOWN_S    = 2.0     # Refractory Cooldown Lockout sau kích hoạt (giây)
WAKEWORD_HYSTERESIS_MS = 300     # Cửa sổ làm mượt Hysteresis (ms)

# ---------------------------------------------------------------------------
# Module 3 – Streaming ASR (GPU, ~900MB VRAM)
# Model: PhoWhisper-small weights qua Faster-Whisper / CTranslate2
# Đặc tả: blueprint Mục 6 – WER ≤ 9.5%, VRAM ~850-950 MB
# ---------------------------------------------------------------------------
ASR_MODEL_SIZE         = "small"      # PhoWhisper-small (không dùng large-v3 – quá VRAM)
ASR_MODEL_PATH         = str(BASE_DIR / "models" / "faster_whisper" / "phowhisper-small")
ASR_LANGUAGE           = "vi"         # Tiếng Việt
ASR_COMPUTE_TYPE       = "int8_float16"  # CTranslate2 tối ưu CUDA Tensor Cores
ASR_BEAM_SIZE          = 5
ASR_VAD_FILTER         = True
ASR_CHUNK_LENGTH_S     = 30           # Độ dài tối đa mỗi đoạn (giây)

# ---------------------------------------------------------------------------
# Module 4 – NLU (PhoBERT-Joint ONNX)
# ---------------------------------------------------------------------------
NLU_MODEL_PATH         = str(BASE_DIR / "models" / "phobert_nlu" / "model.onnx")
NLU_TOKENIZER_PATH     = str(BASE_DIR / "models" / "phobert_nlu" / "tokenizer")
NLU_INTENT_THRESHOLD   = 0.6    # Ngưỡng tin cậy nhận dạng intent
NLU_ENTITY_THRESHOLD   = 0.5    # Ngưỡng tin cậy nhận dạng entity

# ---------------------------------------------------------------------------
# Module 5 – Context Manager (MemGPT-style Hierarchical Memory, CPU, 0 MB VRAM)
# Đặc tả: module5_specification.md, Mục 2, 3, 6
# ---------------------------------------------------------------------------
CONTEXT_MAX_WORKING_TURNS       = 12     # Giới hạn số lượt thoại ngắn hạn (FIFO: 10-15 turns)
CONTEXT_MAX_TOKEN_BUDGET        = 3000   # Token budget tối đa (Knapsack constraint B_max = 3000)
CONTEXT_MEMORY_PRESSURE_RATIO   = 0.70   # Ngưỡng kích hoạt Eviction (>= 70% capacity)
CONTEXT_TOPIC_SHIFT_THRESHOLD   = 0.42   # Ngưỡng phát hiện chuyển đổi chủ đề (Cosine sim < 0.42)
CONTEXT_EXPONENTIAL_DECAY_GAMMA = 0.2    # Hệ số suy giảm mũ theo thời gian gamma = 0.2
CONTEXT_HYBRID_ALPHA            = 0.5    # Trọng số tương đồng ngữ nghĩa (alpha = 0.5)
CONTEXT_HYBRID_BETA             = 0.3    # Trọng số có mặt thực thể (beta = 0.3)
CONTEXT_HYBRID_DELTA            = 0.2    # Trọng số độ tươi thời gian (delta = 0.2)
CONTEXT_WORKING_TOKENS          = 3000   # Alias tương thích
CONTEXT_ROLLING_TURNS           = 12     # Alias tương thích
CONTEXT_ARCHIVAL_PATH           = str(BASE_DIR / "data" / "archival_memory.db")

# ---------------------------------------------------------------------------
# Module 6 – Brain / LLM (GPU, ~2.2 GB VRAM)
# Model: Qwen2.5-3B-Instruct GGUF int4 qua llama-cpp-python
# Đặc tả: blueprint Mục 6 – VMLU 62.5pts, TTFT ≤ 350ms trên RTX 3050
# ---------------------------------------------------------------------------
LLM_MODEL_PATH          = str(BASE_DIR / "models" / "qwen2.5")
LLM_MODEL_NAME          = "Qwen2.5-3B-Instruct"    # KHÔNG dùng 7B – vượt VRAM 4GB
LLM_MODEL_FILE          = "Qwen2.5-3B-Instruct-Q4_K_M.gguf"  # GGUF int4 ~2.2 GB
LLM_MAX_NEW_TOKENS      = 512
LLM_TEMPERATURE         = 0.7
LLM_TOP_P               = 0.9
LLM_GPU_LAYERS          = -1     # -1 = toàn bộ layer lên GPU (CUDA)
LLM_GRAMMAR_MASKING     = True   # Enforce JSON Function Calling output format
LLM_CONTEXT_WINDOW      = 4096   # Context window Qwen2.5-3B

# RAG Engine (Dense BGE-M3 + Sparse BM25 + RRF)
RAG_VECTOR_DB_PATH      = str(BASE_DIR / "data" / "vector_store")
RAG_DENSE_MODEL         = "BAAI/bge-m3"   # Dense embedding (blueprint Mục 6)
RAG_SPARSE_MODEL        = "BM25"          # Okapi BM25
RAG_TOP_K               = 5              # K trong Hit Rate@K=3 KPI

# ---------------------------------------------------------------------------
# Module 7 – Output Generator (CPU, 0 MB VRAM)
# TTS: Kokoro-82M Vietnamese ONNX + Punctuation Splitter
# Đặc tả: blueprint Mục 6 – MOS ≥ 4.18, TTFS < 100ms trên CPU
# ---------------------------------------------------------------------------
TTS_MODEL_PATH          = str(BASE_DIR / "models" / "kokoro_tts" / "kokoro-82m-vi.onnx")
TTS_VOICE               = "vi_female_1"   # Giọng đọc chuẩn 3 miền
TTS_SPEED               = 1.0
TTS_PUNCTUATION_SPLIT   = True  # Punctuation Splitter: tổng hợp từng vế câu để giảm TTFS
WEBSOCKET_HOST          = "0.0.0.0"
WEBSOCKET_PORT          = 8765

# ---------------------------------------------------------------------------
# API Keys (nạp từ biến môi trường để bảo mật)
# ---------------------------------------------------------------------------
OPENAI_API_KEY          = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY          = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL               = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_DIR                 = str(BASE_DIR / "logs")
