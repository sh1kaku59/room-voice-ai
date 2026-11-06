"""
scripts/setup_production_models.py
===================================
Kịch bản tự động kiểm tra, tải và xác thực trọng số Production cho Hệ thống RooM Voice AI (v1.1).
Bảo đảm tuân thủ nghiêm ngặt Ngân sách 4.0 GB VRAM trên GPU NVIDIA RTX 3050 Laptop:
  - Module 2 (KWS)  : models/ai_star_light_openwakeword.onnx          (CPU ONNX INT8, 0 MB VRAM)
  - Module 3 (ASR)  : models/phowhisper-small-ct2/model.bin           (GPU CTranslate2 FP16, ~950 MB VRAM)
  - Module 4 (NLU)  : models/phobert_joint_meeting_nlu.onnx           (CPU ONNX INT8, 0 MB VRAM)
  - Module 6 (RAG)  : BAAI/bge-m3                                     (CPU SentenceTransformer, 0 MB VRAM)
  - Module 6 (LLM)  : models/qwen2.5/Qwen2.5-3B-Instruct-Q4_K_M.gguf  (GPU llama.cpp GGUF Q4_K_M, ~2.2 GB VRAM)
  - Module 7 (TTS)  : models/kokoro_tts/kokoro-82m-vi.onnx            (CPU ONNX VITS/Kokoro, 0 MB VRAM)

Cách sử dụng:
  python room_voice_ai/scripts/setup_production_models.py --check
  python room_voice_ai/scripts/setup_production_models.py --download-llm
  python room_voice_ai/scripts/setup_production_models.py --download-bge
  python room_voice_ai/scripts/setup_production_models.py --download-tts
  python room_voice_ai/scripts/setup_production_models.py --all
"""

from __future__ import annotations

import argparse
import importlib
import io
import logging
import sys
from pathlib import Path
from typing import Any

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("setup_production_models")

# Kho lưu trữ chính thức trên Hugging Face cho Qwen2.5-3B-Instruct GGUF Q4_K_M
QWEN_GGUF_REPOS: list[tuple[str, str]] = [
    ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"),
    ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
]
BGE_M3_REPO_ID = "BAAI/bge-m3"
VI_TTS_REPO_ID = "rhasspy/piper-voices"
VI_TTS_ONNX_SUBPATH = "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
VI_TTS_JSON_SUBPATH = "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"


def _check_bge_cached() -> tuple[bool, float]:
    """Kiểm tra nhanh trọng số BAAI/bge-m3 đã lưu trong Hugging Face cache cục bộ chưa."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-m3" / "snapshots"
    if hf_cache.exists():
        for snap in hf_cache.iterdir():
            for weight_name in ("pytorch_model.bin", "model.safetensors"):
                w_file = snap / weight_name
                if w_file.exists() and w_file.stat().st_size > 100 * 1024 * 1024:
                    return True, round(w_file.stat().st_size / (1024 * 1024), 2)
    return False, 0.0


def check_model_inventory() -> dict[str, dict[str, Any]]:
    """Kiểm tra trạng thái hiện diện và kích thước của toàn bộ các mô hình trong dự án."""
    wakeword_path = Path(settings.WAKEWORD_MODEL_PATH)
    asr_bin_path = Path(settings.ASR_MODEL_PATH) / "model.bin"
    nlu_path = Path(settings.NLU_MODEL_PATH)
    nlu_vocab_path = Path(settings.NLU_TOKENIZER_PATH) / "vocab.txt"
    llm_gguf_path = Path(settings.LLM_MODEL_PATH) / settings.LLM_MODEL_FILE
    tts_onnx_path = Path(settings.TTS_MODEL_PATH)
    tts_json_path = Path(f"{settings.TTS_MODEL_PATH}.json")

    bge_cached, bge_size_mb = _check_bge_cached()

    inventory: dict[str, dict[str, Any]] = {
        "Module 2 (Wake-Word ONNX INT8)": {
            "path": str(wakeword_path),
            "exists": wakeword_path.exists() and wakeword_path.stat().st_size > 0,
            "size_mb": round(wakeword_path.stat().st_size / (1024 * 1024), 2) if wakeword_path.exists() else 0.0,
            "device": "CPU (0 MB VRAM)",
        },
        "Module 3 (PhoWhisper-small CTranslate2)": {
            "path": str(asr_bin_path),
            "exists": asr_bin_path.exists() and asr_bin_path.stat().st_size > 0,
            "size_mb": round(asr_bin_path.stat().st_size / (1024 * 1024), 2) if asr_bin_path.exists() else 0.0,
            "device": f"CUDA (~{settings.VRAM_ASR_MB} MB VRAM)",
        },
        "Module 4 (PhoBERT-Joint ONNX INT8)": {
            "path": str(nlu_path),
            "exists": nlu_path.exists() and nlu_vocab_path.exists(),
            "size_mb": round(nlu_path.stat().st_size / (1024 * 1024), 2) if nlu_path.exists() else 0.0,
            "device": "CPU (0 MB VRAM)",
        },
        "Module 6 (RAG Dense BAAI/bge-m3)": {
            "path": BGE_M3_REPO_ID,
            "exists": bge_cached,
            "size_mb": bge_size_mb,
            "device": "CPU Pinned (0 MB VRAM)",
        },
        "Module 6 (Qwen2.5-3B-Instruct GGUF Q4_K_M)": {
            "path": str(llm_gguf_path),
            "exists": llm_gguf_path.exists() and llm_gguf_path.stat().st_size > 100 * 1024 * 1024,
            "size_mb": round(llm_gguf_path.stat().st_size / (1024 * 1024), 2) if llm_gguf_path.exists() else 0.0,
            "device": f"CUDA (~2250 MB <= {settings.VRAM_LLM_MB} MB VRAM)",
        },
        "Module 7 (Vietnamese Neural TTS ONNX)": {
            "path": str(tts_onnx_path),
            "exists": tts_onnx_path.exists() and tts_onnx_path.stat().st_size > 1024 * 1024 and tts_json_path.exists(),
            "size_mb": round(tts_onnx_path.stat().st_size / (1024 * 1024), 2) if tts_onnx_path.exists() else 0.0,
            "device": "CPU (0 MB VRAM)",
        },
    }
    return inventory


def print_inventory_report(inventory: dict[str, dict[str, Any]]) -> None:
    """In bảng tổng hợp trạng thái trọng số và phân bổ VRAM."""
    print("=" * 82)
    print("  ROOM VOICE AI (v1.1) – PRODUCTION MODEL INVENTORY & VRAM BUDGET AUDIT")
    print("=" * 82)
    for name, info in inventory.items():
        status = "READY" if info["exists"] else "MISSING (Fallback Active)"
        print(f"  [{status:<25}] {name}")
        print(f"      Path   : {info['path']}")
        print(f"      Size   : {info['size_mb']:.2f} MB | Execution: {info['device']}")
    total_vram_mb = settings.VRAM_ASR_MB + 2250
    print("-" * 82)
    print(
        f"  Estimated Peak GPU VRAM : {total_vram_mb} MB / {int(settings.VRAM_BUDGET_GB * 1000)} MB "
        f"(RTX 3050 4GB Safe Margin: {int(settings.VRAM_BUDGET_GB * 1000) - total_vram_mb} MB)"
    )
    print("=" * 82)


def download_qwen_gguf(target_dir: Path | None = None) -> Path:
    """Tải trọng số Qwen2.5-3B-Instruct-Q4_K_M.gguf về thư mục models/qwen2.5/."""
    dest_dir = target_dir or Path(settings.LLM_MODEL_PATH)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / settings.LLM_MODEL_FILE

    if dest_file.exists() and dest_file.stat().st_size > 100 * 1024 * 1024:
        logger.info(f"[LLM] Đã tồn tại trọng số GGUF tại: {dest_file} ({dest_file.stat().st_size / 1e9:.2f} GB)")
        return dest_file

    hf_hub: Any = importlib.import_module("huggingface_hub")
    hf_hub_download: Any = hf_hub.hf_hub_download

    last_err: Exception | None = None
    for repo_id, filename in QWEN_GGUF_REPOS:
        try:
            logger.info(f"[LLM] Đang tải '{filename}' từ Hugging Face repo '{repo_id}'...")
            downloaded_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(dest_dir),
                )
            )
            if downloaded_path != dest_file and downloaded_path.exists():
                downloaded_path.replace(dest_file)
            logger.info(f"[LLM] Hoàn tất tải Qwen2.5 GGUF tại: {dest_file}")
            return dest_file
        except Exception as e:
            last_err = e
            logger.warning(f"[LLM] Không thể tải từ '{repo_id}/{filename}': {e}")

    raise RuntimeError(f"Không thể tải Qwen2.5 GGUF từ Hugging Face: {last_err}")


def download_bge_m3_cpu() -> bool:
    """Tải và khởi tạo bộ nhúng BAAI/bge-m3 ghim cứng trên CPU (0 MB VRAM)."""
    logger.info(f"[RAG] Đang tải và xác thực '{BGE_M3_REPO_ID}' trên thiết bị CPU (0 MB VRAM)...")
    st_mod: Any = importlib.import_module("sentence_transformers")
    SentenceTransformer: Any = st_mod.SentenceTransformer

    model = SentenceTransformer(BGE_M3_REPO_ID, device="cpu")
    test_vec = model.encode(["Kiến trúc hệ thống RooM Voice AI"], normalize_embeddings=True)
    logger.info(f"[RAG] Nạp thành công '{BGE_M3_REPO_ID}' trên CPU! Vector shape = {test_vec.shape}")
    return True


def download_tts_onnx(target_onnx_path: Path | None = None) -> Path:
    """Tải trọng số Vietnamese Neural TTS VITS/Kokoro ONNX + JSON phoneme map về models/kokoro_tts/."""
    dest_onnx = target_onnx_path or Path(settings.TTS_MODEL_PATH)
    dest_json = Path(f"{dest_onnx}.json")
    dest_dir = dest_onnx.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    if dest_onnx.exists() and dest_onnx.stat().st_size > 1024 * 1024 and dest_json.exists():
        logger.info(f"[TTS] Đã tồn tại trọng số Neural TTS ONNX tại: {dest_onnx} ({dest_onnx.stat().st_size / 1e6:.2f} MB)")
        return dest_onnx

    hf_hub: Any = importlib.import_module("huggingface_hub")
    hf_hub_download: Any = hf_hub.hf_hub_download

    logger.info(f"[TTS] Đang tải Vietnamese Neural TTS ONNX từ '{VI_TTS_REPO_ID}'...")
    raw_onnx = Path(
        hf_hub_download(
            repo_id=VI_TTS_REPO_ID,
            filename=VI_TTS_ONNX_SUBPATH,
            local_dir=str(dest_dir),
        )
    )
    raw_json = Path(
        hf_hub_download(
            repo_id=VI_TTS_REPO_ID,
            filename=VI_TTS_JSON_SUBPATH,
            local_dir=str(dest_dir),
        )
    )

    if raw_onnx != dest_onnx and raw_onnx.exists():
        raw_onnx.replace(dest_onnx)
    if raw_json != dest_json and raw_json.exists():
        raw_json.replace(dest_json)

    # Dọn dẹp thư mục con tạm do hf_hub_download tạo ra
    vi_subdir = dest_dir / "vi"
    if vi_subdir.exists():
        import shutil

        shutil.rmtree(vi_subdir, ignore_errors=True)

    logger.info(f"[TTS] Hoàn tất tải Vietnamese Neural TTS ONNX tại: {dest_onnx} ({dest_onnx.stat().st_size / 1e6:.2f} MB)")
    return dest_onnx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Công cụ kiểm tra và tải trọng số Production cho RooM Voice AI (v1.1)."
    )
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra trạng thái các model hiện có.")
    parser.add_argument("--download-llm", action="store_true", help="Tải Qwen2.5-3B-Instruct-Q4_K_M.gguf.")
    parser.add_argument("--download-bge", action="store_true", help="Tải BAAI/bge-m3 và ghim cache trên CPU.")
    parser.add_argument("--download-tts", action="store_true", help="Tải Vietnamese Neural TTS ONNX trên CPU.")
    parser.add_argument("--all", action="store_true", help="Tải toàn bộ các trọng số Production còn thiếu.")
    args = parser.parse_args()

    if args.download_llm or args.all:
        download_qwen_gguf()

    if args.download_bge or args.all:
        download_bge_m3_cpu()

    if args.download_tts or args.all:
        download_tts_onnx()

    inventory = check_model_inventory()
    print_inventory_report(inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

