"""
scripts/training/train_module4_phobert_joint.py
================================================
TRAIN SCRIPT: PhoBERT-Joint Intent+Slot NLU cho Module 4
Theo đặc tả: Doc/module4_specification.md

Pipeline:
  1. Download vinai/phobert-base-v2 (base model)
  2. Xây dựng synthetic Vietnamese meeting dataset (~3000 mẫu)
  3. Fine-tune dual-head joint model (Intent + BIO Slot Tagging)
     - Loss: L_joint = 0.5 * L_intent + 0.5 * L_slot
  4. Export PyTorch → ONNX
  5. Quantize ONNX float32 → int8 (~115MB)
  6. Validate: latency ≤ 35ms trên CPU

Chạy:
  cd room_voice_ai
  python scripts/training/train_module4_phobert_joint.py
"""

from __future__ import annotations

import io
import json
import logging
import random
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
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent.parent  # room_voice_ai/
MODEL_DIR     = BASE_DIR / "models" / "phobert_nlu"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
ONNX_PATH     = BASE_DIR / "models" / "phobert_joint_meeting_nlu.onnx"
ONNX_INT8     = BASE_DIR / "models" / "phobert_joint_meeting_nlu_int8.onnx"
DATA_DIR      = BASE_DIR / "data" / "nlu_training"
CHECKPOINT    = MODEL_DIR / "checkpoint_joint.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Label spaces (theo module4_specification.md §6)
# ─────────────────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "Summarize_Meeting",
    "Query_Project_Knowledge",
    "Assign_Action_Item",
    "Schedule_Followup",
    "Explain_Technical_Concept",
    "Chitchat_Clarification",
]

SLOT_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-TIME", "I-TIME",
    "B-TASK", "I-TASK",
    "B-TOPIC", "I-TOPIC",
    "B-TOOL", "I-TOOL",
]

INTENT2ID = {lbl: i for i, lbl in enumerate(INTENT_LABELS)}
SLOT2ID   = {lbl: i for i, lbl in enumerate(SLOT_LABELS)}
ID2INTENT = {i: lbl for lbl, i in INTENT2ID.items()}
ID2SLOT   = {i: lbl for lbl, i in SLOT2ID.items()}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Synthetic Dataset Generation
# ─────────────────────────────────────────────────────────────────────────────

# Template database tiếng Việt cho từng Intent
TEMPLATES = {
    "Assign_Action_Item": [
        ("Giao cho {PER} {TASK} trước {TIME}", ["PER", "TASK", "TIME"]),
        ("Nhờ {PER} {TASK} vào {TIME}", ["PER", "TASK", "TIME"]),
        ("Anh {PER} phụ trách {TASK} hạn {TIME}", ["PER", "TASK", "TIME"]),
        ("{PER} ơi làm {TASK} trước {TIME} nhé", ["PER", "TASK", "TIME"]),
        ("Giao {PER} xử lý {TASK}", ["PER", "TASK"]),
        ("Task cho {PER}: {TASK}", ["PER", "TASK"]),
        ("Nhờ anh {PER} kiểm tra {TASK} ngay hôm nay", ["PER", "TASK"]),
        ("Bạn {PER} viết báo cáo {TASK} trước {TIME}", ["PER", "TASK", "TIME"]),
        ("Chị {PER} deploy {TASK} trước {TIME}", ["PER", "TASK", "TIME"]),
        ("Action item: {PER} phải hoàn thành {TASK} vào {TIME}", ["PER", "TASK", "TIME"]),
    ],
    "Summarize_Meeting": [
        ("Tóm tắt lại những gì đã bàn", []),
        ("Tổng hợp cuộc họp hôm nay", []),
        ("Cho tôi nghe kết luận của buổi họp", []),
        ("Liệt kê các điểm chính đã thống nhất", []),
        ("Recap lại nội dung họp từ đầu", []),
        ("Tóm tắt tiến độ dự án {TOPIC}", ["TOPIC"]),
        ("Kết luận cuộc thảo luận về {TOPIC}", ["TOPIC"]),
        ("Tổng hợp quyết định liên quan đến {TOPIC}", ["TOPIC"]),
        ("Tóm gọn những gì đã bàn về {TOPIC} hôm nay", ["TOPIC"]),
        ("Highlight các action items vừa được đề ra", []),
    ],
    "Query_Project_Knowledge": [
        ("Tìm tài liệu về {TOPIC}", ["TOPIC"]),
        ("Tra cứu thông tin liên quan đến {TOPIC}", ["TOPIC"]),
        ("Hồ sơ thiết kế {TOPIC} ở đâu", ["TOPIC"]),
        ("Cho tôi xem tài liệu kỹ thuật {TOPIC}", ["TOPIC"]),
        ("Lấy thông tin về {TOPIC} từ cơ sở tri thức", ["TOPIC"]),
        ("Kiến thức về {TOPIC} là gì", ["TOPIC"]),
        ("Có tài liệu nào về {TOPIC} không", ["TOPIC"]),
        ("Tra cứu {TOPIC} trong hệ thống", ["TOPIC"]),
        ("Tìm kiếm {TOPIC} trong kho tài liệu", ["TOPIC"]),
        ("Tài liệu về {TOPIC} đã được cập nhật chưa", ["TOPIC"]),
    ],
    "Schedule_Followup": [
        ("Lên lịch họp tiếp theo vào {TIME}", ["TIME"]),
        ("Đặt lịch cuộc họp {TIME}", ["TIME"]),
        ("Hẹn gặp lại vào {TIME}", ["TIME"]),
        ("Chốt lịch họp cho {TIME}", ["TIME"]),
        ("Thêm cuộc họp vào lịch ngày {TIME}", ["TIME"]),
        ("Book phòng họp lúc {TIME}", ["TIME"]),
        ("Nhắc nhở họp vào {TIME}", ["TIME"]),
        ("Tạo cuộc họp follow-up vào {TIME}", ["TIME"]),
        ("Lên lịch review sprint vào {TIME}", ["TIME"]),
        ("Meeting tiếp theo đặt vào {TIME} nhé", ["TIME"]),
    ],
    "Explain_Technical_Concept": [
        ("Giải thích {TOPIC} là gì", ["TOPIC"]),
        ("Kiến trúc {TOPIC} hoạt động thế nào", ["TOPIC"]),
        ("{TOPIC} là gì và dùng để làm gì", ["TOPIC"]),
        ("Cho tôi hiểu về {TOPIC}", ["TOPIC"]),
        ("Giải thích khái niệm {TOPIC} đơn giản", ["TOPIC"]),
        ("Tại sao chúng ta dùng {TOPIC}", ["TOPIC"]),
        ("So sánh {TOPIC} với các giải pháp khác", ["TOPIC"]),
        ("Cách implement {TOPIC} trong dự án", ["TOPIC"]),
        ("Ưu điểm của {TOPIC} là gì", ["TOPIC"]),
        ("Hướng dẫn sử dụng {TOPIC}", ["TOPIC"]),
    ],
    "Chitchat_Clarification": [
        ("Bạn có hiểu không", []),
        ("Ý tôi là gì bạn có nắm không", []),
        ("Đã rõ chưa", []),
        ("Mọi người đồng ý không", []),
        ("Ai có câu hỏi không", []),
        ("Tiếp theo chúng ta làm gì", []),
        ("Cảm ơn mọi người", []),
        ("Cuộc họp kết thúc", []),
        ("Không có vấn đề gì", []),
        ("Oke vậy nhé", []),
    ],
}

# Danh sách fill-in values cho slots
SLOT_VALUES = {
    "PER": [
        "Tuấn", "Minh", "Lan", "Hùng", "Phương", "Đức", "Thu",
        "Hải", "Quân", "Linh", "Nam", "Hoa", "Bình", "Dũng", "Trang",
        "Khoa", "Thành", "Cường", "Nga", "Hiếu",
    ],
    "TASK": [
        "kiểm tra lỗi API", "viết unit test", "review code", "deploy backend",
        "sửa bug login", "thiết kế database", "viết tài liệu", "fix lỗi frontend",
        "tối ưu query", "cập nhật README", "deploy staging", "merge pull request",
        "refactor module auth", "viết script migration", "kiểm tra bảo mật",
        "implement OAuth", "test API endpoint", "build docker image",
        "cấu hình CI/CD", "debug websocket",
    ],
    "TIME": [
        "5h chiều nay", "3h chiều nay", "trưa nay", "tối nay",
        "sáng mai", "chiều mai", "ngày mai", "thứ Hai tuần sau",
        "cuối tuần này", "trước 17h", "trước 12h", "15 phút nữa",
        "30 phút nữa", "1 tiếng nữa", "ngày 15", "tuần sau",
    ],
    "TOPIC": [
        "gRPC", "REST API", "WebSocket", "Docker", "Kubernetes",
        "microservices", "authentication", "JWT", "database schema",
        "CI/CD pipeline", "React frontend", "FastAPI backend", "Redis cache",
        "PostgreSQL", "MongoDB", "NGINX", "load balancing",
        "OAuth 2.0", "message queue", "Kafka",
    ],
    "TOOL": [
        "Jira", "Slack", "GitHub", "GitLab", "Confluence",
        "Notion", "Trello", "Zoom", "Teams", "Google Meet",
    ],
}


def fill_template(template: str, slot_types: list[str]) -> tuple[str, dict[str, str]]:
    """Điền giá trị vào template và trả về câu + slot dict."""
    filled = template
    slots: dict[str, str] = {}
    for slot_type in slot_types:
        value = random.choice(SLOT_VALUES.get(slot_type, ["unknown"]))
        placeholder = "{" + slot_type + "}"
        filled = filled.replace(placeholder, value, 1)
        slots[slot_type] = value
    return filled, slots


def text_to_bio_labels(
    text: str,
    slots: dict[str, str],
    tokenizer,
) -> tuple[list[str], list[str]]:
    """
    Chuyển text + slot dict → tokens + BIO labels.
    Sử dụng word-level tokenization đơn giản.
    """
    words = text.lower().split()
    labels = ["O"] * len(words)

    for slot_type, slot_value in slots.items():
        slot_words = slot_value.lower().split()
        # Tìm vị trí slot trong words
        for i in range(len(words) - len(slot_words) + 1):
            if words[i:i + len(slot_words)] == slot_words:
                labels[i] = f"B-{slot_type}"
                for j in range(1, len(slot_words)):
                    labels[i + j] = f"I-{slot_type}"
                break

    return words, labels


def generate_dataset(n_per_intent: int = 500) -> list[dict]:
    """Sinh synthetic dataset cho NLU training."""
    dataset = []

    for intent, templates in TEMPLATES.items():
        intent_id = INTENT2ID[intent]
        count = 0
        while count < n_per_intent:
            template, slot_types = random.choice(templates)
            text, slots = fill_template(template, slot_types)

            words = text.lower().split()
            bio_labels = ["O"] * len(words)

            for slot_type, slot_value in slots.items():
                slot_words = slot_value.lower().split()
                for i in range(len(words) - len(slot_words) + 1):
                    if words[i:i + len(slot_words)] == slot_words:
                        bio_labels[i] = f"B-{slot_type}"
                        for j in range(1, len(slot_words)):
                            if i + j < len(bio_labels):
                                bio_labels[i + j] = f"I-{slot_type}"
                        break

            dataset.append({
                "text": text,
                "intent": intent,
                "intent_id": intent_id,
                "words": words,
                "bio_labels": bio_labels,
                "slot_ids": [SLOT2ID.get(lbl, 0) for lbl in bio_labels],
                "slots": slots,
            })
            count += 1

    random.shuffle(dataset)
    logger.info(f"[Dataset] Generated {len(dataset)} samples across {len(INTENT_LABELS)} intents.")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: PhoBERT-Joint Model Definition
# ─────────────────────────────────────────────────────────────────────────────

def build_joint_model(num_intents: int = 6, num_slots: int = 11):
    """
    Xây dựng PhoBERT-Joint với Dual Task Heads.
    Theo spec module4_specification.md §2.2:
      - Shared Encoder: 12-layer RoBERTa (PhoBERT-base-v2)
      - Intent Head: Linear(768, 6) + Softmax trên [CLS]
      - Slot Head: Linear(768, 11) BIO sequence tagger
    """
    import transformers as _transformers
    from torch import nn

    transformers_mod: Any = _transformers
    AutoModel: Any = transformers_mod.AutoModel

    class PhoBERTJointModel(nn.Module):
        """PhoBERT-base-v2 + Dual Task Heads cho Joint NLU."""

        def __init__(self, model_name: str, num_intents: int, num_slots: int):
            super().__init__()
            # Shared encoder
            self.bert = AutoModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size  # 768 cho phobert-base-v2

            # Intent classification head – dùng [CLS] token
            self.intent_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_intents),
            )

            # Slot tagging head – dùng toàn bộ token sequence
            self.slot_head = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_slots),
            )

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            token_type_ids: Any | None = None,
        ):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            sequence_output = outputs.last_hidden_state  # (B, L, 768)
            cls_output      = sequence_output[:, 0, :]   # (B, 768)

            intent_logits = self.intent_head(cls_output)        # (B, 6)
            slot_logits   = self.slot_head(sequence_output)     # (B, L, 11)

            return intent_logits, slot_logits

    return PhoBERTJointModel


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def collate_batch(
    samples: list[dict],
    tokenizer,
    max_length: int = 64,
) -> dict:
    """Tạo batch tensors từ list samples."""
    import torch as _torch

    torch: Any = _torch
    all_input_ids    = []
    all_attn_mask    = []
    all_intent_ids   = []
    all_slot_ids     = []

    for sample in samples:
        text = sample["text"]
        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids   = enc["input_ids"][0]
        attn_mask   = enc["attention_mask"][0]

        # Align slot labels với tokenized length
        raw_slot_ids = sample["slot_ids"]
        seq_len = len(input_ids)
        # Pad/truncate slot labels: [CLS]=O, tokens..., [SEP]=O, PAD=O
        aligned_slots = [0]  # [CLS] = O
        for sid in raw_slot_ids[:max_length - 2]:
            aligned_slots.append(sid)
        aligned_slots.append(0)  # [SEP] = O
        # Pad
        while len(aligned_slots) < seq_len:
            aligned_slots.append(-100)  # ignore index
        aligned_slots = aligned_slots[:seq_len]

        all_input_ids.append(input_ids)
        all_attn_mask.append(attn_mask)
        all_intent_ids.append(sample["intent_id"])
        all_slot_ids.append(torch.tensor(aligned_slots, dtype=torch.long))

    return {
        "input_ids":    torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attn_mask),
        "intent_ids":   torch.tensor(all_intent_ids, dtype=torch.long),
        "slot_ids":     torch.stack(all_slot_ids),
    }


def train_joint_model(
    dataset: list[dict],
    model_name: str = "vinai/phobert-base-v2",
    num_epochs: int = 6,
    batch_size: int = 32,
    lr: float = 2e-5,
    alpha: float = 0.5,  # Joint loss weight (spec §3.4)
):
    """
    Fine-tune PhoBERT-Joint với Joint Multi-Task Loss.
    L_joint = α * L_intent + (1-α) * L_slot   (α=0.5 theo spec)
    """
    import torch as _torch
    import transformers as _transformers
    from torch import nn
    from torch.optim import AdamW

    torch: Any = _torch
    transformers_mod: Any = _transformers
    AutoTokenizer: Any = transformers_mod.AutoTokenizer
    device = torch.device("cpu")  # CPU-only theo spec Module 4
    logger.info(f"[Train] Device: {device}")

    # Load tokenizer
    logger.info(f"[Train] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Build model
    logger.info("[Train] Building PhoBERT-Joint model...")
    PhoBERTJoint = build_joint_model(
        num_intents=len(INTENT_LABELS),
        num_slots=len(SLOT_LABELS),
    )
    model = PhoBERTJoint(
        model_name=model_name,
        num_intents=len(INTENT_LABELS),
        num_slots=len(SLOT_LABELS),
    ).to(device)

    # Split train/val (90/10)
    split = int(len(dataset) * 0.9)
    train_data = dataset[:split]
    val_data   = dataset[split:]
    logger.info(f"[Train] Train: {len(train_data)} | Val: {len(val_data)}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Loss functions
    intent_loss_fn = nn.CrossEntropyLoss()
    slot_loss_fn   = nn.CrossEntropyLoss(ignore_index=-100)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        random.shuffle(train_data)

        for i in range(0, len(train_data), batch_size):
            batch_samples = train_data[i:i + batch_size]
            batch = collate_batch(batch_samples, tokenizer)

            input_ids    = batch["input_ids"].to(device)
            attn_mask    = batch["attention_mask"].to(device)
            intent_ids   = batch["intent_ids"].to(device)
            slot_ids     = batch["slot_ids"].to(device)

            # Forward pass
            intent_logits, slot_logits = model(input_ids, attn_mask)

            # Joint Multi-Task Loss (spec §3.4): L = α*L_intent + (1-α)*L_slot
            l_intent = intent_loss_fn(intent_logits, intent_ids)
            l_slot   = slot_loss_fn(
                slot_logits.view(-1, len(SLOT_LABELS)),
                slot_ids.view(-1),
            )
            loss = alpha * l_intent + (1 - alpha) * l_slot

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        model.eval()
        correct_intent = 0
        total = 0
        with torch.no_grad():
            for j in range(0, len(val_data), batch_size):
                vbatch = collate_batch(val_data[j:j + batch_size], tokenizer)
                vi_logits, _ = model(
                    vbatch["input_ids"].to(device),
                    vbatch["attention_mask"].to(device),
                )
                preds = vi_logits.argmax(dim=-1)
                correct_intent += (preds == vbatch["intent_ids"].to(device)).sum().item()
                total += len(vbatch["intent_ids"])

        val_acc = correct_intent / max(total, 1)
        avg_loss = epoch_loss / max(len(train_data) // batch_size, 1)
        logger.info(
            f"[Epoch {epoch+1:02d}/{num_epochs}] Loss={avg_loss:.4f} | "
            f"Val Intent Acc={val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT)
            logger.info(f"  → Saved best checkpoint (acc={best_val_acc:.4f})")

    logger.info(f"[Train] ✅ Training complete. Best Val Intent Acc: {best_val_acc:.4f}")

    # Load best model
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(TOKENIZER_DIR)
        logger.info(f"[Train] Tokenizer saved → {TOKENIZER_DIR}")

    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Export to ONNX
# ─────────────────────────────────────────────────────────────────────────────

def export_to_onnx(model, tokenizer, onnx_path: Path):
    """Export PyTorch model → ONNX."""
    import torch as _torch

    torch: Any = _torch
    model.eval()
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input
    dummy_text = "Giao cho anh Tuấn kiểm tra lỗi API"
    enc = tokenizer(dummy_text, return_tensors="pt", max_length=64,
                    padding="max_length", truncation=True)
    dummy_input_ids  = enc["input_ids"]
    dummy_attn_mask  = enc["attention_mask"]

    logger.info(f"[ONNX] Exporting model → {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attn_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["intent_logits", "slot_logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "intent_logits":  {0: "batch"},
                "slot_logits":    {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    logger.info(f"[ONNX] ✅ Exported → {onnx_path}")


def quantize_onnx_int8(onnx_fp32: Path, onnx_int8: Path):
    """Quantize ONNX float32 → int8 (~115MB target)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore

    logger.info(f"[ONNX-INT8] Quantizing {onnx_fp32.name} → {onnx_int8.name} ...")
    quantize_dynamic(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        weight_type=QuantType.QInt8,
    )
    size_mb = onnx_int8.stat().st_size / (1024 * 1024)
    logger.info(f"[ONNX-INT8] ✅ Quantized → {onnx_int8} ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Validate ONNX Inference
# ─────────────────────────────────────────────────────────────────────────────

def validate_onnx_inference(onnx_path: Path, tokenizer):
    """Kiểm tra ONNX inference latency và output correctness."""
    import onnxruntime as ort  # type: ignore

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    test_cases = [
        ("Giao cho anh Tuấn kiểm tra lỗi API trước 5h chiều", "Assign_Action_Item"),
        ("Tóm tắt cuộc họp hôm nay", "Summarize_Meeting"),
        ("Tìm tài liệu về gRPC trong hệ thống", "Query_Project_Knowledge"),
        ("Lên lịch họp vào chiều mai", "Schedule_Followup"),
        ("Giải thích WebSocket là gì", "Explain_Technical_Concept"),
        ("Mọi người có câu hỏi không", "Chitchat_Clarification"),
    ]

    logger.info("\n[Validate] === ONNX INFERENCE VALIDATION ===")
    latencies = []
    correct = 0

    for text, expected_intent in test_cases:
        enc = tokenizer(text, return_tensors="np", max_length=64,
                        padding="max_length", truncation=True)
        input_feed = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }

        t0 = time.perf_counter()
        raw_outs: Any = session.run(None, input_feed)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        intent_arr: np.ndarray = np.asarray(raw_outs[0])
        pred_intent_id = int(np.argmax(intent_arr[0]))
        pred_intent    = ID2INTENT[pred_intent_id]
        is_correct     = pred_intent == expected_intent
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        logger.info(
            f"  {status} [{latency_ms:5.1f}ms] '{text[:40]}...'\n"
            f"       Pred: {pred_intent} | Expected: {expected_intent}"
        )

    avg_lat = float(np.mean(latencies))
    acc = correct / len(test_cases)
    kpi_lat = avg_lat <= 35.0

    logger.info("\n[Validate] === KẾT QUẢ ===")
    logger.info(f"  Intent Accuracy : {acc:.2%} ({correct}/{len(test_cases)})")
    logger.info(f"  Avg Latency     : {avg_lat:.2f} ms  (KPI ≤ 35ms)")
    logger.info(f"  Latency KPI     : {'✅ ĐẠT' if kpi_lat else '❌ KHÔNG ĐẠT'}")

    # Copy int8 model to primary ONNX path
    import shutil
    if ONNX_INT8.exists():
        shutil.copy2(ONNX_INT8, ONNX_PATH)
        logger.info(f"\n[Deploy] ✅ Copied int8 model → {ONNX_PATH}")
    return kpi_lat, avg_lat, acc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 65)
    logger.info("  TRAIN MODULE 4: PhoBERT-Joint NLU")
    logger.info("  Theo đặc tả: Doc/module4_specification.md")
    logger.info("=" * 65)

    start_total = time.time()

    # STEP 1: Generate dataset
    logger.info("\n[Step 1] Tạo synthetic Vietnamese meeting dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(n_per_intent=500)  # 500 × 6 = 3,000 mẫu

    # Save dataset
    dataset_path = DATA_DIR / "meeting_nlu_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"[Step 1] ✅ Dataset saved → {dataset_path} ({len(dataset)} samples)")

    # STEP 2: Fine-tune or load checkpoint
    if CHECKPOINT.exists() and TOKENIZER_DIR.exists():
        logger.info(f"\n[Step 2] ✅ Đã tìm thấy checkpoint đã train hoàn tất 10 epochs → {CHECKPOINT}")
        logger.info("  Nạp checkpoint và tokenizer để tiến hành Export ONNX...")
        import torch as _torch
        import transformers as _transformers

        torch_mod: Any = _torch
        transformers_mod: Any = _transformers
        AutoTokenizer: Any = transformers_mod.AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=False)
        PhoBERTJoint = build_joint_model(num_intents=len(INTENT_LABELS), num_slots=len(SLOT_LABELS))
        model = PhoBERTJoint(model_name="vinai/phobert-base-v2", num_intents=len(INTENT_LABELS), num_slots=len(SLOT_LABELS))
        model.load_state_dict(torch_mod.load(CHECKPOINT, map_location="cpu"))
    else:
        logger.info("\n[Step 2] Fine-tuning PhoBERT-Joint (CPU, 10 epochs)...")
        logger.info("  Base model: vinai/phobert-base-v2 (downloading from HuggingFace...)")
        model, tokenizer = train_joint_model(
            dataset=dataset,
            model_name="vinai/phobert-base-v2",
            num_epochs=10,
            batch_size=16,
            lr=2e-5,
            alpha=0.5,  # Joint loss α = 0.5 theo spec §3.4
        )

    # STEP 3: Export ONNX float32
    onnx_fp32 = MODEL_DIR / "phobert_joint_fp32.onnx"
    if not onnx_fp32.exists():
        logger.info("\n[Step 3] Export ONNX float32...")
        export_to_onnx(model, tokenizer, onnx_fp32)
    else:
        logger.info(f"\n[Step 3] ✅ Đã tìm thấy ONNX float32 → {onnx_fp32}")

    # STEP 4: Quantize to int8
    logger.info("\n[Step 4] Quantize → ONNX int8 (~115MB)...")
    quantize_onnx_int8(onnx_fp32, ONNX_INT8)

    # STEP 5: Validate
    logger.info("\n[Step 5] Validate ONNX inference (latency KPI ≤ 35ms)...")
    kpi_ok, avg_lat, acc = validate_onnx_inference(ONNX_INT8, tokenizer)

    total_time = time.time() - start_total
    logger.info("\n" + "=" * 65)
    logger.info("  ✅ TRAIN MODULE 4 HOÀN TẤT")
    logger.info(f"  Model: {ONNX_PATH}")
    logger.info(f"  Intent Accuracy: {acc:.2%}")
    logger.info(f"  Avg Latency: {avg_lat:.2f} ms  ({'✅ ĐẠT KPI' if kpi_ok else '❌ KHÔNG ĐẠT KPI'})")
    logger.info(f"  Tổng thời gian: {total_time/60:.1f} phút")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
