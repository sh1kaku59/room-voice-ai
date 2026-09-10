"""
verify_module2.py
=================
Tập lệnh kiểm tra độc lập nhanh (Sanity & KPI Verification) cho Module 2:
Wake-word Detection (Call Name Engine).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# Đảm bảo đường dẫn import gốc
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position,protected-access,duplicate-code
import io

from src.module2_wakeword.detector import (
    BUFFER_SIZE,
    COOLDOWN_FRAMES,
    COOLDOWN_SECONDS,
    FRAME_SIZE,
    SAMPLE_RATE,
    WAKEWORD_THRESHOLD,
    WINDOW_DURATION,
    Module2WakeWordDetector,
)

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SEP = "=" * 55


def status(is_ok: bool) -> str:
    """Trả về nhãn trạng thái OK hoặc FAIL."""
    return "OK  " if is_ok else "FAIL"


def run_verification() -> bool:  # pylint: disable=too-many-locals,too-many-statements
    """Chạy toàn bộ các bước kiểm tra cho Module 2."""
    print(SEP)
    print("  MODULE 2 – VERIFICATION CHECK (WAKE-WORD DETECTOR)")
    print(SEP)

    # ── 1. Constants ────────────────────────────────────────────
    print("\n[1] CONSTANTS CHECK:")
    checks = [
        ("SAMPLE_RATE",        SAMPLE_RATE,        16000),
        ("FRAME_SIZE",         FRAME_SIZE,         512),
        ("WINDOW_DURATION",    WINDOW_DURATION,    1.5),
        ("BUFFER_SIZE",        BUFFER_SIZE,        24000),
        ("WAKEWORD_THRESHOLD", WAKEWORD_THRESHOLD, 0.75),
        ("COOLDOWN_SECONDS",   COOLDOWN_SECONDS,   2.0),
        ("COOLDOWN_FRAMES",    COOLDOWN_FRAMES,    62),
    ]
    all_const_ok = True
    for name, val, expect in checks:
        chk_ok = abs(float(val) - float(expect)) < 1e-6
        print(f"  {status(chk_ok)}  {name} = {val}  (expect {expect})")
        if not chk_ok:
            all_const_ok = False

    # ── 2. Init & RingBuffer Size ───────────────────────────────
    print("\n[2] INIT & RINGBUFFER SIZE:")
    detector = Module2WakeWordDetector(session_id="verify_mod2")
    buf_len = len(detector.audio_buffer)
    buf_ok = buf_len == 24000
    cd_init_ok = detector.cooldown_counter == 0
    print(f"  {status(buf_ok)}  Circular RingBuffer size = {buf_len} samples (expect 24000 / 1.5s)")
    print(f"  {status(cd_init_ok)}  Initial cooldown = {detector.cooldown_counter} (expect 0)")

    # ── 3. Circular Buffer FIFO Sliding ─────────────────────────
    print("\n[3] CIRCULAR BUFFER FIFO SLIDING:")
    detector.reset_audio_buffer()
    ones_frame = np.ones(FRAME_SIZE, dtype=np.float32)
    detector.process_speech_frame(ones_frame)
    buf_list = list(detector.audio_buffer)
    last_samples_ok = all(s == 1.0 for s in buf_list[-FRAME_SIZE:])
    first_samples_ok = all(s == 0.0 for s in buf_list[:FRAME_SIZE])
    fifo_ok = last_samples_ok and first_samples_ok and len(buf_list) == 24000
    print(f"  {status(fifo_ok)}  FIFO push 512 samples: last 512 are 1.0, first are 0.0")

    # ── 4. Temporal Smoothing Spike Rejection ───────────────────
    print("\n[4] TEMPORAL SMOOTHING SPIKE REJECTION:")
    det_spike = Module2WakeWordDetector(session_id="spike_test")
    frame_idx = 0

    def mock_spike_inf(_feat: np.ndarray, _aud: np.ndarray) -> float:
        nonlocal frame_idx
        prob = 1.0 if frame_idx == 50 else 0.05
        frame_idx += 1
        return prob

    det_spike._run_onnx_inference = mock_spike_inf  # type: ignore

    spike_triggered = False
    dummy_f = np.random.uniform(-0.01, 0.01, FRAME_SIZE).astype(np.float32)
    for _ in range(100):
        trig, _ = det_spike.process_speech_frame(dummy_f)
        if trig:
            spike_triggered = True
            break
    spike_ok = not spike_triggered
    print(f"  {status(spike_ok)}  Single spike (prob=1.0 at frame 50) rejected (expect False)")

    # ── 5. Hysteresis & Cooldown Lockout (2.0s / 62 frames) ─────
    print("\n[5] HYSTERESIS & COOLDOWN LOCKOUT:")
    det_cd = Module2WakeWordDetector(session_id="cd_test")
    det_cd._run_onnx_inference = MagicMock(return_value=0.95)  # type: ignore

    activated = False
    for _ in range(20):
        trig, _ = det_cd.process_speech_frame(dummy_f)
        if trig:
            activated = True
            break

    cd_set_ok = det_cd.cooldown_counter == 62
    cd_val_after_trig = det_cd.cooldown_counter
    trig_next, _ = det_cd.process_speech_frame(dummy_f)
    lockout_ok = (trig_next is False) and (det_cd.cooldown_counter == 61)
    cd_test_ok = activated and cd_set_ok and lockout_ok
    print(f"  {status(activated)}  Wake-word activated on high prob stream (0.95)")
    print(f"  {status(cd_set_ok)}  Cooldown set to {cd_val_after_trig} frames (expect 62)")
    print(f"  {status(lockout_ok)}  Next frame locked out (triggered=False, counter=61)")

    # ── 6. Dummy Stream KPI Performance ─────────────────────────
    print("\n[6] DUMMY STREAM KPI PERFORMANCE (5 sec, 150 frames):")
    det_kpi = Module2WakeWordDetector(session_id="kpi_mod2")
    latencies: list[float] = []

    for _ in range(150):
        t0 = time.perf_counter()
        det_kpi.process_speech_frame(dummy_f)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))
    kpi_lat = avg_lat <= 3.5

    print(f"  {status(kpi_lat)}  KPI Avg latency  : {avg_lat:.3f} ms  (KPI <= 3.5 ms)")
    print(f"        Max latency  : {max_lat:.3f} ms")

    print()
    print(SEP)
    all_pass = (
        all_const_ok and buf_ok and cd_init_ok and fifo_ok and
        spike_ok and cd_test_ok and kpi_lat
    )
    result_str = (
        "ALL PASS - READY TO INTEGRATE" if all_pass
        else "SOME CHECKS FAILED - REVIEW NEEDED"
    )
    print(f"  RESULT: {result_str}")
    print(SEP)
    return all_pass


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
