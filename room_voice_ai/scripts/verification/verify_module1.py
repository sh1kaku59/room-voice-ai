"""
verify_module1.py
=================
Tập lệnh kiểm tra độc lập nhanh (Sanity & KPI Verification) cho Module 1:
Audio Preprocessing & Signal Enhancement.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Ensure root import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# pylint: disable=wrong-import-position
import io

from src.module1_audio.enhancer import (
    AGC_TARGET_DBFS,
    FRAME_SIZE,
    HANGOVER_FRAMES,
    HPF_CUTOFF_HZ,
    SAMPLE_RATE,
    VAD_THETA_END,
    VAD_THETA_START,
    Module1AudioPreprocessor,
)
from src.module1_audio.utils import (
    compute_rms_dbfs,
    generate_dummy_stream,
    split_into_frames,
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
    """Chạy toàn bộ các bước kiểm tra cho Module 1."""
    print(SEP)
    print("  MODULE 1 – VERIFICATION CHECK")
    print(SEP)

    # ── 1. Constants ────────────────────────────────────────────
    print("\n[1] CONSTANTS CHECK:")
    checks = [
        ("SAMPLE_RATE",     SAMPLE_RATE,     16000),
        ("FRAME_SIZE",      FRAME_SIZE,      512),
        ("HANGOVER_FRAMES", HANGOVER_FRAMES, 8),
        ("VAD_THETA_START", VAD_THETA_START, 0.50),
        ("VAD_THETA_END",   VAD_THETA_END,   0.35),
        ("AGC_TARGET_DBFS", AGC_TARGET_DBFS, -18.0),
        ("HPF_CUTOFF_HZ",   HPF_CUTOFF_HZ,   80.0),
    ]
    all_const_ok = True
    for name, val, expect in checks:
        chk_ok = abs(float(val) - float(expect)) < 1e-6
        print(f"  {status(chk_ok)}  {name} = {val}  (expect {expect})")
        if not chk_ok:
            all_const_ok = False

    # ── 2. Init ─────────────────────────────────────────────────
    print("\n[2] INIT CHECK (no model load):")
    proc = Module1AudioPreprocessor(session_id="verify")
    print("  OK    Module1AudioPreprocessor instantiated (fallback mode)")

    # ── 3. Hangover state machine ────────────────────────────────
    print("\n[3] HANGOVER STATE MACHINE:")
    proc.hangover_counter = 0
    proc.vad_is_open = False

    r0 = proc.vad_state_machine(0.05)
    print(f"  {status(r0 is False)}  P=0.05, counter=0  -> {r0}  (expect False)")

    r1 = proc.vad_state_machine(0.80)
    print(
        f"  {status(r1 is True)}   P=0.80 -> {r1}, "
        f"counter={proc.hangover_counter}  (expect True, counter=8)"
    )

    hangover_results = []
    for _ in range(HANGOVER_FRAMES):
        r = proc.vad_state_machine(0.10)
        hangover_results.append(r)
    all_hangover_true = all(hangover_results)
    print(f"  {status(all_hangover_true)}  8 hangover frames -> all True: {all_hangover_true}")

    r_close = proc.vad_state_machine(0.10)
    print(f"  {status(r_close is False)}  After hangover -> {r_close}  (expect False)")

    # ── 4. HPF + AGC ─────────────────────────────────────────────
    print("\n[4] HPF + AGC:")
    dc = np.full(FRAME_SIZE, 0.3, dtype=np.float32)
    hpf_out = proc.apply_hpf(dc)
    dc_mean = abs(float(np.mean(hpf_out)))
    hpf_ok = dc_mean < 0.05
    print(f"  {status(hpf_ok)}  HPF removes DC: mean_after={dc_mean:.5f}  (expect < 0.05)")

    # AGC test: input RMS trong vùng gain không bị cap [0.5, 8.0]
    agc_input = np.random.normal(0, 0.04, FRAME_SIZE * 10).astype(np.float32)
    agc_out = Module1AudioPreprocessor.apply_agc(agc_input, target_dbfs=-18.0)
    agc_rms = compute_rms_dbfs(agc_out)
    agc_ok = abs(agc_rms - (-18.0)) < 3.0
    print(
        f"  {status(agc_ok)}  AGC target -18dBFS: got {agc_rms:.1f} dBFS  (expect ~-18, tol ±3dB)"
    )

    # ── 5. process_stream_chunk ──────────────────────────────────
    print("\n[5] PROCESS_STREAM_CHUNK:")
    silent = np.zeros(FRAME_SIZE, dtype=np.float32)
    enh, sp_silent = proc.process_stream_chunk(silent)
    len_ok = len(enh) == FRAME_SIZE
    dtype_ok = enh.dtype == np.float32
    print(f"  {status(len_ok)}    Silent frame: len={len(enh)} (expect {FRAME_SIZE})")
    print(f"  {status(dtype_ok)}    dtype={enh.dtype}  (expect float32)")
    print(f"        is_speech={sp_silent}")

    t_arr = np.linspace(0, FRAME_SIZE / SAMPLE_RATE, FRAME_SIZE, endpoint=False)
    speech = (0.15 * np.sin(2 * np.pi * 300 * t_arr)).astype(np.float32)
    _, sp_speech = proc.process_stream_chunk(speech)
    print(f"        Speech frame: is_speech={sp_speech}")

    # ── 6. Dummy stream KPI ──────────────────────────────────────
    print("\n[6] DUMMY STREAM KPI (5 sec, ~156 frames):")
    audio = generate_dummy_stream(total_seconds=5.0)
    frames = split_into_frames(audio, FRAME_SIZE, pad_last=False)
    rng = np.random.default_rng(42)

    latencies: list[float] = []
    timeline: list[tuple[float, bool]] = []
    proc2 = Module1AudioPreprocessor(session_id="kpi")

    for i, frame in enumerate(frames):
        ref = rng.normal(0, 0.002, FRAME_SIZE).astype(np.float32)
        t0 = time.perf_counter()
        _, is_sp = proc2.process_stream_chunk(frame, far_end_ref=ref)
        latencies.append((time.perf_counter() - t0) * 1000)
        timeline.append((i * FRAME_SIZE / SAMPLE_RATE, is_sp))

    avg_lat = float(np.mean(latencies))
    max_lat = float(np.max(latencies))
    silence_frames  = [r[1] for r in timeline if r[0] < 1.5]
    speech_frames   = [r[1] for r in timeline if 1.5 <= r[0] < 3.5]
    hangover_frames_v = [r[1] for r in timeline if 3.5 <= r[0] < 3.76]

    kpi1 = avg_lat <= 5.0
    drop_rate = 1 - sum(silence_frames) / max(len(silence_frames), 1)
    kpi2 = drop_rate >= 0.90
    active_rate = sum(speech_frames) / max(len(speech_frames), 1)
    kpi3 = active_rate >= 0.80
    kpi4 = any(hangover_frames_v) if hangover_frames_v else True

    print(f"  {status(kpi1)}  KPI-1 Avg latency  : {avg_lat:.3f} ms  (KPI <= 5.0 ms)")
    print(f"        Max latency  : {max_lat:.2f} ms")
    print(f"  {status(kpi2)}  KPI-2 Silence drop : {drop_rate*100:.0f}%  (KPI >= 90%)")
    print(f"  {status(kpi3)}  KPI-3 Speech active: {active_rate*100:.0f}%  (KPI >= 80%)")
    print(f"  {status(kpi4)}  KPI-4 Hangover kept: {any(hangover_frames_v)}")

    print()
    print(SEP)
    all_pass = (
        all_const_ok and hpf_ok and agc_ok and len_ok and
        dtype_ok and kpi1 and kpi2 and kpi3 and kpi4
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
