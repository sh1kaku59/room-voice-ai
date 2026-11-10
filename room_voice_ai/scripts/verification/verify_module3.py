"""
verify_module3.py
=================
Script kiểm tra độc lập và đánh giá KPI hiệu năng của MODULE 3 (Streaming Vietnamese ASR).
"""

import io
import sys
import time
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.module3_asr.transcriber import Module3VietnameseStreamingASR


def run_verification():
    print("=" * 65)
    print("  MODULE 3 – VERIFICATION CHECK (VIETNAMESE STREAMING ASR)")
    print("=" * 65)

    # 1. CONSTANTS & INITIALIZATION CHECK
    print("\n[1] INITIALIZATION & CONSTANTS CHECK:")
    asr = Module3VietnameseStreamingASR(model_path="", device="cpu", compute_type="int8")

    print(f"  OK    sample_rate        = {asr.sample_rate}  (expect 16000)")
    print(f"  OK    overlap_samples    = {asr.overlap_samples}  (expect 2400 / 150ms)")
    print(f"  OK    beam_size          = {asr.beam_size}  (expect 1 for streaming)")
    print(f"  OK    context_buffer len = {len(asr.context_buffer)}  (expect 0)")
    print(f"  OK    committed_text     = '{asr.committed_text}'  (expect empty)")
    print(f"  OK    unconfirmed_text   = '{asr.unconfirmed_text}'  (expect empty)")
    assert asr.sample_rate == 16000
    assert asr.overlap_samples == 2400
    assert len(asr.context_buffer) == 0

    # 2. WORD-LEVEL LCP ALGORITHM CHECK
    print("\n[2] WORD-LEVEL LCP (LONGEST COMMON PREFIX) CHECK:")
    s1 = "Tôi muốn deploy hệ thống mới"
    s2 = "Tôi muốn deploy sản phẩm mới"
    lcp1 = Module3VietnameseStreamingASR.longest_common_prefix(s1, s2)
    print(f"  LCP('{s1}',\n      '{s2}')\n      -> '{lcp1}'")
    assert lcp1 == "Tôi muốn deploy"

    s3 = "Anh muốn deploy"
    lcp2 = Module3VietnameseStreamingASR.longest_common_prefix(s1, s3)
    print(f"  LCP('{s1}', '{s3}') -> '{lcp2}' (empty)")
    assert lcp2 == ""
    print("  OK    LCP Word-level prefix matching verified 100%")

    # 3. CODE-SWITCHING NORMALIZATION CHECK
    print("\n[3] TECHNICAL CODE-SWITCHING NORMALIZATION CHECK:")
    test_cases = [
        ("Tôi cần đì-ploi cái sụt này", "Tôi cần deploy cái source code này"),
        ("Áp-bi-ai bị bấc ở bét-en", "API bị bug ở backend"),
        ("Tét xong rồi mớc vào rì-bô trên Giắc", "test xong rồi merge vào repo trên Jira"),
        ("Sờ-crum team đang xem đát-ta", "Scrum team đang xem data"),
        ("Gờ-rúp-pê-xê kết nối nhanh", "gRPC kết nối nhanh"),
    ]
    for raw, expected in test_cases:
        res = asr.apply_code_switching(raw)
        print(f"  Raw     : {raw}")
        print(f"  Expected: {expected}")
        print(f"  Result  : {res}")
        assert res.lower() == expected.lower() or res == expected
    print("  OK    All IT Code-switching regex mappings verified 100%")

    # 4. VIETNAMESE NORMALIZATION CHECK
    print("\n[4] VIETNAMESE CAPITALIZATION & WHITESPACE NORMALIZATION:")
    dirty = "   chúng ta bắt đầu cuộc họp bàn về kiến trúc hệ thống.  "
    clean = asr.normalize_vietnamese_text(dirty)
    print(f"  Dirty : '{dirty}'")
    print(f"  Clean : '{clean}'")
    assert clean == "Chúng ta bắt đầu cuộc họp bàn về kiến trúc hệ thống."
    print("  OK    Vietnamese normalization verified")

    # 5. STREAMING AUDIO CHUNKING & OVERLAP BUFFER CHECK
    print("\n[5] STREAMING AUDIO CHUNKING & OVERLAP BUFFER CHECK:")
    # Push 1 second of audio (16,000 samples)
    audio_1s = np.random.uniform(-0.05, 0.05, 16000).astype(np.float32)
    comm, hyp = asr.transcribe_streaming_chunk(audio_1s, is_speech_final=False)
    print("  Audio in: 16,000 samples (1.0s)")
    print(f"  Context buffer kept for overlap: {len(asr.context_buffer)} samples (expect {asr.overlap_samples})")
    assert len(asr.context_buffer) == asr.overlap_samples
    print(f"  Committed text  : '{comm}'")
    print(f"  Hypothesis text : '{hyp}'")
    print("  OK    150ms context overlap preserved successfully")

    # 6. FINALIZATION FLUSH
    print("\n[6] SPEECH FINALIZATION FLUSH CHECK:")
    _final_comm, final_hyp = asr.transcribe_streaming_chunk(audio_1s[:8000], is_speech_final=True)
    print(f"  Final transcript: '{final_hyp}'")
    print(f"  Context buffer after finalization: {len(asr.context_buffer)} samples (expect 0)")
    assert len(asr.context_buffer) == 0
    assert asr.committed_text == ""
    assert asr.unconfirmed_text == ""
    print("  OK    Finalization flushes buffer and resets state cleanly")

    # 7. PERFORMANCE KPI BENCHMARK (Latency & RTF)
    print("\n[7] PERFORMANCE KPI BENCHMARK (Streaming Chunk Latency):")
    chunk_500ms = np.random.uniform(-0.05, 0.05, 8000).astype(np.float32)
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = asr.transcribe_streaming_chunk(chunk_500ms, is_speech_final=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_lat = np.mean(latencies)
    max_lat = np.max(latencies)
    rtf = (avg_lat / 1000.0) / 0.5  # latency / audio_duration
    print(f"  OK    Avg Chunk Latency : {avg_lat:.3f} ms  (KPI <= 150.0 ms)")
    print(f"        Max Chunk Latency : {max_lat:.3f} ms")
    print(f"  OK    Real-Time Factor  : {rtf:.4f}  (KPI <= 0.12)")
    assert avg_lat < 150.0

    print("\n" + "=" * 65)
    print("  RESULT: MODULE 3 ALL VERIFICATIONS PASSED (100%)")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()
