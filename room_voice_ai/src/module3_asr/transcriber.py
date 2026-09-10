"""
src/module3_asr/transcriber.py
================================
MODULE 3: REAL-TIME VIETNAMESE ASR (STREAMING SPEECH-TO-TEXT)
Tối ưu hóa Faster-Whisper và CTranslate2 float16 trên GPU NVIDIA (< 1.0GB VRAM)
Tích hợp thuật toán Local Agreement Prefix Alignment và bộ chuẩn hóa Code-switching.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import AudioChunkPayload, TranscriptPayload, WakeWordPayload

logger = logging.getLogger(__name__)


class Module3VietnameseStreamingASR:
    """
    Class nhận dạng giọng nói tiếng Việt thời gian thực hỗ trợ Streaming,
    ổn định hiển thị phụ đề bằng LCP và Việt hóa thuật ngữ IT.
    """

    # Kích thước buffer tối thiểu trước khi gửi lên Whisper (giây)
    MIN_BUFFER_SEC: float = 0.5
    # Kích thước buffer tối đa (giây) – force transcribe
    MAX_BUFFER_SEC: float = 10.0

    def __init__(
        self,
        model_path: str = "vinai/PhoWhisper-small",
        device: str = "cuda",
        compute_type: str = "float16",
        beam_size: int = 1,
        context_overlap_ms: int = 150,
        session_id: str = "",
    ) -> None:
        """Khởi tạo Session giải mã Faster-Whisper."""
        self.session_id = session_id
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.sample_rate = settings.AUDIO_SAMPLE_RATE if hasattr(settings, "AUDIO_SAMPLE_RATE") else 16000

        # Ngưỡng chồng lấn ngữ cảnh âm học (150ms = 2400 samples)
        self.overlap_samples = int(context_overlap_ms * self.sample_rate / 1000)

        # Khởi tạo các biến lưu trữ trạng thái bộ đệm
        self.context_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.committed_text: str = ""
        self.unconfirmed_text: str = ""

        # Trạng thái lắng nghe của pipeline
        self._listening: bool = False
        self._audio_buffer: list[np.ndarray] = []
        self._confirmed_prefix: str = ""

        # Mô phỏng nạp mô hình trong khối chạy thử, thực tế nạp model từ faster-whisper
        self.model: Any | None = None
        self.model_path = model_path
        if model_path:
            self._init_model_engine(model_path)

        # Từ điển ánh xạ chuẩn hóa Code-switching IT Việt - Anh
        self.code_switching_dict: dict[str, str] = {
            r"\báp[- ]bi[- ]ai\b|\ba[- ]pi\b": "API",
            r"\bbấc\b|\bbắc\b": "bug",
            r"\bđì[- ]ploi\b|\bđi[- ]ploi\b": "deploy",
            r"\bbét[- ]en\b|\bbách[- ]en\b": "backend",
            r"\bgờ[- ]rúp[- ]pê[- ]xê\b|\bgê[- ]pê[- ]xê\b": "gRPC",
            r"\bsụt\b|\bxuốc\b|\bsoóc\b": "source code",
            r"\bmớc\b|\bmơ[- ]gơ\b": "merge",
            r"\brì[- ]bô\b|\bri[- ]po\b": "repo",
            r"\bgiắc\b|\bdi[- ]ra\b": "Jira",
            r"\bđát[- ]ta\b|\bđa[- ]ta\b": "data",
            r"\btét\b|\btét[- ]ting\b": "test",
            r"\bsờ[- ]crum\b|\bích[- ]crum\b": "Scrum",
        }

        # Đăng ký lắng nghe EventBus
        self.bus = EventBus.get_instance()
        self.bus.subscribe(EventChannel.AUDIO_CHUNK, self._on_audio_chunk)
        self.bus.subscribe(EventChannel.WAKE_WORD, self._on_wake_word)
        self.bus.subscribe(EventChannel.PIPELINE_RESET, self._on_reset)
        logger.info(f"[Module 3] Initialized (device={device}, compute_type={compute_type}).")

    def _init_model_engine(self, model_path: str) -> None:
        """Nạp engine Faster-Whisper thực tế nếu thư viện có sẵn, nếu không giả lập mock."""
        try:
            # Tự động nạp CUDA DLLs từ torch/lib nếu chạy trên Windows + CUDA
            if self.device == "cuda" and hasattr(os, "add_dll_directory"):
                try:
                    import torch  # type: ignore

                    torch_file = getattr(torch, "__file__", None)
                    if torch_file:
                        torch_lib = os.path.join(os.path.dirname(torch_file), "lib")
                        if os.path.exists(torch_lib):
                            os.add_dll_directory(torch_lib)
                            os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
                except Exception:
                    pass

            # Kiểm tra xem đường dẫn cục bộ models/phowhisper-small-ct2 có tồn tại không
            local_model_path = Path(__file__).resolve().parent.parent.parent / "models" / "phowhisper-small-ct2"
            if local_model_path.exists() and (local_model_path / "model.bin").exists():
                model_path = str(local_model_path)

            from faster_whisper import WhisperModel  # type: ignore

            logger.info(f"[Module 3] Đang nạp PhoWhisper từ {model_path} trên {self.device} ({self.compute_type})...")
            self.model = WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
            )
            logger.info("[Module 3] Nạp mô hình Faster-Whisper thành công!")
        except Exception as e:
            logger.warning(f"[Module 3] [Cảnh báo] Không thể nạp faster_whisper ({e}). Sử dụng Mock Engine để kiểm thử.")
            self.model = None

    def load(self) -> None:
        """Tải mô hình Faster-Whisper lên thiết bị thực thi (GPU/CPU)."""
        if self.model is None and self.model_path:
            self._init_model_engine(self.model_path)

    @staticmethod
    def longest_common_prefix(str1: str, str2: str) -> str:
        """Thuật toán tìm Tiền tố chung dài nhất (LCP) theo cấp độ Từ (Word-level LCP)."""
        words1 = str1.strip().split()
        words2 = str2.strip().split()

        lcp_words = []
        for w1, w2 in zip(words1, words2):
            if w1.lower() == w2.lower():
                lcp_words.append(w1)
            else:
                break

        return " ".join(lcp_words)

    def normalize_vietnamese_text(self, text: str) -> str:
        """Bộ chuẩn hóa chính tả, phục hồi viết hoa chữ cái đầu và xóa khoảng trắng thừa."""
        text = text.strip()
        if not text:
            return ""

        # Loại bỏ các khoảng trắng trùng lặp
        text = re.sub(r"\s+", " ", text)

        # Viết hoa chữ cái đầu tiên của câu
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        return text

    def apply_code_switching(self, text: str) -> str:
        """Thay thế các thuật ngữ IT phiên âm tiếng Việt thành từ gốc tiếng Anh chuẩn xác."""
        normalized_text = text
        for pattern, replacement in self.code_switching_dict.items():
            normalized_text = re.sub(pattern, replacement, normalized_text, flags=re.IGNORECASE)
        return normalized_text

    def transcribe_streaming_chunk(
        self,
        audio_chunk_16k: np.ndarray,
        is_speech_final: bool = False,
    ) -> tuple[str, str]:
        """
        Xử lý từng audio chunk streaming từ Module 1.
        Returns:
            committed_text (str): Đoạn văn bản đã được khóa cố định (Committed)
            hypothesis_text (str): Toàn bộ câu đang nhận diện (Hypothesis)
        """
        # 1. Ghép nối audio chunk mới vào bộ đệm ngữ cảnh
        self.context_buffer = np.append(self.context_buffer, audio_chunk_16k)

        # Nếu chưa đủ độ dài tối thiểu 0.5s và chưa kết thúc câu thì chưa giải mã
        if len(self.context_buffer) < 8000 and not is_speech_final:
            return self.committed_text, (self.committed_text + (" " if self.committed_text else "") + self.unconfirmed_text).strip()

        # 2. Suy luận qua mô hình (Giả lập nếu mô hình rỗng, chạy thật nếu có)
        if self.model is not None:
            segments, _ = self.model.transcribe(
                self.context_buffer,
                language="vi",
                beam_size=self.beam_size if not is_speech_final else 3,
                temperature=0.0,
                vad_filter=False,
            )
            raw_hypothesis = " ".join([seg.text.strip() for seg in segments])
        else:
            raw_hypothesis = (
                "Tôi muốn đì ploi cái sụt áp bi ai này lên bét en để sửa bấc"
                if not is_speech_final
                else "Tôi muốn deploy cái source code API này lên backend để sửa bug."
            )

        # 3. Chuẩn hóa văn bản và xử lý Code-switching
        raw_hypothesis = self.normalize_vietnamese_text(raw_hypothesis)
        raw_hypothesis = self.apply_code_switching(raw_hypothesis)

        # 4. Khi người nói kết thúc câu (VAD = False) -> Flush toàn bộ bộ đệm
        if is_speech_final:
            final_text = (self.committed_text + " " if self.committed_text else "") + raw_hypothesis
            final_text = self.normalize_vietnamese_text(final_text)

            self.committed_text = ""
            self.unconfirmed_text = ""
            self.context_buffer = np.array([], dtype=np.float32)

            return final_text, final_text

        # 5. Áp dụng thuật toán Local Agreement Prefix Alignment
        if self.unconfirmed_text == "":
            self.unconfirmed_text = raw_hypothesis

        lcp = self.longest_common_prefix(self.unconfirmed_text, raw_hypothesis)

        if lcp:
            self.committed_text = (self.committed_text + " " if self.committed_text else "") + lcp
            self.committed_text = self.normalize_vietnamese_text(self.committed_text)
            self.unconfirmed_text = raw_hypothesis[len(lcp) :].strip()
        else:
            self.unconfirmed_text = raw_hypothesis

        # Duy trì Overlap Buffer
        if len(self.context_buffer) > self.overlap_samples:
            self.context_buffer = self.context_buffer[-self.overlap_samples :]

        full_hypothesis = (self.committed_text + " " if self.committed_text else "") + self.unconfirmed_text
        return self.committed_text, full_hypothesis.strip()

    # ------------------------------------------------------------------
    # EventBus integration
    # ------------------------------------------------------------------
    async def _on_wake_word(self, _payload: WakeWordPayload | None = None) -> None:
        """Bật chế độ lắng nghe khi phát hiện wake-word."""
        self._listening = True
        self._audio_buffer.clear()
        self._confirmed_prefix = ""
        self.context_buffer = np.array([], dtype=np.float32)
        self.committed_text = ""
        self.unconfirmed_text = ""
        logger.info("[Module 3] 👂 Listening started.")

    async def _on_reset(self, _payload: object | None = None) -> None:
        """Reset trạng thái về idle."""
        self._listening = False
        self._audio_buffer.clear()
        self._confirmed_prefix = ""
        self.context_buffer = np.array([], dtype=np.float32)
        self.committed_text = ""
        self.unconfirmed_text = ""
        logger.info("[Module 3] Reset to idle.")

    async def _on_audio_chunk(self, payload: AudioChunkPayload) -> None:
        """Tích lũy chunk và kích hoạt transcription."""
        if not self._listening or not payload.is_speech:
            return

        chunk_np = np.frombuffer(payload.audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.append(chunk_np)

        buffer_samples = sum(len(c) for c in self._audio_buffer)
        buffer_sec = buffer_samples / self.sample_rate

        if buffer_sec >= self.MIN_BUFFER_SEC:
            await self._transcribe(is_final=(buffer_sec >= self.MAX_BUFFER_SEC))

    async def _transcribe(self, is_final: bool = False) -> None:
        """Chạy inference và phát TranscriptPayload lên EventBus."""
        if not self._audio_buffer:
            return

        audio_np = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()

        committed, hyp = await asyncio.get_event_loop().run_in_executor(
            None, self.transcribe_streaming_chunk, audio_np, is_final
        )

        display_text = hyp if hyp else committed
        if not display_text.strip():
            return

        payload = TranscriptPayload(
            session_id=self.session_id,
            text=display_text,
            is_final=is_final,
            language="vi",
        )
        await self.bus.publish(EventChannel.TRANSCRIPT, payload)
        logger.debug(f"[Module 3] Transcript({'FINAL' if is_final else 'partial'}): {display_text}")

    def _align_prefix(self, new_text: str) -> str:
        if not self._confirmed_prefix:
            return new_text
        if new_text.startswith(self._confirmed_prefix):
            return new_text
        self._confirmed_prefix = ""
        return new_text

    async def finalize(self) -> None:
        """Ép transcribe toàn bộ buffer còn lại và đánh dấu is_final=True."""
        if self._listening and (len(self._audio_buffer) > 0 or len(self.context_buffer) > 0):
            await self._transcribe(is_final=True)
        self._listening = False


# Alias tương thích ngược với hệ thống
StreamingTranscriber = Module3VietnameseStreamingASR
