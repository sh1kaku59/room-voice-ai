"""
main_pipeline.py
=================
File điều phối luồng chạy tích hợp cho hệ thống RooM Voice AI.
Sử dụng State Machine + Event Orchestrator để quản lý toàn bộ vòng đời.

Sơ đồ trạng thái:
  IDLE ──[wake-word]──► LISTENING ──[silence/timeout]──► PROCESSING
                                                              │
                                           ◄──[response done]─┘
                              RESPONDING ◄─────────────────────
                                  │
                           IDLE ◄─┘
"""

from __future__ import annotations

import asyncio
import logging
import sys
from enum import Enum, auto
from typing import Any

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import (
    BrainResponsePayload,
    NLUOutputPayload,
    OutputPayload,
    TranscriptPayload,
    WakeWordPayload,
)

# Modules
from src.module1_audio.enhancer import AudioEnhancer
from src.module2_wakeword.detector import WakeWordDetector
from src.module3_asr.transcriber import StreamingTranscriber
from src.module4_nlu.interpreter import NLUInterpreter
from src.module5_context.manager import ContextManager
from src.module6_brain.brain import BrainCore
from src.module7_output.generator import OutputGenerator

# Logging setup
logging.basicConfig(
    level  = getattr(logging, settings.LOG_LEVEL),
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{settings.LOG_DIR}/room_ai.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("main_pipeline")


# ---------------------------------------------------------------------------
# Trạng thái hệ thống
# ---------------------------------------------------------------------------
class PipelineState(Enum):
    IDLE        = auto()   # Đang chờ wake-word
    LISTENING   = auto()   # Đang nghe câu lệnh người dùng
    PROCESSING  = auto()   # Đang xử lý NLU + Context + Brain
    RESPONDING  = auto()   # Đang phát phản hồi (TTS + WebSocket)
    ERROR       = auto()   # Gặp lỗi nghiêm trọng


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class RooMPipeline:
    """
    Điều phối toàn bộ pipeline theo mô hình State Machine.
    Sử dụng EventBus để kết nối các module bất đồng bộ.
    """

    # Timeout chờ người dùng nói (giây) sau khi wake-word & timeout xử lý
    LISTEN_TIMEOUT = float(getattr(settings, "PIPELINE_LISTEN_TIMEOUT_SEC", 10.0))
    PROCESSING_TIMEOUT = float(getattr(settings, "PIPELINE_PROCESSING_TIMEOUT_SEC", 15.0))
    POST_TTS_COOLDOWN_SEC = float(getattr(settings, "AUDIO_POST_TTS_COOLDOWN_MS", 250)) / 1000.0

    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = session_id
        self.bus        = EventBus.get_instance()
        self.state      = PipelineState.IDLE
        self._listen_timer: asyncio.TimerHandle | None = None
        self._processing_timer: asyncio.TimerHandle | None = None
        self._playback_unlock_task: asyncio.Task[None] | None = None
        self._bg_tasks: set[asyncio.Task[Any]] = set()

        # Khởi tạo tất cả module
        self.module1 = AudioEnhancer(session_id=session_id)
        self.module2 = WakeWordDetector(session_id=session_id)
        self.module3 = StreamingTranscriber(session_id=session_id)
        self.module4 = NLUInterpreter(session_id=session_id)
        self.module5 = ContextManager(session_id=session_id)
        self.module6 = BrainCore(session_id=session_id)
        self.module7 = OutputGenerator(session_id=session_id)

        # Lắng nghe các sự kiện điều phối
        self.bus.subscribe(EventChannel.WAKE_WORD,      self._on_wake_word)
        self.bus.subscribe(EventChannel.TRANSCRIPT,     self._on_transcript)
        self.bus.subscribe(EventChannel.NLU_OUTPUT,     self._on_nlu_output)
        self.bus.subscribe(EventChannel.BRAIN_RESPONSE, self._on_brain_response)
        self.bus.subscribe(EventChannel.FINAL_OUTPUT,   self._on_final_output)
        self.bus.subscribe(EventChannel.SYSTEM_ERROR,   self._on_error)

        logger.info(f"[Pipeline] RooMPipeline initialized. Session: {session_id}")

    # ------------------------------------------------------------------
    # Lifecycle: load → run → shutdown
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """Tải tất cả model vào bộ nhớ (gọi một lần)."""
        logger.info("[Pipeline] Loading all modules...")
        self.module1.load()
        self.module2.load()
        self.module3.load()
        self.module4.load()
        self.module5.load()
        self.module6.load()
        self.module7.load()
        logger.info("[Pipeline] ✅ All modules loaded.")

    async def run(self, audio_source=None) -> None:
        """
        Vòng lặp chính của pipeline.

        Args:
            audio_source: Object có method `async read(n) -> bytes`
                          Nếu None, sử dụng mock source.
        """
        logger.info("[Pipeline] 🚀 RooM Voice AI started.")
        self._log_state()

        tasks = [
            self.module7.start_websocket_server(),
        ]

        if audio_source:
            tasks.append(self.module1.run_stream(audio_source))
        else:
            logger.warning("[Pipeline] No audio source provided. Running in mock mode.")

        await asyncio.gather(*tasks)

    async def shutdown(self) -> None:
        """Dừng toàn bộ pipeline và giải phóng tài nguyên."""
        logger.info("[Pipeline] Shutting down...")
        self._cancel_all_timers()
        self.module1.stop()
        self.module5.close()
        await self.bus.publish(EventChannel.PIPELINE_RESET, None)
        logger.info("[Pipeline] Shutdown complete.")

    # ------------------------------------------------------------------
    # State Machine transitions & Playback Lock
    # ------------------------------------------------------------------
    async def _on_wake_word(self, payload: WakeWordPayload) -> None:
        if self.state != PipelineState.IDLE:
            return
        await self._transition_and_gate(PipelineState.LISTENING, reason="WAKE_WORD")
        self._start_listen_timer()

    async def _on_transcript(self, payload: TranscriptPayload) -> None:
        if not payload.is_final:
            return
        if self.state == PipelineState.LISTENING:
            self._cancel_listen_timer()
            await self._transition_and_gate(PipelineState.PROCESSING, reason="TRANSCRIPT_FINAL")
            self._start_processing_timer()

    async def _on_nlu_output(self, payload: NLUOutputPayload) -> None:
        if self.state == PipelineState.PROCESSING and not payload.route_to_context_manager:
            self._cancel_processing_timer()
            await self._transition_and_gate(PipelineState.IDLE, reason="NLU_SKIPPED")

    async def _on_brain_response(self, payload: BrainResponsePayload) -> None:
        if payload.is_streaming:
            self._cancel_processing_timer()
            if self.state != PipelineState.RESPONDING:
                self._cancel_playback_unlock()
                await self._transition_and_gate(PipelineState.RESPONDING, reason="BRAIN_STREAMING")
        else:
            self._cancel_processing_timer()
            if self.state == PipelineState.PROCESSING:
                await self._transition_and_gate(PipelineState.RESPONDING, reason="BRAIN_DONE")
            # Fallback mở khóa nếu chạy thử không có Module 7 phát FINAL_OUTPUT
            self._schedule_playback_unlock(playback_seconds=0.5, override_existing=False)

    async def _on_final_output(self, payload: OutputPayload) -> None:
        """Khóa Wake-Word & ASR đúng theo thời lượng audio TTS thực tế + cooldown xả vang."""
        self._cancel_processing_timer()
        if self.state != PipelineState.RESPONDING:
            await self._transition_and_gate(PipelineState.RESPONDING, reason="TTS_PLAYBACK")
        audio_bytes = getattr(payload, "audio_bytes", b"") or b""
        # Output audio chuẩn 24,000 Hz, mono, int16 (2 bytes/sample)
        audio_duration_sec = len(audio_bytes) / float(24000 * 2)
        self._schedule_playback_unlock(playback_seconds=audio_duration_sec, override_existing=True)

    async def _on_error(self, error_payload) -> None:
        logger.error(f"[Pipeline] System error: {error_payload}")
        self._cancel_all_timers()
        await self._transition_and_gate(PipelineState.ERROR, reason="SYSTEM_ERROR")
        await asyncio.sleep(3.0)
        await self.bus.publish(EventChannel.PIPELINE_RESET, None)
        await self._transition_and_gate(PipelineState.IDLE, reason="ERROR_RECOVERY")

    # ------------------------------------------------------------------
    # Timers (Listen / Processing / Playback Unlock)
    # ------------------------------------------------------------------
    def _track_task(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    def _start_listen_timer(self) -> None:
        self._cancel_listen_timer()
        loop = asyncio.get_event_loop()
        self._listen_timer = loop.call_later(
            self.LISTEN_TIMEOUT,
            lambda: self._track_task(self._on_listen_timeout()),
        )

    def _cancel_listen_timer(self) -> None:
        if self._listen_timer:
            self._listen_timer.cancel()
            self._listen_timer = None

    def _start_processing_timer(self) -> None:
        self._cancel_processing_timer()
        loop = asyncio.get_event_loop()
        self._processing_timer = loop.call_later(
            self.PROCESSING_TIMEOUT,
            lambda: self._track_task(self._on_processing_timeout()),
        )

    def _cancel_processing_timer(self) -> None:
        if self._processing_timer:
            self._processing_timer.cancel()
            self._processing_timer = None

    def _schedule_playback_unlock(self, playback_seconds: float, override_existing: bool = True) -> None:
        if self._playback_unlock_task and not self._playback_unlock_task.done():
            if not override_existing:
                return
            self._playback_unlock_task.cancel()
        self._playback_unlock_task = asyncio.ensure_future(
            self._unlock_after_playback(playback_seconds)
        )

    def _cancel_playback_unlock(self) -> None:
        if self._playback_unlock_task and not self._playback_unlock_task.done():
            self._playback_unlock_task.cancel()
        self._playback_unlock_task = None

    def _cancel_all_timers(self) -> None:
        self._cancel_listen_timer()
        self._cancel_processing_timer()
        self._cancel_playback_unlock()

    async def _on_listen_timeout(self) -> None:
        """Hết thời gian chờ người dùng nói → force finalize và chống treo LISTENING."""
        if self.state == PipelineState.LISTENING:
            logger.info("[Pipeline] Listen timeout – forcing finalize.")
            await self.module3.finalize()
            if self.state == PipelineState.LISTENING:
                logger.info("[Pipeline] No speech finalized after timeout → returning to IDLE.")
                await self._transition_and_gate(PipelineState.IDLE, reason="LISTEN_TIMEOUT_EMPTY")

    async def _on_processing_timeout(self) -> None:
        """Chống deadlock nếu quá thời gian xử lý NLU/RAG/LLM mà chưa có phản hồi."""
        if self.state == PipelineState.PROCESSING:
            logger.warning("[Pipeline] Processing timeout → resetting pipeline back to IDLE.")
            await self.bus.publish(EventChannel.PIPELINE_RESET, None)
            await self._transition_and_gate(PipelineState.IDLE, reason="PROCESSING_TIMEOUT")

    async def _unlock_after_playback(self, playback_seconds: float) -> None:
        """Giữ Playback Lock đủ thời lượng audio TTS + khoảng xả vang trước khi mở lại IDLE."""
        try:
            wait_sec = max(0.25, playback_seconds) + self.POST_TTS_COOLDOWN_SEC
            await asyncio.sleep(wait_sec)
            if self.state == PipelineState.RESPONDING:
                await self._transition_and_gate(PipelineState.IDLE, reason="PLAYBACK_COMPLETED")
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _gate_payload_for_state(self, state: PipelineState, reason: str = "") -> dict[str, Any]:
        return {
            "wakeword_enabled": state == PipelineState.IDLE,
            "asr_enabled": state in (PipelineState.IDLE, PipelineState.LISTENING),
            "reset_buffers": state == PipelineState.IDLE,
            "state": state.name,
            "reason": reason,
        }

    async def _transition_and_gate(self, new_state: PipelineState, reason: str = "") -> None:
        logger.info(f"[Pipeline] State: {self.state.name} → {new_state.name}")
        self.state = new_state
        self._log_state()
        await self.bus.publish(
            EventChannel.AUDIO_GATE_CONTROL,
            self._gate_payload_for_state(new_state, reason=reason),
        )

    def _transition(self, new_state: PipelineState) -> None:
        logger.info(f"[Pipeline] State: {self.state.name} → {new_state.name}")
        self.state = new_state
        self._log_state()
        try:
            asyncio.get_running_loop()
            self._track_task(
                self.bus.publish(
                    EventChannel.AUDIO_GATE_CONTROL,
                    self._gate_payload_for_state(new_state),
                )
            )
        except RuntimeError:
            pass

    def _log_state(self) -> None:
        icons = {
            PipelineState.IDLE:       "😴",
            PipelineState.LISTENING:  "👂",
            PipelineState.PROCESSING: "🧠",
            PipelineState.RESPONDING: "🔊",
            PipelineState.ERROR:      "❌",
        }
        logger.info(f"[Pipeline] {icons.get(self.state, '?')} Current state: {self.state.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    import os
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    os.makedirs(settings.RAG_VECTOR_DB_PATH, exist_ok=True)
    os.makedirs(str(settings.BASE_DIR / "data"), exist_ok=True)

    pipeline = RooMPipeline(session_id="session_001")
    pipeline.load_all()

    try:
        await pipeline.run(audio_source=None)
    except KeyboardInterrupt:
        logger.info("[Pipeline] Interrupted by user.")
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
