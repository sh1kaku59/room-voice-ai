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

from configs import settings
from src.common.event_bus import EventBus, EventChannel
from src.common.payloads import BrainResponsePayload, TranscriptPayload, WakeWordPayload

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

    # Timeout chờ người dùng nói (giây) sau khi wake-word
    LISTEN_TIMEOUT = 10.0

    def __init__(self, session_id: str = "default_session") -> None:
        self.session_id = session_id
        self.bus        = EventBus.get_instance()
        self.state      = PipelineState.IDLE
        self._listen_timer: asyncio.TimerHandle | None = None

        # Khởi tạo tất cả module
        self.module1 = AudioEnhancer(session_id=session_id)
        self.module2 = WakeWordDetector(session_id=session_id)
        self.module3 = StreamingTranscriber(session_id=session_id)
        self.module4 = NLUInterpreter(session_id=session_id)
        self.module5 = ContextManager(session_id=session_id)
        self.module6 = BrainCore(session_id=session_id)
        self.module7 = OutputGenerator(session_id=session_id)

        # Lắng nghe các sự kiện điều phối
        self.bus.subscribe(EventChannel.WAKE_WORD,     self._on_wake_word)
        self.bus.subscribe(EventChannel.TRANSCRIPT,    self._on_transcript)
        self.bus.subscribe(EventChannel.BRAIN_RESPONSE, self._on_brain_response)
        self.bus.subscribe(EventChannel.SYSTEM_ERROR,  self._on_error)

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
        self.module1.stop()
        self.module5.close()
        await self.bus.publish(EventChannel.PIPELINE_RESET, None)
        logger.info("[Pipeline] Shutdown complete.")

    # ------------------------------------------------------------------
    # State Machine transitions
    # ------------------------------------------------------------------
    async def _on_wake_word(self, payload: WakeWordPayload) -> None:
        if self.state != PipelineState.IDLE:
            return
        self._transition(PipelineState.LISTENING)
        self._start_listen_timer()

    async def _on_transcript(self, payload: TranscriptPayload) -> None:
        if not payload.is_final:
            return
        if self.state == PipelineState.LISTENING:
            self._cancel_listen_timer()
            self._transition(PipelineState.PROCESSING)

    async def _on_brain_response(self, payload: BrainResponsePayload) -> None:
        if payload.is_streaming and self.state != PipelineState.RESPONDING:
            self._transition(PipelineState.RESPONDING)
        elif not payload.is_streaming:
            # Phản hồi hoàn chỉnh → trở về IDLE
            await asyncio.sleep(1.0)   # Chờ TTS phát xong
            self._transition(PipelineState.IDLE)

    async def _on_error(self, error_payload) -> None:
        logger.error(f"[Pipeline] System error: {error_payload}")
        self._transition(PipelineState.ERROR)
        await asyncio.sleep(3.0)
        await self.bus.publish(EventChannel.PIPELINE_RESET, None)
        self._transition(PipelineState.IDLE)

    # ------------------------------------------------------------------
    # Listen timer (timeout khi người dùng im lặng)
    # ------------------------------------------------------------------
    def _start_listen_timer(self) -> None:
        loop = asyncio.get_event_loop()
        self._listen_timer = loop.call_later(
            self.LISTEN_TIMEOUT,
            lambda: asyncio.ensure_future(self._on_listen_timeout())
        )

    def _cancel_listen_timer(self) -> None:
        if self._listen_timer:
            self._listen_timer.cancel()
            self._listen_timer = None

    async def _on_listen_timeout(self) -> None:
        """Hết thời gian chờ người dùng nói → force finalize."""
        if self.state == PipelineState.LISTENING:
            logger.info("[Pipeline] Listen timeout – forcing finalize.")
            await self.module3.finalize()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _transition(self, new_state: PipelineState) -> None:
        logger.info(f"[Pipeline] State: {self.state.name} → {new_state.name}")
        self.state = new_state
        self._log_state()

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
