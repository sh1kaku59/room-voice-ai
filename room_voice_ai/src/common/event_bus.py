"""
src/common/event_bus.py
=======================
Bộ điều phối sự kiện (Event Broker) kết nối các module theo mô hình Pub/Sub.
Sử dụng asyncio.Queue để truyền payload giữa các module bất đồng bộ.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tên các kênh sự kiện chuẩn
# ---------------------------------------------------------------------------
class EventChannel:  # pylint: disable=too-few-public-methods
    """Hằng số tên kênh dùng chung, tránh lỗi typo."""
    AUDIO_CHUNK      = "audio.chunk"          # Module1 → Module2, Module3
    WAKE_WORD        = "wakeword.detected"    # Module2 → Pipeline
    TRANSCRIPT       = "asr.transcript"       # Module3 → Module4
    NLU_OUTPUT       = "nlu.output"           # Module4 → Module5, Module6
    CONTEXT_READY    = "context.ready"        # Module5 → Module6
    BRAIN_RESPONSE   = "brain.response"       # Module6 → Module7
    FINAL_OUTPUT     = "output.final"         # Module7 → Client
    OUTPUT_READY     = "output.ready"         # Module7 → Audio/UI Client
    SYSTEM_ERROR     = "system.error"         # Bất kỳ module nào → Logger
    PIPELINE_RESET   = "pipeline.reset"       # Reset toàn bộ state machine


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------
class EventBus:
    """
    Singleton Event Bus bất đồng bộ.

    Sử dụng:
        bus = EventBus.get_instance()

        # Đăng ký lắng nghe
        bus.subscribe(EventChannel.AUDIO_CHUNK, my_async_handler)

        # Phát sự kiện
        await bus.publish(EventChannel.AUDIO_CHUNK, payload)
    """

    _instance: EventBus | None = None

    def __init__(self) -> None:
        # channel_name → danh sách coroutine handlers
        self._subscribers: dict[str, list[Callable[..., Coroutine[Any, Any, Any]]]] = defaultdict(list)
        # channel_name → asyncio.Queue (cho mô hình queue-based)
        self._queues: dict[str, asyncio.Queue[Any]] = {}

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls) -> EventBus:
        """Lấy singleton instance của EventBus."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Xoá singleton (dùng trong unit test)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Pub / Sub API
    # ------------------------------------------------------------------
    def subscribe(self, channel: str, handler: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Đăng ký handler cho một kênh sự kiện."""
        self._subscribers[channel].append(handler)
        logger.debug("[EventBus] Subscribed '%s' → channel '%s'", handler.__qualname__, channel)

    def unsubscribe(self, channel: str, handler: Callable[..., Coroutine[Any, Any, Any]]) -> None:
        """Huỷ đăng ký handler."""
        if handler in self._subscribers[channel]:
            self._subscribers[channel].remove(handler)

    async def publish(self, channel: str, payload: Any) -> None:
        """
        Phát sự kiện đến tất cả subscribers của kênh.
        Mỗi handler được gọi song song (gather).
        """
        handlers = self._subscribers.get(channel, [])
        if not handlers:
            logger.debug("[EventBus] No subscribers for channel '%s'", channel)
            return

        await asyncio.gather(
            *[handler(payload) for handler in handlers],
            return_exceptions=True
        )

    # ------------------------------------------------------------------
    # Queue-based API (cho mô hình producer-consumer)
    # ------------------------------------------------------------------
    def get_queue(self, channel: str, maxsize: int = 100) -> asyncio.Queue:
        """
        Lấy (hoặc tạo mới) Queue gắn với kênh.
        Dùng khi module muốn tự consume theo tốc độ riêng.
        """
        if channel not in self._queues:
            self._queues[channel] = asyncio.Queue(maxsize=maxsize)
        return self._queues[channel]

    async def enqueue(self, channel: str, payload: Any) -> None:
        """Đưa payload vào Queue của kênh (non-blocking, drop nếu đầy)."""
        q = self.get_queue(channel)
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("[EventBus] Queue full on channel '%s', dropping payload.", channel)

    async def dequeue(self, channel: str) -> Any:
        """Chờ và lấy payload từ Queue của kênh."""
        q = self.get_queue(channel)
        return await q.get()
