from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any

from .asr_service import AsrService
from .config import DEFAULT_SETTINGS, Settings
from .conversation import ConversationManager
from .mcie_service import McieService
from .recognition_service import Person1RecognizerLoader, RecognitionService


class EventHub:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    async def publish(self, event: dict[str, Any]) -> None:
        for queue in tuple(self._subscribers):
            await queue.put(event)


@dataclass
class Runtime:
    conversation: ConversationManager
    recognition: Any
    asr: Any
    mcie: Any
    events: EventHub

    @classmethod
    def create(cls, settings: Settings = DEFAULT_SETTINGS) -> "Runtime":
        gpu_lock = threading.Lock()
        return cls(
            conversation=ConversationManager("local-session"),
            recognition=RecognitionService(loader=Person1RecognizerLoader(settings)),
            asr=AsrService(settings, gpu_lock=gpu_lock),
            mcie=McieService(settings, gpu_lock=gpu_lock),
            events=EventHub(),
        )
