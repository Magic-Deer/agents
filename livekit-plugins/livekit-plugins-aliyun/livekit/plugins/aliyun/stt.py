from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

import aiohttp
from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, stt, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

DEFAULT_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_MODEL = "qwen3-asr-flash-realtime"
DEFAULT_SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CHUNK_MS = 50


@dataclass
class TurnDetection:
    threshold: float = 0.5
    silence_duration_ms: int = 800
    prefix_padding_ms: int = 300


@dataclass
class _STTOptions:
    model: str
    language: str | None
    sample_rate: int
    turn_detection: TurnDetection | None
    api_key: str
    base_url: str
    use_openai_beta_header: bool


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        language: str | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        turn_detection: TurnDetection | None = TurnDetection(),
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
        use_openai_beta_header: bool = True,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                aligned_transcript=False,
                offline_recognize=False,
            ),
        )

        resolved_api_key = api_key if is_given(api_key) else os.environ.get("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Aliyun DashScope API key is required, either as argument or set "
                "DASHSCOPE_API_KEY environment variable"
            )

        self._opts = _STTOptions(
            model=model,
            language=language,
            sample_rate=sample_rate,
            turn_detection=turn_detection,
            api_key=resolved_api_key,
            base_url=base_url,
            use_openai_beta_header=use_openai_beta_header,
        )
        self._session: aiohttp.ClientSession | None = None
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "aliyun"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("Aliyun STT only supports streaming recognition")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = replace(self._opts)
        if is_given(language):
            opts.language = language
        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
        if is_given(base_url):
            self._opts.base_url = base_url

        for stream in self._streams:
            stream.update_options(
                model=model,
                language=language,
                sample_rate=sample_rate,
                turn_detection=turn_detection,
                base_url=base_url,
            )

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._session and not self._session.closed:
            await self._session.close()


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: _STTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._session = http_session
        self._reconnect_event = asyncio.Event()
        self._speaking = False
        self._audio_duration_current = 0.0
        self._duration_queue = deque[float]()
        self._request_id = ""

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._needed_sr = sample_rate
            self._resampler = None
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
        if is_given(base_url):
            self._opts.base_url = base_url
        self._reconnect_event.set()

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"event_id": utils.shortuuid(), "type": event_type}
        payload.update(fields)
        return payload

    def _build_session_update(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "input_audio_format": "pcm",
            "sample_rate": self._opts.sample_rate,
        }
        if self._opts.language:
            session["input_audio_transcription"] = {"language": self._opts.language}
        if self._opts.turn_detection is None:
            session["turn_detection"] = None
        else:
            session["turn_detection"] = {
                "type": "server_vad",
                "threshold": self._opts.turn_detection.threshold,
                "silence_duration_ms": self._opts.turn_detection.silence_duration_ms,
                "prefix_padding_ms": self._opts.turn_detection.prefix_padding_ms,
            }
        return self._event("session.update", session=session)

    def _build_ws_url(self) -> str:
        base_url = self._opts.base_url
        if base_url.startswith("http://"):
            base_url = "ws://" + base_url[len("http://") :]
        elif base_url.startswith("https://"):
            base_url = "wss://" + base_url[len("https://") :]
        if "?" in base_url:
            return f"{base_url}&model={self._opts.model}"
        return f"{base_url}?model={self._opts.model}"

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = self._build_ws_url()
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "User-Agent": "LiveKit Agents",
        }
        if self._opts.use_openai_beta_header:
            headers["OpenAI-Beta"] = "realtime=v1"
        try:
            return await asyncio.wait_for(
                self._session.ws_connect(url, headers=headers),
                timeout=self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Aliyun STT") from e

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=self._opts.sample_rate * CHUNK_MS // 1000,
            )
            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                should_commit = False
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())
                    should_commit = True

                for frame in frames:
                    self._audio_duration_current += frame.duration
                    await ws.send_str(
                        json.dumps(
                            self._event(
                                "input_audio_buffer.append",
                                audio=base64.b64encode(frame.data.tobytes()).decode("utf-8"),
                            )
                        )
                    )

                if should_commit and self._opts.turn_detection is None:
                    if self._audio_duration_current > 0:
                        self._duration_queue.append(self._audio_duration_current)
                        self._audio_duration_current = 0.0
                    await ws.send_str(json.dumps(self._event("input_audio_buffer.commit")))

            closing_ws = True
            try:
                await ws.send_str(json.dumps(self._event("session.finish")))
            except Exception:
                logger.debug("failed to send session.finish", exc_info=True)

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:
                        return
                    raise APIStatusError(
                        message="Aliyun STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        request_id=self._request_id,
                        body=f"{msg.data=} {msg.extra=}",
                    )
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                    if self._handle_event(data):
                        return
                except Exception:
                    logger.exception("failed to process Aliyun STT message")
                    raise

        while True:
            closing_ws = False
            ws = await self._connect_ws()
            try:
                await ws.send_str(json.dumps(self._build_session_update()))
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()
                    if wait_reconnect_task not in done:
                        break
                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()
            finally:
                if not ws.closed:
                    await ws.close()

    def _emit_interim(self, text: str, *, request_id: str = "") -> None:
        if not self._speaking:
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._speaking = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(language=self._opts.language or "", text=text),
                ],
            )
        )

    def _emit_final(self, text: str, *, request_id: str = "") -> None:
        if not self._speaking:
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))
            self._speaking = True
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=request_id,
                alternatives=[
                    stt.SpeechData(language=self._opts.language or "", text=text),
                ],
            )
        )

    def _emit_usage(self, duration: float, *, request_id: str = "") -> None:
        if duration <= 0:
            return
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.RECOGNITION_USAGE,
                request_id=request_id,
                alternatives=[],
                recognition_usage=stt.RecognitionUsage(audio_duration=duration),
            )
        )

    def _handle_event(self, data: dict[str, Any]) -> bool:
        msg_type = data.get("type")
        if msg_type == "session.finished":
            return True
        if msg_type == "error":
            raise APIStatusError(
                message="Aliyun STT error",
                status_code=-1,
                request_id=self._request_id,
                body=data.get("error"),
            )
        if msg_type == "input_audio_buffer.speech_started":
            if not self._speaking:
                self._speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            return False
        if msg_type == "input_audio_buffer.speech_stopped":
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
            return False
        if msg_type == "input_audio_buffer.committed":
            duration = (
                self._duration_queue.popleft()
                if self._duration_queue
                else self._audio_duration_current
            )
            self._audio_duration_current = 0.0
            self._emit_usage(duration, request_id=data.get("event_id", ""))
            if self._speaking:
                self._speaking = False
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                )
            return False
        if msg_type == "conversation.item.input_audio_transcription.text":
            text = data.get("text", "")
            stash = data.get("stash", "")
            combined = f"{stash}{text}" if stash else text
            if combined:
                self._emit_interim(combined, request_id=data.get("item_id", ""))
            return False
        if msg_type == "conversation.item.input_audio_transcription.completed":
            transcript = data.get("transcript", "")
            if transcript:
                self._emit_final(transcript, request_id=data.get("item_id", ""))
            return False
        if msg_type == "conversation.item.input_audio_transcription.failed":
            raise APIStatusError(
                message="Aliyun STT transcription failed",
                status_code=-1,
                request_id=data.get("item_id", ""),
                body=data.get("error"),
            )
        return False
