from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given

from .log import logger

SUPPORTED_SAMPLE_RATES = {8000, 16000}
_OPENAI_BETA_HEADER = {"OpenAI-Beta": "realtime=v1"}
_DEFAULT_LANGUAGE = LanguageCode("und")
_WS_CLOSE_TYPES = (
    aiohttp.WSMsgType.CLOSED,
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSING,
)


@dataclass
class AliyunTurnDetectionOptions:
    threshold: float = 0.0
    silence_duration_ms: int = 400

    def __post_init__(self) -> None:
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("turn_detection.threshold must be between -1 and 1")
        if not 200 <= self.silence_duration_ms <= 6000:
            raise ValueError("turn_detection.silence_duration_ms must be between 200 and 6000")


@dataclass
class AliyunSTTOptions:
    api_key: str
    model: str
    language: LanguageCode | None
    sample_rate: int
    interim_results: bool
    turn_detection: AliyunTurnDetectionOptions | None
    base_url: str


@dataclass
class _ItemState:
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None
    last_preflight_text: str = ""
    manual_audio_duration: float | None = None
    saw_start_of_speech: bool = False
    sent_end_of_speech: bool = False


_DEFAULT_TURN_DETECTION = AliyunTurnDetectionOptions()


def _copy_turn_detection(
    turn_detection: AliyunTurnDetectionOptions | None,
) -> AliyunTurnDetectionOptions | None:
    if turn_detection is None:
        return None
    return dataclasses.replace(turn_detection)


def _to_aliyun_language(language: str | LanguageCode) -> str:
    normalized = LanguageCode(str(language))
    return normalized.language


def _resolve_speech_language(
    event_language: Any, default_language: LanguageCode | None
) -> LanguageCode:
    if isinstance(event_language, str) and event_language:
        return LanguageCode(event_language)
    if default_language is not None:
        return default_language
    return _DEFAULT_LANGUAGE


def _load_message(data: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as e:
        raise APIStatusError(
            message="Aliyun STT returned invalid JSON",
            status_code=-1,
            body=data,
        ) from e

    if not isinstance(payload, dict):
        raise APIStatusError(
            message="Aliyun STT returned a non-object message",
            status_code=-1,
            body=payload,
        )
    return payload


def _message_request_id(data: dict[str, Any]) -> str | None:
    item_id = data.get("item_id")
    if isinstance(item_id, str) and item_id:
        return item_id

    event_id = data.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id

    error = data.get("error")
    if isinstance(error, dict):
        nested_event_id = error.get("event_id")
        if isinstance(nested_event_id, str) and nested_event_id:
            return nested_event_id

    return None


def _status_error_from_event(data: dict[str, Any], default_message: str) -> APIStatusError:
    message = default_message
    error = data.get("error")
    if isinstance(error, dict):
        error_message = error.get("message")
        if isinstance(error_message, str) and error_message:
            message = error_message

    return APIStatusError(
        message=message,
        status_code=-1,
        request_id=_message_request_id(data),
        body=data,
    )


class STT(stt.STT):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "qwen3-asr-flash-realtime",
        language: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: int = 16000,
        interim_results: bool = True,
        turn_detection: AliyunTurnDetectionOptions | None = _DEFAULT_TURN_DETECTION,
        base_url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=interim_results,
                diarization=False,
                aligned_transcript=False,
                offline_recognize=False,
            )
        )

        resolved_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not resolved_api_key:
            raise ValueError("DashScope API key is required")

        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            raise ValueError("sample_rate must be one of 8000 or 16000")

        normalized_language: LanguageCode | None = None
        if is_given(language):
            normalized_language = LanguageCode(_to_aliyun_language(language))

        self._opts = AliyunSTTOptions(
            api_key=resolved_api_key,
            model=model,
            language=normalized_language,
            sample_rate=sample_rate,
            interim_results=interim_results,
            turn_detection=_copy_turn_detection(turn_detection),
            base_url=base_url,
        )
        self._http_session = http_session
        self._streams = weakref.WeakSet[SpeechStream]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Aliyun"

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        raise NotImplementedError(
            "Aliyun realtime STT does not support offline recognize(); use stream() instead"
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = dataclasses.replace(
            self._opts,
            turn_detection=_copy_turn_detection(self._opts.turn_detection),
        )
        if is_given(language):
            opts.language = LanguageCode(_to_aliyun_language(language))

        stream = SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
            http_session=self.session,
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        self._streams.clear()


class SpeechStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: STT,
        opts: AliyunSTTOptions,
        conn_options: APIConnectOptions,
        http_session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)

        self._stt = stt
        self._opts = opts
        self._session = http_session
        self._speaking = False
        self._request_id = uuid4().hex
        self._closed_by_finish = False
        self._item_state: dict[str, _ItemState] = {}
        self._pending_commit_durations: deque[float] = deque()

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        ws: aiohttp.ClientWebSocketResponse | None = None
        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(self._send_task(ws)),
                asyncio.create_task(self._recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await utils.aio.gracefully_cancel(*tasks)
        finally:
            if ws is not None:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = f"{self._opts.base_url}?{urlencode({'model': self._opts.model})}"
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            **_OPENAI_BETA_HEADER,
        }

        ws: aiohttp.ClientWebSocketResponse | None = None
        connected = False
        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )

            first_event = await self._receive_handshake_event(ws)
            first_type = first_event.get("type")
            if first_type == "error":
                raise _status_error_from_event(first_event, "Aliyun STT error")
            if first_type != "session.created":
                raise APIStatusError(
                    message="Aliyun STT expected session.created during handshake",
                    status_code=-1,
                    request_id=_message_request_id(first_event),
                    body=first_event,
                )

            await self._send_event(ws, self._build_session_update_event())

            while True:
                event = await self._receive_handshake_event(ws)
                event_type = event.get("type")
                if event_type == "session.updated":
                    connected = True
                    return ws
                if event_type == "error":
                    raise _status_error_from_event(event, "Aliyun STT error")

                logger.debug("ignoring Aliyun STT handshake event %s", event_type)

        except APITimeoutError:
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Aliyun STT request timed out") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message or "Aliyun STT request failed",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError("Failed to connect to Aliyun STT") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Aliyun STT connection error") from e
        finally:
            if ws is not None and not connected:
                await ws.close()

    def _build_session_update_event(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "sample_rate": self._opts.sample_rate,
            "turn_detection": None,
        }

        if self._opts.language is not None:
            session["input_audio_transcription"] = {"language": str(self._opts.language)}

        if self._opts.turn_detection is not None:
            session["turn_detection"] = {
                "type": "server_vad",
                "threshold": self._opts.turn_detection.threshold,
                "silence_duration_ms": self._opts.turn_detection.silence_duration_ms,
            }

        return {
            "type": "session.update",
            "session": session,
        }

    async def _receive_handshake_event(self, ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
        try:
            msg = await asyncio.wait_for(ws.receive(), self._conn_options.timeout)
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Aliyun STT handshake timed out") from e

        if msg.type in _WS_CLOSE_TYPES:
            raise APIStatusError(
                message="Aliyun STT WebSocket closed during handshake",
                status_code=ws.close_code or -1,
                body={"data": msg.data, "extra": msg.extra},
            )

        if msg.type != aiohttp.WSMsgType.TEXT:
            raise APIStatusError(
                message="Aliyun STT received a non-text handshake message",
                status_code=-1,
                body={"type": msg.type},
            )

        return _load_message(msg.data)

    async def _send_event(
        self, ws: aiohttp.ClientWebSocketResponse, payload: dict[str, Any]
    ) -> None:
        await ws.send_json({"event_id": uuid4().hex, **payload})

    @utils.log_exceptions(logger=logger)
    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=self._opts.sample_rate // 10,
        )
        pending_manual_duration = 0.0

        async for data in self._input_ch:
            frames: list[rtc.AudioFrame]
            should_commit = False
            if isinstance(data, rtc.AudioFrame):
                frames = audio_bstream.write(data.data.tobytes())
                if self._opts.turn_detection is None:
                    pending_manual_duration += data.duration
            else:
                frames = audio_bstream.flush()
                should_commit = self._opts.turn_detection is None

            for frame in frames:
                await self._send_event(
                    ws,
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame.data.tobytes()).decode("utf-8"),
                    },
                )

            if should_commit:
                if pending_manual_duration > 0:
                    self._pending_commit_durations.append(pending_manual_duration)
                await self._send_event(ws, {"type": "input_audio_buffer.commit"})
                pending_manual_duration = 0.0

        for frame in audio_bstream.flush():
            await self._send_event(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame.data.tobytes()).decode("utf-8"),
                },
            )

        await self._send_event(ws, {"type": "session.finish"})

    @utils.log_exceptions(logger=logger)
    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in _WS_CLOSE_TYPES:
                if self._closed_by_finish:
                    return
                self._clear_session_state()
                raise APIStatusError(
                    message="Aliyun STT WebSocket closed unexpectedly",
                    status_code=ws.close_code or -1,
                    body={"data": msg.data, "extra": msg.extra},
                )

            if msg.type != aiohttp.WSMsgType.TEXT:
                logger.warning("unexpected Aliyun STT message type %s", msg.type)
                continue

            data = _load_message(msg.data)
            if self._process_event(data):
                return

    def _process_event(self, data: dict[str, Any]) -> bool:
        msg_type = data.get("type")

        if msg_type == "session.finished":
            self._closed_by_finish = True
            self._clear_session_state()
            return True

        if msg_type in {"session.created", "session.updated"}:
            return False

        if msg_type == "input_audio_buffer.committed":
            self._handle_committed(data)
            return False

        if msg_type == "conversation.item.created":
            item = data.get("item")
            item_id = item.get("id") if isinstance(item, dict) else None
            if isinstance(item_id, str) and item_id:
                self._get_or_create_item_state(item_id)
            return False

        if msg_type == "input_audio_buffer.speech_started":
            self._handle_speech_started(data)
            return False

        if msg_type == "input_audio_buffer.speech_stopped":
            self._handle_speech_stopped(data)
            return False

        if msg_type == "conversation.item.input_audio_transcription.text":
            self._handle_transcript_preview(data)
            return False

        if msg_type == "conversation.item.input_audio_transcription.completed":
            self._handle_transcript_completed(data)
            return False

        if msg_type == "conversation.item.input_audio_transcription.failed":
            self._handle_transcript_failed(data)
            return False

        if msg_type == "error":
            self._clear_session_state()
            raise _status_error_from_event(data, "Aliyun STT error")

        logger.debug("ignoring Aliyun STT event %s", msg_type)
        return False

    def _handle_speech_started(self, data: dict[str, Any]) -> None:
        if self._opts.turn_detection is None:
            return

        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        item_state = self._get_or_create_item_state(item_id)
        audio_start_ms = data.get("audio_start_ms")
        if isinstance(audio_start_ms, (int, float)):
            item_state.audio_start_ms = int(audio_start_ms)

        if self._speaking:
            return

        self._speaking = True
        self._request_id = item_id
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                request_id=item_id,
            )
        )

    def _handle_speech_stopped(self, data: dict[str, Any]) -> None:
        if self._opts.turn_detection is None:
            return

        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        item_state = self._get_or_create_item_state(item_id)
        audio_end_ms = data.get("audio_end_ms")
        if isinstance(audio_end_ms, (int, float)):
            item_state.audio_end_ms = int(audio_end_ms)

        if not self._speaking:
            return

        self._speaking = False
        self._request_id = item_id
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                request_id=item_id,
            )
        )

    def _handle_transcript_preview(self, data: dict[str, Any]) -> None:
        if not self._opts.interim_results:
            return

        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        item_state = self._get_or_create_item_state(item_id)
        stable_text = str(data.get("text") or "")
        stash = str(data.get("stash") or "")
        preview_text = stable_text + stash
        if preview_text:
            self._maybe_emit_manual_start(item_id, has_text=True)

        language = _resolve_speech_language(data.get("language"), self._opts.language)

        # emotion = data.get("emotion")
        # if emotion is not None:
        #     logger.debug("Aliyun STT preview emotion=%s item_id=%s", emotion, item_id)

        if stable_text and stable_text != item_state.last_preflight_text:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.PREFLIGHT_TRANSCRIPT,
                    request_id=item_id,
                    alternatives=[
                        stt.SpeechData(
                            language=language,
                            text=stable_text,
                        )
                    ],
                )
            )
            item_state.last_preflight_text = stable_text

        if preview_text:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=item_id,
                    alternatives=[
                        stt.SpeechData(
                            language=language,
                            text=preview_text,
                        )
                    ],
                )
            )

    def _handle_transcript_completed(self, data: dict[str, Any]) -> None:
        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        item_state = self._get_or_create_item_state(item_id)
        transcript = str(data.get("transcript") or "")
        if self._opts.turn_detection is None and transcript:
            self._maybe_emit_manual_start(item_id, has_text=True)

        language = _resolve_speech_language(data.get("language"), self._opts.language)

        emotion = data.get("emotion")
        if emotion is not None:
            logger.debug("Aliyun STT final emotion=%s item_id=%s", emotion, item_id)

        self._request_id = item_id
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                request_id=item_id,
                alternatives=[
                    stt.SpeechData(
                        language=language,
                        text=transcript,
                        confidence=0.0,
                    )
                ],
            )
        )

        audio_duration: float | None = item_state.manual_audio_duration
        if (
            audio_duration is None
            and item_state.audio_start_ms is not None
            and item_state.audio_end_ms is not None
            and item_state.audio_end_ms > item_state.audio_start_ms
        ):
            audio_duration = (item_state.audio_end_ms - item_state.audio_start_ms) / 1000.0

        if audio_duration is not None:
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.RECOGNITION_USAGE,
                    request_id=item_id,
                    recognition_usage=stt.RecognitionUsage(
                        audio_duration=audio_duration,
                        input_tokens=0,
                        output_tokens=0,
                    ),
                )
            )

        self._maybe_emit_manual_end(item_id)
        self._item_state.pop(item_id, None)

    def _handle_transcript_failed(self, data: dict[str, Any]) -> None:
        item_id = data.get("item_id")
        item_request_id = item_id if isinstance(item_id, str) and item_id else None
        if item_request_id is not None:
            self._maybe_emit_manual_end(item_request_id)
        error = _status_error_from_event(data, "Aliyun transcription failed")
        self._emit_error(error, recoverable=True)
        if item_request_id is not None:
            self._item_state.pop(item_request_id, None)

    def _get_or_create_item_state(self, item_id: str) -> _ItemState:
        return self._item_state.setdefault(item_id, _ItemState())

    def _maybe_emit_manual_start(self, item_id: str, *, has_text: bool) -> None:
        if self._opts.turn_detection is not None or not has_text:
            return

        item_state = self._get_or_create_item_state(item_id)
        if item_state.saw_start_of_speech:
            return

        item_state.saw_start_of_speech = True
        self._speaking = True
        self._request_id = item_id
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.START_OF_SPEECH,
                request_id=item_id,
            )
        )

    def _maybe_emit_manual_end(self, item_id: str) -> None:
        if self._opts.turn_detection is not None:
            return

        item_state = self._get_or_create_item_state(item_id)
        if not item_state.saw_start_of_speech or item_state.sent_end_of_speech:
            return

        item_state.sent_end_of_speech = True
        self._speaking = False
        self._request_id = item_id
        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.END_OF_SPEECH,
                request_id=item_id,
            )
        )

    def _handle_committed(self, data: dict[str, Any]) -> None:
        item_id = data.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            return

        item_state = self._get_or_create_item_state(item_id)
        if self._pending_commit_durations:
            item_state.manual_audio_duration = self._pending_commit_durations.popleft()
            return

        logger.debug("Aliyun STT committed without pending manual duration item_id=%s", item_id)

    def _clear_session_state(self) -> None:
        self._item_state.clear()
        self._pending_commit_durations.clear()
        self._speaking = False
