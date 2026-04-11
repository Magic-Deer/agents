from __future__ import annotations

import asyncio
import base64
import binascii
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)

from .log import logger

_DEFAULT_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
_DEFAULT_MODEL = "qwen3-tts-flash-realtime"
_DEFAULT_VOICE = "Cherry"
_DEFAULT_LANGUAGE_TYPE = "Auto"
_FIXED_RESPONSE_FORMAT = "pcm"
_FIXED_SAMPLE_RATE = 24000
_FIXED_NUM_CHANNELS = 1
_WS_CLOSE_TYPES = (
    aiohttp.WSMsgType.CLOSED,
    aiohttp.WSMsgType.CLOSE,
    aiohttp.WSMsgType.CLOSING,
)
_SUPPORTED_LANGUAGE_TYPES = {
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
}


@dataclass
class AliyunTTSOptions:
    api_key: str
    model: str
    voice: str
    language_type: str
    base_url: str


@dataclass
class _AttemptState:
    commit_sent: bool = False
    finish_sent: bool = False
    response_id: str | None = None
    segment_started: bool = False
    audio_done: bool = False
    response_done: bool = False


def _load_message(data: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as e:
        raise APIStatusError(
            message="Aliyun TTS returned invalid JSON",
            status_code=-1,
            body=data,
        ) from e

    if not isinstance(payload, dict):
        raise APIStatusError(
            message="Aliyun TTS returned a non-object message",
            status_code=-1,
            body=payload,
        )
    return payload


def _message_request_id(data: dict[str, Any]) -> str | None:
    response = data.get("response")
    if isinstance(response, dict):
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            return response_id

    top_level_response_id = data.get("response_id")
    if isinstance(top_level_response_id, str) and top_level_response_id:
        return top_level_response_id

    event_id = data.get("event_id")
    if isinstance(event_id, str) and event_id:
        return event_id

    error = data.get("error")
    if isinstance(error, dict):
        nested_event_id = error.get("event_id")
        if isinstance(nested_event_id, str) and nested_event_id:
            return nested_event_id

    return None


def _event_response_id(data: dict[str, Any]) -> str | None:
    response = data.get("response")
    if isinstance(response, dict):
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            return response_id

    top_level_response_id = data.get("response_id")
    if isinstance(top_level_response_id, str) and top_level_response_id:
        return top_level_response_id

    return None


def _require_str(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if isinstance(value, str) and value:
        return value

    raise APIStatusError(
        message=f"Aliyun TTS {context} is missing a valid {key}",
        status_code=-1,
        request_id=_message_request_id(data),
        body=data,
    )


def _decode_audio_delta(data: dict[str, Any]) -> bytes:
    delta = _require_str(data, "delta", context="response.audio.delta")
    try:
        return base64.b64decode(delta, validate=True)
    except binascii.Error as e:
        raise APIStatusError(
            message="Aliyun TTS response.audio.delta contained invalid base64 audio",
            status_code=-1,
            request_id=_message_request_id(data),
            body=data,
        ) from e


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
        retryable=False,
    )


async def _recv_event(
    ws: aiohttp.ClientWebSocketResponse,
    timeout: float,
) -> dict[str, Any]:
    try:
        msg = await asyncio.wait_for(ws.receive(), timeout)
    except asyncio.TimeoutError as e:
        raise APITimeoutError("Aliyun TTS request timed out") from e

    if msg.type in _WS_CLOSE_TYPES:
        if ws.exception() is not None:
            raise APIConnectionError("Aliyun TTS connection error") from ws.exception()
        raise APIStatusError(
            message="Aliyun TTS WebSocket closed unexpectedly",
            status_code=ws.close_code or -1,
            body={"data": msg.data, "extra": msg.extra},
        )

    if msg.type == aiohttp.WSMsgType.ERROR:
        if ws.exception() is not None:
            raise APIConnectionError("Aliyun TTS connection error") from ws.exception()
        raise APIConnectionError("Aliyun TTS connection error")

    if msg.type != aiohttp.WSMsgType.TEXT:
        raise APIStatusError(
            message="Aliyun TTS received a non-text message",
            status_code=-1,
            body={"type": msg.type},
        )

    return _load_message(msg.data)


async def _send_event(
    ws: aiohttp.ClientWebSocketResponse,
    payload: dict[str, Any],
) -> None:
    if "event_id" in payload:
        raise ValueError("_send_event payload must not include event_id")

    event = dict(payload)
    event["event_id"] = uuid4().hex
    await ws.send_json(event)


def _require_active_response_id(data: dict[str, Any], *, response_id: str | None, context: str) -> str:
    if response_id is None:
        raise APIStatusError(
            message=f"Aliyun TTS received {context} before response.created",
            status_code=-1,
            request_id=_message_request_id(data),
            body=data,
        )

    event_response_id = _event_response_id(data)
    if event_response_id is not None and event_response_id != response_id:
        raise APIStatusError(
            message=f"Aliyun TTS received {context} for unexpected response_id",
            status_code=-1,
            request_id=_message_request_id(data),
            body=data,
        )

    return response_id


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        voice: str = _DEFAULT_VOICE,
        language_type: str = _DEFAULT_LANGUAGE_TYPE,
        base_url: str = _DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not resolved_api_key:
            raise ValueError("DashScope API key is required")

        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not voice or not voice.strip():
            raise ValueError("voice must be a non-empty string")
        if language_type not in _SUPPORTED_LANGUAGE_TYPES:
            raise ValueError(f"language_type must be one of {sorted(_SUPPORTED_LANGUAGE_TYPES)}")
        if "?" in base_url:
            raise ValueError("base_url must not include query parameters; pass model separately")

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True, aligned_transcript=False),
            sample_rate=_FIXED_SAMPLE_RATE,
            num_channels=_FIXED_NUM_CHANNELS,
        )

        self._opts = AliyunTTSOptions(
            api_key=resolved_api_key,
            model=model,
            voice=voice,
            language_type=language_type,
            base_url=base_url,
        )
        self._http_session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

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

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        await asyncio.gather(*(stream.aclose() for stream in list(self._streams)))
        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = dataclasses.replace(tts._opts)

    @utils.log_exceptions(logger=logger)
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=_FIXED_SAMPLE_RATE,
            num_channels=_FIXED_NUM_CHANNELS,
            stream=True,
            mime_type="audio/pcm",
        )

        self._connection_reused = False

        ws: aiohttp.ClientWebSocketResponse | None = None
        tasks: list[asyncio.Task[None]] = []
        state = _AttemptState()
        try:
            ws = await self._connect_ws()
            tasks = [
                asyncio.create_task(self._send_task(ws, state)),
                asyncio.create_task(self._recv_task(ws, output_emitter, state)),
            ]
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            if ws is not None and not ws.closed:
                await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = f"{self._opts.base_url}?{urlencode({'model': self._opts.model})}"
        headers = {"Authorization": f"Bearer {self._opts.api_key}"}

        ws: aiohttp.ClientWebSocketResponse | None = None
        connected = False
        started_at = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                self._tts.session.ws_connect(url, headers=headers),
                self._conn_options.timeout,
            )

            first_event = await self._receive_handshake_event(ws)
            first_type = first_event.get("type")
            if first_type == "error":
                raise _status_error_from_event(first_event, "Aliyun TTS error")
            if first_type != "session.created":
                raise APIStatusError(
                    message="Aliyun TTS expected session.created during handshake",
                    status_code=-1,
                    request_id=_message_request_id(first_event),
                    body=first_event,
                )

            await _send_event(ws, self._build_session_update_event())

            while True:
                event = await self._receive_handshake_event(ws)
                event_type = event.get("type")
                if event_type == "session.updated":
                    connected = True
                    self._acquire_time = time.perf_counter() - started_at
                    return ws
                if event_type == "error":
                    raise _status_error_from_event(event, "Aliyun TTS error")

                logger.debug("ignoring Aliyun TTS handshake event %s", event_type)
        except APITimeoutError:
            raise
        except asyncio.TimeoutError as e:
            raise APITimeoutError("Aliyun TTS request timed out") from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message or "Aliyun TTS request failed",
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except aiohttp.ClientConnectorError as e:
            raise APIConnectionError("Failed to connect to Aliyun TTS") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Aliyun TTS connection error") from e
        finally:
            if ws is not None and not connected:
                await ws.close()

    async def _receive_handshake_event(
        self,
        ws: aiohttp.ClientWebSocketResponse,
    ) -> dict[str, Any]:
        return await _recv_event(ws, timeout=self._conn_options.timeout)

    def _build_session_update_event(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "voice": self._opts.voice,
                "mode": "commit",
                "language_type": self._opts.language_type,
                "response_format": _FIXED_RESPONSE_FORMAT,
                "sample_rate": _FIXED_SAMPLE_RATE,
            },
        }

    @utils.log_exceptions(logger=logger)
    async def _send_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        state: _AttemptState,
    ) -> None:
        has_text = False

        async for data in self._input_ch:
            if isinstance(data, str):
                if not data:
                    continue
                await _send_event(
                    ws,
                    {
                        "type": "input_text_buffer.append",
                        "text": data,
                    },
                )
                has_text = True
                continue

            if not has_text or state.commit_sent:
                continue

            self._mark_started()
            await _send_event(ws, {"type": "input_text_buffer.commit"})
            state.commit_sent = True

        if not state.finish_sent:
            await _send_event(ws, {"type": "session.finish"})
            state.finish_sent = True

    @utils.log_exceptions(logger=logger)
    async def _recv_task(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        state: _AttemptState,
    ) -> None:
        while True:
            event = await _recv_event(ws, timeout=self._conn_options.timeout)
            event_type = event.get("type")

            if event_type in {
                "session.created",
                "session.updated",
                "input_text_buffer.committed",
                "response.output_item.added",
                "response.content_part.added",
                "response.content_part.done",
                "response.output_item.done",
            }:
                continue

            if event_type == "response.created":
                response = event.get("response")
                if not isinstance(response, dict):
                    raise APIStatusError(
                        message="Aliyun TTS response.created missing response object",
                        status_code=-1,
                        request_id=_message_request_id(event),
                        body=event,
                    )

                response_id = _require_str(response, "id", context="response.created.response")
                if state.response_id is not None and state.response_id != response_id:
                    raise APIStatusError(
                        message="Aliyun TTS received a second active response.created",
                        status_code=-1,
                        request_id=_message_request_id(event),
                        body=event,
                    )

                state.response_id = response_id
                continue

            if event_type == "response.audio.delta":
                response_id = _require_active_response_id(
                    event,
                    response_id=state.response_id,
                    context="response.audio.delta",
                )
                if not state.segment_started:
                    output_emitter.start_segment(segment_id=response_id)
                    state.segment_started = True

                output_emitter.push(_decode_audio_delta(event))
                continue

            if event_type == "response.audio.done":
                _require_active_response_id(
                    event,
                    response_id=state.response_id,
                    context="response.audio.done",
                )
                if state.segment_started and not state.audio_done:
                    output_emitter.end_segment()
                state.audio_done = True
                continue

            if event_type == "response.done":
                _require_active_response_id(
                    event,
                    response_id=state.response_id,
                    context="response.done",
                )

                response = event.get("response")
                if not isinstance(response, dict):
                    raise APIStatusError(
                        message="Aliyun TTS response.done missing response object",
                        status_code=-1,
                        request_id=_message_request_id(event),
                        body=event,
                    )

                usage = response.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise APIStatusError(
                            message="Aliyun TTS response.done usage must be an object",
                            status_code=-1,
                            request_id=_message_request_id(event),
                            body=event,
                        )

                    token_usage: dict[str, int] = {}
                    for key in ("input_tokens", "output_tokens"):
                        if key not in usage:
                            continue
                        value = usage[key]
                        if type(value) is not int:
                            raise APIStatusError(
                                message=f"Aliyun TTS response.done usage.{key} must be an integer",
                                status_code=-1,
                                request_id=_message_request_id(event),
                                body=event,
                            )
                        token_usage[key] = value

                    if token_usage:
                        self._set_token_usage(**token_usage)

                state.response_done = True
                continue

            if event_type == "session.finished":
                if state.response_done and state.segment_started and not state.audio_done:
                    logger.warning(
                        "Aliyun TTS session.finished arrived before response.audio.done; closing segment"
                    )
                    output_emitter.end_segment()
                    state.audio_done = True

                if state.response_done:
                    return

                if not state.commit_sent and state.finish_sent:
                    return

                raise APIStatusError(
                    message="Aliyun TTS session.finished arrived before response.done",
                    status_code=-1,
                    request_id=_message_request_id(event),
                    body=event,
                )

            if event_type == "error":
                raise _status_error_from_event(event, "Aliyun TTS error")

            logger.debug("ignoring Aliyun TTS event %s", event_type)
