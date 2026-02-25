from __future__ import annotations

import asyncio
import gzip
import json
import os
import uuid
import weakref
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Literal

import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

PROTOCOL_VERSION = 0b0001
DEFAULT_HEADER_SIZE = 0b0001

MESSAGE_TYPE_CLIENT_FULL = 0b0001
MESSAGE_TYPE_SERVER_FULL = 0b1001
MESSAGE_TYPE_SERVER_AUDIO = 0b1011
MESSAGE_TYPE_SERVER_ERROR = 0b1111

NO_SEQUENCE = 0b0000
POS_SEQUENCE = 0b0001
NEG_SEQUENCE = 0b0010
NEG_WITH_SEQUENCE = 0b0011

MSG_WITH_EVENT = 0b0100

SERIALIZATION_RAW = 0b0000
SERIALIZATION_JSON = 0b0001

COMPRESSION_NONE = 0b0000
COMPRESSION_GZIP = 0b0001

EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_CONNECTION_FINISHED = 52

EVENT_START_SESSION = 100
EVENT_CANCEL_SESSION = 101
EVENT_FINISH_SESSION = 102
EVENT_SESSION_STARTED = 150
EVENT_SESSION_CANCELED = 151
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153

EVENT_TASK_REQUEST = 200
EVENT_TTS_SENTENCE_START = 350
EVENT_TTS_SENTENCE_END = 351
EVENT_TTS_RESPONSE = 352

CODE_OK = 20000000

DEFAULT_BASE_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
DEFAULT_RESOURCE_ID = "seed-tts-2.0"
DEFAULT_MODEL = "seed-tts-2.0-standard"
DEFAULT_SPEAKER = "zh_female_vv_uranus_bigtts"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_FORMAT = "pcm"

TTSFormat = Literal["pcm", "mp3", "ogg_opus"]

_MIME_TYPES: dict[str, str] = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "ogg_opus": "audio/ogg",
}


def _json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _maybe_additions(additions: dict[str, Any] | str | None) -> str | None:
    if additions is None:
        return None
    if isinstance(additions, str):
        return additions
    return json.dumps(additions, ensure_ascii=False)


def _build_header(
    *,
    message_type: int,
    message_type_specific_flags: int = MSG_WITH_EVENT,
    serialization: int = SERIALIZATION_JSON,
    compression: int = COMPRESSION_NONE,
) -> bytearray:
    header = bytearray()
    header.append((PROTOCOL_VERSION << 4) | DEFAULT_HEADER_SIZE)
    header.append((message_type << 4) | message_type_specific_flags)
    header.append((serialization << 4) | compression)
    header.append(0x00)
    return header


def _build_request(
    *,
    event: int,
    payload_bytes: bytes,
    session_id: str | None = None,
    serialization: int = SERIALIZATION_JSON,
    compression: int = COMPRESSION_NONE,
) -> bytearray:
    frame = _build_header(
        message_type=MESSAGE_TYPE_CLIENT_FULL,
        message_type_specific_flags=MSG_WITH_EVENT,
        serialization=serialization,
        compression=compression,
    )
    frame.extend(int(event).to_bytes(4, "big"))
    if session_id is not None:
        session_bytes = session_id.encode("utf-8")
        frame.extend(len(session_bytes).to_bytes(4, "big"))
        frame.extend(session_bytes)
    frame.extend(len(payload_bytes).to_bytes(4, "big"))
    frame.extend(payload_bytes)
    return frame


def _build_start_connection() -> bytearray:
    return _build_request(event=EVENT_START_CONNECTION, payload_bytes=b"{}")


def _build_finish_connection() -> bytearray:
    return _build_request(event=EVENT_FINISH_CONNECTION, payload_bytes=b"{}")


def _build_start_session(session_id: str, payload: dict[str, Any]) -> bytearray:
    return _build_request(
        event=EVENT_START_SESSION,
        session_id=session_id,
        payload_bytes=_json_dumps(payload),
    )


def _build_finish_session(session_id: str) -> bytearray:
    return _build_request(event=EVENT_FINISH_SESSION, session_id=session_id, payload_bytes=b"{}")


def _build_cancel_session(session_id: str) -> bytearray:
    return _build_request(event=EVENT_CANCEL_SESSION, session_id=session_id, payload_bytes=b"{}")


def _build_task_request(session_id: str, payload: dict[str, Any]) -> bytearray:
    return _build_request(
        event=EVENT_TASK_REQUEST,
        session_id=session_id,
        payload_bytes=_json_dumps(payload),
    )


def _parse_response(data: bytes) -> dict[str, Any]:
    if not data:
        return {}

    header_size = data[0] & 0x0F
    message_type = data[1] >> 4
    message_flags = data[1] & 0x0F
    serialization = data[2] >> 4
    compression = data[2] & 0x0F

    payload = data[header_size * 4 :]
    result: dict[str, Any] = {
        "message_type": message_type,
        "message_flags": message_flags,
        "serialization": serialization,
        "compression": compression,
    }

    if message_type == MESSAGE_TYPE_SERVER_ERROR:
        if len(payload) < 8:
            result["error"] = {"code": -1, "message": "invalid error payload"}
            return result
        code = int.from_bytes(payload[:4], "big", signed=False)
        size = int.from_bytes(payload[4:8], "big", signed=False)
        msg = payload[8 : 8 + size]
        if compression == COMPRESSION_GZIP:
            msg = gzip.decompress(msg)
        try:
            msg_data = json.loads(msg.decode("utf-8"))
        except Exception:
            msg_data = {"message": msg.decode("utf-8", errors="ignore")}
        result["error"] = {"code": code, "message": msg_data}
        return result

    idx = 0
    if message_flags & NEG_SEQUENCE:
        if len(payload) >= 4:
            result["seq"] = int.from_bytes(payload[:4], "big", signed=False)
            idx += 4

    if message_flags & MSG_WITH_EVENT:
        if len(payload) >= idx + 4:
            result["event"] = int.from_bytes(payload[idx : idx + 4], "big", signed=False)
            idx += 4

    if len(payload) >= idx + 4:
        id_len = int.from_bytes(payload[idx : idx + 4], "big", signed=False)
        idx += 4
        if id_len > 0 and len(payload) >= idx + id_len:
            id_bytes = payload[idx : idx + id_len]
            idx += id_len
            result["id"] = id_bytes.decode("utf-8", errors="ignore")

    if len(payload) < idx + 4:
        return result

    payload_size = int.from_bytes(payload[idx : idx + 4], "big", signed=False)
    idx += 4
    payload_msg = payload[idx : idx + payload_size]

    if compression == COMPRESSION_GZIP:
        payload_msg = gzip.decompress(payload_msg)

    if serialization == SERIALIZATION_JSON:
        try:
            payload_msg = json.loads(payload_msg.decode("utf-8"))
        except Exception:
            payload_msg = {}

    result["payload_msg"] = payload_msg
    result["payload_size"] = payload_size
    return result


@dataclass
class _TTSOptions:
    base_url: str
    app_id: str
    access_token: str
    resource_id: str
    speaker: str
    model: str | None
    format: TTSFormat | str
    sample_rate: int
    bit_rate: int | None
    speech_rate: int
    loudness_rate: int
    emotion: str | None
    emotion_scale: int | None
    additions: dict[str, Any] | str | None
    require_usage_tokens_return: str | None

    def get_ws_headers(self) -> dict[str, str]:
        headers = {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        if self.require_usage_tokens_return:
            headers["X-Control-Require-Usage-Tokens-Return"] = self.require_usage_tokens_return
        return headers

    def build_start_session_payload(self, *, session_id: str) -> dict[str, Any]:
        audio_params: dict[str, Any] = {
            "format": self.format,
            "sample_rate": self.sample_rate,
        }
        if self.bit_rate is not None:
            audio_params["bit_rate"] = self.bit_rate
        if self.speech_rate is not None:
            audio_params["speech_rate"] = self.speech_rate
        if self.loudness_rate is not None:
            audio_params["loudness_rate"] = self.loudness_rate
        if self.emotion is not None:
            audio_params["emotion"] = self.emotion
        if self.emotion_scale is not None:
            audio_params["emotion_scale"] = self.emotion_scale

        req_params: dict[str, Any] = {
            "speaker": self.speaker,
            "audio_params": audio_params,
        }
        if self.model:
            req_params["model"] = self.model

        additions = _maybe_additions(self.additions)
        if additions is not None:
            req_params["additions"] = additions

        return {
            "user": {"uid": session_id},
            "event": EVENT_START_SESSION,
            "namespace": "BidirectionalTTS",
            "req_params": req_params,
        }

    def build_task_request_payload(self, *, session_id: str, text: str) -> dict[str, Any]:
        return {
            "user": {"uid": session_id},
            "event": EVENT_TASK_REQUEST,
            "namespace": "BidirectionalTTS",
            "req_params": {"text": text},
        }


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        app_id: str | None = None,
        access_token: str | None = None,
        resource_id: str = DEFAULT_RESOURCE_ID,
        speaker: str = DEFAULT_SPEAKER,
        model: str = DEFAULT_MODEL,
        format: TTSFormat | str = DEFAULT_FORMAT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bit_rate: int | None = None,
        speech_rate: int = 0,
        loudness_rate: int = 0,
        emotion: str | None = None,
        emotion_scale: int | None = None,
        additions: dict[str, Any] | str | None = None,
        require_usage_tokens_return: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: float = 600,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )

        app_id = app_id or os.getenv("VOLCENGINE_APP_ID")
        access_token = access_token or os.getenv("VOLCENGINE_ACCESS_TOKEN")
        if not app_id:
            raise ValueError("VOLCENGINE_APP_ID must be set")
        if not access_token:
            raise ValueError("VOLCENGINE_ACCESS_TOKEN must be set")

        self._session = http_session
        self._opts = _TTSOptions(
            base_url=base_url,
            app_id=app_id,
            access_token=access_token,
            resource_id=resource_id,
            speaker=speaker,
            model=model,
            format=format,
            sample_rate=sample_rate,
            bit_rate=bit_rate,
            speech_rate=speech_rate,
            loudness_rate=loudness_rate,
            emotion=emotion,
            emotion_scale=emotion_scale,
            additions=additions,
            require_usage_tokens_return=require_usage_tokens_return,
        )

        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=max_session_duration,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        ws = await asyncio.wait_for(
            session.ws_connect(
                self._opts.base_url,
                headers=self._opts.get_ws_headers(),
                max_msg_size=1000000000,
            ),
            timeout=timeout,
        )

        await ws.send_bytes(_build_start_connection())
        msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        if msg.type != aiohttp.WSMsgType.BINARY:
            await ws.close()
            raise APIStatusError("Volcengine TTS start connection failed", status_code=-1)
        resp = _parse_response(msg.data)
        event = resp.get("event")
        if event == EVENT_CONNECTION_FAILED:
            payload = resp.get("payload_msg", {})
            status = payload.get("status_code") if isinstance(payload, dict) else None
            message = payload.get("message") if isinstance(payload, dict) else "connection failed"
            await ws.close()
            raise APIStatusError(message, status_code=status)
        if event != EVENT_CONNECTION_STARTED:
            await ws.close()
            raise APIStatusError("Volcengine TTS connection not started", status_code=-1)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        with suppress(asyncio.CancelledError, aiohttp.ClientConnectionError):
            await ws.send_bytes(_build_finish_connection())
            msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
            if msg.type == aiohttp.WSMsgType.BINARY:
                _ = _parse_response(msg.data)
        # Connection already closing/closed; ignore to avoid noisy errors on idle timeouts
        with suppress(aiohttp.ClientConnectionError):
            with suppress(aiohttp.ClientConnectionResetError):
                await ws.close()

    @property
    def model(self) -> str:
        return self._opts.model or "unknown"

    @property
    def provider(self) -> str:
        return "volcengine"

    def update_options(
        self,
        *,
        resource_id: NotGivenOr[str] = NOT_GIVEN,
        speaker: NotGivenOr[str] = NOT_GIVEN,
        model: NotGivenOr[str | None] = NOT_GIVEN,
        format: NotGivenOr[TTSFormat | str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        speech_rate: NotGivenOr[int] = NOT_GIVEN,
        loudness_rate: NotGivenOr[int] = NOT_GIVEN,
        emotion: NotGivenOr[str | None] = NOT_GIVEN,
        emotion_scale: NotGivenOr[int | None] = NOT_GIVEN,
        additions: NotGivenOr[dict[str, Any] | str | None] = NOT_GIVEN,
    ) -> None:
        if is_given(resource_id) and resource_id != self._opts.resource_id:
            self._opts.resource_id = resource_id
            self._pool.invalidate()
        if is_given(speaker):
            self._opts.speaker = speaker
        if is_given(model):
            self._opts.model = model
        if is_given(format):
            self._opts.format = format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._sample_rate = sample_rate
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(speech_rate):
            self._opts.speech_rate = speech_rate
        if is_given(loudness_rate):
            self._opts.loudness_rate = loudness_rate
        if is_given(emotion):
            self._opts.emotion = emotion
        if is_given(emotion_scale):
            self._opts.emotion_scale = emotion_scale
        if is_given(additions):
            self._opts.additions = additions

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        self._pool.prewarm()

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        mime_type = _MIME_TYPES.get(str(self._opts.format), "audio/pcm")
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=mime_type,
            stream=True,
        )

        session_id = utils.shortuuid()
        output_emitter.start_segment(segment_id=session_id)
        session_started = asyncio.Event()
        session_finished = asyncio.Event()
        started_marked = False

        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Volcengine TTS websocket closed unexpectedly",
                        status_code=ws.close_code or -1,
                    )

                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue

                resp = _parse_response(msg.data)
                if "error" in resp:
                    err = resp["error"]
                    raise APIStatusError(str(err.get("message")), status_code=err.get("code"))

                event = resp.get("event")
                payload = resp.get("payload_msg")

                if event == EVENT_SESSION_STARTED:
                    session_started.set()
                elif event == EVENT_SESSION_FINISHED:
                    status_code = None
                    message = ""
                    if isinstance(payload, dict):
                        status_code = payload.get("status_code")
                        message = payload.get("message", "")
                    if status_code is not None and status_code != CODE_OK:
                        raise APIStatusError(message or "session finished with error", status_code)
                    session_finished.set()
                    break
                elif event in (EVENT_SESSION_FAILED, EVENT_SESSION_CANCELED):
                    status_code = None
                    message = "session failed"
                    if isinstance(payload, dict):
                        status_code = payload.get("status_code")
                        message = payload.get("message", message)
                    raise APIStatusError(message, status_code)
                elif event == EVENT_TTS_RESPONSE or event is None:
                    if isinstance(payload, (bytes, bytearray)):
                        output_emitter.push(bytes(payload))
                else:
                    continue

        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal started_marked
            await session_started.wait()
            async for item in self._input_ch:
                if session_finished.is_set():
                    break
                if isinstance(item, str):
                    if not item:
                        continue
                    payload = self._opts.build_task_request_payload(
                        session_id=session_id, text=item
                    )
                    await ws.send_bytes(_build_task_request(session_id, payload))
                    if not started_marked:
                        self._mark_started()
                        started_marked = True
                else:
                    await ws.send_bytes(_build_finish_session(session_id))
                    return

            if not session_finished.is_set():
                await ws.send_bytes(_build_finish_session(session_id))

        async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
            recv_t = asyncio.create_task(recv_task(ws))
            send_t: asyncio.Task | None = None
            started_task = asyncio.create_task(session_started.wait())
            try:
                start_payload = self._opts.build_start_session_payload(session_id=session_id)
                await ws.send_bytes(_build_start_session(session_id, start_payload))

                done, _ = await asyncio.wait(
                    [started_task, recv_t],
                    timeout=self._conn_options.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if recv_t in done:
                    recv_t.result()
                if started_task not in done:
                    raise APITimeoutError()

                send_t = asyncio.create_task(send_task(ws))

                done, _ = await asyncio.wait(
                    [send_t, recv_t],
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for task in done:
                    task.result()

                if not session_finished.is_set():
                    try:
                        await asyncio.wait_for(
                            session_finished.wait(), timeout=self._conn_options.timeout
                        )
                    except asyncio.TimeoutError as e:
                        raise APITimeoutError() from e

                await recv_t
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except APIStatusError:
                raise
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                if send_t is not None:
                    await utils.aio.gracefully_cancel(send_t)
                await utils.aio.gracefully_cancel(recv_t, started_task)
                output_emitter.end_segment()
