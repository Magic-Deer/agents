from __future__ import annotations

import asyncio
import base64
import json
import os
import weakref
from dataclasses import dataclass, replace
from typing import Any

import aiohttp
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, tts, utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger

DEFAULT_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_MODEL = "qwen3-tts-flash-realtime"
DEFAULT_VOICE = "Cherry"
DEFAULT_LANGUAGE_TYPE = "zh"
DEFAULT_RESPONSE_FORMAT = "pcm"
DEFAULT_SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    model: str
    voice: str
    language_type: str
    response_format: str
    sample_rate: int
    mode: str
    instructions: str | None
    optimize_instructions: bool | None
    speech_rate: float | None
    volume: int | None
    pitch_rate: int | None
    bit_rate: int | None
    api_key: str
    base_url: str


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        language_type: str = DEFAULT_LANGUAGE_TYPE,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        mode: str = "server_commit",
        instructions: str | None = None,
        optimize_instructions: bool | None = None,
        speech_rate: float | None = None,
        volume: int | None = None,
        pitch_rate: int | None = None,
        bit_rate: int | None = None,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        resolved_api_key = api_key if is_given(api_key) else os.environ.get("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Aliyun DashScope API key is required, either as argument or set "
                "DASHSCOPE_API_KEY environment variable"
            )

        self._opts = _TTSOptions(
            model=model,
            voice=voice,
            language_type=language_type,
            response_format=response_format,
            sample_rate=sample_rate,
            mode=mode,
            instructions=instructions,
            optimize_instructions=optimize_instructions,
            speech_rate=speech_rate,
            volume=volume,
            pitch_rate=pitch_rate,
            bit_rate=bit_rate,
            api_key=resolved_api_key,
            base_url=base_url,
        )

        self._session: aiohttp.ClientSession | None = None
        self._streams = weakref.WeakSet[SynthesizeStream]()

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

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language_type: NotGivenOr[str] = NOT_GIVEN,
        response_format: NotGivenOr[str] = NOT_GIVEN,
        sample_rate: NotGivenOr[int] = NOT_GIVEN,
        mode: NotGivenOr[str] = NOT_GIVEN,
        instructions: NotGivenOr[str | None] = NOT_GIVEN,
        optimize_instructions: NotGivenOr[bool | None] = NOT_GIVEN,
        speech_rate: NotGivenOr[float | None] = NOT_GIVEN,
        volume: NotGivenOr[int | None] = NOT_GIVEN,
        pitch_rate: NotGivenOr[int | None] = NOT_GIVEN,
        bit_rate: NotGivenOr[int | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language_type):
            self._opts.language_type = language_type
        if is_given(response_format):
            self._opts.response_format = response_format
        if is_given(sample_rate):
            self._opts.sample_rate = sample_rate
            self._sample_rate = sample_rate
        if is_given(mode):
            self._opts.mode = mode
        if is_given(instructions):
            self._opts.instructions = instructions
        if is_given(optimize_instructions):
            self._opts.optimize_instructions = optimize_instructions
        if is_given(speech_rate):
            self._opts.speech_rate = speech_rate
        if is_given(volume):
            self._opts.volume = volume
        if is_given(pitch_rate):
            self._opts.pitch_rate = pitch_rate
        if is_given(bit_rate):
            self._opts.bit_rate = bit_rate
        if is_given(base_url):
            self._opts.base_url = base_url

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return self._synthesize_with_stream(text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._session and not self._session.closed:
            await self._session.close()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts = tts
        self._opts = replace(tts._opts)
        self._session = tts._ensure_session()

    def _build_ws_url(self) -> str:
        base_url = self._opts.base_url
        if base_url.startswith("http://"):
            base_url = "ws://" + base_url[len("http://") :]
        elif base_url.startswith("https://"):
            base_url = "wss://" + base_url[len("https://") :]
        if "?" in base_url:
            return f"{base_url}&model={self._opts.model}"
        return f"{base_url}?model={self._opts.model}"

    def _event(self, event_type: str, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"event_id": utils.shortuuid(), "type": event_type}
        payload.update(fields)
        return payload

    def _build_session_update(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "mode": self._opts.mode,
            "voice": self._opts.voice,
            "language_type": self._opts.language_type,
            "response_format": self._opts.response_format,
            "sample_rate": self._opts.sample_rate,
        }
        if self._opts.instructions:
            session["instructions"] = self._opts.instructions
        if self._opts.optimize_instructions is not None:
            session["optimize_instructions"] = self._opts.optimize_instructions
        if self._opts.speech_rate is not None:
            session["speech_rate"] = self._opts.speech_rate
        if self._opts.volume is not None:
            session["volume"] = self._opts.volume
        if self._opts.pitch_rate is not None:
            session["pitch_rate"] = self._opts.pitch_rate
        if self._opts.bit_rate is not None:
            session["bit_rate"] = self._opts.bit_rate
        return self._event("session.update", session=session)

    def _mime_type(self) -> str:
        fmt = self._opts.response_format.lower()
        if fmt == "pcm":
            return "audio/pcm"
        if fmt == "wav":
            return "audio/wav"
        if fmt == "mp3":
            return "audio/mpeg"
        if fmt == "opus":
            return "audio/opus"
        return f"audio/{fmt}"

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        url = self._build_ws_url()
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "User-Agent": "LiveKit Agents",
        }
        try:
            return await asyncio.wait_for(
                self._session.ws_connect(url, headers=headers),
                timeout=self._conn_options.timeout,
            )
        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
            raise APIConnectionError("failed to connect to Aliyun TTS") from e

    @utils.log_exceptions(logger=logger)
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=self._mime_type(),
            stream=True,
        )

        ws = await self._connect_ws()
        try:
            await ws.send_str(json.dumps(self._build_session_update()))
            closing_ws = False
            segment_started = False
            segment_id = ""

            async def input_task() -> None:
                nonlocal closing_ws
                pending_text = False
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        if pending_text:
                            await ws.send_str(
                                json.dumps(self._event("input_text_buffer.commit"))
                            )
                            pending_text = False
                        continue

                    if not data:
                        continue
                    self._mark_started()
                    pending_text = True
                    await ws.send_str(
                        json.dumps(
                            self._event("input_text_buffer.append", text=data)
                        )
                    )

                if pending_text:
                    await ws.send_str(json.dumps(self._event("input_text_buffer.commit")))
                closing_ws = True
                await ws.send_str(json.dumps(self._event("session.finish")))

            async def recv_task() -> None:
                nonlocal closing_ws, segment_started, segment_id
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
                            message="Aliyun TTS connection closed unexpectedly",
                            status_code=ws.close_code or -1,
                            request_id=request_id,
                            body=f"{msg.data=} {msg.extra=}",
                        )
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue

                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "error":
                        raise APIStatusError(
                            message="Aliyun TTS error",
                            status_code=-1,
                            request_id=request_id,
                            body=data.get("error"),
                        )
                    if msg_type == "session.finished":
                        return
                    if msg_type == "response.created":
                        response = data.get("response", {})
                        segment_id = response.get("id") or data.get("response_id") or segment_id
                        if not segment_started:
                            output_emitter.start_segment(segment_id=segment_id or request_id)
                            segment_started = True
                        continue
                    if msg_type == "response.audio.delta":
                        if not segment_started:
                            output_emitter.start_segment(segment_id=segment_id or request_id)
                            segment_started = True
                        audio_b64 = data.get("delta", "")
                        if audio_b64:
                            output_emitter.push(base64.b64decode(audio_b64))
                        continue
                    if msg_type == "response.audio.done":
                        if segment_started:
                            output_emitter.end_segment()
                        continue
                    if msg_type == "response.done":
                        return

            input_atask = asyncio.create_task(input_task())
            recv_atask = asyncio.create_task(recv_task())

            try:
                done, _ = await asyncio.wait(
                    (input_atask, recv_atask),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if recv_atask in done:
                    recv_atask.result()
                    if not input_atask.done():
                        await utils.aio.gracefully_cancel(input_atask)
                else:
                    await recv_atask
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            finally:
                await utils.aio.gracefully_cancel(input_atask, recv_atask)
        finally:
            if not ws.closed:
                await ws.close()
