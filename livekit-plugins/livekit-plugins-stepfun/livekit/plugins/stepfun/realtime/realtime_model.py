from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal, overload
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from openai.types.beta.realtime import (
    ConversationItem,
    ConversationItemContent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferClearEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    SessionUpdateEvent,
    session_update_event,
)
from openai.types.beta.realtime.response_create_event import Response
from openai.types.beta.realtime.session import (
    InputAudioNoiseReduction,
    InputAudioTranscription,
    Tracing,
    TracingTracingConfiguration,
)
from pydantic import BaseModel, ValidationError

from livekit import rtc
from livekit.agents import APIConnectionError, APIError, io, llm, utils
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.agents.voice.generation import remove_instructions

from ..log import logger
from ..tools import StepfunTool

STEPFUN_BASE_URL = "https://api.stepfun.com/v1"
DEFAULT_MODEL = "step-audio-2"
DEFAULT_VOICE = "qingchunshaonv"
MAX_TOOLS = 128


SAMPLE_RATE = 24000
NUM_CHANNELS = 1

lk_stepfun_debug = int(os.getenv("LK_STEPFUN_DEBUG", 0))


@dataclass
class TurnDetection:
    """StepFun turn detection configuration.

    Parameters match StepFun's API specification:
    https://platform.stepfun.com/docs/zh/api-reference/realtime/chat
    """

    type: Literal["server_vad"] = "server_vad"
    """Type of turn detection, currently only 'server_vad' is supported."""

    prefix_padding_ms: int = 500
    """Duration in milliseconds to include before speech starts. Default: 500ms."""

    silence_duration_ms: int = 100
    """Duration of silence in milliseconds to consider speech ended. Default: 100ms."""

    energy_awakeness_threshold: int = 500
    """Energy threshold (0-5000) for speech detection. Default: 500."""

    def __post_init__(self) -> None:
        if self.type != "server_vad":
            raise ValueError("TurnDetection.type must be 'server_vad'")
        if not (0 <= self.energy_awakeness_threshold <= 5000):
            raise ValueError("TurnDetection.energy_awakeness_threshold must be between 0 and 5000")

@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    temperature: float
    tool_choice: llm.ToolChoice | None
    input_audio_transcription: InputAudioTranscription | None
    input_audio_noise_reduction: InputAudioNoiseReduction | None
    turn_detection: TurnDetection | None
    max_response_output_tokens: int | Literal["inf"] | None
    speed: float | None
    tracing: Tracing | None
    api_key: str | None
    base_url: str
    modalities: list[Literal["text", "audio"]]
    max_session_duration: float | None
    """reset the connection after this many seconds if provided"""
    conn_options: APIConnectOptions


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    audio_transcript: str = ""


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    messages: dict[str, _MessageGeneration]

    _done_fut: asyncio.Future[None]
    _created_timestamp: float
    """timestamp when the response was created"""
    _first_token_timestamp: float | None = None
    """timestamp when the first token was received"""


DEFAULT_TEMPERATURE = 0.8
DEFAULT_INPUT_AUDIO_TRANSCRIPTION = InputAudioTranscription(
    model="gpt-4o-mini-transcribe",
)
DEFAULT_TOOL_CHOICE = "auto"
DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS = "inf"

DEFAULT_MAX_SESSION_DURATION = 20 * 60  # 20 minutes


@dataclass
class _FunctionCallState:
    item_id: str
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""
    done: bool = False
    emitted: bool = False


class RealtimeModel(llm.RealtimeModel):
    @overload
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None: ...

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        modalities: NotGivenOr[list[Literal["text", "audio"]]] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: InputAudioNoiseReduction | None = None,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        modalities = modalities if is_given(modalities) else ["text", "audio"]
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=turn_detection is not None,
                user_transcription=input_audio_transcription is not None,
                auto_tool_reply_generation=False,
                audio_output="audio" in modalities,
                manual_function_calls=True,
            )
        )

        api_key = api_key or os.environ.get("STEP_API_KEY") or os.environ.get("STEPFUN_API_KEY")
        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the STEP_API_KEY/STEPFUN_API_KEY "
                "environment variable"
            )

        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            temperature=temperature if is_given(temperature) else DEFAULT_TEMPERATURE,
            tool_choice=tool_choice or None,
            modalities=modalities,
            input_audio_transcription=input_audio_transcription
            if is_given(input_audio_transcription)
            else DEFAULT_INPUT_AUDIO_TRANSCRIPTION,
            input_audio_noise_reduction=input_audio_noise_reduction,
            turn_detection=turn_detection if is_given(turn_detection) else TurnDetection(),
            api_key=api_key,
            base_url=base_url if is_given(base_url) else STEPFUN_BASE_URL,
            max_response_output_tokens=DEFAULT_MAX_RESPONSE_OUTPUT_TOKENS,  # type: ignore
            speed=speed if is_given(speed) else None,
            tracing=tracing if is_given(tracing) else None,
            max_session_duration=max_session_duration
            if is_given(max_session_duration)
            else DEFAULT_MAX_SESSION_DURATION,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._http_session_owned = False
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self._opts.base_url).netloc

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[InputAudioNoiseReduction | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        if is_given(voice):
            self._opts.voice = voice

        if is_given(temperature):
            self._opts.temperature = temperature

        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection

        if is_given(tool_choice):
            self._opts.tool_choice = tool_choice

        if is_given(input_audio_transcription):
            self._opts.input_audio_transcription = input_audio_transcription

        if is_given(input_audio_noise_reduction):
            self._opts.input_audio_noise_reduction = input_audio_noise_reduction

        if is_given(max_response_output_tokens):
            self._opts.max_response_output_tokens = max_response_output_tokens

        if is_given(speed):
            self._opts.speed = speed

        if is_given(tracing):
            self._opts.tracing = tracing

        for sess in self._sessions:
            sess.update_options(
                voice=voice,
                temperature=temperature,
                turn_detection=turn_detection,
                tool_choice=tool_choice,
                input_audio_transcription=input_audio_transcription,
                max_response_output_tokens=max_response_output_tokens,
                speed=speed,
                tracing=tracing,
            )

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            try:
                self._http_session = utils.http_context.http_session()
            except RuntimeError:
                self._http_session = aiohttp.ClientSession()
                self._http_session_owned = True

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        if self._http_session_owned and self._http_session:
            await self._http_session.close()


def process_base_url(url: str, model: str) -> str:
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # Ensure "/realtime" only for base paths.
    if not parsed_url.path or parsed_url.path.rstrip("/") in ["", "/v1"]:
        path = parsed_url.path.rstrip("/") + "/realtime"
    else:
        path = parsed_url.path

    if "model" not in query_params:
        query_params["model"] = [model]

    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", new_query, ""))

    return new_url


class RealtimeSession(
    llm.RealtimeSession[
        Literal[
            "openai_server_event_received",
            "openai_client_event_queued",
            "stepfun_server_event_received",
            "stepfun_client_event_queued",
        ]
    ]
):
    """
    A session for the Stepfun Realtime API.

    This class is used to interact with the Stepfun Realtime API.
    It is responsible for sending events to and receiving events from Stepfun.

    It exposes raw compatibility events:
    - openai_server_event_received: expose raw server payloads for compatibility with existing hooks
    - openai_client_event_queued: expose raw client payloads for compatibility with existing hooks
    - stepfun_server_event_received: same payload as openai_server_event_received
    - stepfun_client_event_queued: same payload as openai_client_event_queued
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[RealtimeClientEvent | dict[str, Any]]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._instructions: str | None = None
        self._main_atask = asyncio.create_task(self._main_task(), name="RealtimeSession._main_task")
        self.send_event(self._create_session_update_event())

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}
        self._pending_response_create_ids: list[str] = []

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        self._function_call_states: dict[str, _FunctionCallState] = {}
        self._function_item_by_call_id: dict[str, str] = {}

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        # 100ms chunks
        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=SAMPLE_RATE // 10
        )
        self._pushed_duration_s: float = 0
        self.on("openai_server_event_received", self._handle_server_event)
        self.on(
            "openai_server_event_received",
            lambda event: self.emit("stepfun_server_event_received", event),
        )
        self.on(
            "openai_client_event_queued",
            lambda event: self.emit("stepfun_client_event_queued", event),
        )

    def send_event(self, event: RealtimeClientEvent | dict[str, Any]) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        num_retries: int = 0
        max_retries = self._realtime_model._opts.conn_options.max_retry

        async def _reconnect() -> None:
            logger.debug(
                "reconnecting to Stepfun Realtime API",
                extra={"max_session_duration": self._realtime_model._opts.max_session_duration},
            )

            events: list[RealtimeClientEvent | dict[str, Any]] = []

            # options and instructions
            events.append(self._create_session_update_event())

            # tools
            tools = self._tools.flatten()
            if tools:
                events.append(self._create_tools_update_event(tools))

            # chat context
            chat_ctx = self.chat_ctx.copy(
                exclude_function_call=True,
                exclude_instructions=True,
                exclude_empty_message=True,
                exclude_handoff=True,
                exclude_config_update=True,
            )
            old_chat_ctx = self._remote_chat_ctx
            self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
            events.extend(self._create_update_chat_ctx_events(chat_ctx))

            try:
                for ev in events:
                    msg: dict[str, Any]
                    if isinstance(ev, BaseModel):
                        msg = ev.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )
                    else:
                        msg = ev
                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))
            except Exception as e:
                self._remote_chat_ctx = old_chat_ctx  # restore the old chat context
                raise APIConnectionError(
                    message=(
                        "Failed to send message to Stepfun Realtime API during session "
                        "re-connection"
                    ),
                ) from e

            logger.debug("reconnected to Stepfun Realtime API")
            self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

        reconnecting = False
        while not self._msg_ch.closed:
            try:
                ws_conn = await self._create_ws_conn()
                if reconnecting:
                    await _reconnect()
                    num_retries = 0  # reset the retry counter
                await self._run_ws(ws_conn)

            except APIError as e:
                if max_retries == 0 or not e.retryable:
                    self._emit_error(e, recoverable=False)
                    raise
                elif num_retries == max_retries:
                    self._emit_error(e, recoverable=False)
                    raise APIConnectionError(
                        f"Stepfun Realtime API connection failed after {num_retries} attempts",
                    ) from e
                else:
                    self._emit_error(e, recoverable=True)

                    retry_interval = self._realtime_model._opts.conn_options._interval_for_retry(
                        num_retries
                    )
                    logger.warning(
                        f"Stepfun Realtime API connection failed, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={"attempt": num_retries, "max_retries": max_retries},
                    )
                    await asyncio.sleep(retry_interval)
                num_retries += 1

            except Exception as e:
                self._emit_error(e, recoverable=False)
                raise

            reconnecting = True

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._realtime_model._opts.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        url = process_base_url(self._realtime_model._opts.base_url, self._realtime_model._opts.model)

        if lk_stepfun_debug:
            logger.debug(f"connecting to Realtime API: {url}")

        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(url=url, headers=headers),
                self._realtime_model._opts.conn_options.timeout,
            )
        except aiohttp.ClientError as e:
            raise APIConnectionError("Stepfun Realtime API client connection error") from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError(
                message="Stepfun Realtime API connection timed out",
            ) from e

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    if isinstance(msg, BaseModel):
                        msg = msg.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )

                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))

                    if lk_stepfun_debug:
                        msg_copy = msg.copy()
                        if msg_copy["type"] == "input_audio_buffer.append":
                            msg_copy = {**msg_copy, "audio": "..."}

                        logger.debug(f">>> {msg_copy}")
                except Exception:
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing:  # closing is expected, see _send_task
                        return

                    # this will trigger a reconnection
                    raise APIConnectionError(message="Stepfun S2S connection closed unexpectedly")

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)

                # emit the raw json dictionary instead of the BaseModel because different
                # providers can have different event types that are not part of typed models.
                self.emit("openai_server_event_received", event)

                try:
                    if lk_stepfun_debug:
                        event_copy = event.copy()
                        if event_copy["type"] == "response.audio.delta":
                            event_copy = {**event_copy, "delta": "..."}

                        logger.debug(f"<<< {event_copy}")

                    if event["type"] == "input_audio_buffer.speech_started":
                        self._handle_input_audio_buffer_speech_started(
                            InputAudioBufferSpeechStartedEvent.construct(**event)
                        )
                    elif event["type"] == "input_audio_buffer.speech_stopped":
                        self._handle_input_audio_buffer_speech_stopped(
                            InputAudioBufferSpeechStoppedEvent.construct(**event)
                        )
                    elif event["type"] == "response.created":
                        self._handle_response_created(ResponseCreatedEvent.construct(**event))
                    elif event["type"] == "response.output_item.added":
                        self._handle_response_output_item_added(
                            ResponseOutputItemAddedEvent.construct(**event)
                        )
                    elif event["type"] == "response.content_part.added":
                        self._handle_response_content_part_added(
                            ResponseContentPartAddedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.created":
                        self._handle_conversion_item_created(
                            ConversationItemCreatedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.deleted":
                        self._handle_conversion_item_deleted(
                            ConversationItemDeletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.completed":
                        self._handle_conversion_item_input_audio_transcription_completed(
                            ConversationItemInputAudioTranscriptionCompletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.failed":
                        self._handle_conversion_item_input_audio_transcription_failed(
                            ConversationItemInputAudioTranscriptionFailedEvent.construct(**event)
                        )
                    elif event["type"] == "response.text.delta":
                        self._handle_response_text_delta(ResponseTextDeltaEvent.construct(**event))
                    elif event["type"] == "response.text.done":
                        self._handle_response_text_done(ResponseTextDoneEvent.construct(**event))
                    elif event["type"] == "response.audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(event)
                    elif event["type"] == "response.audio.delta":
                        self._handle_response_audio_delta(
                            ResponseAudioDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.done":
                        self._handle_response_audio_transcript_done(
                            ResponseAudioTranscriptDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio.done":
                        self._handle_response_audio_done(ResponseAudioDoneEvent.construct(**event))
                    elif event["type"] == "response.output_item.done":
                        self._handle_response_output_item_done(
                            ResponseOutputItemDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.done":
                        self._handle_response_done(ResponseDoneEvent.construct(**event))
                    elif event["type"] == "error":
                        self._handle_error(ErrorEvent.construct(**event))
                except Exception:
                    if event["type"] == "response.audio.delta":
                        event["delta"] = event["delta"][:10] + "..."
                    logger.exception("failed to handle event", extra={"event": event})

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        wait_reconnect_task: asyncio.Task | None = None
        if self._realtime_model._opts.max_session_duration is not None:
            wait_reconnect_task = asyncio.create_task(
                asyncio.sleep(self._realtime_model._opts.max_session_duration),
                name="_timeout_task",
            )
            tasks.append(wait_reconnect_task)
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # propagate exceptions from completed tasks
            for task in done:
                if task != wait_reconnect_task:
                    task.result()

            if wait_reconnect_task and wait_reconnect_task in done and self._current_generation:
                # wait for the current generation to complete before reconnecting
                await self._current_generation._done_fut
                closing = True

        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            return

        # Alias events before the base event switch runs.
        if event_type == "conversation.item.added":
            event["type"] = "conversation.item.created"
            event_type = "conversation.item.created"
        elif event_type == "response.output_item.created":
            event["type"] = "response.output_item.added"
            event_type = "response.output_item.added"

        if event_type == "error":
            err = event.get("error")
            if isinstance(err, str):
                event["error"] = {"type": "server_error", "message": err}

        if event_type == "conversation.item.created":
            item = event.get("item")
            if isinstance(item, dict):
                if item.get("type") == "message" and item.get("content") is None:
                    item["content"] = []
                if item.get("type") == "function_call":
                    self._record_function_call(
                        item_id=self._safe_str(item.get("id")),
                        call_id=self._safe_str(item.get("call_id")),
                        name=self._safe_str(item.get("name")),
                        arguments=self._safe_str(item.get("arguments")),
                        done=item.get("arguments") is not None,
                    )
                    item_id = self._safe_str(item.get("id"))
                    if item_id:
                        self._emit_function_call_if_ready(item_id)
            return

        if event_type == "response.function_call_arguments.delta":
            self._handle_response_function_call_arguments_delta(event)
            return

        if event_type == "response.function_call_arguments.done":
            self._handle_response_function_call_arguments_done(event)
            return

        # Intentionally ignored; raw event already exposed through *server_event_received.
        if event_type in ("response.thinking.delta", "response.thinking.done"):
            return

    @staticmethod
    def _safe_str(value: Any) -> str | None:
        return value if isinstance(value, str) else None

    def _record_function_call(
        self,
        *,
        item_id: str | None,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        append_arguments: bool = False,
        done: bool = False,
    ) -> _FunctionCallState | None:
        if item_id is None and call_id is not None:
            item_id = self._function_item_by_call_id.get(call_id)
        if item_id is None:
            return None

        state = self._function_call_states.get(item_id)
        if state is None:
            state = _FunctionCallState(item_id=item_id)
            self._function_call_states[item_id] = state

        if call_id is not None:
            state.call_id = call_id
            self._function_item_by_call_id[call_id] = item_id
        if name is not None:
            state.name = name
        if arguments is not None:
            if append_arguments:
                state.arguments += arguments
            else:
                state.arguments = arguments
        if done:
            state.done = True

        return state

    def _emit_function_call_if_ready(self, item_id: str) -> None:
        if self._current_generation is None:
            return

        state = self._function_call_states.get(item_id)
        if state is None or state.emitted:
            return
        if state.call_id is None or state.name is None:
            return
        if not state.done and not state.arguments:
            return

        self._current_generation.function_ch.send_nowait(
            llm.FunctionCall(
                id=item_id,
                call_id=state.call_id,
                name=state.name,
                arguments=state.arguments,
            )
        )
        state.emitted = True

    def _handle_response_function_call_arguments_delta(self, event: dict[str, Any]) -> None:
        state = self._record_function_call(
            item_id=self._safe_str(event.get("item_id")),
            call_id=self._safe_str(event.get("call_id")),
            name=self._safe_str(event.get("name")),
            arguments=self._safe_str(event.get("delta")),
            append_arguments=True,
            done=False,
        )
        if state is not None:
            self._emit_function_call_if_ready(state.item_id)

    def _handle_response_function_call_arguments_done(self, event: dict[str, Any]) -> None:
        arguments = self._safe_str(event.get("arguments"))
        state = self._record_function_call(
            item_id=self._safe_str(event.get("item_id")),
            call_id=self._safe_str(event.get("call_id")),
            name=self._safe_str(event.get("name")),
            arguments=arguments,
            append_arguments=False,
            done=True,
        )
        if state is not None:
            self._emit_function_call_if_ready(state.item_id)

    @staticmethod
    def _item_field(item: Any, field: str) -> Any:
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)

    def _hydrate_function_calls_from_response_done(self, event: ResponseDoneEvent) -> None:
        response_output = getattr(event.response, "output", None)
        if not response_output:
            return

        for out_item in response_output:
            if self._item_field(out_item, "type") != "function_call":
                continue
            item_id = self._safe_str(self._item_field(out_item, "id"))
            state = self._record_function_call(
                item_id=item_id,
                call_id=self._safe_str(self._item_field(out_item, "call_id")),
                name=self._safe_str(self._item_field(out_item, "name")),
                arguments=self._safe_str(self._item_field(out_item, "arguments")),
                done=True,
            )
            if state is not None:
                self._emit_function_call_if_ready(state.item_id)

    def _create_session_update_event(self) -> SessionUpdateEvent:
        input_audio_transcription_opts = self._realtime_model._opts.input_audio_transcription
        input_audio_transcription = (
            session_update_event.SessionInputAudioTranscription.model_validate(
                input_audio_transcription_opts.model_dump(
                    by_alias=True,
                    exclude_unset=True,
                    exclude_defaults=True,
                )
            )
            if input_audio_transcription_opts
            else None
        )

        turn_detection_opts = self._realtime_model._opts.turn_detection
        turn_detection = None
        if turn_detection_opts:
            turn_detection = session_update_event.SessionTurnDetection.model_construct(
                **asdict(turn_detection_opts)
            )

        tracing_opts = self._realtime_model._opts.tracing
        if isinstance(tracing_opts, TracingTracingConfiguration):
            tracing: session_update_event.SessionTracing | None = (
                session_update_event.SessionTracingTracingConfiguration.model_validate(
                    tracing_opts.model_dump(
                        by_alias=True,
                        exclude_unset=True,
                        exclude_defaults=True,
                    )
                )
            )
        else:
            tracing = tracing_opts

        kwargs: dict[str, Any] = {
            "model": self._realtime_model._opts.model,
            "voice": self._realtime_model._opts.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "modalities": self._realtime_model._opts.modalities,
            "turn_detection": turn_detection,
            "input_audio_transcription": input_audio_transcription,
            "input_audio_noise_reduction": self._realtime_model._opts.input_audio_noise_reduction,
            "temperature": self._realtime_model._opts.temperature,
            "tool_choice": _to_oai_tool_choice(self._realtime_model._opts.tool_choice),
        }
        if self._instructions is not None:
            kwargs["instructions"] = self._instructions

        if self._realtime_model._opts.speed is not None:
            kwargs["speed"] = self._realtime_model._opts.speed

        if tracing:
            kwargs["tracing"] = tracing

        # initial session update
        return SessionUpdateEvent(
            type="session.update",
            # Using model_construct because typed voices are restrictive across providers.
            session=session_update_event.Session.model_construct(**kwargs),
            event_id=utils.shortuuid("session_update_"),
        )

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection | None] = NOT_GIVEN,
        max_response_output_tokens: NotGivenOr[int | Literal["inf"] | None] = NOT_GIVEN,
        input_audio_transcription: NotGivenOr[InputAudioTranscription | None] = NOT_GIVEN,
        input_audio_noise_reduction: NotGivenOr[InputAudioNoiseReduction | None] = NOT_GIVEN,
        speed: NotGivenOr[float] = NOT_GIVEN,
        tracing: NotGivenOr[Tracing | None] = NOT_GIVEN,
    ) -> None:
        kwargs: dict[str, Any] = {}

        if is_given(tool_choice):
            self._realtime_model._opts.tool_choice = tool_choice
            kwargs["tool_choice"] = _to_oai_tool_choice(tool_choice)

        if is_given(voice):
            self._realtime_model._opts.voice = voice
            kwargs["voice"] = voice

        if is_given(temperature):
            self._realtime_model._opts.temperature = temperature
            kwargs["temperature"] = temperature

        if is_given(turn_detection):
            self._realtime_model._opts.turn_detection = turn_detection
            if turn_detection:
                kwargs["turn_detection"] = session_update_event.SessionTurnDetection.model_construct(
                    **asdict(turn_detection)
                )
            else:
                kwargs["turn_detection"] = None

        if is_given(max_response_output_tokens):
            self._realtime_model._opts.max_response_output_tokens = max_response_output_tokens
            kwargs["max_response_output_tokens"] = max_response_output_tokens

        if is_given(input_audio_transcription):
            self._realtime_model._opts.input_audio_transcription = input_audio_transcription
            kwargs["input_audio_transcription"] = input_audio_transcription

        if is_given(input_audio_noise_reduction):
            self._realtime_model._opts.input_audio_noise_reduction = input_audio_noise_reduction
            kwargs["input_audio_noise_reduction"] = input_audio_noise_reduction

        if is_given(speed):
            self._realtime_model._opts.speed = speed
            kwargs["speed"] = speed

        if is_given(tracing):
            self._realtime_model._opts.tracing = tracing
            kwargs["tracing"] = tracing

        if kwargs:
            self.send_event(
                SessionUpdateEvent(
                    type="session.update",
                    session=session_update_event.Session.model_construct(**kwargs),
                    event_id=utils.shortuuid("options_update_"),
                )
            )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        chat_ctx = chat_ctx.copy(
            exclude_handoff=True,
            exclude_config_update=True,
        )
        remove_instructions(chat_ctx)

        try:
            async with self._update_chat_ctx_lock:
                events = self._create_update_chat_ctx_events(chat_ctx)
                futs: list[asyncio.Future[None]] = []

                for ev in events:
                    if isinstance(ev, ConversationItemDeleteEvent):
                        futs.append(f := asyncio.Future[None]())
                        self._item_delete_future[ev.item_id] = f
                    elif isinstance(ev, ConversationItemCreateEvent):
                        assert ev.item.id is not None
                        # Stepfun may not send conversation.item.created ack for function_call_output items
                        # so we don't wait for them to avoid timeout, and sync optimistically immediately
                        if ev.item.type == "function_call_output":
                            logger.debug(f"Sending conversation.item.create (function_call_output): item_id={ev.item.id}, call_id={ev.item.call_id}, output={ev.item.output}")
                            # Optimistically sync this item immediately
                            item = _realtime_item_to_livekit_item(ev.item)
                            try:
                                self._remote_chat_ctx.insert(ev.previous_item_id if ev.previous_item_id != "root" else None, item)
                            except ValueError as e:
                                logger.warning(f"failed to optimistically insert function_call_output item: {e}")
                        else:
                            futs.append(f := asyncio.Future[None]())
                            self._item_create_future[ev.item.id] = f
                            logger.debug(f"Sending conversation.item.create: item_id={ev.item.id}, type={ev.item.type}")
                    self.send_event(ev)

                if not futs:
                    return

                await asyncio.wait_for(asyncio.gather(*futs, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Stepfun update_chat_ctx ack timed out; continuing with optimistic sync"
            )
            self._sync_remote_ctx_optimistically(chat_ctx)

    def _sync_remote_ctx_optimistically(self, chat_ctx: llm.ChatContext) -> None:
        remote_ctx = llm.remote_chat_context.RemoteChatContext()
        previous_id: str | None = None
        for item in chat_ctx.items:
            try:
                remote_ctx.insert(previous_id, item)
                previous_id = item.id
            except ValueError:
                continue

        self._remote_chat_ctx = remote_ctx

    def _create_update_chat_ctx_events(
        self, chat_ctx: llm.ChatContext
    ) -> list[ConversationItemCreateEvent | ConversationItemDeleteEvent]:
        events: list[ConversationItemCreateEvent | ConversationItemDeleteEvent] = []
        diff_ops = llm.utils.compute_chat_ctx_diff(self._remote_chat_ctx.to_chat_ctx(), chat_ctx)

        def _delete_item(msg_id: str) -> None:
            events.append(
                ConversationItemDeleteEvent(
                    type="conversation.item.delete",
                    item_id=msg_id,
                    event_id=utils.shortuuid("chat_ctx_delete_"),
                )
            )

        def _create_item(previous_msg_id: str | None, msg_id: str) -> None:
            chat_item = chat_ctx.get_by_id(msg_id)
            assert chat_item is not None
            events.append(
                ConversationItemCreateEvent(
                    type="conversation.item.create",
                    item=_livekit_item_to_realtime_item(chat_item),
                    previous_item_id=("root" if previous_msg_id is None else previous_msg_id),
                    event_id=utils.shortuuid("chat_ctx_create_"),
                )
            )

        for msg_id in diff_ops.to_remove:
            _delete_item(msg_id)

        for previous_msg_id, msg_id in diff_ops.to_create:
            _create_item(previous_msg_id, msg_id)

        # update the items with the same id but different content
        for previous_msg_id, msg_id in diff_ops.to_update:
            _delete_item(msg_id)
            _create_item(previous_msg_id, msg_id)

        return events

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        async with self._update_fnc_ctx_lock:
            ev, retained_tools = self._build_tools_update_event(tools)
            self.send_event(ev)
            self._tools = llm.ToolContext(retained_tools)

    def _create_tools_update_event(
        self, tools: list[llm.Tool]
    ) -> SessionUpdateEvent:
        full_tools = list(tools)
        for provider_tool in self._tools.provider_tools:
            if provider_tool not in full_tools:
                full_tools.append(provider_tool)

        event, _ = self._build_tools_update_event(full_tools)
        return event

    def _build_tools_update_event(
        self, tools: list[llm.Tool]
    ) -> tuple[SessionUpdateEvent, list[llm.Tool]]:
        tool_payloads: list[dict[str, Any]] = []
        retained_tools: list[llm.Tool] = []

        for tool in tools:
            if isinstance(tool, llm.FunctionTool):
                tool_desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                stepfun_tool = self._to_stepfun_function_tool(tool_desc)
                if stepfun_tool is None:
                    logger.error(
                        "Stepfun Realtime API doesn't support this tool",
                        extra={"tool": tool_desc},
                    )
                    continue
                tool_payloads.append(stepfun_tool)
                retained_tools.append(tool)
            elif isinstance(tool, llm.RawFunctionTool):
                tool_desc = dict(tool.info.raw_schema)
                tool_desc.pop("meta", None)
                tool_desc["type"] = "function"
                stepfun_tool = self._to_stepfun_function_tool(tool_desc)
                if stepfun_tool is None:
                    logger.error(
                        "Stepfun Realtime API doesn't support this tool",
                        extra={"tool": tool_desc},
                    )
                    continue
                tool_payloads.append(stepfun_tool)
                retained_tools.append(tool)
            elif isinstance(tool, llm.ProviderTool):
                provider_payload = self._provider_tool_to_dict(tool)
                if provider_payload is None:
                    logger.error(
                        "Stepfun Realtime API doesn't support this provider tool",
                        extra={"tool": tool},
                    )
                    continue
                tool_payloads.append(provider_payload)
                retained_tools.append(tool)
            else:
                logger.error(
                    "Stepfun Realtime API doesn't support this tool type",
                    extra={"tool": tool},
                )

        if len(tool_payloads) > MAX_TOOLS:
            logger.warning(
                "Stepfun Realtime API supports at most %d tools, truncating extra tools",
                MAX_TOOLS,
                extra={"total_tools": len(tool_payloads)},
            )
            tool_payloads = tool_payloads[:MAX_TOOLS]

            allowed_names: set[str] = set()
            allowed_provider_types: set[str] = set()
            for payload in tool_payloads:
                tool_type = payload.get("type")
                if tool_type == "function":
                    name = payload.get("name")
                    if isinstance(name, str):
                        allowed_names.add(name)
                elif isinstance(tool_type, str):
                    allowed_provider_types.add(tool_type)

            retained_tools = [
                t
                for t in retained_tools
                if (
                    isinstance(t, (llm.FunctionTool, llm.RawFunctionTool))
                    and t.info.name in allowed_names
                )
                or (
                    isinstance(t, StepfunTool)
                    and t.to_dict().get("type") in allowed_provider_types
                )
            ]

        session_tools: list[session_update_event.SessionTool] = []
        for payload in tool_payloads:
            try:
                session_tools.append(session_update_event.SessionTool.model_construct(**payload))
            except ValidationError:
                logger.error(
                    "Stepfun Realtime API doesn't support this tool payload",
                    extra={"tool": payload},
                )

        event = SessionUpdateEvent(
            type="session.update",
            session=session_update_event.Session.model_construct(
                model=self._realtime_model._opts.model,  # type: ignore
                tools=session_tools,
            ),
            event_id=utils.shortuuid("tools_update_"),
        )
        return event, retained_tools

    def _provider_tool_to_dict(self, tool: llm.ProviderTool) -> dict[str, Any] | None:
        if not isinstance(tool, StepfunTool):
            return None

        payload = tool.to_dict()
        if not isinstance(payload.get("type"), str):
            return None
        if payload["type"] not in {"web_search", "retrieval"}:
            return None
        if payload["type"] == "retrieval" and not isinstance(payload.get("vector_store_id"), str):
            return None

        return payload

    def _to_stepfun_function_tool(self, tool_desc: dict[str, Any]) -> dict[str, Any] | None:
        if tool_desc.get("type") != "function":
            return None

        function_desc = tool_desc.get("function", {})
        name = tool_desc.get("name") or function_desc.get("name")
        if not isinstance(name, str) or not name:
            return None

        description = tool_desc.get("description") or function_desc.get("description") or ""
        if not isinstance(description, str):
            description = ""

        parameters = tool_desc.get("parameters") or function_desc.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}

        return {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

    async def update_instructions(self, instructions: str) -> None:
        event_id = utils.shortuuid("instructions_update_")
        # f = asyncio.Future()
        # self._response_futures[event_id] = f
        self.send_event(
            SessionUpdateEvent(
                type="session.update",
                session=session_update_event.Session.model_construct(instructions=instructions),
                event_id=event_id,
            )
        )
        self._instructions = instructions

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.write(data):
                self.send_event(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(nf.data).decode("utf-8"),
                    )
                )
                self._pushed_duration_s += nf.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        if self._pushed_duration_s > 0.1:
            self.send_event(InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))
            self._pushed_duration_s = 0

    def clear_audio(self) -> None:
        self.send_event(InputAudioBufferClearEvent(type="input_audio_buffer.clear"))
        self._pushed_duration_s = 0

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[event_id] = fut
        self._pending_response_create_ids.append(event_id)
        self.send_event(
            ResponseCreateEvent(
                type="response.create",
                event_id=event_id,
                response=Response(
                    instructions=instructions or None,
                    metadata={"client_event_id": event_id},
                ),
            )
        )

        def _on_timeout() -> None:
            if fut and not fut.done():
                self._response_created_futures.pop(event_id, None)
                with contextlib.suppress(ValueError):
                    self._pending_response_create_ids.remove(event_id)
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut

    def interrupt(self) -> None:
        self.send_event(ResponseCancelEvent(type="response.cancel"))

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if "audio" in modalities:
            self.send_event(
                ConversationItemTruncateEvent(
                    type="conversation.item.truncate",
                    content_index=0,
                    item_id=message_id,
                    audio_end_ms=audio_end_ms,
                )
            )
        elif utils.is_given(audio_transcript):
            # sync the forwarded text to the remote chat ctx
            chat_ctx = self.chat_ctx.copy(
                exclude_handoff=True,
                exclude_config_update=True,
            )
            if (idx := chat_ctx.index_by_id(message_id)) is not None:
                new_item = copy.copy(chat_ctx.items[idx])
                assert new_item.type == "message"

                new_item.content = [audio_transcript]
                chat_ctx.items[idx] = new_item
                events = self._create_update_chat_ctx_events(chat_ctx)
                for ev in events:
                    self.send_event(ev)

    async def aclose(self) -> None:
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        user_transcription_enabled = (
            self._realtime_model._opts.input_audio_transcription is not None
        )
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=user_transcription_enabled),
        )

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        assert event.response.id is not None, "response.id is None"

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
            _created_timestamp=time.time(),
            _done_fut=asyncio.Future(),
        )

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )

        fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        client_event_id: str | None = None
        if isinstance(event.response.metadata, dict):
            client_event_id_value = event.response.metadata.get("client_event_id")
            if isinstance(client_event_id_value, str):
                client_event_id = client_event_id_value

        if client_event_id is not None:
            with contextlib.suppress(ValueError):
                self._pending_response_create_ids.remove(client_event_id)
            fut = self._response_created_futures.pop(client_event_id, None)
        else:
            while self._pending_response_create_ids:
                fallback_id = self._pending_response_create_ids.pop(0)
                fut = self._response_created_futures.pop(fallback_id, None)
                if fut is not None:
                    break

        if fut is not None:
            if not fut.done():
                generation_ev.user_initiated = True
                fut.set_result(generation_ev)
            else:
                logger.warning("response of generate_reply received after it's timed out.")

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "message":
            item_generation = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
                modalities=asyncio.Future(),
            )
            if not self._realtime_model.capabilities.audio_output:
                item_generation.audio_ch.close()
                item_generation.modalities.set_result(["text"])

            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_generation.text_ch,
                    audio_stream=item_generation.audio_ch,
                    modalities=item_generation.modalities,
                )
            )
            self._current_generation.messages[item_id] = item_generation

    def _handle_response_content_part_added(self, event: ResponseContentPartAddedEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item_id) is not None, "item_id is None"
        assert (item_type := event.part.type) is not None, "part.type is None"

        if item_type == "text" and self._realtime_model.capabilities.audio_output:
            logger.warning("Text response received from Stepfun Realtime API in audio modality.")

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation.messages[item_id].modalities.set_result(
                ["text"] if item_type == "text" else ["audio", "text"]
            )

    def _handle_conversion_item_created(self, event: ConversationItemCreatedEvent) -> None:
        assert event.item.id is not None, "item.id is None"

        logger.debug(f"Received conversation.item.created: item_id={event.item.id}, type={event.item.type}")

        item = _realtime_item_to_livekit_item(event.item)
        previous_item_id = event.previous_item_id
        # Some providers may send an invalid previous_item_id (e.g. self-reference or stale id).
        if previous_item_id == event.item.id:
            previous_item_id = None

        try:
            self._remote_chat_ctx.insert(previous_item_id, item)
        except ValueError:
            fallback_previous_item_id: str | None = None
            existing_items = self._remote_chat_ctx.to_chat_ctx().items
            if existing_items:
                fallback_previous_item_id = existing_items[-1].id
                if fallback_previous_item_id == event.item.id:
                    fallback_previous_item_id = None

            try:
                self._remote_chat_ctx.insert(fallback_previous_item_id, item)
            except ValueError as e:
                logger.warning(
                    f"failed to insert item `{event.item.id}`: {str(e)}",
                )
            else:
                logger.debug(
                    "inserted item with fallback previous_item_id",
                    extra={
                        "item_id": event.item.id,
                        "requested_previous_item_id": event.previous_item_id,
                        "fallback_previous_item_id": fallback_previous_item_id,
                    },
                )

        if fut := self._item_create_future.pop(event.item.id, None):
            fut.set_result(None)

    def _handle_conversion_item_deleted(self, event: ConversationItemDeletedEvent) -> None:
        assert event.item_id is not None, "item_id is None"

        try:
            self._remote_chat_ctx.delete(event.item_id)
        except ValueError as e:
            logger.warning(
                f"failed to delete item `{event.item_id}`: {str(e)}",
            )

        if fut := self._item_delete_future.pop(event.item_id, None):
            fut.set_result(None)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            remote_item.item.content.append(event.transcript)

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=event.item_id,
                transcript=event.transcript,
                is_final=True,
            ),
        )

    def _handle_conversion_item_input_audio_transcription_failed(
        self, event: ConversationItemInputAudioTranscriptionFailedEvent
    ) -> None:
        logger.error(
            "Stepfun Realtime API failed to transcribe input audio",
            extra={"error": event.error},
        )

    def _handle_response_text_delta(self, event: ResponseTextDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]
        if (
            item_generation.audio_ch.closed
            and self._current_generation._first_token_timestamp is None
        ):
            # only if audio is not available
            self._current_generation._first_token_timestamp = time.time()

        item_generation.text_ch.send_nowait(event.delta)
        item_generation.audio_transcript += event.delta

    def _handle_response_text_done(self, event: ResponseTextDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        assert self._current_generation is not None, "current_generation is None"

        item_id = event["item_id"]
        delta = event["delta"]

        if (start_time := event.get("start_time")) is not None:
            delta = io.TimedString(delta, start_time=start_time)

        item_generation = self._current_generation.messages[item_id]
        item_generation.text_ch.send_nowait(delta)
        item_generation.audio_transcript += delta

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]
        if self._current_generation._first_token_timestamp is None:
            self._current_generation._first_token_timestamp = time.time()

        if not item_generation.modalities.done():
            item_generation.modalities.set_result(["audio", "text"])

        data = base64.b64decode(event.delta)
        item_generation.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_response_audio_transcript_done(self, _: ResponseAudioTranscriptDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "function_call":
            item = event.item
            state = self._record_function_call(
                item_id=item_id,
                call_id=item.call_id,
                name=item.name,
                arguments=item.arguments,
                done=True,
            )
            if state is not None:
                self._emit_function_call_if_ready(state.item_id)
        elif item_type == "message":
            item_generation = self._current_generation.messages[item_id]
            item_generation.text_ch.close()
            item_generation.audio_ch.close()
            if not item_generation.modalities.done():
                # in case message modalities is not set, this shouldn't happen
                item_generation.modalities.set_result(self._realtime_model._opts.modalities)

    def _handle_response_done(self, event: ResponseDoneEvent) -> None:
        if self._current_generation is None:
            self._function_call_states.clear()
            self._function_item_by_call_id.clear()
            return  # Provider may emit response.done without response.created during interruption.

        assert self._current_generation is not None, "current_generation is None"

        try:
            self._hydrate_function_calls_from_response_done(event)
        except Exception:
            logger.exception("failed to hydrate function calls from response.done")

        created_timestamp = self._current_generation._created_timestamp
        first_token_timestamp = self._current_generation._first_token_timestamp

        for generation in self._current_generation.messages.values():
            # close all messages that haven't been closed yet
            if not generation.text_ch.closed:
                generation.text_ch.close()
            if not generation.audio_ch.closed:
                generation.audio_ch.close()
            if not generation.modalities.done():
                generation.modalities.set_result(self._realtime_model._opts.modalities)

        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        for item_id, item_generation in self._current_generation.messages.items():
            if (remote_item := self._remote_chat_ctx.get(item_id)) and isinstance(
                remote_item.item, llm.ChatMessage
            ):
                remote_item.item.content.append(item_generation.audio_transcript)

        with contextlib.suppress(asyncio.InvalidStateError):
            self._current_generation._done_fut.set_result(None)
        self._current_generation = None
        self._function_call_states.clear()
        self._function_item_by_call_id.clear()

        # calculate metrics
        usage = (
            event.response.usage.model_dump(exclude_defaults=True) if event.response.usage else {}
        )
        ttft = first_token_timestamp - created_timestamp if first_token_timestamp else -1
        duration = time.time() - created_timestamp
        metrics = RealtimeModelMetrics(
            timestamp=created_timestamp,
            request_id=event.response.id or "",
            ttft=ttft,
            duration=duration,
            cancelled=event.response.status == "cancelled",
            label=self._realtime_model.label,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            tokens_per_second=usage.get("output_tokens", 0) / duration if duration > 0 else 0,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(
                audio_tokens=usage.get("input_token_details", {}).get("audio_tokens", 0),
                cached_tokens=usage.get("input_token_details", {}).get("cached_tokens", 0),
                text_tokens=usage.get("input_token_details", {}).get("text_tokens", 0),
                cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                    text_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("text_tokens", 0),
                    audio_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("audio_tokens", 0),
                    image_tokens=usage.get("input_token_details", {})
                    .get("cached_tokens_details", {})
                    .get("image_tokens", 0),
                ),
                image_tokens=usage.get("input_token_details", {}).get("image_tokens", 0),
            ),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=usage.get("output_token_details", {}).get("text_tokens", 0),
                audio_tokens=usage.get("output_token_details", {}).get("audio_tokens", 0),
                image_tokens=usage.get("output_token_details", {}).get("image_tokens", 0),
            ),
            metadata=Metadata(
                model_name=self._realtime_model.model,
                model_provider=self._realtime_model.provider,
            ),
        )
        self.emit("metrics_collected", metrics)
        self._handle_response_done_but_not_complete(event)

    def _handle_response_done_but_not_complete(self, event: ResponseDoneEvent) -> None:
        """Handle response done but not complete, i.e. cancelled, incomplete or failed.

        For example this method will emit an error if we receive a "failed" status, e.g.
        with type "invalid_request_error" due to code "inference_rate_limit_exceeded".

        In other failures it will emit a debug level log.
        """
        if event.response.status == "completed":
            return

        if event.response.status == "failed":
            if event.response.status_details and hasattr(event.response.status_details, "error"):
                error_type = getattr(event.response.status_details.error, "type", "unknown")
                error_body = event.response.status_details.error
                message = f"Stepfun Realtime API response failed with error type: {error_type}"
            else:
                error_body = None
                message = "Stepfun Realtime API response failed with unknown error"
            self._emit_error(
                APIError(
                    message=message,
                    body=error_body,
                    retryable=True,
                ),
                # All failures are not documented; assume optimistic retryability.
                recoverable=True,
            )
        elif event.response.status in {"cancelled", "incomplete"}:
            logger.debug(
                "Stepfun Realtime API response done but not complete with status: %s",
                event.response.status,
                extra={
                    "event_id": event.response.id,
                    "event_response_status": event.response.status,
                },
            )
        else:
            logger.debug("Unknown response status: %s", event.response.status)

    def _handle_error(self, event: ErrorEvent) -> None:
        err_message = event.error.message or ""
        if err_message.startswith("Cancellation failed") or (
            "no ongoing response to cancel" in err_message.lower()
        ):
            return

        logger.error(
            "Stepfun Realtime API returned an error",
            extra={"error": event.error},
        )
        self._emit_error(
            APIError(
                message="Stepfun Realtime API returned an error",
                body=event.error,
                retryable=True,
            ),
            recoverable=True,
        )

        # TODO: set exception for the response future if it exists

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )


def _livekit_item_to_realtime_item(item: llm.ChatItem) -> ConversationItem:
    conversation_item = ConversationItem(
        id=item.id,
    )

    if item.type == "function_call":
        conversation_item.type = "function_call"
        conversation_item.call_id = item.call_id
        conversation_item.name = item.name
        conversation_item.arguments = item.arguments

    elif item.type == "function_call_output":
        conversation_item.type = "function_call_output"
        conversation_item.call_id = item.call_id
        conversation_item.output = item.output
        logger.debug(f"Creating function_call_output item: call_id={item.call_id}, output={item.output}")

    elif item.type == "message":
        role = "system" if item.role == "developer" else item.role
        conversation_item.type = "message"
        conversation_item.role = role

        content_list: list[ConversationItemContent] = []
        for c in item.content:
            if isinstance(c, str):
                content_list.append(
                    ConversationItemContent(
                        type=("text" if role == "assistant" else "input_text"),
                        text=c,
                    )
                )

            elif isinstance(c, llm.ImageContent):
                continue  # not supported for now
            elif isinstance(c, llm.AudioContent):
                if conversation_item.role == "user":
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )

                    content_list.append(
                        ConversationItemContent(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )

        conversation_item.content = content_list

    return conversation_item


def _realtime_item_to_livekit_item(item: ConversationItem) -> llm.ChatItem:
    assert item.id is not None, "id is None"

    if item.type == "function_call":
        assert item.call_id is not None, "call_id is None"
        assert item.name is not None, "name is None"

        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments or "",
        )

    if item.type == "function_call_output":
        assert item.call_id is not None, "call_id is None"
        assert item.output is not None, "output is None"

        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if item.type == "message":
        assert item.role is not None, "role is None"
        assert item.content is not None, "content is None"

        content: list[llm.ChatContent] = []

        for c in item.content:
            if c.type == "text" or c.type == "input_text":
                assert c.text is not None, "text is None"
                content.append(c.text)

        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")


def _to_oai_tool_choice(tool_choice: llm.ToolChoice | None) -> str:
    if isinstance(tool_choice, str):
        return tool_choice

    elif isinstance(tool_choice, dict) and tool_choice["type"] == "function":
        return tool_choice["function"]["name"]

    return "auto"
