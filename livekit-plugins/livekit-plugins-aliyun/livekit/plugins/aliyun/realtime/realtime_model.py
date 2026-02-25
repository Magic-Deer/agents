# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen Realtime API implementation for LiveKit Agents.

This module implements the RealtimeModel interface for Aliyun Cloud's Qwen Realtime API.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, Literal

import aiohttp
from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from . import models
from ..log import logger
from . import api_proto, utils as realtime_utils


@dataclass
class TurnDetection:
    """Turn detection configuration for VAD."""
    threshold: float = 0.5  # [-1.0, 1.0]
    silence_duration_ms: int = 800  # [200, 6000]
    prefix_padding_ms: int = 300


@dataclass
class _RealtimeOptions:
    """Internal options for RealtimeModel."""
    model: str
    voice: str
    modalities: list[Literal["text", "audio"]]
    instructions: str | None
    turn_detection: TurnDetection | None
    temperature: float
    top_p: float
    top_k: int | None
    max_tokens: int | None
    smooth_output: bool | None
    api_key: str
    base_url: str
    max_session_duration: float | None


class RealtimeModel(llm.RealtimeModel):
    """Qwen Realtime API implementation."""

    def __init__(
        self,
        *,
        model: str = models.DEFAULT_MODEL,
        voice: str = models.DEFAULT_VOICE,
        modalities: list[Literal["text", "audio"]] = ["text", "audio"],
        instructions: str | None = None,
        turn_detection: TurnDetection | None = TurnDetection(),
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        max_tokens: int | None = None,
        smooth_output: bool | None = None,
        api_key: str | None = None,
        base_url: str = models.BASE_URL,
        max_session_duration: float | None = 110 * 60,  # 110 minutes
    ) -> None:
        # Get API key from environment if not provided
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY must be set in environment or passed as api_key parameter"
            )

        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            modalities=modalities,
            instructions=instructions,
            turn_detection=turn_detection,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            smooth_output=smooth_output,
            api_key=api_key,
            base_url=base_url,
            max_session_duration=max_session_duration,
        )

        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,  # Qwen doesn't support message truncation
                turn_detection=turn_detection is not None,
                user_transcription=True,  # Qwen supports input transcription
                auto_tool_reply_generation=False,
                audio_output="audio" in modalities,
                manual_function_calls=True,
            )
        )

        self._sessions: weakref.WeakSet[RealtimeSession] = weakref.WeakSet()
        self._http_session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "aliyun"

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        # Close all active sessions
        for sess in list(self._sessions):
            await sess.aclose()

        # Close HTTP session
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession()
        return self._http_session


@dataclass
class _MessageGeneration:
    """Internal message generation state."""
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    audio_transcript: str = ""


@dataclass
class _CurrentGeneration:
    """Current generation state."""
    response_id: str
    user_initiated: bool
    messages: dict[str, _MessageGeneration]
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    _done_fut: asyncio.Future[None]
    _first_token_timestamp: float | None = None


class RealtimeSession(llm.RealtimeSession):
    """Qwen Realtime API session."""

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch: utils.aio.Chan[api_proto.ClientEvent | dict[str, Any]] = utils.aio.Chan()

        # Audio processing
        self._input_resampler: rtc.AudioResampler | None = None
        self._bstream = utils.audio.AudioByteStream(
            sample_rate=models.INPUT_SAMPLE_RATE,
            num_channels=models.NUM_CHANNELS,
            samples_per_channel=models.CHUNK_SIZE_SAMPLES,
        )
        self._pushed_duration_s = 0.0

        # State management
        self._chat_ctx = llm.ChatContext()
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()
        self._current_generation: _CurrentGeneration | None = None
        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}

        # Start main task
        self._main_atask = asyncio.create_task(self._main_task())

        # Send initial session update
        self.send_event(self._create_session_update_event())

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools

    def send_event(self, event: api_proto.ClientEvent | dict[str, Any]) -> None:
        """Send a client event to the server."""
        self._msg_ch.send_nowait(event)

    def _create_session_update_event(self) -> api_proto.SessionUpdateEvent:
        """Create a session.update event with current configuration."""
        opts = self._realtime_model._opts
        event: api_proto.SessionUpdateEvent = {
            "type": "session.update",
            "modalities": opts.modalities,
            "voice": opts.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": realtime_utils.get_output_audio_format(opts.model),
            "temperature": opts.temperature,
            "top_p": opts.top_p,
        }

        if opts.instructions:
            event["instructions"] = opts.instructions

        if opts.turn_detection:
            event["turn_detection"] = realtime_utils.to_turn_detection_config(
                threshold=opts.turn_detection.threshold,
                silence_duration_ms=opts.turn_detection.silence_duration_ms,
                prefix_padding_ms=opts.turn_detection.prefix_padding_ms,
            )
        else:
            event["turn_detection"] = None

        if opts.top_k is not None:
            event["top_k"] = opts.top_k

        if opts.max_tokens is not None:
            event["max_tokens"] = opts.max_tokens

        if opts.smooth_output is not None:
            event["smooth_output"] = opts.smooth_output

        return event

    # ========================================================================
    # Abstract method implementations
    # ========================================================================

    async def update_instructions(self, instructions: str) -> None:
        """Update system instructions."""
        self._realtime_model._opts.instructions = instructions
        event = self._create_session_update_event()
        self.send_event(event)

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        """Update chat context."""
        self._chat_ctx = chat_ctx
        # Note: Qwen doesn't support deleting messages, so we can only append new ones
        # For full context reset, we would need to reconnect

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        """Update available tools."""
        self._tools = llm.ToolContext(tools=tools)
        # Note: Qwen doesn't have native tool calling support
        # Tools need to be described in instructions

    def update_options(
        self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN
    ) -> None:
        """Update session options."""
        # Qwen doesn't support tool_choice parameter
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Push audio frame to input buffer."""
        for f in self._resample_audio(frame):
            data = f.data.tobytes()
            for nf in self._bstream.push(data):
                self.send_event(
                    api_proto.InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(nf.data).decode("utf-8"),
                    )
                )
                self._pushed_duration_s += nf.duration

    def push_video(self, frame: rtc.VideoFrame) -> None:
        """Push video frame (as image) to input buffer."""
        # Convert video frame to JPEG
        from PIL import Image
        import io

        # Convert frame to PIL Image
        img = Image.frombytes(
            "RGB",
            (frame.width, frame.height),
            bytes(frame.data),
        )

        # Compress to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_data = buffer.getvalue()

        # Send to server
        self.send_event(
            api_proto.InputImageBufferAppendEvent(
                type="input_image_buffer.append",
                image=base64.b64encode(image_data).decode("utf-8"),
            )
        )

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Generate a reply (manual mode)."""
        response_id = realtime_utils.generate_response_id()
        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._response_created_futures[response_id] = fut

        # Send response.create event
        self.send_event(api_proto.ResponseCreateEvent(type="response.create"))

        return fut

    def commit_audio(self) -> None:
        """Commit audio buffer."""
        if self._pushed_duration_s > 0.1:  # At least 100ms
            self.send_event(
                api_proto.InputAudioBufferCommitEvent(type="input_audio_buffer.commit")
            )
            self._pushed_duration_s = 0

    def clear_audio(self) -> None:
        """Clear audio buffer."""
        self.send_event(
            api_proto.InputAudioBufferClearEvent(type="input_audio_buffer.clear")
        )
        self._pushed_duration_s = 0

    def interrupt(self) -> None:
        """Interrupt current response."""
        self.send_event(api_proto.ResponseCancelEvent(type="response.cancel"))

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Truncate message (not supported by Qwen)."""
        logger.warning("Message truncation is not supported by Qwen Realtime API")

    async def aclose(self) -> None:
        """Close the session."""
        self._msg_ch.close()
        await self._main_atask

    # ========================================================================
    # Audio processing
    # ========================================================================

    def _resample_audio(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        """Resample audio to 16kHz mono."""
        # Check if we need to reset the resampler due to sample rate change
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # Input audio changed to a different sample rate
                self._input_resampler = None

        # Create resampler if needed
        if self._input_resampler is None and (
            frame.sample_rate != models.INPUT_SAMPLE_RATE
            or frame.num_channels != models.NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=models.INPUT_SAMPLE_RATE,
                num_channels=models.NUM_CHANNELS,
            )

        # Resample or pass through
        if self._input_resampler:
            return self._input_resampler.push(frame)
        else:
            return [frame]

    # ========================================================================
    # WebSocket connection management
    # ========================================================================

    async def _main_task(self) -> None:
        """Main task loop with reconnection logic."""
        max_retries = 5
        num_retries = 0

        while True:
            try:
                ws_conn = await self._create_ws_conn()
                num_retries = 0  # Reset on successful connection

                await self._run_ws(ws_conn)

                # If we exit normally, break
                break

            except Exception as e:
                if num_retries >= max_retries:
                    logger.error(f"Max retries reached, giving up: {e}")
                    self._emit_error(e, recoverable=False)
                    raise

                logger.warning(f"Connection error, retrying ({num_retries + 1}/{max_retries}): {e}")
                self._emit_error(e, recoverable=True)

                # Exponential backoff
                retry_delay = min(2 ** num_retries, 30)
                await asyncio.sleep(retry_delay)
                num_retries += 1

    async def _create_ws_conn(self) -> aiohttp.ClientWebSocketResponse:
        """Create WebSocket connection."""
        opts = self._realtime_model._opts
        url = f"{opts.base_url}?model={opts.model}"

        headers = {
            "Authorization": f"Bearer {opts.api_key}",
            "User-Agent": "LiveKit Agents",
        }

        logger.info(f"Connecting to {url}")

        http_session = self._realtime_model._ensure_http_session()
        ws = await http_session.ws_connect(url, headers=headers, timeout=30)

        logger.info("WebSocket connected")
        return ws

    async def _run_ws(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Run WebSocket send/receive loop."""
        send_task = asyncio.create_task(self._send_task(ws_conn), name="send_task")
        recv_task = asyncio.create_task(self._recv_task(ws_conn), name="recv_task")

        # Optional: session timeout reconnection
        timeout_task = None
        if self._realtime_model._opts.max_session_duration:
            timeout_task = asyncio.create_task(
                asyncio.sleep(self._realtime_model._opts.max_session_duration),
                name="timeout_task",
            )

        tasks = [send_task, recv_task]
        if timeout_task:
            tasks.append(timeout_task)

        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check for errors
            for task in done:
                if task.exception():
                    raise task.exception()  # type: ignore

        finally:
            if not ws_conn.closed:
                await ws_conn.close()

    async def _send_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Send task: forward client events to server."""
        async for event in self._msg_ch:
            if ws_conn.closed:
                break

            try:
                await ws_conn.send_str(json.dumps(event))
            except Exception as e:
                logger.error(f"Error sending event: {e}")
                raise

    async def _recv_task(self, ws_conn: aiohttp.ClientWebSocketResponse) -> None:
        """Receive task: handle server events."""
        async for msg in ws_conn:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    event = json.loads(msg.data)
                    self._handle_server_event(event)
                except Exception as e:
                    logger.error(f"Error handling server event: {e}")
                    self._emit_error(e, recoverable=True)

            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws_conn.exception()}")
                raise ws_conn.exception() or Exception("WebSocket error")

            elif msg.type == aiohttp.WSMsgType.CLOSED:
                logger.info("WebSocket closed by server")
                break

    # ========================================================================
    # Server event handling
    # ========================================================================

    def _handle_server_event(self, event: dict[str, Any]) -> None:
        """Dispatch server events to appropriate handlers."""
        event_type = event.get("type")

        if event_type == "error":
            self._handle_error(event)
        elif event_type == "session.created":
            self._handle_session_created(event)
        elif event_type == "session.updated":
            self._handle_session_updated(event)
        elif event_type == "input_audio_buffer.speech_started":
            self._handle_input_audio_buffer_speech_started(event)
        elif event_type == "input_audio_buffer.speech_stopped":
            self._handle_input_audio_buffer_speech_stopped(event)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            self._handle_input_audio_transcription_completed(event)
        elif event_type == "response.created":
            self._handle_response_created(event)
        elif event_type == "response.text.delta":
            self._handle_response_text_delta(event)
        elif event_type == "response.audio.delta":
            self._handle_response_audio_delta(event)
        elif event_type == "response.audio_transcript.delta":
            self._handle_response_audio_transcript_delta(event)
        elif event_type == "response.output_item.added":
            self._handle_response_output_item_added(event)
        elif event_type == "response.done":
            self._handle_response_done(event)
        else:
            logger.debug(f"Unhandled event type: {event_type}")

    def _handle_error(self, event: dict[str, Any]) -> None:
        """Handle error event."""
        error_data = event.get("error", {})
        error_msg = error_data.get("message", "Unknown error")
        logger.error(f"Server error: {error_msg}")
        self._emit_error(Exception(error_msg), recoverable=True)

    def _handle_session_created(self, event: dict[str, Any]) -> None:
        """Handle session.created event."""
        logger.info("Session created")

    def _handle_session_updated(self, event: dict[str, Any]) -> None:
        """Handle session.updated event."""
        logger.debug("Session updated")

    def _handle_input_audio_buffer_speech_started(self, event: dict[str, Any]) -> None:
        """Handle VAD speech start."""
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(self, event: dict[str, Any]) -> None:
        """Handle VAD speech stop."""
        user_transcription_enabled = self._realtime_model._opts.turn_detection is not None
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(
                user_transcription_enabled=user_transcription_enabled
            ),
        )

    def _handle_input_audio_transcription_completed(self, event: dict[str, Any]) -> None:
        """Handle input audio transcription completed."""
        item_id = event.get("item_id", "")
        transcript = event.get("transcript", "")

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=transcript,
                is_final=True,
            ),
        )

    def _handle_response_created(self, event: dict[str, Any]) -> None:
        """Handle response.created event."""
        response = event.get("response", {})
        response_id = response.get("id", realtime_utils.generate_response_id())

        # Check if this is a user-initiated response
        user_initiated = response_id in self._response_created_futures

        # Create new generation
        self._current_generation = _CurrentGeneration(
            response_id=response_id,
            user_initiated=user_initiated,
            messages={},
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            function_ch=utils.aio.Chan[llm.FunctionCall](),
            _done_fut=asyncio.Future(),
        )

        # Create GenerationCreatedEvent
        gen_event = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=user_initiated,
            response_id=response_id,
        )

        # Emit event
        self.emit("generation_created", gen_event)

        # Resolve future if user-initiated
        if user_initiated and response_id in self._response_created_futures:
            self._response_created_futures[response_id].set_result(gen_event)
            del self._response_created_futures[response_id]

    def _handle_response_output_item_added(self, event: dict[str, Any]) -> None:
        """Handle response.output_item.added event."""
        if not self._current_generation:
            return

        item = event.get("item", {})
        item_id = item.get("id", realtime_utils.generate_item_id())

        # Create message generation
        msg_gen = _MessageGeneration(
            message_id=item_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            modalities=asyncio.Future(),
        )

        self._current_generation.messages[item_id] = msg_gen

        # Send to message stream
        self._current_generation.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=item_id,
                text_stream=msg_gen.text_ch,
                audio_stream=msg_gen.audio_ch,
                modalities=msg_gen.modalities,
            )
        )

    def _handle_response_text_delta(self, event: dict[str, Any]) -> None:
        """Handle response.text.delta event."""
        if not self._current_generation:
            return

        item_id = event.get("item_id", "")
        delta = event.get("delta", "")

        if item_id in self._current_generation.messages:
            msg_gen = self._current_generation.messages[item_id]

            # Record first token timestamp
            if self._current_generation._first_token_timestamp is None:
                self._current_generation._first_token_timestamp = time.time()

            # Set modalities if not set
            if not msg_gen.modalities.done():
                msg_gen.modalities.set_result(["text"])

            # Send text delta
            msg_gen.text_ch.send_nowait(delta)

    def _handle_response_audio_delta(self, event: dict[str, Any]) -> None:
        """Handle response.audio.delta event."""
        if not self._current_generation:
            return

        item_id = event.get("item_id", "")
        delta = event.get("delta", "")

        if item_id in self._current_generation.messages:
            msg_gen = self._current_generation.messages[item_id]

            # Record first token timestamp
            if self._current_generation._first_token_timestamp is None:
                self._current_generation._first_token_timestamp = time.time()

            # Set modalities if not set
            if not msg_gen.modalities.done():
                msg_gen.modalities.set_result(["audio", "text"])

            # Decode and send audio
            audio_data = base64.b64decode(delta)
            sample_rate = realtime_utils.get_output_sample_rate(
                self._realtime_model._opts.model
            )

            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=sample_rate,
                num_channels=models.NUM_CHANNELS,
                samples_per_channel=len(audio_data) // models.BYTES_PER_SAMPLE,
            )

            msg_gen.audio_ch.send_nowait(frame)

    def _handle_response_audio_transcript_delta(self, event: dict[str, Any]) -> None:
        """Handle response.audio_transcript.delta event."""
        if not self._current_generation:
            return

        item_id = event.get("item_id", "")
        delta = event.get("delta", "")

        if item_id in self._current_generation.messages:
            msg_gen = self._current_generation.messages[item_id]
            msg_gen.audio_transcript += delta

    def _handle_response_done(self, event: dict[str, Any]) -> None:
        """Handle response.done event."""
        if not self._current_generation:
            return

        # Close all message channels
        for msg_gen in self._current_generation.messages.values():
            msg_gen.text_ch.close()
            msg_gen.audio_ch.close()

            # Set modalities if not set
            if not msg_gen.modalities.done():
                msg_gen.modalities.set_result([])

        # Close generation channels
        self._current_generation.message_ch.close()
        self._current_generation.function_ch.close()

        # Mark as done
        self._current_generation._done_fut.set_result(None)

        # Collect metrics
        response = event.get("response", {})
        usage = response.get("usage", {})

        if self._current_generation._first_token_timestamp:
            # Calculate metrics (simplified)
            logger.debug(
                f"Response completed - tokens: {usage.get('total_tokens', 0)}"
            )

        self._current_generation = None

    def _emit_error(self, error: Exception, recoverable: bool) -> None:
        """Emit error event."""
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model._label,
                error=error,
                recoverable=recoverable,
            ),
        )
