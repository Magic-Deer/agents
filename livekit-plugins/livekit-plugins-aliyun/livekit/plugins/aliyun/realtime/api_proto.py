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
Qwen Realtime API protocol definitions.

Based on: https://help.aliyun.com/zh/model-studio/client-events
          https://help.aliyun.com/zh/model-studio/server-events
"""

import sys
from typing import Any, Literal, TypedDict

if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired


# ============================================================================
# Client Events (Client → Server)
# ============================================================================

class TurnDetectionConfig(TypedDict):
    """VAD configuration for turn detection."""
    type: Literal["server_vad"]
    threshold: NotRequired[float]  # [-1.0, 1.0], default 0.5
    silence_duration_ms: NotRequired[int]  # [200, 6000], default 800
    prefix_padding_ms: NotRequired[int]  # default 300


class SessionUpdateEvent(TypedDict):
    """Update session configuration."""
    type: Literal["session.update"]
    modalities: NotRequired[list[Literal["text", "audio"]]]
    voice: NotRequired[str]
    input_audio_format: NotRequired[Literal["pcm16"]]
    output_audio_format: NotRequired[Literal["pcm16", "pcm24"]]
    smooth_output: NotRequired[bool | None]  # Qwen3-Omni-Flash only
    instructions: NotRequired[str]
    turn_detection: NotRequired[TurnDetectionConfig | None]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    max_tokens: NotRequired[int]
    repetition_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    seed: NotRequired[int]


class ResponseCreateEvent(TypedDict):
    """Create a response (manual mode)."""
    type: Literal["response.create"]


class ResponseCancelEvent(TypedDict):
    """Cancel ongoing response."""
    type: Literal["response.cancel"]


class InputAudioBufferAppendEvent(TypedDict):
    """Append audio data to input buffer."""
    type: Literal["input_audio_buffer.append"]
    audio: str  # Base64-encoded PCM16 audio


class InputAudioBufferCommitEvent(TypedDict):
    """Commit audio buffer to create user message."""
    type: Literal["input_audio_buffer.commit"]


class InputAudioBufferClearEvent(TypedDict):
    """Clear audio buffer."""
    type: Literal["input_audio_buffer.clear"]


class InputImageBufferAppendEvent(TypedDict):
    """Append image data to buffer."""
    type: Literal["input_image_buffer.append"]
    image: str  # Base64-encoded JPG/JPEG image


# Union type for all client events
ClientEvent = (
    SessionUpdateEvent
    | ResponseCreateEvent
    | ResponseCancelEvent
    | InputAudioBufferAppendEvent
    | InputAudioBufferCommitEvent
    | InputAudioBufferClearEvent
    | InputImageBufferAppendEvent
)


# ============================================================================
# Server Events (Server → Client)
# ============================================================================

class ErrorEvent(TypedDict):
    """Server error event."""
    type: Literal["error"]
    event_id: str
    error: dict[str, Any]


class SessionCreatedEvent(TypedDict):
    """Session created (first event after connection)."""
    type: Literal["session.created"]
    event_id: str
    session: dict[str, Any]


class SessionUpdatedEvent(TypedDict):
    """Session updated successfully."""
    type: Literal["session.updated"]
    event_id: str
    session: dict[str, Any]


class InputAudioBufferSpeechStartedEvent(TypedDict):
    """VAD detected speech start."""
    type: Literal["input_audio_buffer.speech_started"]
    event_id: str


class InputAudioBufferSpeechStoppedEvent(TypedDict):
    """VAD detected speech stop."""
    type: Literal["input_audio_buffer.speech_stopped"]
    event_id: str


class InputAudioBufferCommittedEvent(TypedDict):
    """Audio buffer committed."""
    type: Literal["input_audio_buffer.committed"]
    event_id: str


class InputAudioBufferClearedEvent(TypedDict):
    """Audio buffer cleared."""
    type: Literal["input_audio_buffer.cleared"]
    event_id: str


class ConversationItemCreatedEvent(TypedDict):
    """Conversation item created."""
    type: Literal["conversation.item.created"]
    event_id: str
    item: dict[str, Any]


class ConversationItemInputAudioTranscriptionCompletedEvent(TypedDict):
    """User audio transcription completed."""
    type: Literal["conversation.item.input_audio_transcription.completed"]
    event_id: str
    item_id: str
    transcript: str


class ConversationItemInputAudioTranscriptionFailedEvent(TypedDict):
    """User audio transcription failed."""
    type: Literal["conversation.item.input_audio_transcription.failed"]
    event_id: str
    item_id: str
    error: dict[str, Any]


class ResponseCreatedEvent(TypedDict):
    """Response generation started."""
    type: Literal["response.created"]
    event_id: str
    response: dict[str, Any]


class ResponseDoneEvent(TypedDict):
    """Response generation completed."""
    type: Literal["response.done"]
    event_id: str
    response: dict[str, Any]


class ResponseTextDeltaEvent(TypedDict):
    """Incremental text generation."""
    type: Literal["response.text.delta"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseTextDoneEvent(TypedDict):
    """Text generation completed."""
    type: Literal["response.text.done"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseAudioDeltaEvent(TypedDict):
    """Incremental audio generation."""
    type: Literal["response.audio.delta"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str  # Base64-encoded PCM audio


class ResponseAudioDoneEvent(TypedDict):
    """Audio generation completed."""
    type: Literal["response.audio.done"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseAudioTranscriptDeltaEvent(TypedDict):
    """Incremental audio transcript generation."""
    type: Literal["response.audio_transcript.delta"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseAudioTranscriptDoneEvent(TypedDict):
    """Audio transcript generation completed."""
    type: Literal["response.audio_transcript.done"]
    event_id: str
    response_id: str
    item_id: str
    output_index: int
    content_index: int


class ResponseOutputItemAddedEvent(TypedDict):
    """New output item added to response."""
    type: Literal["response.output_item.added"]
    event_id: str
    response_id: str
    item: dict[str, Any]


class ResponseOutputItemDoneEvent(TypedDict):
    """Output item completed."""
    type: Literal["response.output_item.done"]
    event_id: str
    response_id: str
    item: dict[str, Any]


# Union type for all server events
ServerEvent = (
    ErrorEvent
    | SessionCreatedEvent
    | SessionUpdatedEvent
    | InputAudioBufferSpeechStartedEvent
    | InputAudioBufferSpeechStoppedEvent
    | InputAudioBufferCommittedEvent
    | InputAudioBufferClearedEvent
    | ConversationItemCreatedEvent
    | ConversationItemInputAudioTranscriptionCompletedEvent
    | ConversationItemInputAudioTranscriptionFailedEvent
    | ResponseCreatedEvent
    | ResponseDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseAudioDeltaEvent
    | ResponseAudioDoneEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioTranscriptDoneEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
)
