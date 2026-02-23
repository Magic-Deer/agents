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

"""Utility functions for Qwen Realtime API."""

import secrets
from typing import Literal

from .api_proto import TurnDetectionConfig


def generate_item_id() -> str:
    """Generate a unique item ID."""
    return f"item_{secrets.token_hex(12)}"


def generate_response_id() -> str:
    """Generate a unique response ID."""
    return f"resp_{secrets.token_hex(12)}"


def to_turn_detection_config(
    threshold: float = 0.5,
    silence_duration_ms: int = 800,
    prefix_padding_ms: int = 300,
) -> TurnDetectionConfig:
    """Create a turn detection configuration."""
    return TurnDetectionConfig(
        type="server_vad",
        threshold=threshold,
        silence_duration_ms=silence_duration_ms,
        prefix_padding_ms=prefix_padding_ms,
    )


def get_output_sample_rate(model: str) -> int:
    """Get the output sample rate for a given model."""
    if "turbo" in model.lower():
        return 16000  # Turbo models use 16kHz
    return 24000  # Flash models use 24kHz


def get_output_audio_format(model: str) -> Literal["pcm16", "pcm24"]:
    """Get the output audio format for a given model."""
    if "turbo" in model.lower():
        return "pcm16"  # Turbo models use PCM16
    return "pcm24"  # Flash models use PCM24
