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

from typing import Literal

# Supported voices for Qwen3-Omni-Flash-Realtime-2025-12-01
# See: https://help.aliyun.com/zh/model-studio/realtime
TTSVoices = [
    "Cherry",  # Default voice
    "Chelsie",
    # Add more voices as needed
]

# Supported Qwen Realtime models
RealtimeModels = Literal[
    "qwen3-omni-flash-realtime",
    "qwen3-omni-flash-realtime-2025-12-01",
    "qwen3-omni-flash-realtime-2025-09-15",
    "qwen-omni-turbo-realtime",
]


# Audio format constants
INPUT_SAMPLE_RATE = 16000  # 16kHz for input
OUTPUT_SAMPLE_RATE = 24000  # 24kHz for output (Flash model)
OUTPUT_SAMPLE_RATE_TURBO = 16000  # 16kHz for output (Turbo model)
NUM_CHANNELS = 1  # Mono
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# Audio chunk size (100ms)
CHUNK_DURATION_MS = 100
CHUNK_SIZE_SAMPLES = INPUT_SAMPLE_RATE * CHUNK_DURATION_MS // 1000
