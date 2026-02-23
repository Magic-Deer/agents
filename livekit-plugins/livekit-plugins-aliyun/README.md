# LiveKit Aliyun Cloud Plugin

LiveKit Agents plugin for Aliyun Cloud services, specifically supporting the Qwen Realtime API.

## Features

- **Qwen Realtime API**: Real-time speech-to-speech interaction with Aliyun Cloud's Qwen models
- **Multi-language Support**: Supports 10 languages including Chinese, English, French, German, Russian, Italian, Spanish, Portuguese, Japanese, and Korean
- **Multiple Voices**: 49 voice options available for Qwen3-Omni-Flash-Realtime-2025-12-01
- **VAD Support**: Server-side Voice Activity Detection for automatic turn detection
- **Manual Mode**: Client-controlled audio submission for push-to-talk scenarios

## Installation

```bash
pip install livekit-plugins-aliyun
```

## Quick Start

```python
from livekit.agents import AgentSession
from livekit.plugins import aliyun

# Create an agent session with Qwen Realtime API
session = AgentSession(
    llm=aliyun.realtime.RealtimeModel(
        model="qwen3-omni-flash-realtime",
        voice="Cherry",
        api_key="your-dashscope-api-key",
    )
)
```

## Configuration

### API Key

Set your DashScope API key as an environment variable:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

Or pass it directly:

```python
model = aliyun.realtime.RealtimeModel(api_key="your-api-key")
```

### Supported Models

- `qwen3-omni-flash-realtime` (recommended, stable version)
- `qwen3-omni-flash-realtime-2025-12-01` (snapshot version)
- `qwen3-omni-flash-realtime-2025-09-15` (snapshot version)
- `qwen-omni-turbo-realtime` (legacy version)

### VAD Configuration

```python
from livekit.plugins.aliyun.realtime import TurnDetection

model = aliyun.realtime.RealtimeModel(
    turn_detection=TurnDetection(
        threshold=0.6,  # [-1.0, 1.0], higher = more sensitive
        silence_duration_ms=1000,  # [200, 6000]
        prefix_padding_ms=300,  # Audio before speech start
    )
)
```

### Manual Mode (Push-to-Talk)

```python
model = aliyun.realtime.RealtimeModel(
    turn_detection=None,  # Disable VAD
)

# In your code:
# session.commit_audio()  # Manually commit audio
# session.generate_reply()  # Manually trigger response
```

### Output Modalities

```python
# Text only
model = aliyun.realtime.RealtimeModel(modalities=["text"])

# Text and audio (default)
model = aliyun.realtime.RealtimeModel(modalities=["text", "audio"])
```

## Advanced Usage

### Custom Instructions

```python
model = aliyun.realtime.RealtimeModel(
    instructions="You are a helpful assistant that speaks in a friendly tone."
)
```

### Temperature and Sampling

```python
model = aliyun.realtime.RealtimeModel(
    temperature=0.8,  # [0, 2), higher = more creative
    top_p=0.9,  # [0, 1], nucleus sampling
    top_k=50,  # Top-k sampling
    max_tokens=2000,  # Maximum output tokens
)
```

## Technical Specifications

### Audio Format

- **Input**: PCM16, 16kHz, mono, 16-bit
- **Output**: PCM24 (Flash models) or PCM16 (Turbo models), 24kHz/16kHz, mono, 16-bit

### Session Limits

- Maximum session duration: 120 minutes
- Context length: 65,536 tokens (Flash) / 32,768 tokens (Turbo)

## Implementation Status

### Completed

- Plugin structure and configuration
- API protocol definitions
- Utility functions
- Basic RealtimeModel and RealtimeSession classes

### In Progress

- WebSocket connection management
- Audio input/output processing
- Event handling and dispatching
- VAD integration
- Chat context synchronization

### TODO

- Complete RealtimeSession implementation
- Error handling and reconnection logic
- Metrics collection
- Video input support
- Tool calling support (via prompt engineering)
- Unit and integration tests
- Example applications

## API Reference

See the [official Qwen Realtime API documentation](https://help.aliyun.com/zh/model-studio/realtime) for more details.

## License

Apache 2.0

## Contributing

Contributions are welcome! Please see the [LiveKit Agents repository](https://github.com/livekit/agents) for contribution guidelines.
