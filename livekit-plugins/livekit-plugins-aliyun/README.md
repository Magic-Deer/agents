# Aliyun plugin for LiveKit Agents

Realtime speech-to-text and text-to-speech support for LiveKit Agents using Aliyun
DashScope.

The plugin exposes:

- `livekit.plugins.aliyun.STT`
- `livekit.plugins.aliyun.TTS`

## Installation

```bash
pip install livekit-plugins-aliyun
```

## Pre-requisites

You'll need an API key from Aliyun. It can be set as an environment variable: `DASHSCOPE_API_KEY`

## Quick Start

```python
from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession
from livekit.plugins import aliyun

load_dotenv()

session = AgentSession(
    stt=aliyun.STT(),
    tts=aliyun.TTS(),
)

agent = Agent(instructions="Reply briefly and clearly.")
```

## STT

### Common Options

```python
from livekit.plugins import aliyun

stt = aliyun.STT(
    model="qwen3-asr-flash-realtime",
    language="zh",
    sample_rate=16000,
    interim_results=True,
)
```

- `api_key`: DashScope API key. Overrides the environment variable.
- `model`: Realtime ASR model name.
- `language`: Optional language code such as `zh`, `yue`, or `en`.
- `sample_rate`: Input sample rate. Typical value is `16000`.
- `interim_results`: Whether to emit interim transcripts.
- `turn_detection`: Server-side VAD settings. Set it to `None` to use manual mode.

### Manual Mode

To disable server-side turn detection and use manual mode:

```python
from livekit.plugins import aliyun

stt = aliyun.STT(turn_detection=None)
```

## TTS

### Common Options

```python
from livekit.plugins import aliyun

tts = aliyun.TTS(
    model="qwen3-tts-flash-realtime",
    voice="Cherry",
    language_type="Chinese",
)
```

- `api_key`: DashScope API key. Overrides the environment variable.
- `model`: Realtime TTS model name.
- `voice`: Realtime TTS voice name.
- `language_type`: One of `Auto`, `Chinese`, `English`, `German`, `Italian`,
  `Portuguese`, `Spanish`, `Japanese`, `Korean`, `French`, or `Russian`.
- `base_url`: Override the websocket endpoint. For the international region, use
  `wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime`.
