# Aliyun plugin for LiveKit Agents

Realtime speech-to-text, text-to-speech, and LLM support for LiveKit Agents using
Aliyun DashScope.

The plugin exposes:

- `livekit.plugins.aliyun.STT`
- `livekit.plugins.aliyun.TTS`
- `livekit.plugins.aliyun.LLM`

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
    llm=aliyun.LLM(),
    stt=aliyun.STT(),
    tts=aliyun.TTS(),
)

agent = Agent(instructions="Reply briefly and clearly.")
```

## LLM

### Common Options

```python
from livekit.plugins import aliyun

llm = aliyun.LLM(
    model="qwen-plus",
    enable_thinking=False,
)
```

- `api_key`: DashScope API key. Overrides the environment variable.
- `model`: DashScope OpenAI-compatible chat model name. Defaults to `qwen-plus`.
- `base_url`: Override the OpenAI-compatible endpoint. Defaults to
  `https://dashscope.aliyuncs.com/compatible-mode/v1`.
- `enable_thinking`: Whether to enable Qwen deep-thinking mode. Defaults to `False`.
- `thinking_budget`: Optional token budget for thinking mode. Requires
  `enable_thinking=True`.
- `max_tokens`: Optional maximum output token count.
- `seed`: Optional deterministic sampling seed.
- `tool_choice`: Supports `auto`, `none`, and forcing a specific function tool. The
  `required` option is not supported.
- `preserve_thinking`: Not supported in this version. Reasoning content is not persisted
  or returned as ordinary text.

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
