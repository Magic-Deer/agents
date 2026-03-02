# livekit-plugins-stepfun

Stepfun Realtime plugin for LiveKit Agents.

## Install

```bash
pip install livekit-plugins-stepfun
```

## Usage

```python
from livekit.plugins import stepfun

llm = stepfun.realtime.RealtimeModel(
    model="step-audio-2",
    voice="qingchunshaonv",
)
```

## API Key

Set one of:

- `STEP_API_KEY`
- `STEPFUN_API_KEY`

Or pass `api_key=...` to `RealtimeModel`.

## Built-in Tools

Use Stepfun provider tools with function tools in the same session:

```python
from livekit.plugins.stepfun import tools as stepfun_tools

tool_list = [
    stepfun_tools.WebSearch(),
    stepfun_tools.Retrieval(vector_store_id="vs_123"),
]
```
