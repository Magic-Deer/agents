from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
import openai
from livekit.agents import llm
from livekit.agents.inference.llm import LLMStream as _LLMStream
from livekit.agents.llm import ChatContext, ToolChoice, utils as llm_utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from openai.types.chat import completion_create_params

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


@dataclass
class _LLMOptions:
    model: str
    temperature: NotGivenOr[float]
    top_p: NotGivenOr[float]
    max_tokens: NotGivenOr[int]
    max_completion_tokens: NotGivenOr[int]
    parallel_tool_calls: NotGivenOr[bool]
    tool_choice: NotGivenOr[ToolChoice]


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        model: str = "doubao-seed-2-0-pro-260215",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        client: openai.AsyncClient | None = None,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        max_completion_tokens: NotGivenOr[int] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
    ) -> None:
        super().__init__()

        resolved_api_key = (
            api_key if is_given(api_key) else os.environ.get("VOLCENGINE_API_KEY")
        )
        if not resolved_api_key:
            raise ValueError(
                "Volcengine API key is required, either as argument or set "
                "VOLCENGINE_API_KEY environment variable"
            )

        self._opts = _LLMOptions(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )

        self._owns_client = client is None
        self._client = client or openai.AsyncClient(
            api_key=resolved_api_key,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            max_retries=0,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
                follow_redirects=True,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=50,
                    keepalive_expiry=120,
                ),
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream:
        extra: dict[str, Any] = {}
        if is_given(extra_kwargs):
            extra.update(extra_kwargs)

        # Volcengine streams reasoning_content by default; disable thinking to ensure return content.
        # The OpenAI client doesn't accept `thinking` at the top level, pass it via extra_body.
        thinking_value = extra.pop("thinking", {"type": "disabled"})
        extra_body = extra.setdefault("extra_body", {})
        extra_body["thinking"] = thinking_value

        if is_given(self._opts.temperature):
            extra["temperature"] = self._opts.temperature

        if is_given(self._opts.top_p):
            extra["top_p"] = self._opts.top_p

        if is_given(self._opts.max_tokens):
            extra["max_tokens"] = self._opts.max_tokens

        if is_given(self._opts.max_completion_tokens):
            extra["max_completion_tokens"] = self._opts.max_completion_tokens

        parallel_tool_calls = (
            parallel_tool_calls
            if is_given(parallel_tool_calls)
            else self._opts.parallel_tool_calls
        )
        if is_given(parallel_tool_calls):
            extra["parallel_tool_calls"] = parallel_tool_calls

        tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        if is_given(tool_choice):
            if isinstance(tool_choice, dict):
                extra["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["function"]["name"]},
                }
            elif tool_choice in ("auto", "required", "none"):
                extra["tool_choice"] = tool_choice

        if is_given(response_format):
            extra["response_format"] = llm_utils.to_openai_response_format(response_format)

        if "max_tokens" in extra and "max_completion_tokens" in extra:
            raise ValueError("max_tokens and max_completion_tokens cannot both be set")

        return LLMStream(
            self,
            model=self._opts.model,
            client=self._client,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            extra_kwargs=extra,
        )


class LLMStream(_LLMStream):
    def __init__(
        self,
        llm_instance: LLM,
        *,
        model: str,
        client: openai.AsyncClient,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
        extra_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            llm_instance,
            model=model,
            provider_fmt="openai",
            strict_tool_schema=True,
            client=client,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            extra_kwargs=extra_kwargs,
        )
