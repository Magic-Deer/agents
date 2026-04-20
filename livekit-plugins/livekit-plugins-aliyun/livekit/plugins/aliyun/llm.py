from __future__ import annotations

import os
from typing import Any

import httpx
import openai
from openai.types.chat import completion_create_params

from livekit.agents import llm
from livekit.agents.llm import ToolChoice, utils as llm_utils
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from livekit.plugins.openai.llm import LLM as OpenAILLM, LLMStream as OpenAILLMStream

DEFAULT_MODEL = "qwen-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class LLM(OpenAILLM):
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = DEFAULT_BASE_URL,
        client: openai.AsyncClient | None = None,
        user: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
        top_p: NotGivenOr[float] = NOT_GIVEN,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        max_tokens: NotGivenOr[int] = NOT_GIVEN,
        seed: NotGivenOr[int] = NOT_GIVEN,
        enable_thinking: bool = False,
        thinking_budget: NotGivenOr[int] = NOT_GIVEN,
        extra_body: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        extra_headers: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        extra_query: NotGivenOr[dict[str, str]] = NOT_GIVEN,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        """
        Create a new instance of Aliyun DashScope LLM.

        ``api_key`` must be set to your DashScope API key, either using the argument or by
        setting the ``DASHSCOPE_API_KEY`` environment variable.
        """
        _validate_tool_choice(tool_choice, enable_thinking=enable_thinking)
        merged_body = _merge_extra_body(
            extra_body,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            max_tokens=max_tokens,
            seed=seed,
        )

        self._enable_thinking = enable_thinking
        self._aliyun_extra_body = merged_body

        super().__init__(
            model=model,
            api_key=_get_api_key(api_key),
            base_url=base_url,
            client=client,
            user=user,
            temperature=temperature,
            top_p=top_p,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            extra_body=NOT_GIVEN,
            extra_headers=extra_headers,
            extra_query=extra_query,
            timeout=timeout,
            _strict_tool_schema=False,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "Aliyun"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        response_format: NotGivenOr[
            completion_create_params.ResponseFormat | type[llm_utils.ResponseFormatT]
        ] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> OpenAILLMStream:
        resolved_tool_choice = tool_choice if is_given(tool_choice) else self._opts.tool_choice
        _validate_tool_choice(resolved_tool_choice, enable_thinking=self._enable_thinking)
        merged_extra_kwargs = _merge_call_extra_kwargs(
            extra_kwargs,
            aliyun_extra_body=self._aliyun_extra_body,
            enable_thinking=self._enable_thinking,
        )

        return super().chat(
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            response_format=response_format,
            extra_kwargs=merged_extra_kwargs,
        )


def _get_api_key(key: NotGivenOr[str]) -> str:
    dashscope_api_key = key if is_given(key) else os.environ.get("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required, either as argument or set "
            "DASHSCOPE_API_KEY environment variable"
        )
    return dashscope_api_key


def _merge_extra_body(
    extra_body: NotGivenOr[dict[str, Any]],
    *,
    enable_thinking: bool,
    thinking_budget: NotGivenOr[int],
    max_tokens: NotGivenOr[int],
    seed: NotGivenOr[int],
) -> dict[str, Any]:
    merged_body = dict(extra_body) if is_given(extra_body) else {}
    _validate_no_preserve_thinking(merged_body)

    if is_given(thinking_budget):
        _validate_thinking_budget(thinking_budget, enable_thinking=enable_thinking)
        merged_body["thinking_budget"] = thinking_budget
    if is_given(max_tokens):
        merged_body["max_tokens"] = max_tokens
    if is_given(seed):
        merged_body["seed"] = seed

    merged_body["enable_thinking"] = enable_thinking
    return merged_body


def _validate_tool_choice(
    tool_choice: NotGivenOr[ToolChoice],
    *,
    enable_thinking: bool,
) -> None:
    if not is_given(tool_choice):
        return
    if tool_choice == "required":
        raise ValueError("Aliyun LLM does not support tool_choice='required'")
    if enable_thinking and isinstance(tool_choice, dict):
        raise ValueError(
            "Aliyun LLM does not support forcing a specific tool when enable_thinking=True"
        )


def _merge_call_extra_kwargs(
    extra_kwargs: NotGivenOr[dict[str, Any]],
    *,
    aliyun_extra_body: dict[str, Any],
    enable_thinking: bool,
) -> dict[str, Any]:
    merged_kwargs = dict(extra_kwargs) if is_given(extra_kwargs) else {}
    per_call_body = _get_per_call_extra_body(merged_kwargs)
    _validate_no_preserve_thinking(per_call_body)

    if "thinking_budget" in per_call_body:
        _validate_thinking_budget(per_call_body["thinking_budget"], enable_thinking=enable_thinking)

    final_body = {**aliyun_extra_body, **per_call_body}
    final_body["enable_thinking"] = enable_thinking
    merged_kwargs["extra_body"] = final_body
    return merged_kwargs


def _get_per_call_extra_body(extra_kwargs: dict[str, Any]) -> dict[str, Any]:
    extra_body = extra_kwargs.get("extra_body", NOT_GIVEN)
    if not is_given(extra_body):
        return {}
    if not isinstance(extra_body, dict):
        raise TypeError("extra_kwargs['extra_body'] must be a dict")
    return dict(extra_body)


def _validate_no_preserve_thinking(extra_body: dict[str, Any]) -> None:
    if "preserve_thinking" in extra_body:
        raise ValueError("Aliyun LLM does not support preserve_thinking")


def _validate_thinking_budget(thinking_budget: int, *, enable_thinking: bool) -> None:
    if not enable_thinking:
        raise ValueError("thinking_budget requires enable_thinking=True")
    if thinking_budget < 0:
        raise ValueError("thinking_budget must be greater than or equal to 0")
