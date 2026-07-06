"""Adapter for the Anthropic Messages API.

Implements request building and response parsing for
POST https://api.anthropic.com/v1/messages. Needs ANTHROPIC_API_KEY.
Request construction is covered by offline contract tests; live calls
were not exercised in this repository (no key available).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from .base import AdapterError, MalformedOutputError, ModelAdapter, ModelTurn, Usage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicAdapter(ModelAdapter):
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: httpx.Client | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self._client = client

    # -- request building (pure, contract-tested offline) -----------------

    def build_request(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        system = ""
        out: list[dict[str, Any]] = []
        for m in messages:
            role, content = m["role"], m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                out.append({"role": "user", "content": content})
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in m.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["call_id"],
                        "name": tc["tool"],
                        "input": tc["args"],
                    })
                out.append({"role": "assistant", "content": blocks or content})
            elif role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m["call_id"],
                        "content": json.dumps(m["content"], sort_keys=True),
                    }],
                })
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": out,
        }
        if tools:
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        return body

    def parse_response(self, data: dict[str, Any], latency_ms: float) -> ModelTurn:
        try:
            text = ""
            calls: list[dict[str, Any]] = []
            for block in data["content"]:
                if block["type"] == "text":
                    text += block["text"]
                elif block["type"] == "tool_use":
                    calls.append({
                        "call_id": block["id"],
                        "tool": block["name"],
                        "args": block["input"],
                    })
            usage = Usage(
                input_tokens=data.get("usage", {}).get("input_tokens", 0),
                output_tokens=data.get("usage", {}).get("output_tokens", 0),
                latency_ms=latency_ms,
            )
            return ModelTurn(text=text, tool_calls=calls, usage=usage,
                             model=data.get("model", self.model))
        except (KeyError, TypeError) as exc:
            raise MalformedOutputError(f"unexpected response shape: {exc}",
                                       raw=json.dumps(data)[:2000]) from exc

    # -- transport ---------------------------------------------------------

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ModelTurn:
        if not self.api_key:
            raise AdapterError("ANTHROPIC_API_KEY is not set")
        body = self.build_request(messages, tools)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        client = self._client or httpx.Client(timeout=120)
        start = time.monotonic()
        try:
            resp = client.post(API_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise AdapterError(f"anthropic transport error: {exc}") from exc
        finally:
            if self._client is None:
                client.close()
        latency = (time.monotonic() - start) * 1000
        if resp.status_code >= 500:
            raise AdapterError(f"anthropic HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise AdapterError(f"anthropic HTTP {resp.status_code}: {resp.text[:500]}")
        return self.parse_response(resp.json(), latency)
