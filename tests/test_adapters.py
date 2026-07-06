"""Offline contract tests for the live adapters: request building,
response parsing, and error mapping. No network, no API keys."""

import httpx
import pytest

from agent_runtime.adapters.anthropic_adapter import AnthropicAdapter
from agent_runtime.adapters.base import AdapterError, MalformedOutputError
from agent_runtime.adapters.mock import MockModelAdapter
from agent_runtime.adapters.openai_adapter import OpenAIAdapter

MESSAGES = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "read the sheet"},
    {"role": "assistant", "content": "",
     "tool_calls": [{"call_id": "c1", "tool": "read_file",
                     "args": {"path": "a.csv"}}]},
    {"role": "tool", "call_id": "c1", "content": {"content": "a,b\n1,2"}},
]
TOOLS = [{"name": "read_file", "description": "read a file",
          "parameters": {"type": "object",
                         "properties": {"path": {"type": "string"}},
                         "required": ["path"]}}]


# -- anthropic ---------------------------------------------------------------


def test_anthropic_request_shape():
    body = AnthropicAdapter(api_key="k").build_request(MESSAGES, TOOLS)
    assert body["system"] == "be helpful"
    assert body["model"] and body["max_tokens"] > 0
    assert body["tools"][0]["input_schema"]["required"] == ["path"]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    tool_use = body["messages"][1]["content"][0]
    assert tool_use["type"] == "tool_use" and tool_use["name"] == "read_file"
    tool_result = body["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "c1"


def test_anthropic_parse_response():
    data = {
        "model": "claude-sonnet-4-5",
        "content": [
            {"type": "text", "text": "Reading now."},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file",
             "input": {"path": "a.csv"}},
        ],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }
    turn = AnthropicAdapter(api_key="k").parse_response(data, latency_ms=42.0)
    assert turn.text == "Reading now."
    assert turn.tool_calls == [{"call_id": "toolu_1", "tool": "read_file",
                                "args": {"path": "a.csv"}}]
    assert turn.usage.input_tokens == 12 and not turn.usage.simulated


def test_anthropic_missing_key_is_adapter_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="ANTHROPIC_API_KEY"):
        AnthropicAdapter().complete(MESSAGES, TOOLS)


def test_anthropic_http_500_maps_to_adapter_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(500, text="oops"))
    adapter = AnthropicAdapter(api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(AdapterError, match="500"):
        adapter.complete(MESSAGES, TOOLS)


def test_anthropic_garbage_response_is_malformed():
    with pytest.raises(MalformedOutputError):
        AnthropicAdapter(api_key="k").parse_response({"weird": True}, 0.0)


# -- openai --------------------------------------------------------------------


def test_openai_request_shape():
    body = OpenAIAdapter(api_key="k").build_request(MESSAGES, TOOLS)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    tc = body["messages"][2]["tool_calls"][0]
    assert tc["type"] == "function" and tc["function"]["name"] == "read_file"
    assert body["messages"][3]["tool_call_id"] == "c1"
    assert body["tools"][0]["function"]["parameters"]["required"] == ["path"]


def test_openai_parse_response():
    data = {
        "model": "gpt-4o",
        "choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "call_9", "type": "function",
                            "function": {"name": "read_file",
                                         "arguments": "{\"path\": \"a.csv\"}"}}],
        }}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }
    turn = OpenAIAdapter(api_key="k").parse_response(data, 10.0)
    assert turn.tool_calls[0]["args"] == {"path": "a.csv"}
    assert turn.usage.input_tokens == 20


def test_openai_bad_tool_arguments_json_is_malformed():
    data = {"choices": [{"message": {
        "content": None,
        "tool_calls": [{"id": "c", "type": "function",
                        "function": {"name": "t", "arguments": "{broken"}}],
    }}]}
    with pytest.raises(MalformedOutputError, match="not valid JSON"):
        OpenAIAdapter(api_key="k").parse_response(data, 0.0)


def test_openai_missing_key_is_adapter_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(AdapterError, match="OPENAI_API_KEY"):
        OpenAIAdapter().complete(MESSAGES, TOOLS)


def test_openai_http_500_maps_to_adapter_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(502, text="bad"))
    adapter = OpenAIAdapter(api_key="k", client=httpx.Client(transport=transport))
    with pytest.raises(AdapterError, match="502"):
        adapter.complete(MESSAGES, TOOLS)


# -- mock -----------------------------------------------------------------------


def test_mock_usage_is_deterministic_and_labeled():
    t1 = MockModelAdapter([{"final": "hello there"}]).complete(MESSAGES, TOOLS)
    t2 = MockModelAdapter([{"final": "hello there"}]).complete(MESSAGES, TOOLS)
    assert t1.usage == t2.usage
    assert t1.usage.simulated
    assert t1.usage.input_tokens > 0 and t1.usage.latency_ms > 0


def test_mock_script_exhaustion_is_malformed_error():
    adapter = MockModelAdapter([{"final": "only turn"}])
    adapter.complete(MESSAGES, TOOLS)
    with pytest.raises(MalformedOutputError, match="exhausted"):
        adapter.complete(MESSAGES, TOOLS)
