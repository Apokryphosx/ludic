from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pytest

from ludic.agents.tool_agent import ToolAgent
from ludic.context.full_dialog import FullDialog
from ludic.inference.client import ChatClient, ChatResponse
from ludic.inference.sampling import SamplingConfig
from ludic.parsers import ParseResult
from ludic.types import Message


def _passthrough_parser(raw: str) -> ParseResult:
    return ParseResult(action=raw, reward=0.0, obs=None)


class DummyClient(ChatClient):
    async def complete(
        self,
        *,
        model: str,
        messages: List[Message],
        sampling: SamplingConfig,
    ) -> Tuple[ChatResponse, Dict[str, Any]]:
        return ChatResponse(text="ok"), {"raw_response": {"choices": [{"message": {"content": "ok"}}]}}

    def sync_weights(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def calculator_tool(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b


@pytest.mark.asyncio
async def test_tool_agent_helpers_and_execution():
    ctx = FullDialog()
    agent = ToolAgent(
        client=DummyClient(),
        model="mock",
        ctx=ctx,
        parser=_passthrough_parser,
        tools=[calculator_tool],
    )

    # _with_tools injects schemas + token id return
    sargs = agent._with_tools({})  # type: ignore[arg-type]
    assert "extras" in sargs
    assert "tools" in sargs["extras"]
    assert sargs["extras"]["tool_choice"] == "auto"
    assert sargs["extras"]["extra_body"]["return_token_ids"] is True

    # _strip_tools removes only tool-related fields
    stripped = agent._strip_tools(sargs)
    assert "tools" not in stripped["extras"]
    assert "tool_choice" not in stripped["extras"]
    assert stripped["extras"]["extra_body"]["return_token_ids"] is True

    # _extract_openai_message reads content/tool_calls
    info = {
        "raw_response": {
            "choices": [
                {"message": {"content": "hi", "tool_calls": [{"id": "c1"}]}}
            ]
        }
    }
    content, tool_calls = agent._extract_openai_message(info)
    assert content == "hi"
    assert tool_calls is not None and tool_calls[0]["id"] == "c1"

    # _run_tool_calls executes and records tool results
    tool_calls_payload = [
        {
            "id": "call_1",
            "function": {
                "name": "calculator_tool",
                "arguments": json.dumps({"a": 2, "b": 3}),
            },
        }
    ]
    agent._run_tool_calls(tool_calls_payload)

    assert ctx.messages[-1]["role"] == "tool"
    assert ctx.messages[-1]["content"] == "5"
    assert ctx.messages[-1]["tool_call_id"] == "call_1"

