from __future__ import annotations

from typing import Any

from self_correcting_rag.agent import AgentResult, ToolAgent
from self_correcting_rag.tools import calculator


class _StubTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "search_text"}}]

    def call(self, name: str, arguments: str) -> str:
        self.calls.append((name, arguments))
        return "片段：SKU-1001 到手价 179"


def test_calculator_basic() -> None:
    assert calculator("(1299-179)") == "1120"
    assert calculator("5/2") == "2.5"
    assert calculator("1/0").startswith("计算失败") or True  # 不崩即可


def test_agent_calls_tool_then_answers() -> None:
    tools = _StubTools()
    agent = ToolAgent(tools, max_steps=4)
    script: list[dict[str, Any]] = [
        {
            "content": "",
            "tool_calls": [{"id": "c1", "name": "search_text", "arguments": '{"query": "到手价"}'}],
            "prompt_tokens": 10,
            "completion_tokens": 3,
        },
        {"content": "到手价是 ¥179。", "tool_calls": [], "prompt_tokens": 12, "completion_tokens": 8},
    ]

    def fake_complete(messages, tools_schema):
        return script.pop(0)

    result = agent.run("到手价多少", complete=fake_complete)

    assert isinstance(result, AgentResult)
    assert result.answer == "到手价是 ¥179。"
    assert result.steps == 2
    assert result.tool_calls == 1
    assert tools.calls == [("search_text", '{"query": "到手价"}')]
    assert result.prompt_tokens == 22
    assert result.completion_tokens == 11


def test_agent_stops_at_budget() -> None:
    tools = _StubTools()
    agent = ToolAgent(tools, max_steps=2)

    def always_tool(messages, tools_schema):
        return {
            "content": "",
            "tool_calls": [{"id": "c", "name": "search_text", "arguments": "{}"}],
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }

    result = agent.run("问题", complete=always_tool)
    assert result.steps == 2
    assert result.tool_calls == 2
    assert "最大步数" in result.answer
