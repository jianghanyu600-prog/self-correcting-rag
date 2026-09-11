"""带工具的 Agent 循环（工具调用 + 多轮 + 预算），依赖注入以便离线测试。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

SYSTEM_PROMPT = (
    "你是电商商品分析 Agent。你可以调用工具：search_text（文本检索）、"
    "search_images（图片说明检索）、view_image（让视觉模型直接看原图回答问题）、"
    "calculator（计算）。\n"
    "原则：先检索再作答；图上才有的信息用 view_image；需要算数用 calculator；"
    "材料不足时直接回复：无法根据给定信息回答。回答要给出依据的 doc_id。"
)

# complete(messages, tools) -> {"content", "tool_calls":[{id,name,arguments}], "prompt_tokens", "completion_tokens"}
CompleteFn = Callable[[list[dict], list[dict]], dict[str, Any]]


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int


def openai_complete(client: OpenAI, model: str | None = None) -> CompleteFn:
    def _complete(messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        resp = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=messages,
            tools=tools or None,
            temperature=0.0,
        )
        message = resp.choices[0].message
        usage = resp.usage
        return {
            "content": message.content or "",
            "tool_calls": [
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (message.tool_calls or [])
            ],
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    return _complete


class ToolAgent:
    """最小可用 agent loop：模型决定调工具 → 执行 → 回填 → 再决定，直到给答案或超预算。"""

    def __init__(self, tools, *, system: str = SYSTEM_PROMPT, max_steps: int = 5) -> None:
        self.tools = tools
        self.system = system
        self.max_steps = max_steps

    def run(self, question: str, *, complete: CompleteFn) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": question},
        ]
        prompt_tokens = completion_tokens = tool_calls = 0
        last_content = ""

        for step in range(1, self.max_steps + 1):
            out = complete(messages, self.tools.schemas())
            prompt_tokens += out.get("prompt_tokens", 0)
            completion_tokens += out.get("completion_tokens", 0)
            last_content = out.get("content") or ""

            if not out.get("tool_calls"):
                return AgentResult(last_content, step, tool_calls, prompt_tokens, completion_tokens)

            messages.append(
                {
                    "role": "assistant",
                    "content": last_content,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"] or "{}",
                            },
                        }
                        for tc in out["tool_calls"]
                    ],
                }
            )
            for tc in out["tool_calls"]:
                tool_calls += 1
                try:
                    result = self.tools.call(tc["name"], tc["arguments"] or "{}")
                except Exception as exc:  # noqa: BLE001 - 工具错误要回给模型
                    result = f"工具执行失败：{exc}"
                messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": str(result)}
                )

        return AgentResult(
            last_content or "达到最大步数限制，未能给出可靠答案。",
            self.max_steps,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        )
