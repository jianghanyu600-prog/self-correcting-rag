"""基于检索片段的生成器（受控回答，禁止编造）。"""

from __future__ import annotations

from openai import OpenAI

from .llm import chat

_SYSTEM = (
    "你是电商商品问答助手。只依据给定的商品片段回答，不要编造。"
    "若片段不足以回答，直接回复：无法根据给定信息回答。用中文，尽量简洁。"
)


def answer(
    client: OpenAI,
    question: str,
    contexts: list[str],
    *,
    model: str | None = None,
) -> str:
    block = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"商品片段：\n{block}\n\n问题：{question}",
        },
    ]
    return chat(client, messages, model=model, temperature=0.0)
