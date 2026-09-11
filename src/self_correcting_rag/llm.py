"""极薄的 OpenAI 兼容 LLM 客户端（默认 DeepSeek）。"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


def build_client() -> OpenAI:
    load_dotenv()
    base = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base)


def chat(
    client: OpenAI,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """同步单轮对话，返回文本。judge/generate 都走它。"""
    model = model or os.getenv("MODEL_NAME", "deepseek-chat")
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature
    )
    return (resp.choices[0].message.content or "").strip()
