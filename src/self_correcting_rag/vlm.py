"""VLM（看图）客户端：读 VISION_* 环境变量，支持「描述图」与「带图作答」。"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_PROMPT = (
    "请仔细阅读这张商品图片，逐条列出图中的全部可见文字与关键信息"
    "（价格、尺码、容量、活动、说明等），不要遗漏、不要推测。"
)

_ANSWER_SYSTEM = (
    "你是电商商品问答助手。你会看到若干文本片段与商品图片。"
    "只依据给定材料回答，不要编造；若材料不足以回答，直接回复：无法根据给定信息回答。"
    "用中文，尽量简洁。"
)


def build_vision_client() -> OpenAI:
    load_dotenv()
    key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY")
    base = os.getenv("VISION_API_BASE") or os.getenv("OPENAI_API_BASE")
    return OpenAI(api_key=key, base_url=base)


def vision_model() -> str:
    return os.getenv("VISION_MODEL", "qwen-vl-plus")


def _data_url(path: str | Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def describe_image(
    client: OpenAI,
    path: str | Path,
    *,
    prompt: str = DEFAULT_PROMPT,
    model: str | None = None,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model or vision_model(), messages=messages, temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip()


def answer_with_images(
    client: OpenAI,
    question: str,
    text_contexts: list[str],
    image_paths: list[str | Path],
    *,
    model: str | None = None,
) -> str:
    """回答阶段真看图：文本片段 + 原图一起交给多模态模型。"""
    block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(text_contexts, start=1))
    content: list[dict] = [
        {
            "type": "text",
            "text": f"文本片段：\n{block or '（无）'}\n\n问题：{question}",
        }
    ]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    messages = [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": content},
    ]
    resp = client.chat.completions.create(
        model=model or vision_model(), messages=messages, temperature=0.0
    )
    return (resp.choices[0].message.content or "").strip()
