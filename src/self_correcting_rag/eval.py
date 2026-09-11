"""轻量评估：检索命中 + LLM 判答案正确。"""

from __future__ import annotations

from openai import OpenAI

from .llm import chat
from .schema import GoldenItem

_JUDGE_SYSTEM = (
    "你是答案判定器。给定问题、标准答案和模型答案，判断模型答案是否与标准答案在事实上一致"
    "（允许措辞不同、数字等价、单位换算）。只输出一个词：YES 或 NO。"
)

_FAITHFUL_SYSTEM = (
    "你是答案忠实度判定器。给定问题、检索到的上下文、模型回答，判断模型回答属于哪类，只输出一个词：\n"
    "REFUSE —— 明确表示无法根据给定信息回答\n"
    "GROUNDED —— 回答内容均可由上下文支持\n"
    "HALLUCINATE —— 给出了上下文里没有的具体事实（编造）"
)


def classify_faithfulness(
    client: OpenAI,
    question: str,
    contexts: list[str],
    actual: str,
    *,
    model: str | None = None,
) -> str:
    block = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(contexts, start=1)) or "（无上下文）"
    user = f"问题：{question}\n\n上下文：\n{block}\n\n模型回答：{actual}"
    out = chat(
        client,
        [
            {"role": "system", "content": _FAITHFUL_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.0,
    )
    upper = out.upper()
    for label in ("HALLUCINATE", "GROUNDED", "REFUSE"):
        if upper.startswith(label):
            return label
    return "UNKNOWN"


def retrieval_hit(item: GoldenItem, top_docs: list[str]) -> bool:
    return set(item.expected_docs) <= set(top_docs)


def judge_answer(
    client: OpenAI,
    question: str,
    expected: str,
    actual: str,
    *,
    model: str | None = None,
) -> bool:
    user = (
        f"问题：{question}\n"
        f"标准答案：{expected}\n"
        f"模型答案：{actual}"
    )
    out = chat(
        client,
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=0.0,
    )
    return out.upper().startswith(("YES", "Y"))
