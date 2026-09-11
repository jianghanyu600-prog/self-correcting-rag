"""检索质量自评（grade_context）与生成后校验（validate_claims）的 prompt 骨架。

judge 通过注入的 `llm_fn(messages) -> str` 调用，便于测试用假实现、不碰网络。
"""

from __future__ import annotations

from collections.abc import Callable

LLMFn = Callable[[list[dict[str, str]]], str]

_GRADER_SYSTEM = (
    "你是检索质量评估器。给定一个用户问题与若干检索到的片段，"
    "判断能否仅凭这些片段给出忠实、完整的回答。只输出一个词：\n"
    "SUFFICIENT —— 片段足以回答\n"
    "INSUFFICIENT —— 需要更多信息（片段相关但不够/不全）\n"
    "NO_REL —— 片段与问题无关"
)


class ContextGrader:
    """判断已召回片段是否足够作答。"""

    def __init__(self, llm_fn: LLMFn) -> None:
        self._llm_fn = llm_fn

    def grade(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n---\n\n".join(
            f"[{index}] {text}" for index, text in enumerate(contexts)
        )
        if not context_block:
            context_block = "（无召回）"
        user = f"问题：{question}\n\n检索到的片段：\n{context_block}"
        return self._llm_fn(
            [
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user},
            ]
        ).strip().upper()


def _is_supported(claims: list[str], contexts: list[str]) -> list[bool]:
    """占位：生成后校验把答案拆成断言，逐一对照来源。（D5 接入）"""
    return [False] * len(claims)
