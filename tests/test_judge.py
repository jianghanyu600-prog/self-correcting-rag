from __future__ import annotations

from self_correcting_rag.judge import ContextGrader


def test_grader_returns_classification_from_llm_fn() -> None:
    calls: list[list[dict[str, str]]] = []

    def fake_llm(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return "SUFFICIENT"

    grader = ContextGrader(fake_llm)
    verdict = grader.grade("专业版多少钱", ["billing.md: 专业版每坐席每月 399 元。"])

    assert verdict == "SUFFICIENT"
    assert calls and calls[0][0]["role"] == "system"
    assert "问题：专业版多少钱" in calls[0][1]["content"]


def test_grader_empty_context_passes_through() -> None:
    grader = ContextGrader(lambda messages: "NO_REL")
    assert grader.grade("视频客服支持吗", []) == "NO_REL"
