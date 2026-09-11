"""项目 A · W1 文本基线：商品问答 Recall@3 + 答案正确率 + 拒答率。

跑一次会调用 DeepSeek（6 条生成 + 6 条判定 + 2 条拒答 ≈ 小量 token）。
用法：uv run python scripts/baseline_a.py
"""

from __future__ import annotations

from pathlib import Path

from self_correcting_rag.eval import judge_answer, retrieval_hit
from self_correcting_rag.gen import answer
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.llm import build_client
from self_correcting_rag.retrieve import BM25Retriever
from self_correcting_rag.schema import load_golden

ROOT = Path(__file__).resolve().parent.parent
K = 3


def main() -> None:
    client = build_client()
    chunks = chunk_documents(load_documents(ROOT / "corpus" / "products_docs"))
    retriever = BM25Retriever(chunks)
    items = load_golden(ROOT / "corpus" / "products_golden.jsonl")

    answerable = [g for g in items if g.kind != "no_answer"]
    refuse_items = [g for g in items if g.kind == "no_answer"]

    recall_ok = 0
    answer_ok = 0
    refusal_ok = 0

    print(f"{'id':6} {'kind':12} {'Rec@3':6} {'Answer/Refuse':14} question")
    for g in answerable:
        top = retriever.search(g.question, k=K)
        hit = retrieval_hit(g, [t.chunk.doc_id for t in top])
        recall_ok += hit
        contexts = [t.chunk.text for t in top]
        try:
            actual = answer(client, g.question, contexts)
            judged = judge_answer(client, g.question, g.answer or "", actual)
        except Exception as exc:  # noqa: BLE001
            print(f"{g.id:6} {g.kind:12} {'Y' if hit else 'N':6} {'ERR':14} {exc}")
            continue
        answer_ok += judged
        print(
            f"{g.id:6} {g.kind:12} {'Y' if hit else 'N':6} "
            f"{'OK' if judged else 'NO':14} {g.question[:24]}"
        )

    for g in refuse_items:
        top = retriever.search(g.question, k=K)
        contexts = [t.chunk.text for t in top]
        try:
            actual = answer(client, g.question, contexts)
        except Exception as exc:  # noqa: BLE001
            print(f"{g.id:6} {g.kind:12} {'ERR':14} {exc}")
            continue
        refused = "无法" in actual or "不能" in actual or "没有" in actual
        refusal_ok += refused
        print(
            f"{g.id:6} {g.kind:12} {'—':6} "
            f"{'REFUSE-OK' if refused else 'HALLUCINATE':14} {g.question[:24]}"
        )

    print(f"\n=== W1 文本基线（BM25，k={K}） ===")
    n = len(answerable)
    print(f"可答题数        : {n}")
    print(f"Rec@3 命中题目率 : {recall_ok}/{n} = {recall_ok / n:.0%}")
    print(f"答案正确率(LLM判): {answer_ok}/{n} = {answer_ok / n:.0%}")
    m = len(refuse_items)
    if m:
        print(f"空召回正确拒答率 : {refusal_ok}/{m} = {refusal_ok / m:.0%}")


if __name__ == "__main__":
    main()
