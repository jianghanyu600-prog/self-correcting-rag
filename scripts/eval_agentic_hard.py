"""项目 A · 难例（含 30 篇干扰文档）：固定 RAG vs Agentic。

用法：uv run python scripts/eval_agentic_hard.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from self_correcting_rag.agent import ToolAgent, openai_complete
from self_correcting_rag.eval import judge_answer
from self_correcting_rag.gen import answer
from self_correcting_rag.images import caption_images, vision_caption_fn
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.llm import build_client
from self_correcting_rag.retrieve import BM25Retriever
from self_correcting_rag.schema import load_golden
from self_correcting_rag.tools import Tools
from self_correcting_rag.vlm import build_vision_client

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
IMAGES = CORPUS / "products_images"
K = 3


def main() -> None:
    ds = build_client()
    vl = build_vision_client()

    text_docs = load_documents(CORPUS / "products_docs")
    distract_docs = load_documents(CORPUS / "products_docs_distract")
    text_chunks = chunk_documents({**text_docs, **distract_docs})
    image_chunks = caption_images(IMAGES, vision_caption_fn(vl))
    retriever_all = BM25Retriever(text_chunks + image_chunks)

    tools = Tools(BM25Retriever(text_chunks), BM25Retriever(image_chunks), IMAGES, vl)
    agent = ToolAgent(tools, max_steps=5)
    complete = openai_complete(ds)

    items = load_golden(CORPUS / "products_golden_hard.jsonl")
    print(f"语料：目标 {len(text_docs)} 篇 + 干扰 {len(distract_docs)} 篇 + 图 {len(image_chunks)}")

    fixed_hit = fixed_ok = agent_ok = 0
    steps, toolcalls, tokens, latencies = [], [], [], []

    print(f"\n{'id':6}{'固定RAG命中':12}{'固定RAG':9}{'Agentic':9}{'步':4}{'工具':5} 问题")
    for g in items:
        top = retriever_all.search(g.question, k=K)
        from self_correcting_rag.retrieve import top_docs

        hit = set(g.expected_docs) <= set(top_docs(top))
        fixed_hit += hit
        baseline = answer(ds, g.question, [t.chunk.text for t in top])
        b_ok = judge_answer(ds, g.question, g.answer or "", baseline)

        start = time.perf_counter()
        result = agent.run(g.question, complete=complete)
        latencies.append(time.perf_counter() - start)
        a_ok = judge_answer(ds, g.question, g.answer or "", result.answer)

        fixed_ok += b_ok
        agent_ok += a_ok
        steps.append(result.steps)
        toolcalls.append(result.tool_calls)
        tokens.append(result.prompt_tokens + result.completion_tokens)
        print(
            f"{g.id:6}{'Y' if hit else 'N':12}{'OK' if b_ok else 'NO':9}"
            f"{'OK' if a_ok else 'NO':9}{result.steps:<4}{result.tool_calls:<5} {g.question[:18]}"
        )

    n = len(items)
    print(f"\n=== 难例（含 30 篇干扰，n={n}）===")
    print(f"{'':16}{'固定RAG':>10}{'Agentic':>10}")
    print(f"{'检索命中率':16}{fixed_hit / n:>9.0%}{'-':>10}")
    print(f"{'答案正确率':16}{fixed_ok / n:>9.0%}{agent_ok / n:>10.0%}")
    print(f"{'平均步数':16}{'-':>10}{statistics.mean(steps):>10.2f}")
    print(f"{'平均工具调用':16}{'-':>10}{statistics.mean(toolcalls):>10.2f}")
    print(f"{'平均 token':16}{'-':>10}{statistics.mean(tokens):>10.0f}")
    print(f"{'平均延迟(s)':16}{'-':>10}{statistics.mean(latencies):>10.2f}")


if __name__ == "__main__":
    main()
