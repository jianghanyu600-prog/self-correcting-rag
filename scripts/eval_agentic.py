"""项目 A · Agentic vs 固定 RAG 对比（准确率 / 步数 / 工具调用 / token / 延迟）。

用法：uv run python scripts/eval_agentic.py
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


def _load_items():
    image_items = load_golden(CORPUS / "products_golden_image.jsonl")
    text_items = [g for g in load_golden(CORPUS / "products_golden.jsonl") if g.kind != "no_answer"]
    tempting = load_golden(CORPUS / "products_golden_tempting.jsonl")
    return image_items, text_items, tempting


def main() -> None:
    ds = build_client()
    vl = build_vision_client()

    text_chunks = chunk_documents(load_documents(CORPUS / "products_docs"))
    image_chunks = caption_images(IMAGES, vision_caption_fn(vl))
    retriever_all = BM25Retriever(text_chunks + image_chunks)

    tools = Tools(BM25Retriever(text_chunks), BM25Retriever(image_chunks), IMAGES, vl)
    agent = ToolAgent(tools, max_steps=5)
    complete = openai_complete(ds)

    image_items, text_items, tempting = _load_items()
    items = image_items + text_items + tempting

    fixed_ok = agent_ok = 0
    steps, toolcalls, tokens, latencies = [], [], [], []
    kind_fixed: dict[str, list[bool]] = {}
    kind_agent: dict[str, list[bool]] = {}

    print(f"{'id':7}{'kind':12}{'固定RAG':9}{'Agentic':9}{'步':4}{'工具':5}{'tok':6} 问题")
    for g in items:
        # 固定 RAG 基线（单次检索 + 单次生成）
        top = retriever_all.search(g.question, k=K)
        baseline = answer(ds, g.question, [t.chunk.text for t in top])
        b_ok = judge_answer(ds, g.question, g.answer or "", baseline)

        start = time.perf_counter()
        result = agent.run(g.question, complete=complete)
        elapsed = time.perf_counter() - start
        a_ok = judge_answer(ds, g.question, g.answer or "", result.answer)

        fixed_ok += b_ok
        agent_ok += a_ok
        kind_fixed.setdefault(g.kind, []).append(b_ok)
        kind_agent.setdefault(g.kind, []).append(a_ok)
        steps.append(result.steps)
        toolcalls.append(result.tool_calls)
        tokens.append(result.prompt_tokens + result.completion_tokens)
        latencies.append(elapsed)
        print(
            f"{g.id:7}{g.kind:12}{'OK' if b_ok else 'NO':9}{'OK' if a_ok else 'NO':9}"
            f"{result.steps:<4}{result.tool_calls:<5}{result.prompt_tokens + result.completion_tokens:<6}"
            f" {g.question[:20]}"
        )

    n = len(items)
    print(f"\n=== 固定 RAG vs Agentic（n={n}）===")
    print(f"{'':14}{'固定RAG':>10}{'Agentic':>10}")
    print(f"{'答案正确率':14}{fixed_ok / n:>9.0%}{agent_ok / n:>10.0%}")
    print(f"{'平均步数':14}{'-':>10}{statistics.mean(steps):>10.2f}")
    print(f"{'平均工具调用':14}{'-':>10}{statistics.mean(toolcalls):>10.2f}")
    print(f"{'平均 token':14}{'-':>10}{statistics.mean(tokens):>10.0f}")
    print(f"{'平均延迟(s)':14}{'-':>10}{statistics.mean(latencies):>10.2f}")

    print("\n按类别（固定RAG / Agentic）:")
    for kind in ("image_only", "single_hop", "multi_hop", "tempting"):
        fixed_list = kind_fixed.get(kind)
        agent_list = kind_agent.get(kind)
        if fixed_list:
            print(
                f"  {kind:12} {sum(fixed_list)}/{len(fixed_list)}"
                f"   {sum(agent_list)}/{len(agent_list)}"
            )


if __name__ == "__main__":
    main()
