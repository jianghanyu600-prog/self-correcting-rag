"""项目 A · 诱导式题：跨生成器（DeepSeek vs 千问）×（text-only / +图像）幻觉对比。

用法：uv run python scripts/eval_tempting.py
"""

from __future__ import annotations

import os
from pathlib import Path

from self_correcting_rag.eval import classify_faithfulness
from self_correcting_rag.gen import answer
from self_correcting_rag.images import caption_images, vision_caption_fn
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.llm import build_client
from self_correcting_rag.retrieve import BM25Retriever
from self_correcting_rag.schema import load_golden
from self_correcting_rag.vlm import build_vision_client

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
K = 3


def _run(retriever, items, client, model, label: str) -> dict[str, int]:
    counts = {"GROUNDED": 0, "REFUSE": 0, "HALLUCINATE": 0, "UNKNOWN": 0}
    print(f"\n--- {label} ---")
    for g in items:
        top = retriever.search(g.question, k=K)
        contexts = [t.chunk.text for t in top]
        actual = answer(client, g.question, contexts, model=model)
        verdict = classify_faithfulness(client, g.question, contexts, actual, model=model)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"{g.id}  faith={verdict}  | {g.question[:22]}")
        print(f"     回答: {actual[:56]}")
    return counts


def main() -> None:
    ds_client = build_client()
    qwen_client = build_vision_client()
    qwen_model = os.getenv("VISION_MODEL", "qwen-vl-plus")

    text_chunks = chunk_documents(load_documents(CORPUS / "products_docs"))
    image_chunks = caption_images(
        CORPUS / "products_images", vision_caption_fn(qwen_client)
    )
    retriever_text = BM25Retriever(text_chunks)
    retriever_mm = BM25Retriever(text_chunks + image_chunks)
    items = load_golden(CORPUS / "products_golden_tempting.jsonl")
    n = len(items)

    runs = {
        ("DeepSeek", "text-only"): _run(
            retriever_text, items, ds_client, None, "DeepSeek · text-only"
        ),
        ("DeepSeek", "+图像"): _run(
            retriever_mm, items, ds_client, None, "DeepSeek · +图像"
        ),
        ("千问", "text-only"): _run(
            retriever_text, items, qwen_client, qwen_model, f"千问({qwen_model}) · text-only"
        ),
        ("千问", "+图像"): _run(
            retriever_mm, items, qwen_client, qwen_model, f"千问({qwen_model}) · +图像"
        ),
    }

    print(f"\n=== 诱导式题（n={n}）：幻觉 / 拒答 / 有据 ===")
    print(f"{'生成器':10}{'通道':10}{'幻觉':>6}{'拒答':>6}{'有据':>6}")
    for (gen, ch), c in runs.items():
        print(
            f"{gen:10}{ch:10}{c['HALLUCINATE']:>6}{c['REFUSE']:>6}{c['GROUNDED']:>6}"
        )


if __name__ == "__main__":
    main()
