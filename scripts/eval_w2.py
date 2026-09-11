"""项目 A · W2 核心对比：图题上 text-only vs +图像(caption)，含幻觉率。

用法：uv run python scripts/eval_w2.py
"""

from __future__ import annotations

from pathlib import Path

from self_correcting_rag.eval import classify_faithfulness, judge_answer, retrieval_hit
from self_correcting_rag.gen import answer
from self_correcting_rag.images import caption_images, vision_caption_fn
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.llm import build_client
from self_correcting_rag.retrieve import BM25Retriever, top_docs
from self_correcting_rag.schema import load_golden
from self_correcting_rag.vlm import build_vision_client

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
K = 3


def _run(retriever, items, client, label: str, *, faithfulness: bool) -> tuple[int, int, int]:
    hit_n = ok_n = halluc_n = 0
    print(f"\n--- {label} ---")
    for g in items:
        top = retriever.search(g.question, k=K)
        hit = retrieval_hit(g, top_docs(top))
        contexts = [t.chunk.text for t in top]
        try:
            actual = answer(client, g.question, contexts)
            ok = judge_answer(client, g.question, g.answer or "", actual)
            verdict = (
                classify_faithfulness(client, g.question, contexts, actual)
                if faithfulness
                else "-"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{g.id}: ERR {exc}")
            continue
        hit_n += hit
        ok_n += ok
        halluc_n += verdict == "HALLUCINATE"
        print(
            f"{g.id}  hit={'Y' if hit else 'N'}  correct={'OK' if ok else 'NO'}  "
            f"faith={verdict}  | {g.question[:20]}"
        )
    n = len(items)
    print(
        f"{label}: 命中 {hit_n}/{n}，正确 {ok_n}/{n}={ok_n / n:.0%}，"
        f"幻觉 {halluc_n}/{n}={halluc_n / n:.0%}"
    )
    return hit_n, ok_n, halluc_n


def main() -> None:
    text_client = build_client()
    vision_client = build_vision_client()

    text_chunks = chunk_documents(load_documents(CORPUS / "products_docs"))
    image_chunks = caption_images(
        CORPUS / "products_images", vision_caption_fn(vision_client)
    )
    print(f"文本 chunk = {len(text_chunks)}，图像 caption = {len(image_chunks)}")

    retriever_text = BM25Retriever(text_chunks)
    retriever_mm = BM25Retriever(text_chunks + image_chunks)

    image_items = load_golden(CORPUS / "products_golden_image.jsonl")
    _, ok_text, hall_text = _run(
        retriever_text, image_items, text_client, "图题 · text-only", faithfulness=True
    )
    _, ok_mm, hall_mm = _run(
        retriever_mm, image_items, text_client, "图题 · +图像(caption)", faithfulness=True
    )

    text_items = [
        g for g in load_golden(CORPUS / "products_golden.jsonl") if g.kind != "no_answer"
    ]
    _run(retriever_mm, text_items, text_client, "文本题回归", faithfulness=False)

    n = len(image_items)
    print(f"\n=== W2 结论（图题，n={n}）===")
    print(f"{'':12}{'text-only':>12}{'+图像通道':>12}")
    print(f"{'答案正确率':12}{ok_text / n:>11.0%}{ok_mm / n:>12.0%}")
    print(f"{'幻觉答案数':12}{hall_text:>12}{hall_mm:>12}")
    print(f"{'幻觉率':12}{hall_text / n:>11.0%}{hall_mm / n:>12.0%}")


if __name__ == "__main__":
    main()
