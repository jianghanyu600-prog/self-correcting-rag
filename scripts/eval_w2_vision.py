"""项目 A · W2b：回答阶段「真看图」通道 vs 之前的 caption 通道。

用法：uv run python scripts/eval_w2_vision.py
"""

from __future__ import annotations

from pathlib import Path

from self_correcting_rag.eval import classify_faithfulness, judge_answer, retrieval_hit
from self_correcting_rag.images import caption_images, image_path_for, vision_caption_fn
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.llm import build_client
from self_correcting_rag.retrieve import BM25Retriever, top_docs
from self_correcting_rag.schema import load_golden
from self_correcting_rag.vlm import answer_with_images, build_vision_client

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
IMAGES = CORPUS / "products_images"
K = 3


def main() -> None:
    text_client = build_client()
    vision_client = build_vision_client()

    text_chunks = chunk_documents(load_documents(CORPUS / "products_docs"))
    image_chunks = caption_images(IMAGES, vision_caption_fn(vision_client))
    retriever = BM25Retriever(text_chunks + image_chunks)

    items = load_golden(CORPUS / "products_golden_image.jsonl")
    n = len(items)
    hit_n = ok_n = halluc_n = 0

    print(f"--- 图题 · 回答阶段真看图（k={K}）---")
    for g in items:
        top = retriever.search(g.question, k=K)
        hit = retrieval_hit(g, top_docs(top))
        texts, image_paths = [], []
        for t in top:
            path = image_path_for(t.chunk.doc_id, IMAGES)
            if path is not None and path.exists():
                image_paths.append(path)
            else:
                texts.append(t.chunk.text)
        actual = answer_with_images(vision_client, g.question, texts, image_paths)
        ok = judge_answer(text_client, g.question, g.answer or "", actual)
        contexts = [t.chunk.text for t in top]
        verdict = classify_faithfulness(text_client, g.question, contexts, actual)
        hit_n += hit
        ok_n += ok
        halluc_n += verdict == "HALLUCINATE"
        print(
            f"{g.id}  hit={'Y' if hit else 'N'}  correct={'OK' if ok else 'NO'}  "
            f"faith={verdict}  imgs={len(image_paths)}"
        )
        print(f"     回答: {actual[:64]}")

    print(f"\n=== W2b 结论（图题 n={n}）===")
    print(f"检索命中 {hit_n}/{n}，答案正确 {ok_n}/{n} = {ok_n / n:.0%}，幻觉 {halluc_n}/{n}")
    print("\n对比（同一批题、同一检索）：")
    print("  text-only        : 0/5 = 0%")
    print("  +caption(只看文字) : 5/5 = 100%")
    print(f"  +真看图(回答看图)  : {ok_n}/{n} = {ok_n / n:.0%}")


if __name__ == "__main__":
    main()
