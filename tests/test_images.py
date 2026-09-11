from __future__ import annotations

from pathlib import Path

from self_correcting_rag.images import caption_images
from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.retrieve import BM25Retriever, top_docs

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _fake_caption(path: Path) -> str:
    return {
        "img_p1001_promo.png": "SKU-1001 双十二活动 到手价 179 赠品 收纳袋 截止 12-31",
        "img_p2001_size.png": "SKU-2001 尺码建议 脚长 260mm 选 42 码",
        "img_p5001_capacity.png": "SKU-5001 炸篮实际可用容量 4.2L 标称 5L 清洗 温水 浸泡 10 分钟",
    }.get(path.name, path.name)


def _combined_chunks():
    text_chunks = chunk_documents(load_documents(CORPUS / "products_docs"))
    image_chunks = caption_images(CORPUS / "products_images", _fake_caption)
    return text_chunks + image_chunks


def test_caption_images_produces_chunks() -> None:
    chunks = caption_images(CORPUS / "products_images", _fake_caption)
    assert len(chunks) >= 3
    assert all(c.chunk_id.endswith("::0") for c in chunks)


def test_image_only_question_hits_caption_chunk() -> None:
    retriever = BM25Retriever(_combined_chunks())
    hits = retriever.search("双十二 到手价 赠品 收纳袋", k=3)
    assert "img_p1001_promo.png" in top_docs(hits)


def test_image_content_not_in_text_corpus() -> None:
    """text-only 语料里没有该信息，故此时不会命中图片。"""
    text_only = BM25Retriever(chunk_documents(load_documents(CORPUS / "products_docs")))
    hits = text_only.search("双十二 到手价 赠品 收纳袋", k=3)
    assert "img_p1001_promo.png" not in top_docs(hits)
