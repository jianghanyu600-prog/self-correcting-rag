from __future__ import annotations

from pathlib import Path

from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.retrieve import BM25Retriever, top_docs
from self_correcting_rag.schema import load_golden

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _retriever() -> BM25Retriever:
    docs = load_documents(CORPUS / "products_docs")
    return BM25Retriever(chunk_documents(docs))


def test_product_corpus_loaded() -> None:
    docs = load_documents(CORPUS / "products_docs")
    assert len(docs) >= 5


def test_product_golden_three_kinds() -> None:
    items = load_golden(CORPUS / "products_golden.jsonl")
    kinds = {item.kind for item in items}
    assert {"single_hop", "multi_hop", "no_answer"} <= kinds


def test_fill_material_hits_snow_jacket() -> None:
    hits = _retriever().search("极境雪地羽绒服填充物是什么", k=3)
    assert "p1001_snow_jacket.md" in top_docs(hits)


def test_shoe_weight_hits_trail_shoes() -> None:
    hits = _retriever().search("越野跑鞋单只重量多少克", k=3)
    assert "p2001_trail_shoes.md" in top_docs(hits)


def test_multi_hop_needs_both_docs_in_top_k() -> None:
    hits = _retriever().search(
        "极寒徒步 防泼水越野跑鞋 抗 -30°C 外套 SKU 价格", k=5
    )
    docs = top_docs(hits)
    assert {"p2001_trail_shoes.md", "p1001_snow_jacket.md"} <= set(docs)
