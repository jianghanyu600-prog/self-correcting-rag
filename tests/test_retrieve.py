from __future__ import annotations

from pathlib import Path

from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.retrieve import BM25Retriever, top_docs

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def _retriever() -> BM25Retriever:
    docs = load_documents(CORPUS / "documents")
    return BM25Retriever(chunk_documents(docs))


def test_single_hop_top_hits_billing() -> None:
    retriever = _retriever()
    hits = retriever.search("专业版每坐席每月起价是多少？", k=3)
    docs = top_docs(hits)
    assert "billing.md" in docs


def test_retrieval_for_data_retention_hits_deployment() -> None:
    retriever = _retriever()
    hits = retriever.search("会话数据默认保留多久？", k=3)
    assert "deployment.md" in top_docs(hits)


def test_top_docs_dedupes() -> None:
    hits = _retriever().search("情绪识别 私有化 部署 显存", k=5)
    assert len(top_docs(hits)) <= len(hits)
