from __future__ import annotations

from pathlib import Path

from self_correcting_rag.ingest import chunk_documents, load_documents
from self_correcting_rag.schema import load_golden

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def test_load_documents_finds_all_md() -> None:
    docs = load_documents(CORPUS / "documents")
    assert len(docs) >= 4
    assert all(text.strip() for text in docs.values())


def test_chunking_one_chunk_per_doc() -> None:
    docs = load_documents(CORPUS / "documents")
    chunks = chunk_documents(docs)
    assert {c.doc_id for c in chunks} == set(docs)
    assert all(c.chunk_id.endswith("::0") for c in chunks)


def test_golden_loads_three_kinds() -> None:
    items = load_golden(CORPUS / "golden.jsonl")
    kinds = {item.kind for item in items}
    assert {"single_hop", "multi_hop", "no_answer"} <= kinds
    assert all(item.question.strip() for item in items)
