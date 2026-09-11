"""语料装载：documents/*.md → Chunk 列表。"""

from __future__ import annotations

from pathlib import Path

from .schema import Chunk


def load_documents(directory: str | Path) -> dict[str, str]:
    """返回 {doc_id: 全文}。doc_id = 文件名。"""
    root = Path(directory)
    docs: dict[str, str] = {}
    for path in sorted(root.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    return docs


def chunk_documents(docs: dict[str, str]) -> list[Chunk]:
    """当前粒度：每个文档一个 chunk；后续可替换成按语义分片。"""
    return [
        Chunk(chunk_id=f"{doc_id}::0", doc_id=doc_id, text=text)
        for doc_id, text in docs.items()
    ]
