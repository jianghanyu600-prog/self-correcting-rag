"""检索层：先实现 BM25（中文 bigram 分词），后续在此接口上加 dense/hybrid/rerank。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from rank_bm25 import BM25Okapi

from .schema import Chunk

_ASCII = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """ASCII 按词切分；中文按「字符 bigram」切分，使 BM25 对中文也能匹配。"""
    lowered = text.lower()
    tokens = _ASCII.findall(lowered)
    for cjk_run in _CJK.findall(lowered):
        if len(cjk_run) == 1:
            tokens.append(cjk_run)
        else:
            tokens.extend(cjk_run[i : i + 2] for i in range(len(cjk_run) - 1))
    return tokens


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


class BM25Retriever:
    """BM25 检索器。先索引全部 chunk，再按 tokenize 匹配。"""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [Hit(self.chunks[i], float(scores[i])) for i in order[:k]]


def top_docs(hits: list[Hit]) -> list[str]:
    """去重后的命中文档列表（保序）。"""
    seen: list[str] = []
    for hit in hits:
        if hit.chunk.doc_id not in seen:
            seen.append(hit.chunk.doc_id)
    return seen
