"""共享数据模型：语料、chunk、golden 题。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    """检索的最小单元。当前一个文档一个 chunk，后续可再按语义分片。"""

    chunk_id: str  # 例如 "billing.md::0"
    doc_id: str  # 文档名，如 "billing.md"
    text: str


@dataclass(frozen=True)
class GoldenItem:
    """golden 评测题。kind: single_hop / multi_hop / no_answer。"""

    id: str
    kind: str
    question: str
    expected_docs: tuple[str, ...]
    answer: str | None = None


def load_golden(path: str | Path) -> list[GoldenItem]:
    rows: list[GoldenItem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(
                GoldenItem(
                    id=obj["id"],
                    kind=obj["kind"],
                    question=obj["question"],
                    expected_docs=tuple(obj.get("expected_docs", [])),
                    answer=obj.get("answer"),
                )
            )
    return rows
