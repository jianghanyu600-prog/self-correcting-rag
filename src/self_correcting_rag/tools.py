"""Agent 工具：文本检索 / 图片检索 / 看图（vision-as-a-tool） / 计算器。"""

from __future__ import annotations

import ast
import json
import operator
from pathlib import Path

from openai import OpenAI

from .retrieve import BM25Retriever
from .vlm import answer_with_images

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "在商品文本资料（参数/详情/评价）中检索，返回最相关的条目。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_images",
            "description": "在商品图片的说明文字中检索（价格/尺码/容量等图上信息）。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": "让视觉模型直接查看某张原图并回答关于它的问题（图里有、文字没有时用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string", "description": "图片文件名，如 img_p5001_capacity.png"},
                    "question": {"type": "string"},
                },
                "required": ["doc_id", "question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算算术表达式，如 (1299-179)。",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(int(result)) if result == int(result) else str(result)
    except Exception as exc:  # noqa: BLE001
        return f"计算失败'{expression}'：{exc}"


class Tools:
    """把检索器/图片目录/VLM 打包成 agent 可调用的工具集。"""

    def __init__(
        self,
        text_retriever: BM25Retriever,
        image_retriever: BM25Retriever,
        image_dir: str | Path,
        vlm_client: OpenAI,
        *,
        k: int = 3,
    ) -> None:
        self.text_retriever = text_retriever
        self.image_retriever = image_retriever
        self.image_dir = Path(image_dir)
        self.vlm_client = vlm_client
        self.k = k

    def schemas(self) -> list[dict]:
        return SCHEMAS

    def call(self, name: str, arguments: str | dict) -> str:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        if name == "search_text":
            return self._search(self.text_retriever, args["query"])
        if name == "search_images":
            return self._search(self.image_retriever, args["query"])
        if name == "view_image":
            return self._view_image(args["doc_id"], args["question"])
        if name == "calculator":
            return calculator(args["expression"])
        return f"未知工具：{name}"

    def _search(self, retriever: BM25Retriever, query: str) -> str:
        hits = retriever.search(query, k=self.k)
        if not hits:
            return "（无结果）"
        return "\n".join(
            f"[{i}] {h.chunk.doc_id}: {h.chunk.text[:180]}"
            for i, h in enumerate(hits, start=1)
        )

    def _view_image(self, doc_id: str, question: str) -> str:
        path = self.image_dir / doc_id
        if not path.exists():
            return f"找不到图片：{doc_id}"
        return answer_with_images(self.vlm_client, question, [], [path])
