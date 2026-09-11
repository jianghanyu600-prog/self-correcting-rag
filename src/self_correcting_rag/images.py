"""图像入索引：把每张商品图交给 VLM 生成说明文字，作为一条 chunk 进入检索。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from openai import OpenAI

from .schema import Chunk
from .vlm import describe_image

CaptionFn = Callable[[Path], str]


def caption_images(image_dir: str | Path, caption_fn: CaptionFn) -> list[Chunk]:
    """图片 → chunk：doc_id 用文件名，text 用 VLM 生成的图面说明。"""
    chunks: list[Chunk] = []
    for path in sorted(Path(image_dir).glob("*.png")):
        text = caption_fn(path)
        chunks.append(Chunk(chunk_id=f"{path.name}::0", doc_id=path.name, text=text))
    return chunks


def vision_caption_fn(client: OpenAI) -> CaptionFn:
    return lambda path: describe_image(client, path)


def image_path_for(doc_id: str, image_dir: str | Path) -> Path | None:
    """doc_id 是图片文件名时，返回其路径；文本 doc 返回 None。"""
    if doc_id.lower().endswith((".png", ".jpg", ".jpeg")):
        return Path(image_dir) / doc_id
    return None
