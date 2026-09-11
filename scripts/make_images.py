"""生成合成商品图：图里含「文字语料中故意不写」的信息，用于多模态对比实验。

用法：uv run python scripts/make_images.py
产出：corpus/products_images/*.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "products_images"

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]

IMAGES = [
    (
        "img_p1001_promo.png",
        "SKU-1001 极境雪地羽绒服 · 双十二活动",
        [
            "活动到手价：¥179",
            "赠品：收纳袋 1 个",
            "活动截止：12-31",
            "（本页价格与赠品仅见于此图，商品详情文本未提）",
        ],
    ),
    (
        "img_p2001_size.png",
        "SKU-2001 山脊越野跑鞋 · 尺码建议",
        [
            "脚长 250mm → 建议 41 码",
            "脚长 260mm → 建议 42 码",
            "脚长 270mm → 建议 43 码",
            "（尺码表仅见于此图）",
        ],
    ),
    (
        "img_p5001_capacity.png",
        "SKU-5001 视窗空气炸锅 · 实拍说明",
        [
            "炸篮实际可用容量：4.2L",
            "标称容量：5L",
            "清洗建议：温水 + 中性洗涤剂浸泡 10 分钟",
            "（实际容量与清洗建议仅见于此图）",
        ],
    ),
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    title_font = _font(30)
    body_font = _font(26)
    for name, title, lines in IMAGES:
        width, height = 720, 90 + 56 * len(lines) + 40
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width, 70], fill="#1f6feb")
        draw.text((20, 18), title, fill="white", font=title_font)
        y = 96
        for line in lines:
            draw.text((28, y), line, fill="#111111", font=body_font)
            y += 56
        image.save(OUT / name)
        print(f"generated {name}")


if __name__ == "__main__":
    main()
