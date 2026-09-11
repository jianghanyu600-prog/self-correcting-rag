"""生成 30 篇「干扰商品」文档：大量复用通用词，把目标文档挤出 top-k。

产出：corpus/products_docs_distract/p6xxx_*.md
用法：uv run python scripts/make_distractors.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "products_docs_distract"

_CATEGORIES = [
    ("羽绒服", ["填充物：80% 灰鸭绒", "填充物：90% 鹅绒", "填充物：70% 白鸭绒"]),
    ("越野跑鞋", ["外底：橡胶止滑", "外底：EVA 缓震", "外底：碳板竞速"]),
    ("瑜伽裤", ["面料：涤纶 90% + 氨纶 10%", "面料：棉 95% + 氨纶 5%", "面料：锦纶 70% + 氨纶 30%"]),
    ("空气炸锅", ["容量：3L", "容量：6L", "容量：8L"]),
    ("智能体脂秤", ["供电：2 节 AAA 电池", "供电：USB 充电", "供电：3 节 AA 电池"]),
    ("保温杯", ["保温：6 小时", "保温：12 小时", "保温：24 小时"]),
]

_TEMPLATE = """# SKU-{sku} {name} 商品详情

- 类目：{category}
- 双十二到手价：¥{price}（原价 ¥{origin}）
- 赠品：{gift}
- {spec}
- 尺码建议：脚长 {foot}mm 建议选 {size} 码
- 清洗建议：温水 + 中性洗涤剂浸泡 {minutes} 分钟
- 适用场景：{scene}
- 用户评价摘要：{review}
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for index in range(30):
        category, specs = _CATEGORIES[index % len(_CATEGORIES)]
        sku = 6000 + index
        text = _TEMPLATE.format(
            sku=sku,
            name=f"通用{category}{index:02d}",
            category=category,
            price=99 + index * 7,
            origin=199 + index * 9,
            gift=["收纳袋 1 个", "清洁布 1 张", "备用鞋带 1 副", "无"][index % 4],
            spec=specs[index % len(specs)],
            foot=235 + (index % 6) * 5,
            size=38 + (index % 6),
            minutes=5 + (index % 4) * 5,
            scene=["通勤", "户外", "居家", "运动"][index % 4],
            review=["性价比不错", "一般般", "回购了", "包装好"][index % 4],
        )
        (OUT / f"p{sku}_distractor_{index:02d}.md").write_text(text, encoding="utf-8")
    print(f"generated 30 distractor docs -> {OUT}")


if __name__ == "__main__":
    main()
