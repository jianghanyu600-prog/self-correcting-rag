# corpus

项目 A 的语料与评测集（全部为自造的示例数据，可公开）。

## 目录
- `products_docs/*.md`：5 件商品的目标语料
- `products_docs_distract/*.md`：30 篇干扰商品（词重叠），用于制造"一次检索不够"的难例
- `products_images/*.png`：3 张合成商品图（信息只在图里，文字未提）
- `products_golden.jsonl`：文本题（单跳/多跳/空召回）
- `products_golden_image.jsonl`：图题（答案只在图中）
- `products_golden_tempting.jsonl`：诱导式题（题面埋前提，测忠实度）
- `products_golden_hard.jsonl`：难例（不含 SKU 的模糊题面，配合干扰文档）

## 评测题字段
| 字段 | 含义 |
|---|---|
| id | 题目编号 |
| kind | `single_hop` / `multi_hop` / `no_answer` / `image_only` / `tempting` / `hard` |
| question | 用户问题 |
| expected_docs | 期望命中的文档/图片名；`no_answer` 为空数组 |
| answer | 标准答案；`no_answer` 为 null |
| note | 出题依据说明 |
