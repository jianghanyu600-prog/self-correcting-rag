# corpus

检索语料 + 三类 golden 评测集（可扩展；当前为虚构产品「星云客服平台」seed）。

## 目录
- `documents/*.md`：语料，按文档/分片组织（billing/deployment/models/privacy…）
- `golden.jsonl`：评测题，每行一个 JSON 对象：

| 字段 | 含义 |
|---|---|
| id | 题目编号（A-xx 单跳 / B-xx 多跳 / C-xx 空召回） |
| kind | `single_hop` / `multi_hop` / `no_answer` |
| question | 用户问题 |
| expected_docs | 期望被检索到的文档名（相对 documents/）；`no_answer` 为空数组 |
| answer | 期望答案；`no_answer` 为 null |
| note | 依据说明（写题人备注） |

## 设计约束
- `single_hop`：单文档可答，首轮检索应命中；
- `multi_hop`：答案要拼 ≥2 处文档，首轮检索常漏 → 是自纠错的主要受益对象；
- `no_answer`：语料里确实没有 → 应正确拒答，防止幻觉。

目标规模：~50 题（A/B/C 大致 2:2:1），宁少而准，不凑数。
