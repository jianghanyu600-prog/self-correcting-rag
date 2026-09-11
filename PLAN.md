# PLAN · 电商商品多模态问答（项目 A）— 已完成

## 定位
非客服：图文检索 + 可答性/忠实性评测 + Agent 工具循环。量化**模态缺口**，比较融合策略与 Agentic。

## 已完成（W1–W3）
- **W1**：5 商品语料 + 8 文本 golden；BM25(bigram) 基线 → Rec@3 6/6、正确 6/6、拒答 2/2
- **W2**：3 张「图上才有」合成图 + VLM caption 入索引 → 图题 text-only **0% → +图像 100%**
- **W2b**：回答阶段真看图（`answer_with_images`）→ 100%（与 caption 持平，负结果记录）
- **W3**：升级为 **Agentic**（工具：search_text / search_images / view_image / calculator；步数预算）
  - 易语料 n=14：固定 RAG 100% vs Agentic 100%（Agent 无收益、更贵）
  - **难例 n=6（+30 篇干扰）**：固定 RAG **67%** vs Agentic **83%** ← gap 成立
- 忠实性：无支撑拒答 100%、幻觉 0%；跨模型：DeepSeek 3/3 vs 千问 2/3

## 交付物
- README 六问（Problem/Architecture/Algorithm/Experiment/Ablation/Limitations/复现）
- 20 个离线单测；ruff 干净
- 实验脚本：baseline_a · eval_w2 · eval_w2_vision · eval_tempting · eval_agentic · eval_agentic_hard
- 数据生成：make_images · make_distractors

## 状态：**已完成 / 封板**

## Future Work（不进本轮）
1. Hybrid(向量+BM25) + Reranker
2. 「caption 有损」的密集表格图 → 量化真看图增益
3. Graph 检索（全局主题/关系类问题）作为消融维度
4. 迁移到真实公开数据（ABO / Product1M，注意许可）
5. 扩样本（图题 20+、难例 20+）让数字更稳
