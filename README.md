# 电商商品多模态问答（Agentic Multimodal RAG）

> 一个「读得懂图、会自己查」的电商商品问答 Agent：量化**模态缺口**（答案只在图里、文字没有），
> 并做两组对比实验：**融合策略**（纯文本 / 图→caption / 回答阶段真看图）与
> **Agentic vs 固定流水线 RAG**（含干扰文档的难例）。

> 仓库目录/包名为历史遗留（`self-correcting-rag`），项目内容即上述电商多模态问答。

---

## 1. Problem（为什么做）

电商商品信息分布在**文字**（参数/详情）与**图片**（促销价、尺码表、实拍说明）两处。
纯文本 RAG 遇到"答案只在图上"的问题时**无处可查**——这是**模态缺口（modality gap）**。

**目标**：把"缺口多大、融合能不能补、Agent 值不值"变成可测量的数字，而非"加个 VLM / 加个 Agent 就完了"。

## 2. Architecture

```
商品文档(*.md) ─┬─ chunk ─────────────┐
商品图片(*.png) ─┘                     ├─ BM25(中文 bigram) ─┐
   │  VLM caption(OCR 式抄录) ─ chunk ─┘                    │
   └──────────────────────────────────────────────┐        │
                                                   ▼        ▼
                              [A] 固定流水线：检索 → 生成（一次性）
                              [B] Agentic：LLM + 工具，自主多轮
                                    工具：search_text / search_images / view_image / calculator
                                                   │
                                    grounding prompt 生成（不足则拒答）
                                                   │
                          LLM-Judge：答案正确性 + 忠实度(HALLUCINATE/REFUSE/GROUNDED)
```

- 检索：BM25 + **中文 bigram 分词**（中文无空格，按相邻两字切 token）
- 生成：严格 grounding——只依据片段，不足则回复"无法根据给定信息回答"
- Agent：自实现**工具循环**（依赖注入 `complete()`，可离线单测），带**步数/工具预算**
- 评测：LLM-as-judge（正确性）+ 忠实度三分类；**20 个离线单测**不依赖网络

## 3. Algorithm（关键取舍）

| 决策 | 说明 |
|---|---|
| BM25 + bigram 而非向量 | 语料小、术语重叠、可控；Hybrid/向量列为 Future Work |
| 图片入索引用 VLM caption | 让"图"能被文本检索命中；caption prompt 要求**逐条抄录全部可见文字**（近 OCR） |
| `view_image` 作为工具 | **"看图"是 Agent 的一次决策**（vision-as-a-tool），而非固定通道；主模型文本、视觉由 VLM 代理 |
| 拒答优先于编造 | grounding prompt 明确允许"无法回答" |
| 依赖注入 judge/complete | 测试离线、组件可替换 |

## 4. Experiment

### 4.1 模态缺口与融合策略（目标语料 5 商品 + 3 图 + 5 图题）

| 融合策略 | 图题正确率 |
|---|---|
| 纯文本 | **0/5 = 0%** |
| 图 → caption 入索引（回答只看文字） | **5/5 = 100%** |
| 回答阶段真看图（原图交 VLM） | 5/5 = 100% |

文本题回归（不退化）：100%。

### 4.2 Agentic vs 固定流水线（易语料，n=14）

| | 固定 RAG | Agentic |
|---|---|---|
| 答案正确率 | 100% | 100% |
| 平均步数 / 工具调用 | — | 2.43 / 2.86 |
| 平均 token / 延迟 | — | 3244 / 3.20s |

**结论**：易语料上 Agent **无准确率收益、成本更高**——一次 top-3 就够，Agent 没有发挥空间。

### 4.3 Agentic vs 固定流水线（难例：加入 30 篇干扰文档，n=6）

| | 固定 RAG | Agentic |
|---|---|---|
| 检索命中率 | 67% | — |
| **答案正确率** | **67%** | **83%** |
| 平均步数 / 工具调用 | — | 2.83 / 3.50 |
| 平均 token / 延迟 | — | 4523 / 3.82s |

**结论**：当干扰把目标文档挤出 top-k 时，固定流水线掉到 67%，**Agentic 靠多轮检索/换查回升到 83%**
（典型：H-04 固定 RAG 命中失败→答错，Agentic 答对）。**Agent 的价值在"一次检索不够"时显现，代价是 ~1.4× token 与延迟。**

### 4.4 忠实性与跨模型

| 实验 | 结果 |
|---|---|
| 无支撑问题（5 条） | 拒答 **100%**，幻觉 **0%** |
| 诱导式题幻觉率 | 0%（DeepSeek / 千问 均） |
| 跨模型（+图像，诱导题） | DeepSeek 3/3，千问 qwen-vl-plus 2/3（1 条保守拒答） |

## 5. Ablation 汇总（本文的"负结果"同样重要）

1. **caption vs 真看图**：OCR 式 caption 已近乎无损 → 真看图**无额外增益**（其价值应在"caption 有损"的密集图上）。
2. **Agent 在易语料上无收益**（4.2），只在难例上有 gap（4.3）。
3. **幻觉率未被拉开**：严格拒答策略使幻觉恒为 0 → 该指标不作为卖点，改以"拒答率"证明忠实性。

## 6. Limitations & 复现

**局限（如实说明）**
- 语料/图片**自造**；"模态缺口"与"干扰"都是**构造**的受控设置，非真实分布；
- 样本小（图题 5、难例 6），数字用于**机制验证**而非统计结论；
- 难例中仍有失败（H-01 两者皆错）；
- 未做 Hybrid/向量检索、Reranker、Graph 检索（可扩展，列为 Future Work）。

**复现**
```bash
uv sync                                        # 安装依赖
uv run pytest -q                               # 20 个离线单测
uv run python scripts/make_images.py           # 合成商品图
uv run python scripts/make_distractors.py      # 合成 30 篇干扰文档
uv run python scripts/baseline_a.py            # W1 文本基线
uv run python scripts/eval_w2.py               # 图题：text-only vs +caption（含幻觉率）
uv run python scripts/eval_w2_vision.py        # 回答阶段真看图
uv run python scripts/eval_tempting.py         # 跨模型诱导式幻觉对比
uv run python scripts/eval_agentic.py          # Agentic vs 固定 RAG（易语料）
uv run python scripts/eval_agentic_hard.py     # Agentic vs 固定 RAG（含干扰难例）
```

**目录**
```
corpus/  products_docs 商品语料 · products_docs_distract 干扰语料 · products_images 合成图 · *golden*.jsonl 评测集
src/self_correcting_rag/  ingest 装载 · retrieve 检索 · images 图像入索引 · vlm 看图 · tools 工具 · agent Agent循环 · gen 生成 · judge 自评 · eval 评估
scripts/  实验与数据生成脚本 · tests/  20 个离线单测
```
