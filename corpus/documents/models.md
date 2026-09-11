# 星云客服平台 · 模型与 API

情绪识别使用自研模型 sentiment-v2，输入为对话原文与 ASR 转写文本，输出愤怒/焦虑/满意等情绪标签与置信度。
sentiment-v2 仅在私有化环境可用，通过 POST /v1/emotion 调用，需要 Bearer token。
智能质检使用 qa-checker 模型，依据会话记录自动打分。
语音渠道的 ASR 模型默认使用 stt-zh（中文普通话），可选 stt-dialect 适配方言。
