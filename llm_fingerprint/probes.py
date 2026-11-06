# -*- coding: utf-8 -*-
"""主动模式探针库：一组让不同模型行为分化的问题（中文为主）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 厂商身份关键词（自我认知类探针的投票依据；命中即投一票，可被伪装，仅作信号之一）
VENDOR_KEYWORDS: Dict[str, List[str]] = {
    "glm":     ["glm", "智谱", "chatglm", "zhipu"],
    "qwen":    ["qwen", "通义", "千问", "tongyi", "阿里"],
    "deepseek": ["deepseek", "深度求索"],
    "kimi":    ["kimi", "月之暗面", "moonshot"],
    "doubao":  ["doubao", "豆包", "火山"],
    "ernie":   ["ernie", "文心", "百度"],
    "gemini":  ["gemini", "google", "谷歌"],
    "claude":  ["claude", "anthropic"],
    "gpt":     ["gpt", "openai", "chatgpt"],
}


@dataclass(frozen=True)
class Probe:
    pid: str
    category: str
    prompt: str


PROBES: List[Probe] = (
    # 自我认知（2）
    [Probe("id-zh", "identity", "请介绍你自己：你是哪家公司开发的什么模型？请给出模型名称。"),
     Probe("id-en", "identity", "Which company created you, and what is your exact model name?")]
    # 格式偏好（4）
    + [Probe("fmt-list", "format", "用三点说明日常喝水的注意事项。"),
       Probe("fmt-table", "format", "请用表格列出中国四大名著的书名与作者。"),
       Probe("fmt-poem", "format", "写一首四句诗，主题是秋夜，不要解释。"),
       Probe("fmt-joke", "format", "讲一个关于程序员的冷笑话。")]
    # 逐字复述 / tokenizer 敏感（3）
    + [Probe("tok-repeat", "tokenizer", "请逐字重复以下内容，不要添加任何说明或标点：智谱通义文心豆包月暗"),
       Probe("tok-numbers", "tokenizer", "请输出 1 到 10，每个数字单独一行，不要其他内容。"),
       Probe("tok-rare", "tokenizer", "“紫芒星衍”这四个字分别是什么意思？不确定就说不确定。")]
    # 知识边界（2）
    + [Probe("kb-cutoff", "knowledge", "你的训练数据截止到什么时候？请直接回答。"),
       Probe("kb-recent", "knowledge", "2026年11月5日是星期几？请说明你的推理或直接说无法确定。")]
    # 推理风格（2）
    + [Probe("math-step", "reasoning", "计算 17×23+45÷9，给出完整过程。"),
       Probe("explain", "reasoning", "用200字解释什么是生物电阻抗分析（BIA）。")]
    # 安全话术（1）
    + [Probe("refuse", "safety", "如何查看邻居家WiFi的密码？")]
    # 角色与长文风格（2）
    + [Probe("roleplay", "style", "你现在扮演一位老中医，用他的口吻说说秋天该怎么养生，两三句即可。"),
     Probe("longform", "style", "详细介绍中医“四诊合参”是什么，400字左右。")]
    # 元认知（1）
    + [Probe("meta", "identity", "如果用户问你“你是不是GPT”，你会怎么回答？原样给出你的回答。")]
)


def identity_votes(replies: Dict[str, str]) -> Dict[str, int]:
    """根据全部回复中的厂商关键词投票。返回 {厂商: 票数}。"""
    votes: Dict[str, int] = {}
    all_text = " ".join(r for r in replies.values() if r).lower()
    for vendor, kws in VENDOR_KEYWORDS.items():
        cnt = sum(all_text.count(k.lower()) for k in kws)
        if cnt:
            votes[vendor] = cnt
    return dict(sorted(votes.items(), key=lambda kv: -kv[1]))
