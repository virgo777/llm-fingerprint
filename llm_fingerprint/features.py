# -*- coding: utf-8 -*-
"""风格特征提取：数值风格向量 + 字符三元组稀疏向量。纯标准库。"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

# 数值特征（固定顺序，档案与分类必须一致）
NUMERIC_FEATURES: List[str] = [
    "len_chars", "len_sentences", "sent_len_mean", "sent_len_std", "long_sent_ratio",
    "para_count", "para_len_mean",
    "comma_per_kchar", "period_per_kchar", "dunhao_per_kchar", "semicolon_per_kchar",
    "colon_per_kchar", "quote_per_kchar", "paren_per_kchar", "ellipsis_per_kchar",
    "dash_per_kchar", "exclaim_per_kchar", "question_per_kchar",
    "latin_ratio", "digit_ratio",
    "md_heading_count", "md_bold_count", "md_code_block_count", "md_table_row_count",
    "list_dash_count", "list_star_count", "list_number_count", "emoji_count",
    "conn_word_per_kchar", "first_person_per_kchar", "polite_per_kchar",
]

_CONN = re.compile(
    "首先|其次|再次|然后|此外|另外|同时|然而|因此|所以|总之|综上所述|"
    "需要注意的是|值得注意的是|总的来说|一方面|另一方面")
_FIRST_PERSON = re.compile("我|我们|咱们")
_POLITE = re.compile("希望对你有帮助|很高兴|当然可以|没问题|请告诉我|"
                     "如有任何|谢邀|希望这|如果有任何|随时问我")
_SENT_SPLIT = re.compile(r"[。！？!?;\n]+")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002702-\U000027B0\U0001F900-\U0001F9FF\U00002600-\U000026FF]")
_PUNCTS = {
    "comma_per_kchar": "，,", "period_per_kchar": "。.",
    "dunhao_per_kchar": "、", "semicolon_per_kchar": "；;",
    "colon_per_kchar": "：:", "quote_per_kchar": "“”\"‘’'",
    "paren_per_kchar": "（）()", "dash_per_kchar": "—–-—",
    "exclaim_per_kchar": "！", "question_per_kchar": "？?",
}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _per_kchar(text: str, pattern: re.Pattern) -> float:
    return len(pattern.findall(text)) * 1000.0 / max(len(text), 1)


def extract_features(text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """返回 (数值特征, 字符三元组 tf 向量)。文本过短抛 ValueError。"""
    if not text or not text.strip():
        raise ValueError("输入文本为空，无法提取特征")
    t = text.strip()
    n = len(t)
    f: Dict[str, float] = {}

    # 句子
    sents = [s.strip() for s in _SENT_SPLIT.split(t) if s.strip()]
    slens = [len(s) for s in sents]
    f["len_chars"] = float(n)
    f["len_sentences"] = float(len(sents))
    f["sent_len_mean"] = _safe_div(sum(slens), len(slens))
    mean = f["sent_len_mean"]
    f["sent_len_std"] = math.sqrt(_safe_div(sum((x - mean) ** 2 for x in slens), len(slens)))
    f["long_sent_ratio"] = _safe_div(sum(1 for x in slens if x > 50), len(slens))

    # 段落
    paras = [p for p in t.split("\n") if p.strip()]
    f["para_count"] = float(len(paras))
    f["para_len_mean"] = _safe_div(n, max(len(paras), 1))

    # 标点（每千字）
    for key, chars in _PUNCTS.items():
        f[key] = sum(t.count(c) for c in chars) * 1000.0 / n
    f["ellipsis_per_kchar"] = t.count("……") * 1000.0 / n

    # 字符构成
    latin = sum(1 for c in t if c.isascii() and c.isalpha())
    digit = sum(1 for c in t if c.isascii() and c.isdigit())
    f["latin_ratio"] = _safe_div(latin, n)
    f["digit_ratio"] = _safe_div(digit, n)

    # Markdown / 列表 / emoji
    f["md_heading_count"] = float(len(re.findall(r"^#{1,6}\s", t, re.M)))
    f["md_bold_count"] = float(len(re.findall(r"\*\*[^*\n]+\*\*", t)))
    f["md_code_block_count"] = float(len(re.findall(r"```", t)) // 2)
    f["md_table_row_count"] = float(len(re.findall(r"^\|.*\|$", t, re.M)))
    f["list_dash_count"] = float(len(re.findall(r"^[-—]\s", t, re.M)))
    f["list_star_count"] = float(len(re.findall(r"^\*\s", t, re.M)))
    f["list_number_count"] = float(len(re.findall(r"^\d+[\.、）]\s*", t, re.M)))
    f["emoji_count"] = float(len(_EMOJI.findall(t)))

    # 词汇风格（每千字）
    f["conn_word_per_kchar"] = _per_kchar(t, _CONN)
    f["first_person_per_kchar"] = _per_kchar(t, _FIRST_PERSON)
    f["polite_per_kchar"] = _per_kchar(t, _POLITE)

    # 字符三元组（去空白，保留中英混排次序信息）
    clean = re.sub(r"\s+", "", t)
    grams: Dict[str, float] = {}
    for i in range(len(clean) - 2):
        g = clean[i:i + 3]
        grams[g] = grams.get(g, 0.0) + 1.0
    total = max(len(clean) - 2, 1)
    return f, {k: v / total for k, v in grams.items() if v >= 1}


# ---------- logprobs 概率指纹特征（比表层文本难伪造一个量级） ----------

import math as _math

LP_FEATURES: List[str] = (
    ["lp_n_positions", "lp_top1_prob_mean", "lp_top1_prob_std", "lp_top1_prob_min",
     "lp_entropy_mean", "lp_entropy_std", "lp_mass_top1_mean", "lp_mass_top3_mean",
     "lp_margin_mean", "lp_gap_mean"] +
    [f"lp_hist_{i}" for i in range(8)])


def extract_lp_features(lp_content: list) -> Dict[str, float]:
    """从 top-k logprobs 序列提取概率指纹数值特征。lp_content 见 probe_client 返回。"""
    import math as _m
    if not lp_content:
        raise ValueError("logprobs 为空")
    top1 = []
    ents = []
    margins = []
    for e in lp_content:
        tops = e.get("top") or []
        if not tops:
            continue
        probs = sorted(((_m.exp(t["logprob"]), t["token"]) for t in tops),
                       key=lambda x: -x[0])
        p1 = probs[0][0]
        top1.append(p1)
        z = sum(p for p, _ in probs) or 1.0
        ents.append(-sum((p / z) * _m.log(p / z + 1e-12) for p, _ in probs))
        margins.append(p1 - (probs[1][0] if len(probs) > 1 else 0.0))
    if not top1:
        raise ValueError("logprobs 无有效 top 分布")
    n = len(top1)
    mean1 = sum(top1) / n
    f: Dict[str, float] = {"lp_n_positions": float(n),
                           "lp_top1_prob_mean": mean1,
                           "lp_top1_prob_std": _m.sqrt(sum((x - mean1) ** 2 for x in top1) / n),
                           "lp_top1_prob_min": min(top1),
                           "lp_entropy_mean": sum(ents) / n,
                           "lp_entropy_std": _m.sqrt(sum((x - sum(ents) / n) ** 2 for x in ents) / n),
                           "lp_mass_top1_mean": mean1,
                           "lp_mass_top3_mean": sum(sorted(top1, reverse=True)[:3]) / min(n, 3),
                           "lp_margin_mean": sum(margins) / n,
                           "lp_gap_mean": sum(top1) / n - min(top1)}
    hist = [0.0] * 8
    for x in top1:
        hist[min(int(x * 8), 7)] += 1
    for i in range(8):
        f[f"lp_hist_{i}"] = hist[i] / n
    return f
