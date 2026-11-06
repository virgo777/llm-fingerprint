# -*- coding: utf-8 -*-
"""分类器：数值风格距离 + 三元组余弦 + 语义嵌入余弦，三路融合，开集拒绝。
嵌入器不可用（缺 onnxruntime 等）时自动退化为双路（风格 + 三元组）。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from llm_fingerprint.embedding import get_embedder
from llm_fingerprint.features import extract_features
from llm_fingerprint.profile import ModelProfile, ProfileStore

# 双路权重（无嵌入时）
W_STYLE_2 = 0.4
W_GRAM_2 = 0.6
# 三路权重（有嵌入时）
W_STYLE_3 = 0.25
W_GRAM_3 = 0.25
W_EMB_3 = 0.5

CONFIDENCE_FLOOR = 0.35  # 融合分低于此值 → 拒绝判定（开集保护）
MIN_MARGIN = 1.15        # best/second 比值低于此 → 置信度打折


@dataclass(frozen=True)
class Candidate:
    name: str
    style_score: float
    gram_score: float
    emb_score: float
    combined: float


@dataclass(frozen=True)
class Result:
    best: Optional[Candidate]
    runner_up: Optional[Candidate]
    confidence: float
    rejected: bool
    reason: str
    ranking: List[Candidate]
    mode: str  # "style+gram+embedding" | "style+gram"


def _cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(k, 0.0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def classify_text(text: str, store: ProfileStore, embedder=None, pid: str = None) -> Result:
    """对一段回复文本做闭集判定 + 开集拒绝。

    pid: 该回复对应的探针 id（如 "id-zh"）。提供且所有档案都有该探针的
    条件化质心时，嵌入比较切换到"同探针 vs 同探针"，去除话题语义变量。
    """
    profiles = store.load_all()
    if not profiles:
        raise ValueError("档案库为空：请先用 calibrate 命令校准至少一个模型")
    if embedder is None:
        embedder = get_embedder()
    feats, grams = extract_features(text)

    text_vec = None
    if embedder is not None and all(p.embedding_centroid for p in profiles):
        try:
            text_vec = embedder.embed(text)
        except Exception:
            text_vec = None
    # 探针条件化：全部档案都带该探针子质心时启用
    use_probe = bool(pid) and all(p.probe_emb.get(pid) for p in profiles) \
        and text_vec is not None

    cands: List[Candidate] = []
    for p in profiles:
        style_dist = p.style_distance(feats)
        style_score = 1.0 / (1.0 + style_dist)
        gram_score = _cosine(grams, p.trigram_centroid)
        if text_vec is not None:
            ref = p.probe_emb.get(pid) if use_probe else p.embedding_centroid
            emb = 0.0
            for x, y in zip(text_vec, ref):
                emb += float(x) * y
            combined = (W_STYLE_3 * style_score + W_GRAM_3 * gram_score
                        + W_EMB_3 * emb)
            mode = "style+gram+probe-embedding" if use_probe else "style+gram+embedding"
            emb_score = round(max(emb, 0.0), 4)
        else:
            combined = W_STYLE_2 * style_score + W_GRAM_2 * gram_score
            mode = "style+gram"
            emb_score = 0.0
        cands.append(Candidate(p.name, round(style_score, 4), round(gram_score, 4),
                               emb_score, round(combined, 4)))
    cands.sort(key=lambda c: -c.combined)
    best, second = (cands[0], cands[1]) if len(cands) >= 2 else (cands[0], None)

    if best.combined < CONFIDENCE_FLOOR:
        return Result(best, second, confidence=best.combined, rejected=True,
                      reason=f"最高融合分 {best.combined:.3f} 低于阈值 {CONFIDENCE_FLOOR}，"
                             f"可能不在已校准模型库中（或文本过短/风格被改写）",
                      ranking=cands, mode=mode)

    margin = (best.combined / second.combined) if second and second.combined > 0 else 2.0
    if margin >= MIN_MARGIN:
        conf = min(0.99, best.combined)
    else:
        conf = best.combined * (0.5 + 0.5 * margin / MIN_MARGIN)
    return Result(best, second, confidence=round(conf, 4), rejected=False,
                  reason="" if margin >= MIN_MARGIN else
                  f"前两名分差过小（{best.combined:.3f} vs {second.combined:.3f}），判定仅供参考",
                  ranking=cands, mode=mode)
