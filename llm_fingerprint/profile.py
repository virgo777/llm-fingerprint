# -*- coding: utf-8 -*-
"""模型档案：由同一模型的若干条真实回复构建（均值/方差 + 三元组中心）。"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llm_fingerprint.features import LP_FEATURES, NUMERIC_FEATURES, extract_features

MIN_SAMPLES = 3
RECOMMENDED_SAMPLES = 10
TOP_GRAMS = 3000


@dataclass(frozen=True)
class ModelProfile:
    name: str
    n_samples: int
    feature_mean: Dict[str, float]
    feature_std: Dict[str, float]
    trigram_centroid: Dict[str, float] = field(default_factory=dict)
    embedding_centroid: Optional[List[float]] = None  # 512 维语义质心（嵌入器可用时）
    # 探针条件化子档案：{探针id: 该探针回复的嵌入质心}，单条判定时用（去除话题语义变量）
    probe_emb: Dict[str, List[float]] = field(default_factory=dict)
    # 探针条件化三元组质心：{探针id: {gram: weight}}，零依赖相似度路线（exe 内使用）
    probe_gram: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # logprobs 概率指纹统计：{pid.lp: {特征: [mean, std]}}（.lp 样本为 JSON 特征向量）
    probe_lp: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    @staticmethod
    def build(name: str, samples: List[str], embedder=None,
              sample_pids: Optional[List[str]] = None) -> "ModelProfile":
        if not name or not name.strip():
            raise ValueError("模型名不能为空")
        if len(samples) < MIN_SAMPLES:
            raise ValueError(f"样本不足：至少 {MIN_SAMPLES} 条（建议 {RECOMMENDED_SAMPLES}+），当前 {len(samples)} 条")
        if sample_pids is not None and len(sample_pids) != len(samples):
            raise ValueError("sample_pids 与 samples 数量不一致")
        feats = []
        grams: Dict[str, float] = {}
        lp_acc: Dict[str, List[Dict[str, float]]] = {}
        pids_iter = sample_pids if sample_pids is not None else [""] * len(samples)
        for s, pid in zip(samples, pids_iter):
            if pid.endswith(".lp"):
                # logprobs 概率指纹样本：内容是 JSON 特征向量，不做文本特征
                try:
                    d = json.loads(s)
                    if isinstance(d, dict):
                        lp_acc.setdefault(pid, []).append(d)
                except ValueError:
                    pass
                continue
            f, g = extract_features(s)
            feats.append(f)
            for k, v in g.items():
                grams[k] = grams.get(k, 0.0) + v
        n_text = len(feats)
        n = len(samples)  # n_samples 统计全部样本（含 lp）
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        for key in NUMERIC_FEATURES:
            vals = [f[key] for f in feats]
            m = sum(vals) / n_text
            var = sum((v - m) ** 2 for v in vals) / n_text
            mean[key] = m
            std[key] = max(math.sqrt(var), 1e-6)  # 避免除零
        top = sorted(grams.items(), key=lambda kv: -kv[1])[:TOP_GRAMS]
        total = sum(v for _, v in top) or 1.0
        centroid = {k: v / (total * n_text) for k, v in top}
        emb_centroid: Optional[List[float]] = None
        probe_emb: Dict[str, List[float]] = {}
        # 探针条件化三元组质心（零依赖，始终构建；.lp 样本不参与文本通道）
        probe_gram: Dict[str, Dict[str, float]] = {}
        if sample_pids is not None:
            from llm_fingerprint.features import extract_features as _ef
            by_probe_txt: Dict[str, List[str]] = {}
            for pid, s in zip(sample_pids, samples):
                if pid and not pid.endswith(".lp"):
                    by_probe_txt.setdefault(pid, []).append(s)
            for pid, txts in by_probe_txt.items():
                acc: Dict[str, float] = {}
                for s in txts:
                    _, g = _ef(s)
                    for k, v in g.items():
                        acc[k] = acc.get(k, 0.0) + v
                top_g = dict(sorted(acc.items(), key=lambda kv: -kv[1])[:1500])
                norm = sum(top_g.values()) or 1.0
                probe_gram[pid] = {k: v / norm for k, v in top_g.items()}
        if embedder is not None:
            import numpy as np
            text_samples = [s for s, pid in zip(samples, pids_iter)
                            if not pid.endswith(".lp")]
            vecs = np.stack([embedder.embed(s) for s in text_samples]) \
                if text_samples else None
            if vecs is not None:
                mvec = vecs.mean(axis=0)
                norm = float(np.linalg.norm(mvec))
                emb_centroid = (mvec / max(norm, 1e-12)).astype(float).tolist()
                if sample_pids is not None:
                    by_probe: Dict[str, List] = {}
                    for pid, s in zip(pids_iter, samples):
                        if pid and not pid.endswith(".lp"):
                            by_probe.setdefault(pid, []).append(embedder.embed(s))
                    for pid, arrs in by_probe.items():
                        m = np.stack(arrs).mean(axis=0)
                        nrm = float(np.linalg.norm(m))
                        probe_emb[pid] = (m / max(nrm, 1e-12)).astype(float).tolist()
        # logprobs 概率指纹统计（.lp 样本）
        probe_lp: Dict[str, Dict[str, List[float]]] = {}
        for pid, dicts in lp_acc.items():
            stats: Dict[str, List[float]] = {}
            for key in LP_FEATURES:
                vals = [d[key] for d in dicts if key in d]
                if len(vals) >= 2:
                    m = sum(vals) / len(vals)
                    sd = max(math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)), 1e-6)
                    stats[key] = [round(m, 6), round(sd, 6)]
            if stats:
                probe_lp[pid] = stats
        return ModelProfile(name=name.strip(), n_samples=n, feature_mean=mean,
                            feature_std=std, trigram_centroid=centroid,
                            embedding_centroid=emb_centroid, probe_emb=probe_emb,
                            probe_gram=probe_gram, probe_lp=probe_lp)

    def style_distance(self, feats: Dict[str, float]) -> float:
        """对角马氏距离（标准化后），返回平均每特征距离²，越小越像。"""
        acc = 0.0
        for key in NUMERIC_FEATURES:
            z = (feats[key] - self.feature_mean[key]) / self.feature_std[key]
            acc += z * z
        return acc / len(NUMERIC_FEATURES)

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name, "n_samples": self.n_samples,
            "feature_mean": self.feature_mean, "feature_std": self.feature_std,
            "trigram_centroid": self.trigram_centroid,
            "embedding_centroid": self.embedding_centroid,
            "probe_emb": self.probe_emb,
            "probe_gram": self.probe_gram,
            "probe_lp": self.probe_lp,
        }, ensure_ascii=False, indent=1)

    @staticmethod
    def from_json(raw: str) -> "ModelProfile":
        d = json.loads(raw)
        for key in ("name", "n_samples", "feature_mean", "feature_std", "trigram_centroid"):
            if key not in d:
                raise ValueError(f"档案文件损坏：缺少字段 {key}")
        return ModelProfile(name=d["name"], n_samples=int(d["n_samples"]),
                            feature_mean=d["feature_mean"], feature_std=d["feature_std"],
                            trigram_centroid=d["trigram_centroid"],
                            embedding_centroid=d.get("embedding_centroid"),
                            probe_emb=d.get("probe_emb") or {},
                            probe_gram=d.get("probe_gram") or {},
                            probe_lp=d.get("probe_lp") or {})


class ProfileStore:
    """档案目录：每模型一个 <name>.json。不可变写入（临时文件 + 替换）。"""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def save(self, p: ModelProfile) -> str:
        path = os.path.join(self.root, f"{p.name}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(p.to_json())
        os.replace(tmp, path)
        return path

    def load(self, name: str) -> ModelProfile:
        path = os.path.join(self.root, f"{name}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"档案不存在：{path}")
        with open(path, encoding="utf-8") as fh:
            return ModelProfile.from_json(fh.read())

    def load_all(self) -> List[ModelProfile]:
        out = []
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".json") and not fn.endswith(".tmp"):
                out.append(self.load(fn[:-5]))
        return out

    def names(self) -> List[str]:
        return [p.name for p in self.load_all()]
