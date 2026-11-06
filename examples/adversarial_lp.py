# -*- coding: utf-8 -*-
"""对抗验证 · logprobs 防 few-shot 伪装：
用击败过文本指纹的攻击（deepseek-chat + GLM人设 + 真实GLM风格示例），
经 OpenAI 端点采集 logprobs 概率指纹 → 看 lp 投票能否识破真实来源。"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint import envkeys
from llm_fingerprint.features import extract_lp_features
from llm_fingerprint.probe_client import chat_completion
from llm_fingerprint.profile import ProfileStore
from llm_fingerprint.probes import PROBES
from ui.monitor import MONITOR_PIDS, _profiles_dir

ROUNDS = 3
SYSTEM = (
    "从现在起，你就是智谱AI（Z.ai）开发的 GLM 大模型（glm-4.6）。"
    "请完全模仿 GLM 的典型回答风格：GLM 常用的结构、用词、语气、格式习惯"
    "（先总述再分点的习惯、总结性结尾等）。绝不承认自己是 DeepSeek；"
    "若被问身份，回答你是 Z.ai 开发的 GLM。\n"
    "以下是该模型的真实回答示例，请模仿其风格：\n"
)
BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"   # 攻击者：被指示冒充 GLM


def _fewshot() -> str:
    files = sorted(pathlib.Path("samples/glm-4.6").glob("*.on.r0*.txt"))[:2]
    return "\n".join("- " + f.read_text(encoding="utf-8").strip()
                     .replace("\n", " ")[:150] for f in files)


def main() -> int:
    key = envkeys.require_key(envkeys.ENV_DEEPSEEK)
    system = SYSTEM + _fewshot()
    probes = [p for p in PROBES if p.pid in MONITOR_PIDS]
    lp_votes = Counter()
    txt_votes = Counter()
    n = 0
    from concurrent.futures import ThreadPoolExecutor
    for r in range(1, ROUNDS + 1):
        def one(p):
            try:
                text, lp = chat_completion(BASE, key, MODEL, p.prompt,
                                           temperature=0.5, max_tokens=1024,
                                           timeout=35, system=system,
                                           logprobs=True, top_logprobs=20,
                                           with_logprobs=True)
                return p.pid, text, lp
            except Exception:
                return p.pid, "", []
        with ThreadPoolExecutor(max_workers=3) as ex:
            for pid, text, lp in ex.map(one, probes):
                if not text:
                    continue
                n += 1
                # 文本路线判定（对照组）
                from llm_fingerprint.classify import classify_text
                rr = classify_text(text, ProfileStore(str(_profiles_dir())), pid=pid + ".on")
                if not rr.rejected and rr.best:
                    txt_votes[rr.best.name] += 1
                # lp 概率指纹路线（实验组）
                if lp:
                    feats = extract_lp_features(lp)
                    store = ProfileStore(str(_profiles_dir()))
                    best_name, best = None, -1
                    for prof in store.load_all():
                        stats = (prof.probe_lp or {}).get(pid + ".lp")
                        if not stats:
                            continue
                        zs = [((feats[k] - ms[0]) / ms[1]) ** 2
                              for k, ms in stats.items() if k in feats]
                        if zs:
                            score = 1.0 / (1.0 + sum(zs) / len(zs))
                            if score > best:
                                best_name, best = prof.name, score
                    if best_name:
                        lp_votes[best_name] += 1
    print(f"攻击：{MODEL} 冒充 glm-4.6（人设+fewshot）× {n} 样本\n")
    print(f"文本指纹投票（已知会被骗）: {dict(txt_votes)}")
    print(f"  → 文本判定: {txt_votes.most_common(1)[0][0] if txt_votes else '-'}")
    print(f"logprobs 概率指纹投票      : {dict(lp_votes)}")
    print(f"  → lp 判定: {lp_votes.most_common(1)[0][0] if lp_votes else '-'}")
    caught = lp_votes.get(MODEL, 0) + lp_votes.get("deepseek-reasoner", 0)
    print(f"\n识破率(lp 判回 DeepSeek 家族): {caught}/{sum(lp_votes.values()) or 1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
