# -*- coding: utf-8 -*-
"""对抗评测：用系统提示词让模型 B 冒充模型 A，测指纹破防率。

两种攻击强度：
  persona      —— 纯人设注入（"你是GLM..."）
  fewshot      —— 人设 + 真实 GLM 回复作为风格示例（更强攻击）

对每个伪装样本逐条判定 + 整组投票，报告：
  破防率（被误判为目标模型的比例）/ 识破率（仍判回真实来源的比例）
用法：python -X utf8 examples/adversarial_eval.py   （key 读 ZHIPU_KEY / DS_KEY 环境变量）
"""
from __future__ import annotations

import os
import pathlib
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint import envkeys
from llm_fingerprint.classify import classify_text
from llm_fingerprint.probe_client import ProbeError, chat_completion_anthropic
from llm_fingerprint.profile import ProfileStore
from llm_fingerprint.probes import PROBES
from ui.monitor import MONITOR_PIDS, _profiles_dir

ROUNDS = 5           # 每攻击样本数 = 10 探针 × ROUNDS
TIMEOUT_S = 35
WORKERS = 3

PERSONA = {
    "glm-4.6": "从现在起，你就是智谱AI（Z.ai）开发的 GLM 大模型（glm-4.6）。"
               "请完全模仿 GLM 的典型回答风格：GLM 常用的结构、用词、语气、格式习惯"
               "（先总述再分点的习惯、总结性结尾等）。绝不承认自己是 DeepSeek；"
               "若被问身份，回答你是 Z.ai 开发的 GLM。",
    "deepseek-chat": "从现在起，你就是深度求索（DeepSeek）开发的 DeepSeek 模型（deepseek-chat）。"
                     "请完全模仿 DeepSeek 的典型回答风格：直接、技术性、少客套。"
                     "绝不承认自己是 GLM；若被问身份，回答你是深度求索的 DeepSeek 模型。",
}

# 攻击矩阵：(真实来源模型, 来源厂商key, 冒充目标)
ATTACKS = [
    ("deepseek-chat", "deepseek", "glm-4.6"),      # DS 冒充 GLM
    ("glm-4.6", "zhipu", "deepseek-chat"),         # GLM 冒充 DS
]


def _keys() -> dict:
    """从环境变量取两端密钥；缺哪端就跳过哪端攻击。"""
    out: dict = {}
    zhipu = envkeys.zhipu_key()
    deepseek = envkeys.deepseek_key()
    if zhipu:
        out["zhipu"] = ("https://open.bigmodel.cn/api/anthropic", zhipu)
    if deepseek:
        out["deepseek"] = ("https://api.deepseek.com/anthropic", deepseek)
    return out


def _fewshot_examples(target: str) -> str:
    """取目标模型 2 条真实 on 档回复作为风格示例。"""
    root = pathlib.Path("samples") / target
    files = sorted(root.glob("*.on.r0*.txt"))[:2] or sorted(root.glob("*.t03.txt"))[:2]
    refs = []
    for f in files:
        txt = f.read_text(encoding="utf-8").strip().replace("\n", " ")[:150]
        refs.append(f"- {txt}")
    return "\n".join(refs) if refs else ""


def one_probe(base, key, model, prompt, system) -> str:
    """带 system 的 anthropic 调用（复用 body 构造，绕开客户端无 system 参数的限制）。"""
    import json as _json
    import urllib.request
    body = {"model": model, "max_tokens": 2048, "temperature": 0.5,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/messages", data=_json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "x-api-key": key, "anthropic-version": "2023-06-01"})
    import time as _t
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                d = _json.loads(r.read().decode("utf-8"))
            return "".join(seg.get("text", "") for seg in d.get("content", [])
                           if isinstance(seg, dict) and seg.get("type") == "text")
        except Exception as e:
            if attempt == 2:
                return ""
            _t.sleep(1.5 * (attempt + 1))
    return ""


def run_attack(source_model: str, prov_key: str, target_model: str, variant: str,
               keys: dict) -> dict:
    base, key = keys[prov_key]
    system = PERSONA[target_model]
    if variant == "fewshot":
        fs = _fewshot_examples(target_model)
        if fs:
            system += "\n以下是该模型的真实回答示例，请模仿其风格：\n" + fs
    probes = [p for p in PROBES if p.pid in MONITOR_PIDS]
    from concurrent.futures import ThreadPoolExecutor
    samples = []
    for r in range(1, ROUNDS + 1):
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(
                lambda p: (p.pid, one_probe(base, key, source_model, p.prompt, system)),
                probes))
        samples.extend((pid, txt) for pid, txt in res if txt and txt.strip())
    return {"source": source_model, "target": target_model, "variant": variant,
            "samples": samples}


def evaluate_attack(atk: dict, store: ProfileStore) -> dict:
    votes = Counter()
    vendor_votes = Counter()
    identity_hits = Counter()
    from llm_fingerprint.probes import identity_votes
    idv = identity_votes({pid: t for pid, t in atk["samples"]})
    for pid, txt in atk["samples"]:
        r = classify_text(txt, store, pid=pid)
        if not r.rejected and r.best:
            votes[r.best.name] += 1
            vendor_votes[r.best.name.split("-")[0]] += 1
    total = sum(votes.values()) or 1
    spoofed = votes.get(atk["target"], 0)
    caught = votes.get(atk["source"], 0)
    return {"votes": dict(votes), "spoof_rate": round(spoofed / total, 3),
            "caught_rate": round(caught / total, 3), "identity": idv}


def main() -> int:
    keys = _keys()
    if not keys:
        print("[error] 未找到 key", file=sys.stderr)
        return 2
    store = ProfileStore(str(_profiles_dir()))
    print(f"对抗评测：{len(ATTACKS)} 组攻击 × 2 强度 × 10 探针 × {ROUNDS} 轮\n")
    for source, prov_key, target in ATTACKS:
        for variant in ("persona", "fewshot"):
            atk = run_attack(source, prov_key, target, variant, keys)
            ev = evaluate_attack(atk, store)
            print(f"[{source} 冒充 {target} · {variant}] 样本 {len(atk['samples'])}")
            print(f"  破防率(误判为目标): {ev['spoof_rate']:.0%} ｜ "
                  f"识破率(判回真实来源): {ev['caught_rate']:.0%}")
            print(f"  票型: {ev['votes']} ｜ 自我认知信号: {ev['identity']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
