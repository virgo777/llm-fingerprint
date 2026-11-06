# -*- coding: utf-8 -*-
"""logprobs 概率指纹采集：DeepSeek OpenAI 端点（支持 top-20 logprobs），
chat + reasoner × 10 探针 × N 轮，特征向量存 samples/<m>/<pid>.lp.rNN.txt。
断点续采；key 从环境变量 DS_KEY 读取。
用法：python -X utf8 examples/collect_logprobs.py   （COLLECT_ROUNDS 覆盖，默认 10）
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint import envkeys
from llm_fingerprint.features import extract_lp_features
from llm_fingerprint.probe_client import ProbeError, chat_completion, safe_note
from llm_fingerprint.probes import PROBES
from ui.monitor import MONITOR_PIDS

ROUNDS = int(os.environ.get("COLLECT_ROUNDS", "10"))
TIMEOUT_S = 35
WORKERS = 3
MODELS = ["deepseek-chat", "deepseek-reasoner"]
BASE = "https://api.deepseek.com"


def _key() -> str:
    return envkeys.require_key(envkeys.ENV_DEEPSEEK)


def one(model, key, pid, prompt, out_path) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    try:
        text, lp = chat_completion(BASE, key, model, prompt, temperature=0.5,
                                   max_tokens=1024, timeout=TIMEOUT_S,
                                   logprobs=True, top_logprobs=20,
                                   with_logprobs=True)
        feats = extract_lp_features(lp) if lp else None
    except (ProbeError, ValueError) as e:
        safe_note(f"{model}/{pid}.lp: {e}")
        return "fail"
    if not feats:
        return "fail"
    out_path.write_text(json.dumps(feats, ensure_ascii=False), encoding="utf-8")
    return "ok"


def run_model(model, key) -> int:
    root = pathlib.Path("samples") / model
    root.mkdir(parents=True, exist_ok=True)
    probes = [p for p in PROBES if p.pid in MONITOR_PIDS]
    fails = 0
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        jobs = [(p.pid, p.prompt, root / f"{p.pid}.lp.r{r:02d}.txt") for p in probes]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda j: one(model, key, j[0], j[1], j[2]), jobs))
        fails += sum(1 for x in res if x == "fail")
        print(f"[{model}] round {r}/{ROUNDS} fail={fails} "
              f"elapsed={int(time.time() - t0)}s", flush=True)
    print(f"[{model}] COMPLETE lp_files="
          f"{len(list(root.glob('*.lp.r*.txt')))} fail={fails}", flush=True)
    return fails


def main() -> int:
    key = _key()
    if not key:
        print("[error] 未找到 DS key", file=sys.stderr)
        return 2
    print(f"logprobs 采集：{ROUNDS} 轮 × 10 探针 × {len(MODELS)} 模型", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        fails = list(ex.map(lambda m: run_model(m, key), MODELS))
    print(f"全部完成 总耗时{int(time.time() - t0)}s 失败{sum(fails)}次", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
