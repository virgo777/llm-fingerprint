# -*- coding: utf-8 -*-
"""重复采集：每模型 10 探针 × 50 轮（500 条），构建更强特征档案。

- 可断点续采：已存在且非空的样本文件直接跳过
- key 来源：环境变量 ZHIPU_KEY / DS_KEY（缺失的目标自动跳过）
- 用法：python -X utf8 examples/collect_rounds.py   （在 llm-fingerprint 目录下）
- 轮数可用环境变量 COLLECT_ROUNDS 覆盖（冒烟测试用 1）
输出 samples/<model>/<pid>.r<NN>.txt；每 10 轮打印一次进度。
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint import envkeys
from llm_fingerprint.probe_client import (ProbeError, chat_completion,
                                          chat_completion_anthropic, safe_note)
from llm_fingerprint.probes import PROBES
from ui.monitor import MONITOR_PIDS

ROUNDS = int(os.environ.get("COLLECT_ROUNDS", "50"))
TIMEOUT_S = 35
WORKERS_PER_MODEL = 3
TEMPERATURE = 0.5
SAMPLES_ROOT = pathlib.Path("samples")

MODELS = [("glm-4.6", "zhipu"), ("glm-4.5", "zhipu"),
          ("deepseek-chat", "deepseek"), ("deepseek-reasoner", "deepseek")]


def _keys_from_env() -> dict:
    """只从环境变量取 key：ZHIPU_KEY / DS_KEY；缺哪端就跳过哪端。"""
    out: dict = {}
    zhipu = envkeys.zhipu_key()
    ds = envkeys.deepseek_key()
    if zhipu:
        out["zhipu"] = ("https://open.bigmodel.cn/api/anthropic", "anthropic", zhipu)
    if ds:
        out["deepseek"] = ("https://api.deepseek.com/anthropic", "anthropic", ds)
    return out


def one_call(model: str, prov: tuple, pid: str, prompt: str, out_path: pathlib.Path) -> str:
    """单次调用并落盘。已有非空文件跳过；失败返回 fail（客户端内部已重试）。"""
    base, api, key = prov
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    fn = chat_completion_anthropic if api == "anthropic" else chat_completion
    try:
        txt = fn(base, key, model, prompt, temperature=TEMPERATURE, timeout=TIMEOUT_S)
    except ProbeError as e:
        safe_note(f"{model}/{pid}: {e}")
        return "fail"
    if txt and txt.strip():
        out_path.write_text(txt, encoding="utf-8")
        return "ok"
    return "fail"


def run_model(name: str, prov_name: str, prov: tuple) -> int:
    probes = [p for p in PROBES if p.pid in MONITOR_PIDS]
    root = SAMPLES_ROOT / name
    root.mkdir(parents=True, exist_ok=True)
    fails = 0
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        jobs = [(p, root / f"{p.pid}.r{r:02d}.txt") for p in probes]
        with ThreadPoolExecutor(max_workers=WORKERS_PER_MODEL) as ex:
            res = list(ex.map(
                lambda j: one_call(name, prov, j[0].pid, j[0].prompt, j[1]), jobs))
        fails += sum(1 for x in res if x == "fail")
        if r % 10 == 0 or r == ROUNDS:
            print(f"[{name}] round {r}/{ROUNDS} fail={fails} "
                  f"elapsed={int(time.time() - t0)}s", flush=True)
    total = len(list(root.glob("*.txt")))
    print(f"[{name}] COMPLETE files={total} fail={fails}", flush=True)
    return fails


def main() -> int:
    provs = _keys_from_env()
    if not provs:
        print("[error] 未找到 key（环境变量 ZHIPU_KEY / DS_KEY）",
              file=sys.stderr)
        return 2
    todo = [(m, p) for m, p in MODELS if p in provs]
    print(f"采集开始：{ROUNDS} 轮 × {len(MONITOR_PIDS)} 探针 × {len(todo)} 模型 "
          f"= {ROUNDS * len(MONITOR_PIDS) * len(todo)} 次调用", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
        fails = list(ex.map(lambda t: run_model(t[0], t[1], provs[t[1]]), todo))
    print(f"全部完成 总耗时{int(time.time() - t0)}s 失败{sum(fails)}次", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
