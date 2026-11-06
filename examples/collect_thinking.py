# -*- coding: utf-8 -*-
"""思考等级采集：每模型 50 轮 × 双档（off=关思考 / on=默认）× 10 探针，
on 档同时捕获思考内容（.thk 文件）。可断点续采，key 读 ZHIPU_KEY / DS_KEY 环境变量。

输出 samples/<model>/<pid>.<off|on|thk>.r<NN>.txt
用法：python -X utf8 examples/collect_thinking.py   （COLLECT_ROUNDS 可覆盖轮数）
"""
from __future__ import annotations

import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint import envkeys
from llm_fingerprint.probe_client import ProbeError, chat_completion_anthropic, safe_note
from llm_fingerprint.probes import PROBES
from ui.monitor import MONITOR_PIDS

ROUNDS = int(os.environ.get("COLLECT_ROUNDS", "50"))
TIMEOUT_S = 35
WORKERS = 3
TEMPERATURE = 0.5
SAMPLES_ROOT = pathlib.Path("samples")

# 4 模型全采：chat 的 disabled 参数虽是 no-op，但采集保证档案键结构一致
MODELS = [("glm-4.5", "zhipu"), ("glm-4.6", "zhipu"),
          ("deepseek-chat", "deepseek"), ("deepseek-reasoner", "deepseek")]


def _keys_from_env() -> dict:
    """只从环境变量取 key：ZHIPU_KEY / DS_KEY；缺哪端就跳过哪端。"""
    out: dict = {}
    zhipu = envkeys.zhipu_key()
    ds = envkeys.deepseek_key()
    if zhipu:
        out["zhipu"] = "https://open.bigmodel.cn/api/anthropic", zhipu
    if ds:
        out["deepseek"] = "https://api.deepseek.com/anthropic", ds
    return out


def one_probe(model, base, key, pid, prompt, mode, r) -> str:
    """单探针单档单轮。返回 ok/fail/skip。
    on 档总是写 thk 文件（思考为空则写空文件作标记），保证断点续采幂等。"""
    out_text = SAMPLES_ROOT / model / f"{pid}.{mode}.r{r:02d}.txt"
    out_thk = SAMPLES_ROOT / model / f"{pid}.thk.r{r:02d}.txt" if mode == "on" else None
    if out_text.exists() and out_text.stat().st_size > 0 \
            and (out_thk is None or out_thk.exists()):
        return "skip"
    thinking = {"type": "disabled"} if mode == "off" else None
    try:
        text, think = chat_completion_anthropic(
            base, key, model, prompt, temperature=TEMPERATURE,
            timeout=TIMEOUT_S, thinking=thinking, with_thinking=True)
    except ProbeError as e:
        safe_note(f"{model}/{pid}.{mode}: {e}")
        return "fail"
    if not (text and text.strip()):
        return "fail"
    (SAMPLES_ROOT / model).mkdir(parents=True, exist_ok=True)
    out_text.write_text(text, encoding="utf-8")
    if out_thk is not None:  # on 档：空思考写空标记，非空写内容
        out_thk.write_text(think or "", encoding="utf-8")
    return "ok"


def run_model(name: str, prov: tuple) -> int:
    base, key = prov
    probes = [p for p in PROBES if p.pid in MONITOR_PIDS]
    fails = 0
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        jobs = [(p.pid, p.prompt, m) for p in probes for m in ("off", "on")]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda j: one_probe(name, base, key, j[0], j[1], j[2], r), jobs))
        fails += sum(1 for x in res if x == "fail")
        if r % 10 == 0 or r == ROUNDS:
            print(f"[{name}] round {r}/{ROUNDS} fail={fails} "
                  f"elapsed={int(time.time() - t0)}s", flush=True)
    files = len(list((SAMPLES_ROOT / name).glob("*.r*.txt")))
    print(f"[{name}] COMPLETE files={files} fail={fails}", flush=True)
    return fails


def main() -> int:
    provs = _keys_from_env()
    if not provs:
        print("[error] 未找到 key", file=sys.stderr)
        return 2
    todo = [(m, p) for m, p in MODELS if p in provs]
    print(f"思考等级采集：{ROUNDS} 轮 × 10 探针 × 2 档(off/on) × {len(todo)} 模型 "
          f"= {ROUNDS * 10 * 2 * len(todo)} 次调用", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(todo)) as ex:
        fails = list(ex.map(lambda t: run_model(t[0], provs[t[1]]), todo))
    print(f"全部完成 总耗时{int(time.time() - t0)}s 失败{sum(fails)}次", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
