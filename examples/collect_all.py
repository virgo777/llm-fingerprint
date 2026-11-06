# -*- coding: utf-8 -*-
"""并发采集多模型探针语料 + 校准 + 混淆矩阵评测。"""
import concurrent.futures as cf
import json
import os
import pathlib
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_fingerprint.probe_client import ProbeError, chat_completion_anthropic
from llm_fingerprint.probes import PROBES, identity_votes

ZHIPU_BASE = "https://open.bigmodel.cn/api/anthropic"
DS_BASE = "https://api.deepseek.com/anthropic"
HOST_WORKERS = 3          # 每个目标并发线程数
RUNS = [("t03", 0.3), ("t07", 0.7)]


def target_list():
    zhipu_key = os.environ.get("ZHIPU_KEY", "")
    ds_key = os.environ.get("DS_KEY", "")
    targets = [
        ("glm-4.5", ZHIPU_BASE, "anthropic", "glm-4.5", zhipu_key, 2048),
        ("deepseek-chat", DS_BASE, "anthropic", "deepseek-chat", ds_key, 2048),
        ("deepseek-reasoner", DS_BASE, "anthropic", "deepseek-reasoner", ds_key, 3000),
    ]
    return [t for t in targets if t[4]]  # 无 key 的跳过


def run_battery(name, base, api, model_id, key, max_tokens):
    root = pathlib.Path("samples") / name
    root.mkdir(parents=True, exist_ok=True)
    def one(item):
        run, temp, probe = item
        try:
            txt = chat_completion_anthropic(base, key, model_id, probe.prompt,
                                            temperature=temp, max_tokens=max_tokens)
            return run, probe.pid, txt
        except ProbeError as e:
            print(f"  [warn] {name}/{probe.pid}.{run}: {str(e)[:100]}")
            return run, probe.pid, ""
    items = [(run, temp, p) for run, temp in RUNS for p in PROBES]
    replies = {}
    with cf.ThreadPoolExecutor(max_workers=HOST_WORKERS) as ex:
        for run, pid, txt in ex.map(one, items):
            if txt.strip():
                (root / f"{pid}.{run}.txt").write_text(txt, encoding="utf-8")
                replies[f"{pid}.{run}"] = txt
    ok = len(replies)
    total = len(items)
    print(f"[{name}] 成功 {ok}/{total}，身份信号: {identity_votes(replies)}")
    return ok


def main():
    targets = target_list()
    if not targets:
        print("[error] 未提供任何 key（需要 ZHIPU_KEY / DS_KEY 环境变量）", file=sys.stderr)
        return 2
    with cf.ThreadPoolExecutor(max_workers=len(targets)) as ex:
        oks = list(ex.map(lambda t: run_battery(*t), targets))
    if all(o >= 3 for o in oks):
        print("全部目标采集完成")
        return 0
    print("[warn] 部分目标样本过少")
    return 1


if __name__ == "__main__":
    sys.exit(main())
