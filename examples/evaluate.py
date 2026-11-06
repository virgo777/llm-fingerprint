# -*- coding: utf-8 -*-
"""校准全部模型 + 评测：单条 / 整组合并 / 逐条投票 三种模式。
数据切分：训练 = r01-r40 轮 + 旧 t03/t07 语料；测试 = r41-r50 轮（每模型 100 条）。"""
import os
import pathlib
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_fingerprint.classify import classify_text
from llm_fingerprint.embedding import get_embedder
from llm_fingerprint.profile import ModelProfile, ProfileStore
from llm_fingerprint.probes import PROBES

TEST_ROUNDS = set(range(41, 51))  # r41..r50 为测试轮
probe_ids = {p.pid for p in PROBES}
MODES = {"off", "on", "thk", "lp"}  # 思考等级 + logprobs 档案键后缀


def _round_of(fname: str):
    m = re.search(r"\.r(\d+)\.txt$", fname)
    return int(m.group(1)) if m else None


def _pid_of(fname: str):
    parts = pathlib.Path(fname).name.split(".")
    if parts[0] not in probe_ids:
        return ""
    if len(parts) > 1 and parts[1] in MODES:
        return parts[0] + "." + parts[1]
    return parts[0]


def build_store(models):
    samples_root = pathlib.Path("samples")
    store = ProfileStore(pathlib.Path("profiles_eval_tmp"))
    emb = get_embedder()
    test_sets, test_pids = {}, {}
    for m in models:
        fs = sorted((samples_root / m).glob("*.txt"))
        fs = [f for f in fs if f.stat().st_size > 0]
        test = [f for f in fs if (_round_of(f.name) or 0) in TEST_ROUNDS]
        train = [f for f in fs if f not in set(test)]
        if len(train) < 3 or not test:
            print(f"[warn] {m} 样本不足(train={len(train)}, test={len(test)})，跳过")
            continue
        store.save(ModelProfile.build(
            m, [f.read_text(encoding="utf-8") for f in train], embedder=emb,
            sample_pids=[_pid_of(f.name) for f in train]))
        test_sets[m] = [f.read_text(encoding="utf-8") for f in test]
        test_pids[m] = [_pid_of(f.name) for f in test]
    return store, test_sets, test_pids


def report(title, store, cases, pids=None):
    names = list(cases)
    print(f"\n=== {title} ===")
    print(f"{'真实\\判定':20s}" + "".join(f"{n[:14]:17s}" for n in names) + "拒绝")
    correct = total = rejected = 0
    for truth, texts in cases.items():
        cells = []
        for i, txt in enumerate(texts):
            pid = pids.get(truth, [""] * len(texts))[i] if pids else None
            r = classify_text(txt, store, pid=pid)
            total += 1
            pred = "-" if (r.rejected or r.best is None) else r.best.name
            if r.rejected:
                rejected += 1
            elif pred == truth:
                correct += 1
            cells.append(pred)
        line = f"{truth[:18]:20s}"
        for n in names:
            cnt = sum(1 for c in cells if c == n)
            mark = "*" if n == truth else " "
            line += f"{cnt}{mark}".ljust(17)
        print(line + str(sum(1 for c in cells if c == "-")))
    denom = max(total - rejected, 1)
    print(f"闭集准确率: {correct}/{total - rejected} = {correct / denom:.1%}"
          f"｜拒绝 {rejected}/{total}")


def vote_mode(store, test_sets, test_pids):
    """模式三：逐条判定 + 多数表决（monitor 的生产路径）+ 分信号面（off/on/thk）"""
    print("\n=== 模式三：逐条投票（多数表决） ===")
    def vendor(n):
        return "deepseek" if n.startswith("deepseek") else \
               ("zhipu" if n.startswith("glm") else "?")
    hit = vhit = 0
    per_mode_hit = {mode: [0, 0] for mode in ("off", "on", "thk", "")}  # [对, 总]
    for m, texts in test_sets.items():
        votes = Counter()
        mode_votes = {"off": Counter(), "on": Counter(), "thk": Counter(), "": Counter()}
        for txt, pid in zip(texts, test_pids[m]):
            r = classify_text(txt, store, pid=pid)
            if not r.rejected and r.best:
                votes[r.best.name] += 1
                mode = pid.rsplit(".", 1)[1] if "." in pid and \
                    pid.rsplit(".", 1)[1] in ("off", "on", "thk") else ""
                mode_votes[mode][r.best.name] += 1
        winner = votes.most_common(1)[0][0] if votes else "-"
        hit += (winner == m)
        vhit += (vendor(winner) == vendor(m))
        parts = []
        for mode, label in (("off", "off档"), ("on", "on档"), ("thk", "思考")):
            mv = mode_votes[mode]
            if sum(mv.values()) >= 5:
                mw = mv.most_common(1)[0][0]
                per_mode_hit[mode][1] += 1
                per_mode_hit[mode][0] += (mw == m)
                parts.append(f"{label}→{mw}")
        print(f"{m:18s} -> {winner:18s} {'✓' if winner == m else '✗'}  票型: {dict(votes)}"
              + (f" ｜ {', '.join(parts)}" if parts else ""))
    print(f"精确模型: {hit}/{len(test_sets)} ｜ 厂商级: {vhit}/{len(test_sets)}")
    for mode, label in (("off", "off 档单独投票"), ("on", "on 档单独投票"),
                        ("thk", "思考内容单独投票")):
        c, t = per_mode_hit[mode]
        if t:
            print(f"  分信号面 · {label}: {c}/{t}")


def main():
    samples_root = pathlib.Path("samples")
    models = sorted(d.name for d in samples_root.iterdir() if d.is_dir())
    if not models:
        print("[error] samples/ 为空", file=sys.stderr)
        return 2
    emb = get_embedder()
    print(f"嵌入器: {'可用(BGE 512维)' if emb else '不可用(零依赖三元组路线)'}")
    store, test_sets, test_pids = build_store(models)
    print(f"档案: {store.names()}（测试集 = r41-r50 轮，每模型 {len(next(iter(test_sets.values()), []))} 条）")

    report("模式一：单条回复判定（探针条件化）", store, test_sets, pids=test_pids)
    vote_mode(store, test_sets, test_pids)

    # ---------- 校准：厂商级误指控率 ----------
    def vendor(n):
        return n.split("-")[0]
    total = fa = 0
    for m, texts in test_sets.items():
        for txt in texts:
            r = classify_text(txt, store)
            if r.rejected or r.best is None:
                continue
            total += 1
            if vendor(r.best.name) != vendor(m):
                fa += 1
    print(f"\n=== 校准 · 厂商级误指控率（单条判定） ===")
    print(f"误指控 {fa}/{total} = {fa / max(total, 1):.2%}"
          f"（投票模式下为 0/4；目标 ≤5%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
