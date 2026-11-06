# -*- coding: utf-8 -*-
"""CLI：calibrate / classify / probe / list。"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

from llm_fingerprint.classify import Result, classify_text
from llm_fingerprint.embedding import get_embedder
from llm_fingerprint.probe_client import ProbeError, run_probes
from llm_fingerprint.probes import PROBES, identity_votes
from llm_fingerprint.profile import MIN_SAMPLES, RECOMMENDED_SAMPLES, ModelProfile, ProfileStore


MODE_SUFFIXES = {"off", "on", "thk", "lp"}  # 思考等级 + logprobs 档案键后缀


def _pid_of(filename: str, probe_ids) -> str:
    """从文件名解析档案键：<pid>.<off|on|thk>.rNN.txt → 'pid.mode'；无模式段则裸 pid。"""
    parts = os.path.basename(filename).split(".")
    if not parts or parts[0] not in probe_ids:
        return ""
    if len(parts) > 1 and parts[1] in MODE_SUFFIXES:
        return parts[0] + "." + parts[1]
    return parts[0]


def _load_samples(paths: List[str]):
    """paths 可含目录（取其中 *.txt）与文件；一条回复一个文件。
    文件名 <探针id>[.off|.on|.thk].rNN.txt 自动识别为探针条件化档案键。
    返回 (texts, pids)。"""
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.txt"), recursive=True)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[warn] 路径不存在，已跳过: {p}")
    probe_ids = {p.pid for p in PROBES}
    texts, pids = [], []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            txt = fh.read().strip()
        if not txt:
            continue
        texts.append(txt)
        pids.append(_pid_of(f, probe_ids))
    return texts, pids


def _print_result(r: Result) -> None:
    print("\n===== 判定结果 =====")
    if r.best is None:
        print("（档案库为空）")
        return
    print(f"融合模式: {r.mode}")
    if r.rejected:
        print(f"⚠ 无法确定: {r.reason}")
    else:
        print(f"✓ 最可能模型: {r.best.name}   置信度: {r.confidence:.1%}")
        if r.reason:
            print(f"  注意: {r.reason}")
    if r.runner_up is not None:
        print(f"  次可能: {r.runner_up.name} (融合分 {r.runner_up.combined:.3f}, "
              f"best {r.best.combined:.3f})")
    print("  完整排名(模型 | 风格分 | n-gram分 | 嵌入分 | 融合分):")
    for c in r.ranking[:5]:
        print(f"    {c.name:16s} | {c.style_score:.3f} | {c.gram_score:.3f} | "
              f"{c.emb_score:.3f} | {c.combined:.3f}")


def cmd_calibrate(args) -> int:
    samples, pids = _load_samples(args.samples)
    if len(samples) < MIN_SAMPLES:
        print(f"[error] 有效样本 {len(samples)} 条 < 最低 {MIN_SAMPLES} 条"
              f"（建议每模型 {RECOMMENDED_SAMPLES}+ 条真实回复）", file=sys.stderr)
        return 2
    try:
        p = ModelProfile.build(args.model, samples, embedder=get_embedder(),
                               sample_pids=pids)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    store = ProfileStore(args.store)
    path = store.save(p)
    emb_note = "含语义嵌入质心" if p.embedding_centroid else "未含嵌入（嵌入器不可用）"
    probe_note = f"，探针子档案 {len(p.probe_emb)} 个" if p.probe_emb else ""
    print(f"已生成档案: {path}  (模型 {p.name}, 样本 {p.n_samples} 条, {emb_note}{probe_note})")
    return 0


def cmd_classify(args) -> int:
    if args.text_file and args.text_file != "-":
        with open(args.text_file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
    store = ProfileStore(args.store)
    try:
        r = classify_text(text, store)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    _print_result(r)
    return 0 if not r.rejected else 1


def cmd_probe(args) -> int:
    print(f"对 {args.base_url} (model_id={args.model_id}, api={args.api}) 执行 {len(PROBES)} 个探针…")
    try:
        replies = run_probes(args.base_url, args.api_key, args.model_id,
                             temperature=args.temperature, api=args.api)
    except ProbeError as e:
        print(f"[error] 探测失败: {e}", file=sys.stderr)
        return 2
    ok = {k: v for k, v in replies.items() if v}
    if len(ok) < 3:
        print("[error] 成功探针过少，无法判定", file=sys.stderr)
        return 2
    if args.dump:
        import json
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(replies, fh, ensure_ascii=False, indent=1)
        print(f"探针回复已存: {args.dump}")
    votes = identity_votes(ok)
    if votes:
        print("自我认知信号（可被伪装，仅供参考）:", votes)
    # 探针条件化逐条判定 + 投票
    store = ProfileStore(args.store)
    try:
        from collections import Counter
        tally: Counter = Counter()
        per_probe = []
        for pid, reply in sorted(ok.items()):
            r = classify_text(reply, store, pid=pid)
            if not r.rejected and r.best is not None:
                tally[r.best.name] += 1
                per_probe.append((pid, r.best.name, f"{r.confidence:.0%}"))
        if tally:
            print("\n逐探针判定（探针 -> 模型 @置信度）:")
            for pid, name, conf in per_probe:
                print(f"  {pid:14s} -> {name} @ {conf}")
            winner, cnt = tally.most_common(1)[0]
            print(f"\n投票结果: {dict(tally)}  →  综合判定: {winner}（{cnt}/{sum(tally.values())} 票）")
        merged = "\n\n".join(f"{k}: {v}" for k, v in sorted(ok.items()))
        r = classify_text(merged, store)
        _print_result(r)
        return 0
    except ValueError as e:
        print(f"[info] {e}（仅输出自我认知信号）")
        return 3


def cmd_list(args) -> int:
    store = ProfileStore(args.store)
    names = store.names()
    if not names:
        print(f"档案库为空: {args.store}（先运行 calibrate）")
        return 1
    for n in names:
        p = store.load(n)
        print(f"  {p.name:20s} 样本数 {p.n_samples}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="llm-fingerprint",
                                 description="大模型回复指纹识别（统计风格，闭集判定）")
    ap.add_argument("--store", default="profiles", help="档案目录 (默认 ./profiles)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="用某模型的真实回复样本生成档案")
    c.add_argument("--model", required=True)
    c.add_argument("--samples", nargs="+", required=True,
                   help="样本目录或 .txt 文件（一文件一条回复）")
    c.set_defaults(fn=cmd_calibrate)

    c = sub.add_parser("classify", help="判定一段已有回复来自哪个模型")
    c.add_argument("--text-file", help="回复文本文件（- 表示 stdin）")
    c.set_defaults(fn=cmd_classify)

    c = sub.add_parser("probe", help="主动探测：向端点发探针并判定")
    c.add_argument("--base-url", required=True, help="如 https://api.xxx.com/v1")
    c.add_argument("--api-key", required=True)
    c.add_argument("--model-id", required=True)
    c.add_argument("--api", choices=["openai", "anthropic"], default="openai",
                   help="端点协议类型（智谱 /api/anthropic 用 anthropic）")
    c.add_argument("--temperature", type=float, default=0.5)
    c.add_argument("--dump", help="保存全部探针回复到 JSON（可再用于 calibrate）")
    c.set_defaults(fn=cmd_probe)

    c = sub.add_parser("list", help="列出已校准档案")
    c.set_defaults(fn=cmd_list)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
