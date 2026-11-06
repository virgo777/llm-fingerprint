# -*- coding: utf-8 -*-
"""监测模块：向端点发送 10 个文风探针，与档案库比对，输出每个模型的文本相似度。"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_fingerprint.classify import classify_text
from llm_fingerprint.embedding import get_embedder
from llm_fingerprint.features import extract_features, extract_lp_features
from llm_fingerprint.probe_client import (ProbeError, chat_completion,
                                          chat_completion_anthropic)
from llm_fingerprint.profile import ProfileStore
from llm_fingerprint.probes import PROBES, identity_votes


def _cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * big.get(k, 0.0) for k, v in small.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0

# 10 个文风敏感探针（剔除逐字复述/数数等无文风差异的机械探针）
MONITOR_PIDS = ["id-zh", "fmt-list", "fmt-table", "fmt-poem", "fmt-joke",
                "math-step", "explain", "refuse", "roleplay", "longform"]
MIN_OK_REPLIES = 3
LOW_SIM_NOTE = 0.75  # 顶部平均相似度低于此值 → 提示可能不在档案库中
PROBE_TIMEOUT_S = 17  # 单探针超时（用户要求 17 秒）


def _profiles_dir() -> Path:
    """档案目录：exe 旁的 profiles/ 优先（可外部更新），否则用打包内置副本。"""
    import sys
    if getattr(sys, "frozen", False):
        ext = Path(sys.executable).parent / "profiles"
        if ext.is_dir() and any(ext.glob("*.json")):
            return ext
        return Path(getattr(sys, "_MEIPASS", ".")) / "profiles"
    return Path(__file__).resolve().parent.parent / "profiles"


def rank_models(samples, profiles, embedder) -> dict:
    """纯比对逻辑：文本样本走相似度+投票；.lp 概率指纹样本走 z 距离投票。
    samples: [(pid, text), ...] —— pid 以 .lp 结尾时 text 是 JSON 特征向量。"""
    import json as _json
    votes: Counter = Counter()
    per_probe_acc: dict = {}  # {pid: {model: [sim,...]}} 聚合多轮
    per_probe_n: dict = {}
    lp_acc: dict = {p.name: [] for p in profiles}  # lp 得分按模型聚合
    for pid, text in sorted(samples):
        if pid.endswith(".lp"):
            # logprobs 概率指纹：z 距离（改不了文风就改不了概率分布）
            try:
                lpf = _json.loads(text)
            except ValueError:
                continue
            acc = per_probe_acc.setdefault(pid, {})
            best_name, best_score = None, -1.0
            for p in profiles:
                stats = (p.probe_lp or {}).get(pid)
                if not stats:
                    continue
                zs = []
                for k, ms in stats.items():
                    if k in lpf:
                        z = abs((float(lpf[k]) - ms[0]) / ms[1])
                        zs.append(min(z, 10.0) ** 2)  # 限幅：长度类特征不该一票否决
                score = 1.0 / (1.0 + (sum(zs) / len(zs) if zs else 99.0))
                acc.setdefault(p.name, []).append(score)
                lp_acc[p.name].append(score)
                if score > best_score:
                    best_name, best_score = p.name, score
            if best_name is not None:
                votes[best_name] += 1
            per_probe_n[pid] = per_probe_n.get(pid, 0) + 1
            continue
        r = classify_text(text, profiles_store_wrap(profiles), pid=pid)
        if not r.rejected and r.best is not None:
            votes[r.best.name] += 1
        _, grams = extract_features(text)
        acc = per_probe_acc.setdefault(pid, {})
        if embedder is not None:
            try:
                vec = embedder.embed(text)
            except Exception:
                vec = None
            if vec is not None:
                for p in profiles:
                    ref = (p.probe_emb or {}).get(pid)
                    if ref:
                        sim = sum(float(x) * y for x, y in zip(vec, ref))
                        acc.setdefault(p.name, []).append(max(sim, 0.0))
        else:
            # 零依赖路线：探针条件化三元组余弦
            for p in profiles:
                refg = (p.probe_gram or {}).get(pid)
                if refg:
                    sim = max(_cos(grams, refg), 0.0)
                    acc.setdefault(p.name, []).append(sim)
        per_probe_n[pid] = per_probe_n.get(pid, 0) + 1
    # 探针明细：多轮取均值
    per_probe = []
    for pid in sorted(per_probe_acc):
        acc = per_probe_acc[pid]
        per_probe.append({"pid": pid,
                          "n": per_probe_n[pid],
                          "sims": {name: round(sum(v) / len(v) * 100, 1)
                                   for name, v in acc.items() if v}})
    sim_sum = {p.name: [] for p in profiles}
    for entry in per_probe:
        for name, pct in entry["sims"].items():
            if name in sim_sum:
                sim_sum[name].append(pct / 100.0)
    models = []
    for p in profiles:
        vals = sim_sum[p.name]
        lp_vals = lp_acc[p.name]
        models.append({
            "name": p.name,
            "similarity": round(sum(vals) / len(vals) * 100, 1) if vals else None,
            "n_probes": len(vals),
            "lp_similarity": round(sum(lp_vals) / len(lp_vals) * 100, 1)
            if lp_vals else None,
            "votes": votes.get(p.name, 0),
        })
    models.sort(key=lambda m: -(m["similarity"] if m["similarity"] is not None else -1))
    return {"models": models, "votes": dict(votes), "per_probe": per_probe}


class _StoreShim:
    """把已加载的档案列表包装成 classify_text 需要的 store 接口。"""

    def __init__(self, profiles):
        self._profiles = profiles

    def load_all(self):
        return self._profiles


def profiles_store_wrap(profiles):
    return _StoreShim(profiles)


VENDOR_NAMES = {"glm": "智谱GLM", "deepseek": "DeepSeek"}
SIBLING_MARGIN = 2.5  # 同厂前两名票差低于此倍数 → 判定降级为厂商级


def _vendor(model_name: str) -> str:
    return model_name.split("-")[0]


def verdict_scope(models: list, votes: dict) -> str:
    """同厂相邻版本（如 glm-4.5/4.6）文风信噪比≈0（实测 0.03），
    票差不足 SIBLING_MARGIN 倍时判定只到厂商级。"""
    if len(models) < 2:
        return "model"
    top1, top2 = models[0], models[1]
    if _vendor(top1["name"]) != _vendor(top2["name"]):
        return "model"
    if top1["votes"] >= SIBLING_MARGIN * max(top2["votes"], 1):
        return "model"
    return "vendor"


def _call_probe(base: str, key: str, api: str, model_id: str,
                prompt: str, thinking: dict | None, system: str | None = None):
    """统一探针调用，返回 (正文, 思考内容, lp特征或None)。
    anthropic：思考档可控；openai：附带 top-20 logprobs 概率指纹
    （端点不支持 logprobs 时自动降级为纯文本，不报错）。"""
    if api == "anthropic":
        text, think = chat_completion_anthropic(
            base, key, model_id, prompt, temperature=0.5, max_tokens=2048,
            timeout=PROBE_TIMEOUT_S, thinking=thinking, with_thinking=True,
            system=system)
        return text, think, None
    try:
        text, lp = chat_completion(base, key, model_id, prompt, temperature=0.5,
                                   max_tokens=1024, timeout=PROBE_TIMEOUT_S,
                                   system=system, logprobs=True, top_logprobs=20,
                                   with_logprobs=True)
    except ProbeError as e:
        if "logprobs" in str(e).lower():  # 部分中转站不支持 logprobs → 降级
            text, lp = chat_completion(base, key, model_id, prompt,
                                       temperature=0.5, max_tokens=1024,
                                       timeout=PROBE_TIMEOUT_S, system=system), []
        else:
            raise
    feats = None
    if lp:
        try:
            feats = extract_lp_features(lp)
        except Exception:
            feats = None
    return text, "", feats


def _monitor_tasks(api: str) -> list:
    """检测任务清单：anthropic 每探针双档（off/on），openai 单档（on 键）。"""
    tasks = []
    for p in PROBES:
        if p.pid in MONITOR_PIDS:
            if api == "anthropic":
                tasks.append((f"{p.pid}.off", p.prompt, {"type": "disabled"}))
            tasks.append((f"{p.pid}.on", p.prompt, None))
    return tasks


def run_monitor(base_url: str, api_key: str, api: str, model_id: str,
                store_dir: str | None = None, on_stage=None,
                rounds: int = 50) -> dict:
    """发送 10 探针 × 思考档位 × rounds 轮 → 比对档案库 → 相似度排名。
    on_stage(stage, done, total)：阶段回调；"probes" 的 total = 每轮任务数×rounds，
    另有 "probe:<档案键>:<ok|fail>" 流式事件。on 档的思考内容另存为 .thk 键样本。"""

    def _stage(stage, done=0, total=0):
        if on_stage is not None:
            try:
                on_stage(stage, done, total)
            except Exception:
                pass

    if not model_id or not model_id.strip():
        return {"ok": False, "error": "开始检测需要模型名（请填写中转站的模型名称）"}
    store_dir = Path(store_dir) if store_dir else _profiles_dir()
    if not store_dir.is_dir() or not list(store_dir.glob("*.json")):
        return {"ok": False, "error": f"档案库为空: {store_dir}（先运行 calibrate）"}
    store = ProfileStore(str(store_dir))
    profiles = store.load_all()
    tasks = _monitor_tasks(api)
    total = len(tasks) * rounds
    counter = {"done": 0}
    samples: list = []  # [(档案键, text), ...] 档案键如 "fmt-list.off" / "fmt-list.thk"
    base, key = base_url.strip(), api_key.strip()
    model = model_id.strip()

    def _do(task):
        pid_key, prompt, think_param = task
        try:
            text, think, lp_feats = _call_probe(base, key, api, model,
                                                prompt, think_param)
        except Exception:
            text, think, lp_feats = "", "", None
        actual = pid_key
        # 请求 off 档却返回思考 → 端点忽略思考参数，按 on 档归档（本身也是信号）
        if pid_key.endswith(".off") and think and think.strip():
            actual = pid_key[:-4] + ".on"
        return actual, text, think, lp_feats

    import concurrent.futures as cf
    import json as _json
    for r in range(1, rounds + 1):
        try:
            with cf.ThreadPoolExecutor(max_workers=3) as ex:
                for pid_key, text, think, lp_feats in ex.map(_do, tasks):
                    counter["done"] += 1
                    okflag = "ok" if text and text.strip() else "fail"
                    _stage("probes", counter["done"], total)
                    _stage(f"probe:{pid_key}:{okflag}", counter["done"], total)
                    if text and text.strip():
                        samples.append((pid_key, text))
                        if pid_key.endswith(".on") and think and think.strip():
                            samples.append((pid_key[:-3] + ".thk", think))
                        if lp_feats is not None:  # openai 协议概率指纹
                            base_key = pid_key[:-3] if pid_key.endswith(".on") else pid_key
                            samples.append((base_key + ".lp", _json.dumps(lp_feats)))
        except Exception as e:
            return {"ok": False, "error": f"第 {r}/{rounds} 轮探针发送失败："
                                         f"{type(e).__name__}: {e}"}
    if len(samples) < MIN_OK_REPLIES:
        return {"ok": False, "error": f"有效回复过少（{len(samples)}/{total}），无法比对"}
    _stage("compare", total, total)  # 显示层：done 不超过调用数（样本数含思考副产物）
    ranked = rank_models(samples, profiles, get_embedder())
    sim_winner = ranked["models"][0] if ranked["models"] else None
    # 判定以投票为准；无票时退回相似度冠军
    vote_winner = max(ranked["votes"], key=ranked["votes"].get) if ranked["votes"] else None
    scope = verdict_scope(ranked["models"], ranked["votes"])
    if scope == "vendor" and vote_winner:
        verdict = f"{VENDOR_NAMES.get(_vendor(vote_winner), _vendor(vote_winner))} 家族"
    else:
        verdict = vote_winner or (sim_winner["name"] if sim_winner else None)
    agree = (vote_winner is None) or (sim_winner is not None
                                      and vote_winner == sim_winner["name"])
    low = sim_winner is not None and sim_winner["similarity"] is not None \
        and sim_winner["similarity"] < LOW_SIM_NOTE * 100
    return {
        "ok": True, "model_id": model_id.strip(), "rounds": rounds,
        "n_replies": len(samples), "n_sent": total,
        "models": ranked["models"], "votes": ranked["votes"],
        "per_probe": ranked["per_probe"],
        "identity": identity_votes({pid: t for pid, t in samples}),
        "similarity_winner": sim_winner["name"] if sim_winner else None,
        "vote_winner": vote_winner, "verdict": verdict,
        "verdict_scope": scope, "agree": agree, "low_similarity": low,
    }
