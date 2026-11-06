# -*- coding: utf-8 -*-
"""run_monitor 多轮双档模式的离线测试（mock _call_probe，无网络）。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ui.monitor as mon
from llm_fingerprint.profile import ModelProfile, ProfileStore


def _profiles(tmp_path):
    a = ["首先，明确概念。其次，注意细节。总之，系统推进。"] * 4
    b = ["**指南**\n\n1. 要点 ✅\n\n希望有帮助~"] * 4
    ps = ProfileStore(str(tmp_path))
    ps.save(ModelProfile.build("styleA", a))
    ps.save(ModelProfile.build("styleB", b))
    return ps.load_all()


def test_run_monitor_dual_mode(tmp_path, monkeypatch):
    _profiles(tmp_path)
    calls = {"n": 0}
    events = []

    def fake_call(base, key, api, model_id, prompt, thinking):
        calls["n"] += 1
        if thinking is not None:      # off 档任务
            return "首先，明确概念。其次，注意细节。总之，系统推进。", "", None
        # on 档任务：正文 + 思考内容
        return "**指南**\n\n1. 要点 ✅\n\n希望有帮助~", "思考推理" * 30, None

    monkeypatch.setattr(mon, "_call_probe", fake_call)
    monkeypatch.setattr(mon, "_profiles_dir", lambda: tmp_path)

    def on_stage(stage, done, total):
        events.append((stage, done, total))

    d = mon.run_monitor("https://x", "k", "anthropic", "m1",
                        on_stage=on_stage, rounds=3)
    assert d["ok"], d.get("error")
    # anthropic：每轮 10 探针 × 2 档 = 20 次调用 × 3 轮
    assert calls["n"] == 60
    assert d["n_sent"] == 60
    # 样本 = 20 正文/轮 + 10 思考/轮 = 30/轮
    assert d["n_replies"] == 90
    assert d["rounds"] == 3
    # 思考样本归档为 .thk 键
    pids = {e["pid"] for e in d["per_probe"]}
    assert any(p.endswith(".thk") for p in pids)
    assert any(p.endswith(".off") for p in pids)
    assert any(p.endswith(".on") for p in pids)
    probe_ev = [e for e in events if e[0] == "probes"]
    assert probe_ev[-1][1] == 60 and probe_ev[-1][2] == 60
    assert any(e[0] == "compare" for e in events)
    # 本测试的档案未带探针键 → thk 样本被开集拒绝不投票；off/on 各 30 票
    assert sum(d["votes"].values()) == 60


def test_run_monitor_off_ignored_reclassified(tmp_path, monkeypatch):
    """请求 off 却返回思考 → 样本按 on 归档。"""
    _profiles(tmp_path)

    def fake_call(base, key, api, model_id, prompt, thinking):
        if thinking is not None:
            return "首先，明确概念。其次，注意细节。总之，系统推进。", "端点忽略参数返回的思考", None
        return "首先，明确概念。其次，注意细节。总之，系统推进。", "", None

    monkeypatch.setattr(mon, "_call_probe", fake_call)
    monkeypatch.setattr(mon, "_profiles_dir", lambda: tmp_path)
    d = mon.run_monitor("https://x", "k", "anthropic", "m1", rounds=1)
    assert d["ok"]
    pids = {e["pid"] for e in d["per_probe"]}
    assert not any(p.endswith(".off") for p in pids)   # off 全被改判为 on
    assert any(p.endswith(".on") for p in pids)


def test_run_monitor_zero_rounds_graceful(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("不应发任何调用")
    monkeypatch.setattr(mon, "_call_probe", boom)
    monkeypatch.setattr(mon, "_profiles_dir", lambda: tmp_path)
    ps = ProfileStore(str(tmp_path))
    ps.save(ModelProfile.build("m", ["样本一", "样本二", "样本三"]))
    d = mon.run_monitor("https://x", "k", "anthropic", "m", rounds=0)
    assert d["ok"] is False and "有效回复过少" in d["error"]


def test_verdict_scope_sibling_downgrade():
    """同厂相邻版本票差不足 2.5 倍 → 厂商级；跨厂或票差悬殊 → 模型级。"""
    m = lambda name, votes: {"name": name, "votes": votes, "similarity": 0.9}
    assert mon.verdict_scope([m("glm-4.6", 60), m("glm-4.5", 28)], {}) == "vendor"
    assert mon.verdict_scope([m("glm-4.6", 59), m("glm-4.5", 31)], {}) == "vendor"
    assert mon.verdict_scope([m("deepseek-chat", 95), m("deepseek-reasoner", 4)], {}) == "model"
    assert mon.verdict_scope([m("glm-4.6", 60), m("deepseek-chat", 28)], {}) == "model"
    assert mon.verdict_scope([m("glm-4.6", 75), m("glm-4.5", 28)], {}) == "model"
    assert mon.verdict_scope([m("glm-4.6", 60)], {}) == "model"
