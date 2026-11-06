# -*- coding: utf-8 -*-
"""ui/server 两级探测 + monitor 的单元测试（不发真实网络请求）。"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import urllib.error

import ui.server as srv
import ui.monitor as mon
from llm_fingerprint.profile import ModelProfile


def _payload(**kw):
    p = {"base_url": "https://api.example.com", "api_key": "sk-test", "api": "openai"}
    p.update(kw)
    return p


# ---------- 输入校验 ----------

def test_validate_ok_without_model():
    assert srv.validate_payload(_payload()) is None


@pytest.mark.parametrize("bad,expect_kw", [
    ({"base_url": ""}, "base_url"),
    ({"base_url": "ftp://x"}, "http"),
    ({"api_key": "  "}, "api_key"),
    ({"api": "grpc"}, "协议"),
])
def test_validate_rejects(bad, expect_kw):
    assert expect_kw in srv.validate_payload(_payload(**bad))


# ---------- 两级探测 ----------

def test_stage1_models_ok(monkeypatch):
    monkeypatch.setattr(srv, "_http", lambda *a, **k: (
        200, '{"data":[{"id":"m2"},{"id":"m1"},{"id":"m1"}]}'))
    d = srv.run_test(_payload())
    assert d["ok"] and d["mode"] == "models" and d["n_models"] == 2
    assert d["models"] == ["m1", "m2"] and "latency_ms" in d


def test_stage2_auth_ok(monkeypatch):
    def fake_http(url, key, method="GET", body=None):
        if method == "GET":
            return 404, "not found"
        return 400, '{"error": "body.model: Field required"}'
    monkeypatch.setattr(srv, "_http", fake_http)
    d = srv.run_test(_payload(api="anthropic"))
    assert d["ok"] and d["mode"] == "auth" and "鉴权通过" in d["note"]


def test_stage2_bad_key(monkeypatch):
    monkeypatch.setattr(srv, "_http", lambda *a, **k: (401, '{"error":"令牌已过期"}'))
    d = srv.run_test(_payload())
    assert not d["ok"] and "密钥无效" in d["error"]


def test_network_dead(monkeypatch):
    def boom(*a, **k):
        raise urllib.error.URLError("timed out")
    monkeypatch.setattr(srv, "_http", boom)
    d = srv.run_test(_payload())
    assert not d["ok"] and "无法连通" in d["error"]


def test_stage2_unexpected_400(monkeypatch):
    monkeypatch.setattr(srv, "_http", lambda *a, **k: (400, '{"error":"payload too large"}'))
    d = srv.run_test(_payload())
    assert not d["ok"] and "400" in d["error"]


# ---------- monitor ----------

def test_monitor_requires_model(tmp_path):
    d = mon.run_monitor("https://x", "k", "openai", "", store_dir=str(tmp_path))
    assert not d["ok"] and "模型名" in d["error"]


def test_monitor_empty_store(tmp_path):
    d = mon.run_monitor("https://x", "k", "openai", "m1", store_dir=str(tmp_path))
    assert not d["ok"] and "档案库为空" in d["error"]


def test_rank_models_votes_without_embedder(tmp_path):
    a = ["首先，我们需要明确概念。其次，注意细节。总之，要系统推进。"] * 4
    b = ["**指南**\n\n1. 要点一 ✅\n2. 要点二 ✅\n\n希望对你有帮助~"] * 4
    store = tmp_path / "p"
    store.mkdir()
    from llm_fingerprint.profile import ProfileStore
    ps = ProfileStore(str(store))
    ps.save(ModelProfile.build("styleA", a))
    ps.save(ModelProfile.build("styleB", b))
    profiles = ps.load_all()
    # 同一探针两轮回复（多轮采样形态）
    samples = [("fmt-list", a[0]), ("fmt-list", a[1]),
               ("fmt-joke", b[0]), ("fmt-joke", b[1])]
    ranked = mon.rank_models(samples, profiles, embedder=None)
    names = [m["name"] for m in ranked["models"]]
    assert set(names) == {"styleA", "styleB"}
    assert all(m["similarity"] is None for m in ranked["models"])  # 无嵌入器 → 跳过相似度
    assert sum(ranked["votes"].values()) == 4  # 每条回复各投一票
    # 探针明细聚合：同探针两轮 → n=2
    pn = {e["pid"]: e["n"] for e in ranked["per_probe"]}
    assert pn == {"fmt-list": 2, "fmt-joke": 2}
