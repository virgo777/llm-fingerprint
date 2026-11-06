# -*- coding: utf-8 -*-
"""llm_fingerprint 测试：特征提取 / 档案 / 分类闭环 / 开集拒绝"""
import json
import os
import tempfile

import pytest

from llm_fingerprint.features import extract_features, NUMERIC_FEATURES
from llm_fingerprint.profile import ModelProfile, ProfileStore
from llm_fingerprint.classify import classify_text

# ---------- 合成语料：三种"伪模型"，风格刻意区分 ----------

STYLE_A_TPL = (
    "首先，{topic}的核心在于基础概念的把握。其次，我们需要关注实践层面的细节。"
    "此外，还有一些常见误区需要注意。总之，{topic}是一个循序渐进的过程。"
)
STYLE_B_TPL = "**{topic}指南**\n\n1. 基础概念很重要 📌\n2. 动手实践是关键 ✅\n3. 避开常见坑 ⚠️\n\n希望对你有帮助~"
STYLE_C_TPL = (
    "关于{topic}之问题，兹从如下数端予以说明：其一，概念层面（定义、范畴与边界）；"
    "其二，实践层面——操作规范、注意事项及常见谬误；其三，总结与展望。"
    "综上所述，{topic}需系统化推进。"
)

TOPICS = ["机器学习", "中医养生", "门店运营", "健康管理", "数字人", "脉诊", "体成分", "知识库", "预约系统", "支付", "舌诊", "视觉识别"]


def make_samples(tpl, n):
    return [tpl.format(topic=TOPICS[i % len(TOPICS)]) for i in range(n)]


@pytest.fixture()
def corpora():
    return {
        "modelA": make_samples(STYLE_A_TPL, 12),
        "modelB": make_samples(STYLE_B_TPL, 12),
        "modelC": make_samples(STYLE_C_TPL, 12),
    }


# ---------- 特征提取 ----------

def test_features_deterministic_and_complete():
    t = STYLE_A_TPL.format(topic="测试")
    f1, g1 = extract_features(t)
    f2, g2 = extract_features(t)
    assert f1 == f2, "特征提取必须确定性"
    assert set(f1.keys()) == set(NUMERIC_FEATURES), "数值特征缺项"
    assert len(g1) > 0, "应产出字符 n-gram"


def test_features_reject_empty():
    with pytest.raises(ValueError):
        extract_features("   ")


def test_features_capture_style_difference():
    fa, _ = extract_features(STYLE_A_TPL.format(topic="X"))
    fb, _ = extract_features(STYLE_B_TPL.format(topic="X"))
    assert abs(fa["md_bold_count"] - fb["md_bold_count"]) >= 1
    assert fa["emoji_count"] < fb["emoji_count"]
    assert fb["list_number_count"] > fa["list_number_count"]


# ---------- 档案 ----------

def test_profile_build_and_store_roundtrip(corpora, tmp_path):
    p = ModelProfile.build("modelA", corpora["modelA"])
    assert p.n_samples == 12
    store = ProfileStore(tmp_path)
    store.save(p)
    loaded = store.load("modelA")
    assert loaded.n_samples == 12
    assert set(loaded.feature_mean) == set(p.feature_mean)
    assert loaded.trigram_centroid.keys() == p.trigram_centroid.keys()


def test_profile_reject_too_few_samples():
    with pytest.raises(ValueError):
        ModelProfile.build("bad", ["只有一条样本"])


def test_profile_with_embedding_roundtrip(corpora, tmp_path):
    """带嵌入器的档案必须完整 JSON 往返（回归：曾因变量遮蔽导致 feature_mean 变 ndarray）。"""
    from llm_fingerprint.embedding import get_embedder
    emb = get_embedder()
    if emb is None:
        pytest.skip("嵌入器不可用（缺 onnxruntime/tokenizers）")
    p = ModelProfile.build("modelA", corpora["modelA"], embedder=emb)
    assert isinstance(p.feature_mean, dict), "feature_mean 必须仍是 dict"
    assert p.embedding_centroid is not None and len(p.embedding_centroid) == emb.DIM
    store = ProfileStore(tmp_path)
    store.save(p)  # 内部调用 to_json，ndarray 会在此抛 TypeError
    loaded = store.load("modelA")
    assert loaded.embedding_centroid is not None
    assert len(loaded.embedding_centroid) == emb.DIM


# ---------- 分类闭环（留出法）----------

def test_classify_closed_set(corpora, tmp_path):
    store = ProfileStore(tmp_path)
    held_out = {}
    for name, samples in corpora.items():
        store.save(ModelProfile.build(name, samples[:9]))
        held_out[name] = samples[9:]
    for name, samples in held_out.items():
        for s in samples:
            result = classify_text(s, store)
            assert result.best is not None, f"{name} 样本应能分类"
            assert result.best.name == name, f"误判: {name} -> {result.best.name}"
            assert result.confidence > 0.5


def test_classify_open_set_rejection(corpora, tmp_path):
    """未校准的风格应给出低置信/unknown 判定"""
    store = ProfileStore(tmp_path)
    for name, samples in corpora.items():
        store.save(ModelProfile.build(name, samples[:9]))
    alien = (
        "yo this is totally different english text with no chinese at all "
        "and completely alien style patterns everywhere random words here"
    )
    result = classify_text(alien, store)
    assert result.confidence < 0.6, "异质文本不应给出高置信"


def test_classify_empty_store(tmp_path):
    store = ProfileStore(tmp_path)
    with pytest.raises(ValueError):
        classify_text("任意文本", store)


# ---------- CLI 冒烟 ----------

def test_cli_calibrate_and_classify(corpora):
    from llm_fingerprint.cli import main
    with tempfile.TemporaryDirectory() as td:
        profiles = os.path.join(td, "profiles")
        smp = os.path.join(td, "samples")
        os.makedirs(smp + "/modelA")
        for i, s in enumerate(corpora["modelA"]):
            with open(f"{smp}/modelA/{i}.txt", "w", encoding="utf-8") as fh:
                fh.write(s)
        rc = main(["--store", profiles, "calibrate", "--model", "modelA",
                   "--samples", smp + "/modelA"])
        assert rc == 0 and os.path.isfile(f"{profiles}/modelA.json")
        q = os.path.join(td, "q.txt")
        with open(q, "w", encoding="utf-8") as fh:
            fh.write(corpora["modelA"][11])
        rc = main(["--store", profiles, "classify", "--text-file", q])
        assert rc == 0


# ---------- logprobs 概率指纹 ----------

def _mk_lp(n=6, top1=0.95, ent=0.10):
    """构造合成 logprobs 序列。"""
    out = []
    for i in range(n):
        out.append({"token": f"t{i}", "logprob": __import__("math").log(top1),
                    "top": [
                        {"token": f"t{i}", "logprob": __import__("math").log(top1)},
                        {"token": f"x{i}", "logprob": __import__("math").log((1 - top1) / 2)},
                        {"token": f"y{i}", "logprob": __import__("math").log((1 - top1) / 2)},
                    ]})
    return out


def test_extract_lp_features():
    from llm_fingerprint.features import LP_FEATURES, extract_lp_features
    f = extract_lp_features(_mk_lp())
    assert set(f) == set(LP_FEATURES)
    assert abs(f["lp_top1_prob_mean"] - 0.95) < 1e-6
    assert 0 < f["lp_entropy_mean"] < 1.5
    with pytest.raises(ValueError):
        extract_lp_features([])


def test_profile_lp_roundtrip_and_no_contamination(tmp_path):
    import json
    lp_a = extract_lp = None
    from llm_fingerprint.features import extract_lp_features as elf
    feats_a = [json.dumps(elf(_mk_lp(top1=0.95))) for _ in range(4)]
    feats_b = [json.dumps(elf(_mk_lp(top1=0.55))) for _ in range(4)]
    texts = ["普通文本样本，用于保证文本通道正常。"] * 4
    p = ModelProfile.build("mA", texts + feats_a + feats_b,
                           sample_pids=["fmt-list"] * 4 + ["fmt-list.lp"] * 4 + ["fmt-joke.lp"] * 4)
    assert p.n_samples == 12                       # lp 样本计入
    assert set(p.probe_lp) == {"fmt-list.lp", "fmt-joke.lp"}
    assert not any(k.endswith(".lp") for k in p.probe_gram)  # 无文本通道污染
    store = ProfileStore(tmp_path)
    store.save(p)
    loaded = store.load("mA")
    assert loaded.probe_lp["fmt-list.lp"]["lp_top1_prob_mean"][0] == \
        p.probe_lp["fmt-list.lp"]["lp_top1_prob_mean"][0]


def test_rank_models_lp_vote(tmp_path):
    import json
    from llm_fingerprint.features import extract_lp_features as elf
    from ui.monitor import rank_models
    a = ["甲风格文本样本。"] * 4
    ps = ProfileStore(tmp_path)
    # 模型A：lp 均值 top1=0.95；模型B：lp 均值 top1=0.55（无文本）
    lp_a = [json.dumps(elf(_mk_lp(top1=0.95))) for _ in range(4)]
    lp_b = [json.dumps(elf(_mk_lp(top1=0.55))) for _ in range(4)]
    ps.save(ModelProfile.build("mA", a + lp_a,
                               sample_pids=["fmt-list"] * 4 + ["fmt-list.lp"] * 4))
    ps.save(ModelProfile.build("mB", a + lp_b,
                               sample_pids=["fmt-list"] * 4 + ["fmt-list.lp"] * 4))
    profiles = ps.load_all()
    # 待判样本：top1≈0.95 → 应投 mA
    probe = json.dumps(elf(_mk_lp(top1=0.94)))
    ranked = rank_models([("fmt-list.lp", probe)], profiles, embedder=None)
    assert ranked["votes"].get("mA", 0) == 1
    assert ranked["votes"].get("mB", 0) == 0
    mm = {m["name"]: m for m in ranked["models"]}
    assert mm["mA"]["lp_similarity"] is not None
    assert mm["mA"]["lp_similarity"] > mm["mB"]["lp_similarity"]
