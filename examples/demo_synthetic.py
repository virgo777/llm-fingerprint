# -*- coding: utf-8 -*-
"""演示：三种合成风格 → 校准 → 留出集判定。
注意：合成语料仅用于验证流水线正确性，不代表真实模型准确率。
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_fingerprint.classify import classify_text
from llm_fingerprint.cli import main
from llm_fingerprint.profile import ModelProfile, ProfileStore

A = ("首先，{t}的核心在于基础概念的把握。其次，我们需要关注实践层面的细节。"
     "此外，还有一些常见误区需要注意。总之，{t}是一个循序渐进的过程。")
B = "**{t}指南**\n\n1. 基础概念很重要 📌\n2. 动手实践是关键 ✅\n3. 避开常见坑 ⚠️\n\n希望对你有帮助~"
C = ("关于{t}之问题，兹从如下数端予以说明：其一，概念层面（定义、范畴与边界）；"
     "其二，实践层面——操作规范、注意事项及常见谬误；其三，总结与展望。"
     "综上所述，{t}需系统化推进。")
TOPICS = ["机器学习", "中医养生", "门店运营", "健康管理", "数字人", "脉诊",
          "体成分", "知识库", "预约系统", "支付", "舌诊", "视觉识别"]
MODELS = {"styleA": A, "styleB": B, "styleC": C}


def main_demo() -> int:
    tmp = tempfile.mkdtemp(prefix="fp_demo_")
    store_dir = os.path.join(tmp, "profiles")
    store = ProfileStore(store_dir)
    correct = total = 0
    try:
        for name, tpl in MODELS.items():
            samples = [tpl.format(t=TOPICS[i % len(TOPICS)]) for i in range(12)]
            store.save(ModelProfile.build(name, samples[:9]))
            for s in samples[9:]:            # 留出集 3 条/模型
                r = classify_text(s, store)
                total += 1
                hit = (not r.rejected) and r.best.name == name
                correct += hit
                print(f"  {name} -> 判定 {r.best.name if r.best else '-'}"
                      f" (置信 {r.confidence:.0%}) {'✓' if hit else '✗'}")
        # CLI 冒烟
        rc = main(["--store", store_dir, "list"])
        assert rc == 0, "CLI list 失败"
        print(f"\n留出集准确率: {correct}/{total}")
        return 0 if correct == total else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main_demo())
