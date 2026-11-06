# -*- coding: utf-8 -*-
"""嵌入器懒加载：可用则用 BGE 语义向量，不可用则优雅降级（纯标准库路径）。"""
from __future__ import annotations

from typing import Optional

_embedder = None
_tried = False


def get_embedder():
    """返回共享 BgeEmbedder 实例；依赖/模型缺失或打包环境返回 None（优雅降级）。

    PyInstaller exe 中跳过：onnxruntime 原生 DLL 在 onefile 下不稳定，
    崩溃会直接杀死进程（绕过 Python 异常），因此冻结构建改用纯文本路线。
    """
    global _embedder, _tried
    if _tried:
        return _embedder
    _tried = True
    import sys
    if getattr(sys, "frozen", False):
        return None
    try:
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from text_embedder import BgeEmbedder
        _embedder = BgeEmbedder()
    except Exception:
        _embedder = None
    return _embedder
