# -*- coding: utf-8 -*-
"""BGE 中文文本嵌入器（ONNX INT8，纯 CPU）。

模型：BAAI 公开模型 bge-small-zh-v1.5（MIT 许可）的 ONNX INT8 量化版，
位于本包 models/bge-small-zh-v1.5-onnx/。中文效果好，512 维输出已 L2 归一化。
运行依赖：onnxruntime + tokenizers + numpy（无 torch/transformers/CUDA）。

用法：
    from text_embedder import BgeEmbedder
    emb = BgeEmbedder()
    vec = emb.embed("千紫美尔数字人支持私有化部署")   # (512,) float32, L2-normalized
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except Exception:  # 原生依赖缺失时不阻断 import，实例化时报清晰错误
    ort = None
    Tokenizer = None

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL_DIR = _THIS_DIR / "models" / "bge-small-zh-v1.5-onnx"
_PAD_TOKEN = "[PAD]"


class BgeEmbedder:
    """text → 512 维归一化向量。单例，线程安全（session 串行）。"""

    _instance: Optional["BgeEmbedder"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        inst = cls._instance
        if inst is not None:
            return inst
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self, model_dir: Optional[str] = None, max_length: int = 512) -> None:
        with BgeEmbedder._instance_lock:
            if getattr(self, "_initialized", False):
                return
            try:
                self._init_body(Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR,
                                max_length)
                self._initialized = True
            except Exception:
                BgeEmbedder._instance = None
                raise

    def _init_body(self, model_dir: Path, max_length: int) -> None:
        if ort is None or Tokenizer is None:
            raise RuntimeError("缺少依赖 onnxruntime / tokenizers，"
                               "请 pip install onnxruntime tokenizers")
        onnx_path = model_dir / "model_int8.onnx"
        tok_path = model_dir / "tokenizer.json"
        if not onnx_path.is_file() or not tok_path.is_file():
            raise FileNotFoundError(f"模型文件不完整: {model_dir}")
        self._dim = 512
        cfg = model_dir / "config.json"
        if cfg.is_file():
            try:
                self._dim = int(json.loads(cfg.read_text(encoding="utf-8"))
                                .get("hidden_size", 512))
            except Exception:
                pass
        self._tok = Tokenizer.from_file(str(tok_path))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(pad_token=_PAD_TOKEN)
        self._sess = ort.InferenceSession(str(onnx_path),
                                          providers=["CPUExecutionProvider"])
        self._inputs = [i.name for i in self._sess.get_inputs()]
        self._output = self._sess.get_outputs()[0].name
        self._need_token_type = "token_type_ids" in self._inputs
        self._lock = threading.Lock()

    @property
    def DIM(self) -> int:
        return self._dim

    def _encode(self, texts: List[str]) -> np.ndarray:
        encs = self._tok.encode_batch(list(texts))
        input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        attn = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        feeds = {"input_ids": input_ids, "attention_mask": attn}
        if self._need_token_type:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        with self._lock:
            out = self._sess.run([self._output], feeds)[0]  # (B, L, D)
        mask = attn[..., None].astype(np.float32)
        pooled = (out.astype(np.float32) * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def embed(self, text: str) -> np.ndarray:
        """单条 → (DIM,) float32，L2-normalized。空文本抛 ValueError。"""
        if not text or not text.strip():
            raise ValueError("嵌入文本为空")
        return self._encode([text])[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量 → (N, DIM) float32。空列表 → (0, DIM)。"""
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._encode(texts)
