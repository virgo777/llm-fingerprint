# -*- coding: utf-8 -*-
"""密钥读取：只从环境变量读取，不落盘、不写仓库。

变量约定（见 examples/.env.example）：
  ZHIPU_KEY     智谱 Anthropic 兼容端点
  DS_KEY        DeepSeek（Anthropic 或 OpenAI 兼容端点）
  OPENAI_KEY    任意 OpenAI 兼容端点（可选）
  LLM_EXTRA_KEY 其他厂商（可选）

安全约定：任何脚本都不得读取仓库内的明文密钥文件，也不得打印 key。
"""
from __future__ import annotations

import os

ENV_ZHIPU = "ZHIPU_KEY"
ENV_DEEPSEEK = "DS_KEY"


def _get(name: str) -> str:
    return os.environ.get(name, "").strip()


def zhipu_key() -> str:
    """智谱 key；未配置返回空串，调用方应跳过该目标而不是崩溃。"""
    return _get(ENV_ZHIPU)


def deepseek_key() -> str:
    """DeepSeek key；未配置返回空串。"""
    return _get(ENV_DEEPSEEK)


def require_key(name: str) -> str:
    """取必需 key；缺失时抛出带修复提示的异常（不打印任何密钥内容）。"""
    value = _get(name)
    if not value:
        raise RuntimeError(
            f"缺少环境变量 {name}。请先 export {name}=<your-key>"
            f"（或在本地 .env 中配置后由你的 shell 注入），不要写进仓库。"
        )
    return value
