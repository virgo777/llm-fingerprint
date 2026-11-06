# -*- coding: utf-8 -*-
"""OpenAI 兼容 API 最小客户端（urllib，无第三方依赖）。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

DEFAULT_TIMEOUT = 60
MAX_RETRY = 2


class ProbeError(RuntimeError):
    """API 调用失败（网络/鉴权/限流）。"""


def chat_completion(base_url: str, api_key: str, model_id: str, prompt: str,
                    temperature: float = 0.5, max_tokens: int = 600,
                    timeout: int = DEFAULT_TIMEOUT,
                    system: Optional[str] = None,
                    logprobs: bool = False, top_logprobs: int = 0,
                    with_logprobs: bool = False):
    """调用 POST {base_url}/chat/completions。返回文本；with_logprobs=True 返回 (文本, lp列表)。
    lp列表元素: {"token": str, "logprob": float, "top": [{"token": str, "logprob": float}]}
    """
    if not base_url or not base_url.startswith(("http://", "https://")):
        raise ProbeError(f"非法 base_url: {base_url!r}")
    if not api_key:
        raise ProbeError("缺少 api_key")
    if not model_id:
        raise ProbeError("缺少 model_id")
    url = base_url.rstrip("/") + "/chat/completions"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model_id,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = max(top_logprobs, 1)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    last_err: Optional[str] = None
    for attempt in range(MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices") or []
            if not choices:
                raise ProbeError(f"响应缺少 choices: {str(data)[:200]}")
            msg = choices[0].get("message", {})
            content = msg.get("content")
            text = content if isinstance(content, str) else \
                "".join(seg.get("text", "") for seg in content
                       if isinstance(seg, dict)) if isinstance(content, list) else ""
            if with_logprobs:
                lp_raw = ((choices[0].get("logprobs") or {}).get("content")) or []
                lp = [{"token": e.get("token", ""),
                       "logprob": float(e.get("logprob", 0.0)),
                       "top": [{"token": t.get("token", ""),
                                "logprob": float(t.get("logprob", 0.0))}
                               for t in (e.get("top_logprobs") or [])]}
                      for e in lp_raw if isinstance(e, dict)]
                return text, lp
            if text:
                return text
            raise ProbeError(f"回复内容为空: {str(msg)[:200]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (401, 403, 404):
                break  # 鉴权/路径错误，重试无意义
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    raise ProbeError(f"调用失败（已重试 {MAX_RETRY} 次）：{last_err}")


def chat_completion_anthropic(base_url: str, api_key: str, model_id: str, prompt: str,
                              temperature: float = 0.5, max_tokens: int = 2048,
                              timeout: int = DEFAULT_TIMEOUT,
                              thinking: Optional[dict] = None,
                              with_thinking: bool = False,
                              system: Optional[str] = None):
    """调用 Anthropic 兼容端点 POST {base_url}/v1/messages（智谱/DeepSeek 等）。

    thinking: None=端点默认；{"type":"disabled"} 关思考；{"type":"enabled","budget_tokens":N} 开。
    with_thinking: True 时返回 (正文, 思考内容)；False 只返回正文。
    system: 可选系统提示词。
    """
    if not base_url or not base_url.startswith(("http://", "https://")):
        raise ProbeError(f"非法 base_url: {base_url!r}")
    if not api_key:
        raise ProbeError("缺少 api_key")
    if not model_id:
        raise ProbeError("缺少 model_id")
    url = base_url.rstrip("/") + "/v1/messages"
    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    if thinking is not None:
        payload["thinking"] = thinking
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"})
    last_err: Optional[str] = None
    for attempt in range(MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("content")
            if isinstance(content, list):
                text = "".join(seg.get("text", "") for seg in content
                               if isinstance(seg, dict) and seg.get("type") == "text")
                think = "".join(str(seg.get("thinking", "")) for seg in content
                                if isinstance(seg, dict) and seg.get("type") == "thinking")
                if text or think:
                    if with_thinking:
                        return text, think
                    return text
                raise ProbeError(f"回复 content 为空: {str(data)[:200]}")
            if isinstance(content, str) and content:
                if with_thinking:
                    return content, ""
                return content
            raise ProbeError(f"响应缺少 content: {str(data)[:200]}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (401, 403, 404):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.5 * (attempt + 1))
    raise ProbeError(f"调用失败（已重试 {MAX_RETRY} 次）：{last_err}")


def run_probes(base_url: str, api_key: str, model_id: str,
               temperature: float = 0.5, api: str = "openai",
               pids: list = None, workers: int = 1,
               on_result=None, timeout: int = DEFAULT_TIMEOUT,
               max_retries: int = 0) -> Dict[str, str]:
    """执行探针，返回 {pid: 回复}。pids 非空时只发指定探针。
    workers>1 时并发执行。on_result(pid, text) 在每个探针完成时回调。
    timeout 为单探针超时秒数；max_retries>0 时单探针失败重试。
    单探针失败记为空串，不中断整体。"""
    from llm_fingerprint.probes import PROBES
    if api not in ("openai", "anthropic"):
        raise ProbeError(f"未知 api 类型: {api}")
    fn = chat_completion if api == "openai" else chat_completion_anthropic
    todo = [p for p in PROBES if not pids or p.pid in pids]

    def _one(p):
        txt = ""
        attempts = 1 + max_retries
        for attempt in range(attempts):
            try:
                txt = fn(base_url, api_key, model_id, p.prompt,
                         temperature=temperature, timeout=timeout)
                break
            except ProbeError as e:
                safe_note(f"探针 {p.pid} 失败（第 {attempt + 1}/{attempts} 次）: {e}")
        if on_result is not None:
            try:
                on_result(p.pid, txt)
            except Exception:
                pass
        return p.pid, txt

    if workers <= 1:
        pairs = [_one(p) for p in todo]
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            pairs = list(ex.map(_one, todo))
    return dict(pairs)


def safe_note(msg: str) -> None:
    try:
        print(f"  [warn] {msg}")
    except Exception:
        pass
