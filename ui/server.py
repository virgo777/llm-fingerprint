# -*- coding: utf-8 -*-
"""端点连通性测试面板 —— 本地服务（纯标准库）。
浏览器界面 + 后台两级探测（GET /models → 无 model 鉴权探测），无需模型名称。
key 只在本机内存中流转，不落盘、不写日志。
启动：python ui/server.py  →  http://127.0.0.1:8765
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_fingerprint.probe_client import (ProbeError, chat_completion,
                                          chat_completion_anthropic)

HOST = "127.0.0.1"
PORT = 8765
MONITOR_JOBS: dict = {}  # {id: {t0, status, stage, done, total, result}}


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def _ui_dir() -> Path:
    """index.html 位置：PyInstaller 打包后位于 _MEIPASS/ui/。"""
    if _frozen():
        return Path(getattr(sys, "_MEIPASS", ".")) / "ui"
    return Path(__file__).resolve().parent


def safe_print(*args) -> None:
    try:
        print(*args)
    except Exception:
        pass  # --noconsole 模式下 stdout 可能不可用


_URL_RE = re.compile(r"^https?://[^\s]+$")
_TIMEOUT = 45  # 部分网关首次调用冷启动慢，留足余量


def validate_payload(data) -> str | None:
    """输入校验（系统边界）。返回错误消息，None 表示通过。"""
    if not isinstance(data, dict):
        return "请求体必须是 JSON 对象"
    for field in ("base_url", "api_key"):
        v = data.get(field)
        if not isinstance(v, str) or not v.strip():
            return f"缺少必填项：{field}"
    if not _URL_RE.match(data["base_url"].strip()):
        return "地址必须以 http:// 或 https:// 开头且不含空格"
    if data.get("api") not in ("openai", "anthropic"):
        return "协议类型必须是 openai 或 anthropic"
    if len(data["api_key"]) > 4096:
        return "密钥长度超出限制"
    return None


def _http(url: str, key: str, method: str = "GET", body: dict | None = None):
    """返回 (status, 文本)。网络层异常抛 URLError/Timeout。"""
    headers = {"Authorization": f"Bearer {key}",
               "x-api-key": key,
               "anthropic-version": "2023-06-01"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def _parse_model_ids(text: str) -> list[str]:
    try:
        data = json.loads(text).get("data") or []
        ids = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
        return sorted(set(ids))[:200]
    except (ValueError, AttributeError):
        return []


def probe_models(base: str, key: str, api: str) -> dict:
    """第一级：GET 模型列表。成功即连通，并带回模型清单。"""
    url = (base.rstrip("/") + "/v1/models") if api == "anthropic" else \
          (base.rstrip("/") + "/models")
    status, text = _http(url, key)
    if status == 200:
        models = _parse_model_ids(text)
        if models:  # 200 且解析出模型 → 连通成功
            return {"ok": True, "mode": "models", "models": models,
                    "n_models": len(models)}
        # 200 但无模型数据（部分网关返回空/HTML）→ 降级走鉴权探测
    raise _Fallback(status, text)


class _Fallback(Exception):
    def __init__(self, status: int, text: str):
        super().__init__()
        self.status, self.text = status, text


def probe_auth(base: str, key: str, api: str) -> dict:
    """第二级：无 model 的最小请求。400/422=鉴权通过；401/403=密钥无效。"""
    if api == "anthropic":
        url = base.rstrip("/") + "/v1/messages"
        body = {"max_tokens": 8, "messages": [{"role": "user", "content": "ping"}]}
        model_hint = "model"
    else:
        url = base.rstrip("/") + "/chat/completions"
        body = {"max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
        model_hint = "model"
    try:
        status, text = _http(url, key, method="POST", body=body)
    except (urllib.error.URLError, TimeoutError) as e:
        return {"ok": False, "error": f"无法连通：{_err(e)}"}
    if status in (400, 422):
        low = text.lower()
        if "model" in low or "required" in low or "必填" in text:
            return {"ok": True, "mode": "auth",
                    "note": "端点未提供模型列表，但鉴权通过（对话需补充模型名）"}
        return {"ok": False, "error": f"请求被拒绝（HTTP {status}）：{_snip(text)}"}
    if status in (401, 403):
        return {"ok": False, "error": f"密钥无效或无权限（HTTP {status}）：{_snip(text)}"}
    if status == 404:
        return {"ok": False, "error": "路径不存在（HTTP 404）：请检查地址是否应到 /v1 层"}
    if status == 429:
        return {"ok": False, "error": f"限流或余额不足（HTTP 429）：{_snip(text)}"}
    return {"ok": False, "error": f"意外响应（HTTP {status}）：{_snip(text)}"}


def _snip(text: str) -> str:
    return text.strip().replace("\n", " ")[:120]


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"[:150]


def guess_api(base_url: str) -> str | None:
    """根据地址猜测协议：含 /anthropic → anthropic；/paas/v4、/v1、/openai → openai。"""
    u = (base_url or "").lower()
    if "/anthropic" in u:
        return "anthropic"
    if "/paas/v4" in u or "/v1" in u or "/openai" in u:
        return "openai"
    return None


def run_test(data: dict) -> dict:
    """两级探测：GET /models → 失败则无 model 鉴权探测。网络层失败自动重试一次。"""
    base, key, api = data["base_url"].strip(), data["api_key"].strip(), data["api"]
    if api not in ("openai", "anthropic"):  # 双保险：无效值时按地址猜
        api = guess_api(base) or "openai"
    t0 = time.perf_counter()

    def finish(d: dict) -> dict:
        d["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return d

    def attempt() -> dict:
        try:
            return probe_models(base, key, api)
        except _Fallback:
            return probe_auth(base, key, api)

    try:
        return finish(attempt())
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        import time as _t
        _t.sleep(1.0)  # 冷启动/瞬断重试一次
        try:
            return finish(attempt())
        except (urllib.error.URLError, TimeoutError, ValueError) as e2:
            return finish({"ok": False, "error": f"无法连通（已重试）：{_err(e2)}"})
        except Exception as e2:
            return finish({"ok": False, "error": f"{type(e2).__name__}: {e2}"})
    except Exception as e:
        return finish({"ok": False, "error": f"{type(e).__name__}: {e}"})


def run_concurrent_test(data: dict) -> dict:
    """并发压测：同时发 n 个最小请求，统计成功率与延迟分布。"""
    import concurrent.futures as cf
    import time as _t
    n = data.get("n", 8)
    if not isinstance(n, int) or not 1 <= n <= 64:
        return {"ok": False, "error": "并发数 n 需在 1..64"}
    base, key, api = data["base_url"].strip(), data["api_key"].strip(), data["api"]
    if api not in ("openai", "anthropic"):
        api = guess_api(base) or "openai"
    prompt = "并发测试，请只回复：ok"

    def _one(_):
        t0 = _t.perf_counter()
        try:
            if api == "anthropic":
                chat_completion_anthropic(base, key, data["model_id"].strip(),
                                          prompt, temperature=0.1, max_tokens=512,
                                          timeout=30)
            else:
                # 思考型模型会把预算花在推理上，留足余量保证正文非空
                chat_completion(base, key, data["model_id"].strip(), prompt,
                                temperature=0.1, max_tokens=1500, timeout=30)
            return int((_t.perf_counter() - t0) * 1000), None
        except ProbeError as e:
            return None, str(e)[:120]
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"[:120]

    t0 = _t.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(_one, range(n)))
    wall = int((_t.perf_counter() - t0) * 1000)
    lats = sorted(ms for ms, err in results if ms is not None)
    errs = [err for _, err in results if err]
    p95 = lats[max(0, int(len(lats) * 0.95) - 1)] if lats else None
    return {"ok": True, "n": n, "success": len(lats), "fail": len(errs),
            "avg_ms": round(sum(lats) / len(lats)) if lats else None,
            "p95_ms": p95, "max_ms": lats[-1] if lats else None,
            "wall_ms": wall,
            "errors": list(dict.fromkeys(errs))[:3]}


class Handler(BaseHTTPRequestHandler):





    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, (_ui_dir() / "index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path.startswith("/api/progress"):
            self._handle_progress(self.path.split("?", 1)[-1])
        elif self.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/test":
            handler = self._handle_test
        elif self.path == "/api/monitor":
            handler = self._handle_monitor
        elif self.path == "/api/concurrent":
            handler = self._handle_concurrent
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 65536)
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, json.dumps({"ok": False, "error": "请求体不是合法 JSON"})
                       .encode(), "application/json; charset=utf-8")
            return
        handler(data)

    def _handle_test(self, data: dict) -> None:
        err = validate_payload(data)
        if err:
            self._send(400, json.dumps({"ok": False, "error": err}).encode(),
                       "application/json; charset=utf-8")
            return
        self._send(200, json.dumps(run_test(data), ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _handle_concurrent(self, data: dict) -> None:
        err = validate_payload(data)
        if err:
            self._send(400, json.dumps({"ok": False, "error": err}).encode(),
                       "application/json; charset=utf-8")
            return
        result = run_concurrent_test(data)
        self._send(200, json.dumps(result, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def _handle_monitor(self, data: dict) -> None:
        """异步启动监测任务，立即返回任务 id；进度经 GET /api/progress 轮询。"""
        import uuid
        from ui.monitor import run_monitor
        err = validate_payload(data)
        if err:
            self._send(400, json.dumps({"ok": False, "error": err}).encode(),
                       "application/json; charset=utf-8")
            return
        try:
            rounds = int(data.get("rounds") or 50)
        except (TypeError, ValueError):
            rounds = 50
        rounds = max(1, min(rounds, 100))  # 1..100 轮
        # 清理 2 小时前的旧任务（50 轮任务最长可跑 ~1 小时）
        now = time.time()
        for jid in list(MONITOR_JOBS.keys()):
            if now - MONITOR_JOBS[jid]["t0"] > 7200:
                del MONITOR_JOBS[jid]
        jid = uuid.uuid4().hex[:12]
        MONITOR_JOBS[jid] = {"t0": now, "status": "running", "stage": "probes",
                             "done": 0, "total": 10 * rounds, "rounds": rounds,
                             "result": None, "last_probe": ""}

        def _run():
            job = MONITOR_JOBS[jid]

            def _on_stage(stage, done, total):
                job.update(stage=stage.split(":")[0] if ":" not in stage else "probes",
                           done=done, total=total)
                if ":" in stage:  # probe:<pid>:<ok|fail> → 流式探针事件
                    job["last_probe"] = stage.split(":", 1)[1]
            try:
                result = run_monitor(data["base_url"], data["api_key"], data["api"],
                                    data.get("model_id") or "", on_stage=_on_stage,
                                    rounds=rounds)
            except Exception as e:
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            job["result"] = result
            job["status"] = "done" if result.get("ok") else "error"
            job["stage"] = "done"
            _append_evidence(data, result)

        def _append_evidence(data: dict, result: dict) -> None:
            """证据链：每次判定追加一行 JSONL（绝不记录 api_key）。"""
            import datetime
            try:
                if getattr(sys, "frozen", False):
                    ev_dir = Path(sys.executable).parent / "evidence"
                else:
                    ev_dir = Path(__file__).resolve().parent.parent / "evidence"
                ev_dir.mkdir(exist_ok=True)
                rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
                       "base_url": data.get("base_url", ""),
                       "api": data.get("api", ""), "model_id": data.get("model_id", ""),
                       "rounds": rounds}
                if result.get("ok"):
                    rec.update({"verdict": result.get("verdict"),
                                "verdict_scope": result.get("verdict_scope"),
                                "votes": result.get("votes"),
                                "n_replies": result.get("n_replies"),
                                "n_sent": result.get("n_sent"),
                                "models": [{k: m.get(k) for k in
                                            ("name", "similarity", "lp_similarity", "votes")}
                                           for m in result.get("models", [])],
                                "per_probe": result.get("per_probe"),
                                "identity": result.get("identity")})
                else:
                    rec.update({"error": result.get("error")})
                with open(ev_dir / "evidence.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                safe_print("证据落盘失败（不影响判定）")

        threading.Thread(target=_run, daemon=True).start()
        self._send(200, json.dumps({"ok": True, "id": jid}).encode(),
                   "application/json; charset=utf-8")

    def _handle_progress(self, query: str) -> None:
        qs = urllib.parse.parse_qs(query)
        jid = (qs.get("id") or [""])[0]
        job = MONITOR_JOBS.get(jid)
        if job is None:
            self._send(404, json.dumps({"ok": False, "error": "任务不存在或已过期"})
                       .encode(), "application/json; charset=utf-8")
            return
        payload = {"ok": True, "status": job["status"], "stage": job["stage"],
                   "done": job["done"], "total": job["total"],
                   "rounds": job.get("rounds", 1),
                   "last_probe": job.get("last_probe", "")}
        if job["result"] is not None:
            payload["result"] = job["result"]
        self._send(200, json.dumps(payload, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def log_message(self, fmt: str, *args) -> None:
        safe_print(f"[{self.log_date_time_string()}] {fmt % args}")


def make_server(port: int | None = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((HOST, port or PORT), Handler)


def main() -> int:
    srv = make_server()
    url = f"http://{HOST}:{PORT}"
    safe_print(f"端点测试面板已启动: {url}  （Ctrl+C 退出）")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        safe_print("\n已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
