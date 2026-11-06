# -*- coding: utf-8 -*-
"""桌面客户端入口：后台本地服务 + 原生窗口。
窗口策略（逐级兜底，任何一级失败都写日志）：
  1. pywebview 原生窗口（需 pythonnet/clr 可用；缺失则跳过，避免原生崩溃闪退）
  2. tkinter 控制窗 + 系统浏览器
  3. 纯浏览器 + 常驻
  4. 全部失败 → Windows 原生消息框（ctypes）告知日志位置
启动与各阶段异常均记录到 exe 旁 endpoint_tester.log。
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_TITLE = "端点测试台"


def log_path() -> Path:
    """日志写到 exe（或脚本）所在目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "endpoint_tester.log"
    return Path(__file__).resolve().parent.parent / "endpoint_tester.log"


def log(msg: str) -> None:
    try:
        import datetime
        with open(log_path(), "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except Exception:
        pass


def pick_port(start: int = 8765, tries: int = 20) -> int:
    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("无可用端口")


def start_server() -> tuple[object, int]:
    from ui.server import make_server
    port = pick_port()
    srv = make_server(port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def start_server_with_watchdog(url: str) -> None:
    """服务线程看门狗：意外死亡时记日志并自动重启（最多 5 次）。"""
    state = {"restarts": 0}

    def _watch():
        import time as _t
        while True:
            _t.sleep(5)
            srv_ref = getattr(_watch, "srv", None)
            alive = False
            try:
                import urllib.request as _u
                with _u.urlopen(url, timeout=3):
                    alive = True
            except Exception:
                alive = False
            if not alive and state["restarts"] < 5:
                state["restarts"] += 1
                log(f"服务探活失败，第 {state['restarts']} 次重启")
                try:
                    from ui.server import make_server
                    srv = make_server(pick_port(int(url.rsplit(':', 1)[-1])))
                    threading.Thread(target=srv.serve_forever, daemon=True).start()
                    log(f"服务已重启于 {url}")
                except Exception:
                    log("服务重启失败:\n" + traceback.format_exc())

    threading.Thread(target=_watch, daemon=True).start()


def webview_available() -> bool:
    """pywebview 的 WinForms 后端依赖 pythonnet(clr)；缺 clr 时强上会原生崩溃。"""
    try:
        import clr  # noqa: F401
        return True
    except Exception as e:
        log(f"pywebview 跳过（pythonnet 不可用: {type(e).__name__}）")
        return False


def run_webview(url: str) -> bool:
    if not webview_available():
        return False
    try:
        import webview
        webview.create_window(APP_TITLE, url, width=680, height=920,
                              min_size=(480, 640))
        webview.start()
        return True
    except Exception:
        log("pywebview 启动失败:\n" + traceback.format_exc())
        return False


def run_tk(url: str) -> bool:
    try:
        import tkinter as tk
        import webbrowser
        webbrowser.open(url)
        root = tk.Tk()
        root.title(APP_TITLE)
        root.geometry("420x190")
        root.attributes("-topmost", True)
        tk.Label(root, text="端点测试台 · 运行中",
                 font=("Microsoft YaHei UI", 14, "bold"), pady=16).pack()
        tk.Label(root, text=url, fg="#1a7f4b",
                 font=("Consolas", 10)).pack()
        tk.Label(root, text="界面已在浏览器打开，如未打开请复制上方地址访问\n关闭本窗口即退出程序",
                 fg="#888", pady=14).pack()
        root.mainloop()
        return True
    except Exception:
        log("tkinter 控制窗失败:\n" + traceback.format_exc())
        return False


def message_box_fallback(text: str) -> None:
    """最后兜底：Windows 原生消息框（不依赖任何 GUI 库）。"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0, text, APP_TITLE, 0x10)  # MB_ICONERROR
    except Exception:
        pass


def main() -> int:
    log("=== 启动 ===")
    # WebView2 强制直连：免疫系统代理（Clash/V2Ray 全局模式）导致的
    # 本机 127.0.0.1 fetch 失败（"Failed to fetch"）
    os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
                          "--proxy-server=direct://")
    try:
        srv, port = start_server()
    except Exception:
        log("服务启动失败:\n" + traceback.format_exc())
        message_box_fallback(f"启动失败，详见日志：\n{log_path()}")
        return 1
    url = f"http://127.0.0.1:{port}"
    log(f"服务就绪: {url}")
    start_server_with_watchdog(url)
    if run_webview(url):
        log("正常退出（pywebview 窗口关闭）")
        return 0
    if run_tk(url):
        log("正常退出（控制窗关闭）")
        return 0
    try:
        import webbrowser
        webbrowser.open(url)
        log("以浏览器模式常驻")
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
