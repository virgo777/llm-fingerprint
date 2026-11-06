# 交接文档 · Developer Handover

> 面向接续开发者。项目定位、架构、复现命令与已知边界。

## 一、项目是什么

**大模型指纹识别工具**：输入中转站 URL / key / 模型名 → 测联通 → 发探针 → 与本地档案库比对 → 判定「端点背后是不是它声称的模型」。
- 档案库：4 模型（glm-4.5 / glm-4.6 / deepseek-chat / deepseek-reasoner）× 三信号面（`.off` 关思考正文 / `.on` 默认正文 / `.thk` 思考内容）+ logprobs 面（DeepSeek）
- 桌面客户端：`dist/EndpointTester.exe`（Windows 单文件，约 32 MB）

## 二、实测结论（截至 2026-11-06）

| 项 | 结果 |
|---|---|
| 50 轮采集 | 2000 次调用仅 3 失败，约 50 分钟 |
| 留出测试（r41–r50） | 投票 **3/4 精确、4/4 厂商级**；单条 57% |
| glm-4.5 vs glm-4.6 | **三信号面全部不可区分**（SNR≈0.03）→ 正确降级为「智谱 GLM 家族」 |
| deepseek-reasoner 关思考 | 行为 ≈ deepseek-chat 底座 |
| logprobs 通道 | 智谱端点不支持；DeepSeek 支持 top-20，对抗伪装修复 100% |
| 对抗评测 | few-shot 风格注入破防文本指纹 48%；logprobs 指纹 30/30 识破 |
| 阈值校准 | 投票级误指控 0/4；单条厂商级误指控 18.8%（**不可单独作指控依据**） |
| 测试 | 30/30 通过 |

## 三、密钥管理（重要）

- 所有脚本**只从环境变量读密钥**：`ZHIPU_KEY`、`DS_KEY`（统一入口 `llm_fingerprint/envkeys.py`）
- 仓库内不允许存在明文密钥文件；`.gitignore` 已屏蔽 `keys.md`、`.env` 等
- 采集脚本对缺失的厂商直接跳过，而不是回退读文件
- 证据链 JSONL 只记录时间戳 / 票型 / 分数 / 判定，**不含 key**

## 四、架构速查（改哪找哪）

```
llm_fingerprint/
├── features.py      31 维风格特征 + 字符三元组 + logprobs 特征
├── profile.py       档案（全局质心 + probe_emb 嵌入 + probe_gram 三元组 + lp 质心）
├── classify.py      多路融合判定 + 投票
├── probes.py        17 个探针定义
├── probe_client.py  OpenAI / Anthropic 双协议客户端；run_probes(timeout=, on_result=)
├── envkeys.py       环境变量取 key
├── embedding.py     get_embedder()（frozen 环境返回 None = exe 降级路线）
└── cli.py           calibrate / classify / probe / list
ui/
├── server.py        本地服务：/api/test 两级探测；/api/monitor 异步任务；/api/progress 轮询
├── monitor.py       监测编排：MONITOR_PIDS、PROBE_TIMEOUT_S=17、on_stage 流式回调
├── app.py           桌面入口（pywebview → tkinter → 浏览器三级兜底）
└── index.html       全部 UI
text_embedder/       BGE 嵌入器（Python 环境用，exe 已剔除）
profiles/            4 模型档案（calibrate 重建）
samples/             4 模型多轮语料
tests/               30 个测试
```

## 五、复现命令

```bash
# 测试
python -X utf8 -B -m pytest tests/ -q

# 重新构建 exe（改代码后必跑）
python -X utf8 -m PyInstaller --noconfirm --onefile --noconsole --name EndpointTester ui/app.py \
  --add-data "ui/index.html;ui" --add-data "profiles;profiles" \
  --hidden-import ui.monitor --hidden-import ui.server --hidden-import llm_fingerprint.probes \
  --collect-all pythonnet --collect-all pywebview \
  --exclude-module onnxruntime --exclude-module tokenizers --exclude-module text_embedder

# 采集（断点续采；轮数用 COLLECT_ROUNDS 控制）
export ZHIPU_KEY=... DS_KEY=...
python -X utf8 examples/collect_rounds.py

# 重建档案
for m in glm-4.6 glm-4.5 deepseek-chat deepseek-reasoner; do
  python -X utf8 -m llm_fingerprint.cli --store profiles calibrate --model $m --samples samples/$m
done

# 评测
python examples/evaluate.py

# 开发模式起服务（不依赖窗口）
python -X utf8 ui/server.py    # http://127.0.0.1:8765
```

## 六、接下来可做的

1. **扩充模型库**：每新增一个模型，1 轮校准即可先接入；50 轮采集可显著提高投票稳定性
2. **开放集拒绝**：当前为闭集判定，引入基于距离阈值的「未知模型」拒绝
3. **logprobs 普及路线**：OpenAI 协议端点的中转站通常支持 `top_logprobs`，优先用于防伪装
4. **对抗硬化**：探针随机化（≥200 题 × 3 模板）、似然比判定

## 七、本机环境事实

- Windows 11，Python 3.13（miniconda）；pythonnet + pywebview + pyinstaller
- 智谱 Anthropic 端点 `/v1/models` 可用；GLM 默认带 thinking，`max_tokens ≥ 2048` 才出正文
- DeepSeek OpenAI 路径 `/models` 可用；Anthropic 路径无 models 接口
