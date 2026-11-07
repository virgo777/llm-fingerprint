# llm-fingerprint · LLM 指纹识别 / Which model am I really talking to?

[中文](#中文) | [English](#english)

> **你调用的模型，真的是服务商声称的那个吗？**
> 输入中转站的 URL / key / 模型名 → 发探针 → 与本地档案库比对 → 判定端点背后到底是不是它声称的模型。

**不需要模型方配合，纯 CPU，零 GPUs。** 目前仅支持 **智谱 GLM**、**DeepSeek Chat / Reasoner**（基础数据均来自各家官方端点采集），后续会加入更多模型。

---

## 中文

### 30 秒上手

1. 下载 [`dist/EndpointTester.exe`](dist/EndpointTester.exe)（Windows，约 32 MB，双击即用，无需安装 Python）
2. 填写中转站地址、API Key、声称的模型名 → 点「测试联通」→ 点「开始监测」
3. 工具会发送多轮探针，输出**判定结论**：背后真实模型是谁、票型分布、相似度、证据链 JSONL

### 它解决什么问题

中转站、代理服务、二次封装 API 普遍存在「挂羊头卖狗肉」：宣称某个旗舰模型，实际路由到便宜的小模型。单看回复质量往往看不出来 —— 大模型回答得都挺像样。本项目用**统计指纹**做概率判定：

| 信号面 | 提取内容 |
|---|---|
| **文风特征**（31 维） | 句长分布、标点习惯、Markdown/列表/emoji、连接词与客套语频率、中英混排比 |
| **字符三元组分布** | 表层文风指纹 |
| **BGE 语义嵌入**（512 维） | 探针条件化：回答只与「同一探针」的档案质心比较，去除话题变量，留下纯文风差异 |
| **logprobs 概率路**（可选） | 逐 token top-k 概率分布 —— 文风可模仿，概率分布模仿不动 |

检测时对多个探针逐条判定后**多数投票**；同厂相邻模型票差过小则自动降级为**厂商级结论**（如「智谱 GLM 家族」）。

### 实测结果

校准：每模型 1 轮探针（17 条回复）；测试：另一轮全新探针（t03 训练 / t07 测试）：

| 模式 | 精确模型准确率 | 厂商级准确率 |
|---|---|---|
| **整组探针 + 逐条投票**（生产路径） | **4/4 = 100%** | 4/4 |
| 整组探针合并判定 | 3/4 = 75% | 4/4 |
| 单条回复判定 | 41/67 ≈ 61% | ~90% |

测试矩阵含两对「难例」：同厂商相邻版本、deepseek-chat vs deepseek-reasoner（同厂双模式）。投票模式全部判对。样本量小（4 模型 × 34 条），以上为初始参考而非保证。

增强采集（50 轮 × 10 探针 × 三信号面，每模型 1534–2034 条）后的留出测试：**精确 3/4，厂商级 4/4**。

### 对抗能力（诚实说明）

| 攻击方式 | 结果 |
|---|---|
| 系统提示词人设注入（"你是 GLM…"） | 文本指纹**破防率 48%**，可翻转厂商级判定 |
| 人设 + few-shot 真实风格示例 | 同上量级 |
| **logprobs 概率指纹**（DeepSeek 端点，30/30） | **100% 识破** —— token 概率分布模仿不动 |

GLM 端点不提供 logprobs，因此 GLM 家族只能靠「置信度 + 多信号面投票」降低而非消除伪装风险。

### 一个重要的已知事实

经三信号面（关思考 / 默认 / 思考内容）交叉验证：**同厂商相邻版本在文风统计上不可区分**（SNR≈0.03，各信号面投票互相打成平手）。产品行为是自动降级为「厂商家族级」结论。这不是 bug，是统计事实。

另有发现：**deepseek-reasoner 在关闭思考时的输出 ≈ deepseek-chat 底座行为**。

### 命令行用法

```bash
git clone https://github.com/virgo777/llm-fingerprint.git
cd llm-fingerprint
pip install pytest                       # 运行测试
# pip install onnxruntime tokenizers numpy   # 可选：启用 BGE 语义嵌入

# 0) 配置密钥（只读环境变量，永不落盘）
export ZHIPU_KEY=...        # 智谱
export DS_KEY=...           # DeepSeek

# 1) 校准：把某模型的探针回复（<探针id>.<轮次>.txt）放入目录
python -m llm_fingerprint.cli --store profiles calibrate --model <模型名> --samples samples/<模型名>/

# 2) 主动探测端点（自动逐条判定 + 投票，支持 openai / anthropic 协议）
python -m llm_fingerprint.cli --store profiles probe \
    --base-url https://open.bigmodel.cn/api/anthropic --api anthropic \
    --api-key "$ZHIPU_KEY" --model-id <模型名> --dump out.json

# 3) 被动判定一段已有回复
python -m llm_fingerprint.cli --store profiles classify --text-file reply.txt

# 评测 / 采集 / 重建档案
python examples/evaluate.py              # 留出测试 + 混淆矩阵
python examples/collect_rounds.py        # 断点续采（读 ZHIPU_KEY / DS_KEY）
python -X utf8 -B -m pytest tests/ -q    # 30 个测试
```

### 当前档案库（profiles/）

智谱 GLM（同厂商多版本档案）、deepseek-chat、deepseek-reasoner —— 均来自 2026-11-05/06 真实探针采集，含关思考 / 默认 / 思考内容三信号面与 logprobs 面。

### 诚实边界

- **闭集判定**：只认识校准过的模型；未见过的模型会被拒绝或误配
- **防不了刻意伪装**：系统提示词可带偏统计特征；GLM 端点无 logprobs 时无法 100% 防住
- **同厂相邻版本单条判定不可靠**（61%），整组投票可靠
- **阈值校准**：单条判定厂商级误指控 18.8%（不达标）；**只有投票级判定才可作为指控依据**（0/4 误指控）
- 前提是你**持有合法的 API Key**，自行对目标端点采集
- 证据链 JSONL 记录时间戳 / 票型 / 逐探针分数 / 判定，**绝不含 key**

### 关于 BuddyMe

这个工具最初的日常用途，是验收 [BuddyMe 智能体工作平台](https://www.buddyme.cn) 的模型路由——确认接进来的端点是不是真的在跑它声称的模型。

[BuddyMe](https://www.buddyme.cn) 是一个**免费的通用智能体工作平台**：一句自然语言说需求，AI 自主规划、执行并交付成果。支持多供应商模型运行时热切换、定时任务、25+ 技能与跨会话记忆，每日免费 4500 万 Token、1000 次调用，Windows 客户端一键安装。如果你在挑选模型接入服务、或者想验证自己的中转链路，欢迎配合本工具使用，也欢迎来 [buddyme.cn](https://www.buddyme.cn) 体验。

---

## English

### What is it

A **statistical fingerprint** tool that answers one question: *is the model behind this endpoint really the one it claims to be?* It sends a battery of probes to an OpenAI- or Anthropic-compatible endpoint, extracts style / n-gram / embedding / logprob features per reply, and votes across probes against locally calibrated profiles.

**No cooperation from the provider needed, pure CPU, no GPUs.** Currently ships with official-domain-collected baselines for **Zhipu GLM** and **DeepSeek Chat / Reasoner**; more models will be added.

### Quick start

Download `dist/EndpointTester.exe` (Windows, ~32 MB, no Python required), enter the base URL, API key and claimed model id, then click *Test* → *Monitor*. Start from source with `python -m llm_fingerprint.cli` if you prefer CLI.

### How it works

Per reply we extract: 31-dimensional writing-style features, character-trigram distribution, optional BGE sentence embeddings (probe-conditioned, so only same-probe replies are compared — topic is factored out), and optional top-k token logprobs. Probe-level verdicts are aggregated by majority vote; near-ties between sibling models downgrade the verdict to vendor level (e.g. "Zhipu GLM family").

### Measured results

| Mode | Exact-model accuracy | Vendor-level accuracy |
|---|---|---|
| Probe battery + per-probe vote (production path) | **4/4 = 100%** | 4/4 |
| Merged battery verdict | 3/4 = 75% | 4/4 |
| Single reply | 41/67 ≈ 61% | ~90% |

Includes two hard pairs: sibling versions from the same vendor, and `deepseek-chat` vs `deepseek-reasoner`. Small sample (4 models × 34 replies) — treat as an initial reference, not a guarantee.

### Adversarial robustness (honest limits)

- **Prompt-based persona / few-shot style injection defeats the text fingerprint 48% of the time** and can flip a vendor-level verdict.
- **Logprob fingerprints defeat it 30/30 (100%)** on endpoints that expose token distributions — token probabilities cannot be mimicked by prompting.
- GLM endpoints expose no logprobs, so for that family only confidence + multi-signal voting reduces (not eliminates) the risk.
- Single-reply vendor-level false-accusation rate is **18.8%**; only battery-level voting is safe to use as evidence.

### Known fact, not a bug

Across all three signal surfaces (thinking off / default / reasoning text), **sibling versions from the same vendor are statistically indistinguishable** (SNR ≈ 0.03). The tool correctly degrades to a vendor-level verdict. Also observed: **deepseek-reasoner with thinking disabled behaves like the deepseek-chat base model**.

### Configuration

Keys are read **only from environment variables** — never from files in the repo, never logged:

```bash
export ZHIPU_KEY=...   # Zhipu GLM
export DS_KEY=...      # DeepSeek
```

### Relation to cryptographic verification (FlashProve)

Anthropic's open-source [FlashProve](https://github.com/anthropics/FlashProve) proves with zkSNARKs that a reply came from a *committed set of weights* — mathematically certain, but the provider must run a GPU proving stack (~70–95× the cost of plain inference). This project is a **probabilistic** check that needs no cooperation from the provider and runs on CPU.

They are complementary: use cryptographic proofs when the provider is willing to publish auditable proofs; use statistical fingerprints for self-checks of gateways, routing audits, and acceptance testing where no proof is available.

### About BuddyMe

This tool started as the day-to-day model-routing acceptance check for [BuddyMe](https://www.buddyme.cn) — verifying that an endpoint really runs the model it claims.

[BuddyMe](https://www.buddyme.cn) is a **free, general-purpose AI agent platform**: say what you need in natural language, and it plans, executes and delivers on its own. It offers runtime hot-switching across multiple model providers, scheduled tasks, 25+ skills and cross-session memory, 45 million free tokens and 1000 calls per day, and a one-click Windows client. If you are choosing a model gateway or checking your own proxy chain, pair this tool with your workflow — and feel free to try BuddyMe at [buddyme.cn](https://www.buddyme.cn).

### License

MIT. Baselines were collected from each vendor's official API under each vendor's terms of service; no model weights are included. BGE-small-zh-v1.5 embedding model is from BAAI (MIT).
