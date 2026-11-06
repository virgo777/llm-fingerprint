# 两条路线：密码学证明 vs 统计指纹

> 本文说明本项目与 Anthropic 开源的可证明推理系统 FlashProve 的关系与取舍。

## FlashProve：把推理装上密码学证明

**来源：** Anthropic Frontier Red Team 于 2026-11-02 发布的可证明推理（provable inference）研究原型，Apache 2.0 开源。

- 原文：<https://www.anthropic.com/research/provable-inference>
- 技术报告：<https://www.anthropic.com/document/flashprove-technical-report>
- 代码：<https://github.com/anthropics/FlashProve>

### 它回答什么

用户向模型提问时，无法检查回答是否真的来自开发者声称的那个模型 —— 可能是路由服务为省钱偷偷换上的另一个模型，也可能是攻击者篡改 token 流后的产物。单凭输出判断不可靠，回答质量高也不行。

FlashProve 用 **zkSNARK**（零知识简洁知识论证）给出密码学证明：

1. **离线阶段**：提供方发布模型权重的密码学承诺（一段锁定权重的简短指纹，权重本身保持隐藏）
2. **在线阶段**：对每条回答生成证明，断言「对所承诺的模型执行计算，会生成这些输出 token」

任何持有承诺值、提示词和回答的人，无需权重、算力有限，即可验证。

### 关键数字（Llama 4 Maverick，400B MoE，4-bit）

| 指标 | 数值 |
|---|---|
| 证明速度 | 单张 H100 约 **110 token/s** |
| 证明大小 | 1024 token 请求约 **19 MB** |
| 验证成本 | 24 核 CPU **17 秒**（单线程 21 秒），无需 GPU |
| 相对 vLLM 的 GPU 时间开销 | **70–95×** |
| 离线承诺 | 约 39 分钟，可用于约 1600 万次请求 |

对照此前工作：VerInf 对同一模型约 0.02 token/s、证明 93.6 GB；zkLLM 约 2.5 token/s。FlashProve 提升了约 4 个数量级。

### 四条安全性质

- **完备性**：诚实证明会被接受
- **健全性**：伪造证明在实践中不可能
- **零知识性**：除命题本身外不泄露权重、中间值、专家选择（架构不隐藏）
- **简洁性**：证明小、验证快

并用 Lean 形式化验证了这三条顶层性质，附带经正确性证明的 Lean 验证器。

### 适用场景

- 控制 AI 发展节奏（向监管者证明硬件确实在跑推理而非训练）
- 检测服务器被入侵 / 模型冒充 / token 流篡改
- 文本来源证明（如「失败证明」：某模型消耗 ≥X token 仍没找到漏洞）
- 模型审批合规

### 局限

- 快速可执行代码无形式化保证；证明针对贪心解码的整数模型
- 上下文长度上限 1024 token
- 侧信道（证明耗时）仍可能泄露信息
- **需要提供方部署整套证明栈**，成本约为普通推理的 70–95 倍

## llm-fingerprint：统计指纹

| 维度 | FlashProve | llm-fingerprint |
|---|---|---|
| 保证类型 | 密码学确定性（可证明） | 统计概率（可误判） |
| 是否需要提供方配合 | **必须**（部署证明器、发布承诺） | **不需要**（只发 API 请求） |
| 验证成本 | CPU 数秒 | CPU 数秒，零 GPU |
| 提供方成本 | 推理 × 70–95 | 一次探针调用 |
| 判定对象 | 「承诺的权重」 | 「已知档案库中的哪个模型」 |
| 可否被提示词欺骗 | 不能 | 文本指纹可（48%）；logprobs 面不可 |
| 能否在监管场景单独作指控 | 可以 | **不可以**（单条误指控 18.8%） |

**结论：两者互补，不是替代。**

- 提供商愿意公开可审计 → FlashProve
- 自查网关路由、验收采购、怀疑中转站换模型，且对方不配合 → 本项目

真正需要「确定性」的场景（监管、司法、审计），密码学证明是唯一答案；日常工程判断用统计指纹成本低几个数量级。

## 引用

> Rinberg, R., Allain, A., Chiesa, A., Christ, M., Dalal, R., Fenzi, G., Mayer, B., Weissenberg, G. (2026).
> *Provable Inference: zkSNARKs for Large Language Models* (FlashProve). Anthropic Frontier Red Team.
> <https://www.anthropic.com/research/provable-inference>
