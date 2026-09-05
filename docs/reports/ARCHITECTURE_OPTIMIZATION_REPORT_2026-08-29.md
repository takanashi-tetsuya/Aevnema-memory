# 长期记忆 Demo 架构与 Token 优化报告

日期：2026-08-29（Asia/Tokyo）  
实现版本：`v3.24_balanced_orchestration`

## 1. 结论

本轮没有改动 Source、Episode、Concept、Association 的数据语义，没有降低 float32
精度，也没有取消回答审计或 Association 双审计。优化集中在“同一批证据被模型重复阅读”这一层：

1. 导入时把时间边界与 Episode 粒度复审合并为一次联合复审；
2. 二次理解只发送一个压缩后的多语言推理视图，相邻 Episode 作为共享上下文发送一次；
3. 精确命中已有 Concept alias 时，不再先生成一个随后被丢弃的 embedding；
4. 细粒度 Concept 准入先确定性排除“不可能晋升且不可能安全复用”的低相似单例；
5. 查询重排由固定三遍改成首遍覆盖后按风险分流；
6. 默认并发准备 4 个 Source，数据库写入仍保持有序串行；
7. 增加 `strict / balanced / lean` 三个可逆运行档。

查询端已经完成 23 题在线实测：总 token 从历史 strict 的 1,083,856 降到
779,553，减少 **28.1%**。导入端基于完整导入真实 usage 和真实提示重建得到的保守估算是
8,727,786 → 7,451,362，减少 **14.6%**；这部分还没有做完整语料的 strict/balanced
双份在线导入，因此必须视为估算，不是最终生产数据。

## 2. 专有名词

- **编排（orchestration）**：决定什么时候调用哪一个模型、传入哪些证据，以及何时追加复审的程序逻辑。
- **首轮覆盖（initial coverage）**：让模型从 Candidate@100 中为每个原子问题标出直接证据的第一次重排。
- **独立覆盖复审（independent coverage audit）**：第二个不读取首轮结论的证据侦察，用来降低注意力漂移。
- **压缩器（compressor）**：把覆盖短名单压缩到最终 Top-20，同时保护各证据槽。
- **原子问题（atomic query）**：理论上能由一条或一组明确 Episode 独立证明的最小证据槽。
- **确定性预筛选（deterministic prefilter）**：只用计数和相似度等机器规则决定“不需要问 LLM”的条目。
- **严格路径（strict path）**：首轮覆盖、独立覆盖、压缩器全部执行的历史三遍流程。
- **自适应路径（adaptive path）**：先做首轮覆盖，再根据可见风险决定是否继续调用模型。
- **提示字符数**：模型请求 system + user 文本的字符总量，是 token 的稳定代理，但不等同于供应商实际 token。
- **usage token**：API 响应中报告的实际 `prompt_tokens / completion_tokens / total_tokens`。

## 3. 优化前的成本基线

### 3.1 完整导入

Stage14 完整资产包含 208 个文件、735 个 Source、3,582 个 Episode。导入日志记录了
4,667 次请求和 8,727,786 token：

| 阶段 | 请求数 | 总 token |
|---|---:|---:|
| Episode 提取 | 689 | 2,304,804 |
| 时间边界复审 | 280 | 1,100,274 |
| 粒度复审 | 217 | 719,193 |
| 二次理解 | 356 | 2,139,111 |
| Concept 提取 | 1,064 | 1,896,923 |
| Embedding | 2,034 | 479,744 |
| 关系判断 | 26 | 85,730 |
| 其他 | 1 | 2,007 |

时间/粒度复审与二次理解合计占 3,958,578 token，即完整导入的约 45.4%。因此本轮没有优先优化
NumPy 或 SQLite；真正的大头是重复阅读 Source、候选 Episode 和相邻上下文。

### 3.2 23 题查询

历史 strict 检索使用 152 次请求、1,083,856 token。其中三遍重排本身消耗：

| 重排阶段 | 请求数 | 总 token |
|---|---:|---:|
| 首轮覆盖 | 23 | 382,165 |
| 独立覆盖复审 | 23 | 364,391 |
| Top-20 压缩 | 23 | 212,183 |

三遍合计 958,739 token，占该轮检索总 token 的约 88.5%。查询解析和下一跳规划不是第一优先级。

## 4. 新架构

### 4.1 导入路径

```text
文件适配与自然分片
        ↓
4 路并发 Source 准备
        ↓
Episode 首遍提取
        ↓
时间风险或粒度风险？ ──否──→ 直接进入落库
        │是
        ↓
一次联合 Episode 质量复审
        ↓
Episode float32 embedding + SQLite/RAM
        ↓
Concept 批量提取
        ↓
精确 alias 先复用 ──→ 仅未解析 Concept 批量 embedding
        ↓
关系候选跨 Source 合批
        ↓
SQLite 串行提交
        ↓
必要 Episode 的二次理解
（压缩 Source + 共享相邻上下文）
```

并发只覆盖模型准备阶段。SQLite、Concept 去重、Association 写入和 RAM 索引更新仍然按 Source
顺序执行，因此没有引入多线程共享 sqlite3 cursor 的问题。

### 4.2 查询路径

```text
问题解析 → 初始向量/稀疏召回 → 下一跳规划 → 候选集合
                                           ↓
                                  首轮证据覆盖（必做）
                                           ↓
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
             高复杂/明确缺口         覆盖槽 ≤ 1          其余覆盖充分
                    │                      │                      │
             独立覆盖 + 压缩             仅压缩               直接使用
                    │                      │                      │
                 strict                compress                none
                    └──────────────────────┼──────────────────────┘
                                           ↓
                            Association 遍历/增长、回答与事实审计
```

`balanced` 当前规则：

- 原子问题达到 24 个：`strict`；
- 首轮模型明确报告 `missing_aspects`：`strict`；
- 首轮有效 coverage group 不超过 1 个：`compress`；
- 其他情况：`none`。

这些规则只减少候选重排的重复判断。回答逐主张审计、跨事件连续性审计，以及增长 Association 的主审和
对抗复审没有取消。

## 5. 各项实现和依据

### 5.1 联合 Episode 复审

历史日志中：

- 273 个 Source 触发时间审计；
- 210 个 Source 触发粒度审计；
- 93 个 Source 同时触发两者；
- 两类风险的并集为 390 个 Source。

历史共调用 497 次；联合后首遍理论调用 390 次。用历史候选和 Source 重建新提示，审计提示字符从
2,929,570 降到 2,097,482，减少约 28.4%。联合提示同时保留时间拆分、同场景合并、纯标题清理、
未知人物保护和译文冲突规则。

### 5.2 二次理解共享上下文

旧批量提示把每一个目标 Episode 的 `nearby` 列表都完整放入 item。多个目标相邻时，同一 Episode
摘要会重复很多次；同时该阶段发送的是原始多语言 Source，而首遍提取已经有经过验证的压缩推理视图。

新路径：

- Source 使用与首遍 Episode 提取相同的 zh-CN/ja/en/ko 优先推理视图；
- 对同一个 Source 的所有目标求相邻窗口并集；
- 每个相邻 Episode 按 ID 只发送一次；
- 每个待修订 item 只保留自身 Episode。

对历史 356 个请求重建提示后，字符数从 3,835,835 降到 2,334,501，减少 39.1%。请求数不变，
因此 completion token 不做下降假设。

### 5.3 Concept embedding 延迟到精确 alias 检查之后

旧 `resolve_many_deferred()` 先为一个 Source 的所有 Concept 生成 embedding，随后才检查
canonical name 是否已经是数据库 alias。参与者和常见组织反复出现时，大量向量会立即被丢弃。

新流程先在同一解析锁内处理精确 alias，只有未解析项才组成 embedding batch。它不会改变 Concept
合并语义，只减少 embedding 输入；本报告没有把这部分收益计入 14.6% 导入估算。

### 5.4 Concept 准入预筛

Stage14 细粒度 Concept 实验的准入日志包含 1,024 个新候选、52 个 LLM batch：

- LLM 历史判定 transient 740 个；
- reuse 181 个；
- promote 103 个，但只有满足跨 Episode 图效用门的条目最终能晋升。

在 `promotion_min_distinct_episodes=2` 的实验条件下，历史中“单例但被 LLM 成功 reuse”的最低 Top-1
相似度是 0.5969045。新安全门使用 0.595：低于该值的单例既不能晋升，也没有历史成功复用样本。

历史回放会预筛 54 个候选，其中 40 个原本是 transient，14 个虽被模型叫做 promote，随后仍会被
图效用门降为 transient；没有历史 reuse 被跳过。配合 batch 20 → 48 和提示压缩，估计请求
52 → 21，提示字符减少 11.3%。这是保守门，不追求最大限度删除 LLM 调用。

### 5.5 查询自适应复审

23 题真实在线结果：

| 指标 | strict 历史 | adaptive 在线 | 变化 |
|---|---:|---:|---:|
| 全部 API 请求 | 152 | 130 | -14.5% |
| 重排模型调用 | 69 | 44 | -36.2% |
| Prompt token | 1,030,592 | 732,769 | -28.9% |
| Completion token | 53,264 | 46,784 | -12.2% |
| Total token | 1,083,856 | 779,553 | **-28.1%** |

实际分流是 `strict=9 / compress=3 / none=11`。它和历史日志离线预测的 `8 / 4 / 11` 有一题差异，
原因是模型在新运行中对该题返回了 `missing_aspects`，系统按规则升级到 strict。这证明分流会响应当前
模型结果，而不是只按题目长度硬编码。

## 6. 质量验证与限制

### 6.1 23 题一次在线结果

历史 strict：

- Candidate@100 mean：0.97343；
- Selected@20 mean：0.92150。

本次 adaptive：

- Candidate@100 mean：0.96800；
- Selected@20 mean：0.89734。

不能只看这两个平均值就判定 adaptive 退化：

1. `old_cathedral...` 的 Candidate@100 从 1.0 降到 0.875。Candidate 集合在 adaptive 重排之前已经
   形成，这个变化来自重新调用查询解析/下一跳规划后的随机差异，不是省略重排审计造成的；
2. `paragraph_gap_03` 本次报告缺口并实际走了完整 strict 路径，但 Selected 从 1 降到 0；
3. 同一轮中，日奈复杂题从 0.778 升到 0.889，未花三视角从 0.667 升到 1.0，说明重排有明显随机波动。

### 6.2 冻结输入重复试验

`paragraph_gap_03` 使用历史完全相同的 intent 和 follow-up 重复 3 次：

- 3/3 Candidate=1、Selected=1；
- 2 次走 `compress`；
- 1 次因模型自报缺口走 `strict`。

所以一次在线 0 分没有形成稳定的 adaptive 回归证据。更准确的结论是：本轮观察到约 28.1% 的真实
token 降幅，省略复审的 `none/compress` 路径没有发现稳定失败；但整个重排系统仍有模型方差，正式
对照应固定 intent/follow-up 并做多次重复。

### 6.3 真实导入烟雾测试

使用 `main/31060.json` 和独立数据库：

- 4 Source；
- 18 Episode；
- 0 failed task；
- 19 次 API 请求；
- 32,635 token；
- 230.853 秒；
- Episode 提取 4 次、Concept 提取 4 次、联合质量复审 1 次、关系判断 2 次、embedding 8 次。

该文件没有触发二次理解，因此只能证明联合复审、并发准备、批量关系和落库链路正常；不能把它当成
二次理解压缩的在线 A/B 质量证明。

## 7. 运行档与回退

```dotenv
MEMORY_OPTIMIZATION_PROFILE=balanced
```

- `balanced`：本轮默认；联合导入审计、压缩二次理解、Concept 预筛、自适应查询复审；
- `strict`：恢复单线程准备、拆分时间/粒度审计、完整二次理解上下文、无 Concept 预筛、固定三遍查询；
- `lean`：导入保持优化，查询只做首轮覆盖；不建议用于高风险或正式评测。

旧实验脚本凡是声明完整覆盖审计的地方已经显式固定为 `strict`，避免默认值变化污染历史对照。

## 8. 测试状态

- 131 个确定性单元/集成测试全部通过；
- Python 全量 compileall 通过；
- 新增测试覆盖联合 Episode 审计、自适应复审分流、Concept 预筛和 strict profile 回退；
- 23 题 adaptive 在线评测完成；
- 波动题冻结输入重复 3 次完成；
- 一份真实剧情文件 balanced 导入完成。

## 9. 尚未解决的问题

1. **查询规划仍固定调用。** 23 题下一跳规划约 102,998 token。有 9 个历史问题最终返回空 follow-up，
   但在看到首轮节点之前无法可靠判断其是否为空。下一阶段可把“首轮覆盖 + 是否需要 follow-up”合成一次
   调用；这需要重新安排检索顺序，风险高于本轮优化。
2. **首轮 Candidate 重排仍很贵。** 首轮覆盖约占 38.2 万 token，但它承担质量核心，不能直接删除。
   可以研究按 source_key 分组、短字段名和两级候选摘要，不过必须重新验证 Recall@20。
3. **二次理解压缩缺少在线 A/B。** 历史提示重建显示 -39.1%，但需要选择确实触发未知人物修订的文件，
   分别用 strict/balanced 导入并人工审计修订差异。
4. **联合 Episode 审计需要扩大质量样本。** 当前在线烟雾只触发一次联合复审。应从历史 93 个双风险
   Source 中分层抽样，比较 Episode 数量、时间拆分、未知人物保护和关键事实覆盖。
5. **JSONL 日志仍很大。** 完整日志是用户要求的诊断资产，不消耗 API token，但会增加磁盘与解析成本。
   后续可以在不删除原日志的前提下按运行结束 gzip，或另建紧凑索引；本轮未改变日志完整性。

## 10. 建议的下一阶段

1. 冻结 20—30 个历史双风险 Source，做 strict/balanced 成对导入；
2. 冻结模型输入，分别测 Episode 边界、人物身份、Concept/Association 数量和实际 usage；
3. 对 23 题再做至少 3 轮冻结 intent/follow-up 的 strict/adaptive 对照；
4. 只有在质量置信区间不退化后，才尝试合并下一跳规划与首轮证据覆盖；
5. 保持 float32、SQLite source of truth、Episode/Concept 分离索引和 Association generation 规则不变。

本轮优化已经把最明显的重复推理移除，但没有把所有模型调用都当成坏事。新的原则是：首次证据覆盖必做，
事实安全审计必做，只有“再次阅读同一候选且当前没有可见风险”的调用才被省略。
