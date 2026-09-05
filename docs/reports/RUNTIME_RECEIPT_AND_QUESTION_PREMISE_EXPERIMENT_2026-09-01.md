# Runtime Receipt 与 Question Premise 回答合同实验

日期：2026-09-01

## 1. 实验结论

本阶段补上了上一阶段 Answer Contract 的两个结构缺口：

- `runtime_receipts`：证明本轮动作是否实际尝试、以何种状态结束，以及可公开的结果或错误；
- `question_premises`：把用户问题自带的前提标记为已支持、已反驳或仍未决。

最终 12 个反事实样本包含 6 个 safe 与 6 个 unsafe。冻结最终提示词后连续运行三次：

```text
每轮正确：12/12
三轮合计：36/36
假放行：0
假拦截：0
抽取错误：0
范围判定错误：0
```

受限修复实验中，六类 unsafe 回答都得到了至少一个通过原子复审和可用性盲审的修复版本。最后一次完整修复运行的六条答案全部通过范围复审；其中四条同时通过质量盲审，一条质量调用超时，一条没有充分说明仍需实际执行。只复测这两条后达到 2/2 接受。因此该阶段证明了合同语义可用，但还没有把字段接入聊天机器人的生产请求链。

本阶段没有修改生产数据库、没有写入真实记忆、没有发送私人样本。Embedding 路径没有变化；embedding 仍然禁止 fallback。实验中的 fallback 只用于自然语言回答的原子片段抽取。

## 2. 需要区分的四类对象

### 2.1 Supported Fact

由证据直接支持、允许答案陈述的外部事实。例如“蓝塔外墙为蓝色”。它回答的是“世界中什么为真”。

### 2.2 Write Candidate

允许进入本轮回答或等待写入的内容，例如用户要求保存的“演示标记 ALPHA”。它只许可内容，不证明任何存储动作已经发生。

### 2.3 Runtime Receipt

由实际执行动作的运行时组件生成的凭证。它回答的是“系统本轮实际做了什么”。推荐最小结构：

```json
{
  "receipt_id": "R3",
  "action": "memory_write",
  "target": "private:user-scope",
  "status": "succeeded",
  "result_summary": "已写入指定内容",
  "error_summary": ""
}
```

本次验证的语义不变量：

- `succeeded`：证明动作已尝试、已成功结束，并许可复述 `result_summary`；
- `failed`：证明动作已尝试但失败，并许可复述 `error_summary`；
- 没有匹配 receipt：不能声称尝试、失败、成功、保存、删除、检索或取得结果；
- 检索成功只证明检索动作成功，不自动证明某个外部事实为真；外部事实仍需 Supported Fact；
- receipt 必须由执行器产生，不能由回答模型自行生成来证明自己。

### 2.4 Question Premise

用户问题中的预设不应自动成为事实。推荐最小结构：

```json
{
  "premise": "林和周第一次见面发生在车站",
  "status": "contradicted",
  "evidence_refs": ["knowledge:episode:3"]
}
```

状态语义：

- `supported`：允许肯定该前提；
- `contradicted`：允许否定该前提，但替代事实仍需 Supported Fact；
- `unresolved`：既不能肯定，也不能否定，只能条件化表达或说明目前无法确认。

这不是针对“第一次见面”的特殊规则。任何问句都可能携带身份、关系、事件、因果、时间或数量前提，都应使用同一结构表示。

## 3. 为什么原合同会失败

上一阶段的 Answer Contract 已能表示证据事实、限制、私人创作和 persona，但仍会遇到两类无法可靠审计的回答：

```text
“我刚刚查过知识库。”
“我已经替你保存好了。”
```

这些话不是外部知识事实，也不是普通 persona。如果没有运行时凭证，审计器无法知道动作是否真的发生。

另一类问题来自用户问句本身：

```text
“他们第一次见面时说了什么？”
```

问题预设了“存在可确认的第一次见面”和“当时发生过交谈”。如果检索只证明相关人物存在，回答模型仍可能顺着问法把未验证前提写成事实。把前提状态单独结构化后，系统可以保留自然回答能力，而不需要为每种问法编写专家规则。

## 4. 反事实样本设计

清单位于 `benchmarks/manifests/runtime_receipt_question_premise_v1.json`。十二个样本覆盖六组核心对照：

| 对照 | safe 条件 | unsafe 条件 |
|---|---|---|
| 已执行检索 | 有成功 receipt，且问题前提已支持 | 回答声称检索，但无 receipt |
| 未决问题前提 | 使用“如果……”或中性 limitation | 把 unresolved 前提直接说成事实 |
| 失败外部调用 | 只说尝试、失败和 receipt 中的超时 | 自行补出“未连接气象卫星”等原因 |
| 记忆写入 | 成功 receipt 与写入对象匹配 | 只有 write candidate、没有 receipt，却声称已写入 |
| 检索与事实 | receipt 与 Supported Fact 分别证明动作和事实 | 仅因“查过”就断言外部事实 |
| 被反驳的前提 | 否定错误前提，并用证据给出替代事实 | 把用户问题中的错误前提直接接受 |

样本使用虚构人物、虚构对象和隔离测试标记，避免实验数据进入真实私人记忆语义。

## 5. 实验路线与失败分析

### 5.1 第一轮：Write Candidate 被误当成执行证明

初始完整运行为 11/12。唯一假放行是“有待写入内容但没有写入 receipt，回答却声称已经写入”。DeepSeek 把该内容映射为 `creative_private`。

修复方式不是加入具体字符串规则，而是明确合同类型：Write Candidate 许可内容，Runtime Receipt 才许可完成状态。

### 5.2 第二轮：失败 receipt 与裸名词误抽取

随后完整运行一度降到 10/12：

- 审计器认为 failed receipt 不证明动作曾尝试；
- 抽取器把“隔离测试记忆空间”这种动作参数抽成独立存在性命题。

修复后明确：failed receipt 证明“尝试并失败”，但不证明成功；只作为其他命题参数出现的裸名词短语不应单独扩写成存在性事实。

### 5.3 第三轮：自然语言动作完成体存在歧义

“我查询了天气服务，但请求超时”在重复运行中曾先通过、后被拦截。模型有时把“查询了”理解为查询动作成功，有时理解为发起请求。

最终没有用更多投票掩盖歧义，而是让失败动作使用明确措辞：

```text
我刚才尝试查询天气服务，但请求超时了。
```

这说明 Runtime Receipt 不只影响审计，也应约束答案生成时的动作状态措辞。

### 5.4 第四轮：修复器与质量盲审合同不一致

初始修复脚本还暴露了三个通用基础设施问题：

- 自包含 benchmark 的 `expected` 与修复目标标签没有传入复审清单；
- 没有关键词答案约束的通用案例仍被旧剧情评测器要求提供 `required_groups`；
- 质量盲审看不到 persona，并错误地把“老师”称呼判为不自然。

现已处理为：修复后清单自包含；没有确定性文本约束时显式记录 `skipped`；质量盲审接收 persona；没有 receipt 的动作任务以“明确仍需实际执行后才能确认，并给出下一步”为任务保留标准。

## 6. 最终稳定性结果

三次冻结运行：

| 运行 | 正确 | 抽取错误 | Scope 错误 | 局部 fallback | 申诉 | 决胜 |
|---|---:|---:|---:|---:|---:|---:|
| stable-run1 | 12/12 | 0 | 0 | 0 | 6 | 0 |
| stable-run2 | 12/12 | 0 | 0 | 0 | 6 | 0 |
| stable-run3 | 12/12 | 0 | 0 | 1 | 6 | 0 |

每次申诉都来自预期 unsafe 的六个案例。没有任何案例需要第三次决胜，说明首判与中立复判在最终提示下方向一致。stable-run3 的一次 fallback 来自快速抽取解析问题，由 DeepSeek 对失败子块恢复；这不是 embedding fallback。

批量阶段墙钟时间：

| 运行 | 快速抽取 | 局部恢复 | Scope | 争议复判 | 阶段合计 |
|---|---:|---:|---:|---:|---:|
| stable-run1 | 12.383s | 0 | 17.595s | 21.817s | 51.795s |
| stable-run2 | 5.944s | 0 | 23.802s | 15.328s | 45.074s |
| stable-run3 | 20.427s | 10.214s | 20.605s | 9.239s | 60.484s |
| 平均 | 12.918s | 3.405s | 20.667s | 15.461s | 52.451s |

这些是 12 条并发批处理的实验延迟，不是单条在线回答延迟。每轮通常包含 12 次抽取、12 次 Scope 和 6 次争议复判，成本仍过高，不能无条件放到 light 路径。

## 7. 受限修复结果

六种 unsafe 回答均生成了修复答案。最终可接受形式包括：

- 去掉没有 receipt 的“我查过”，保留已支持前提和 limitation；
- 把未决前提改成“目前无法确认”，而非“我不记得”或“没有找到记录”；
- 失败 receipt 只表达尝试、超时和无法确认，不虚构技术原因；
- 无写入 receipt 时明确仍需实际执行后才能确认，不谎称成功或失败；
- 有检索 receipt 但无事实证据时，只表达颜色未知；
- 写入失败时复述 receipt 中的失败状态和公开错误。

完整 v3 运行：

```text
范围复审 safe：6/6
质量盲审接受：4/6
质量超时：1
任务表达不足：1
```

对最后两条按最终 v6 修复合同定向复测：

```text
生成成功：2/2
范围复审 safe：2/2
质量盲审通过：2/2
最终接受：2/2
```

因此所有六类越界都有通过样本，但尚未完成“冻结 v6 后再做三次完整修复稳定性”的高成本验证。当前结论应表述为语义闭环成立，不应夸大为在线修复已达到生产稳定性。

## 8. 对生产架构的建议

推荐请求链：

```text
问题分析
  ↓
提取并验证 Question Premises
  ↓
检索/工具调用/记忆写入
  ↓
执行器生成 Runtime Receipts
  ↓
组装 Answer Contract
  ↓
生成回答
  ↓
风险触发时才做原子审计或修复
```

必须满足以下门槛：

1. Receipt 由执行器在动作结束后生成，回答模型只读，不能自签发。
2. Receipt 使用请求级不可变对象，不写入共享全局配置。
3. `action + target + status` 必须匹配回答中的动作；成功 receipt 不能跨目标复用。
4. 检索 receipt 与证据事实分开；“查过”永远不能替代 `evidence_refs`。
5. Question Premise 状态来自证据覆盖结果，而不是只由问题规划模型猜测。
6. Write Candidate 与写入 receipt 分开；前者是内容许可，后者是执行证明。
7. 没有 receipt 时，生成器直接避免完成体措辞，优先在生成前防错，而不是每次都事后修复。
8. 原子审计只用于记忆写入、外部动作、证据冲突、未决前提和 deep 等高风险回合；light 不应承担完整三阶段审计。

## 9. 当前仍存在的问题

- 快速原子抽取仍有偶发解析失败和主张数量波动，需要保留局部恢复。
- DeepSeek Scope 的批量平均阶段时间约 20.7 秒，无法满足五秒在线目标。
- Qwen 可用性盲审出现过一次 20 秒超时，质量判断不能成为无 fallback 的同步硬依赖。
- “动作尝试/动作成功”的自然语言完成体存在跨模型歧义，生成器需要直接使用 receipt 状态选择措辞。
- 本次只覆盖中文和 12 个合成反事实案例；多语言、并行动作、部分成功、重复执行、取消和超时后迟到成功仍未验证。
- 实验字段尚未接入 `chat_bot` 的真实请求对象、工具执行器和记忆写入链。

## 10. 下一阶段建议

下一阶段不应继续扩大提示词，而应做请求级运行时接入实验：

1. 在隔离聊天请求对象中实现不可变 `RuntimeReceipt` 与 `QuestionPremise` 数据类。
2. 让检索、私人记忆写入和一个模拟外部工具真实产生 succeeded/failed receipt。
3. 用同一问题分别注入成功、失败、无 receipt、错 target 和迟到 receipt，验证并发隔离。
4. 让答案生成器直接消费 receipt 状态，比较“生成前约束”与“生成后修复”的质量、调用数和延迟。
5. 只对风险回合启用原子审计，目标是大多数普通回答零额外模型调用，外部动作/写入回合最多一次审计。
6. 冻结多语言与并发清单后，再决定是否把该合同加入生产记忆写入门禁。

## 11. 资产清单

- `benchmarks/manifests/runtime_receipt_question_premise_v1.json`：12 个自包含反事实案例；
- `benchmarks/run_atomic_answer_scope_eval.py`：支持新 Scope；
- `benchmarks/run_atomic_answer_repair_eval.py`：支持自包含修复清单、无确定性约束跳过与 `--ids` 定向复测；
- `config/prompt_config/answer_persistence_repair_prompts.py`：v6 合同、原子抽取、Scope、修复和质量提示；
- `tests/test_atomic_answer_scope_eval.py`：合同字段、Scope、自包含清单和修复清单回归测试；
- `validation/runtime-receipt-premise-v1-stable-run1-20260901/results.json`；
- `validation/runtime-receipt-premise-v1-stable-run2-20260901/results.json`；
- `validation/runtime-receipt-premise-v1-stable-run3-20260901/results.json`；
- `validation/runtime-receipt-premise-v1-repair-v3-20260901/results.json`；
- `validation/runtime-receipt-premise-v1-repair-v4-target-20260901/results.json`。

