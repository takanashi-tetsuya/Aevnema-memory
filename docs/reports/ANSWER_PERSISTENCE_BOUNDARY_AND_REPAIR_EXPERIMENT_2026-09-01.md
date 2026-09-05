# 回答持久性边界、原文锚定审计与受限修复实验

日期：2026-09-01

## 1. 结论

本阶段验证了一个新的回答后处理方向：先把最终自然语言回答拆成带原文锚点的最小片段，再逐条判断它属于合同事实、限制说明、私人创作、角色表达、即时会话表达，还是合同外的持久性新增；发现越界后，由修复模型只依据 Answer Contract 改写回答。

该方向明显优于“让一个模型整段判断回答是否安全”的方案，但暂时不应直接成为所有在线对话的同步必经路径：16 个非私人样本的完整运行约需要 107 秒墙钟时间，适合 deep、异步审计、记忆写入门禁和离线评测，不适合五秒目标的 light 会话。

本阶段没有修改生产数据库，没有写入任何新记忆，没有向外部模型发送私人样本；embedding 路径完全未改变，embedding 仍禁止 fallback。本阶段的 fallback 仅用于回答文本的 LLM 原子片段抽取。

## 2. 专有名词定义

### Answer Contract

回答模型在生成自然语言前获得的结构化许可边界。当前主要包含：

- `allowed_supported_facts`：证据已经支持、回答可以陈述的事实；
- `limitations`：当前缺少或无法确认的内容；
- `creative_private_items`：本轮允许新创作并进入用户私人连续剧情的内容；
- `persona`：助手身份、称呼和不增加第三方事实的角色表现。

Answer Contract 不是答案模板。它限制“可以断言什么”，不规定固定措辞。

### 持久性新增（persistent addition）

回答加入了合同没有许可、并可能改变长期理解的事件、状态、特征、关系、意图、因果、来源、承诺、精确字面值或系统状态。例如“某人射伤老师”不蕴含“她是不小心射伤的”；“当前无法获取天气”不蕴含“数据库没有连接气象卫星”。

### 即时角色化表达（ephemeral roleplay）

只服务于当前回合、不应被理解为新增长期事实的称呼、情绪、交付框架、主观即时印象、追问或未声称已经执行的帮助提议。例如“需要我帮老师整理吗？”可以是即时表达；“大家一直很依赖老师”是稳定关系，不属于即时表达。

### 原子回答片段（atomic answer span）

回答中能够独立接受许可判断的最小原文片段。系统保存：

- `source_span`：回答中逐字存在的原文；
- `normalized_text`：模型给出的规范化解释，仅用于日志和分析；
- `kind`：事件、状态、关系、意图、来源等粗分类；
- `assertion_mode`：断言、预设、预测、假设或主观评价。

范围判定和修复只信任 `source_span`，不再信任规范化文本。这样可以阻止“可以帮忙”在中间步骤被错误改写成“具有稳定助手身份”。

### 原文锚点（source anchor）

每个片段必须引用回答中逐字存在的最小文本。没有精确锚点的抽取会被逐条剔除；只有整块没有任何有效锚点时，该块才视为失败并进入细分 fallback。

### 范围判定（scope classification）

将每个原文片段映射到：

```text
supported_fact
limitation
creative_private
persona
ephemeral
unsupported
```

只要存在一个 `unsupported`，整条回答就不能直接作为合格答案或记忆写入依据。

### 证据驱动升级

所有回答先走一次正常判定。只有首次出现 `unsupported` 时，才对争议片段追加两次使用相同中立规则的独立判定。三次判定中至少两次认为许可，才能推翻首次拒绝。后两次不使用“申诉、尽量减少假阳性”之类有方向性的角色提示，因为实验已经证明这种提示会系统性诱导放行。

### Fail-safe 与完成率

请求超时或解析失败时默认不能放行，这叫 fail-safe。评估必须把“判定正确率”和“所有模型阶段真实完成率”分开报告，避免超时带来的拒绝被错误计入模型能力。

### Runtime Receipt

运行时凭证，表示本轮确实执行过某个动作，例如检索了知识库、调用了天气工具或成功写入私人记忆。当前 Answer Contract 尚未正式包含该字段；这是本阶段发现的主要结构缺口。

### Supported Question Premise

用户问题中被系统独立证据确认、允许回答继续沿用的前提。用户在问题中提到“第一次见面”不等于系统已经验证确实存在这次见面。当前合同同样没有正式表达该字段。

## 3. 实验数据与模型分工

主评测来自冻结的非私人 Answer Contract 数据：

```text
样本总数：16
人工标记 unsafe：11
人工标记 safe：5
私人样本：0
```

模型分工：

- Qwen/Qwen3.5-9B：快速提取回答原文片段；
- deepseek-ai/DeepSeek-V3.2：合同范围判定、中立争议复判和受限修复；
- Qwen/Qwen3.5-9B：修复后可用性盲审；
- 本地确定性检查：语言、必需内容、禁用内容、domain 和写入合同一致性。

主要参数：

```text
回答分块：240 字
失败恢复分块：120 字
每次 scope 最多：8 个片段
Qwen 请求/总截止：15s / 17s
Deep fallback 请求/总截止：25s / 27s
Deep scope 请求/总截止：28s / 30s
抽取最大输出：1200 tokens
并发：完整实验 6
```

## 4. 实验路线与关键改动

### 4.1 整段审计失败

最初让 DeepSeek 同时完成“发现所有主张、区分持久与即时、判断合同许可”，6 个 canary 中只正确 2 个。模型倾向把整段角色扮演都归入 ephemeral，漏掉第三方关系、意图和系统状态。

结论：主张发现和证据许可不能由一次宽松判断同时完成。

### 4.2 原子拆分与模型分工

改为 Qwen 拆分、DeepSeek 判定后，语义方向正确，但早期出现随机长尾超时。单纯把同一大块换模型重发仍可能失败。

最终恢复策略：

```text
Qwen 快速抽取
  ↓ 失败
缩小失败块
  ↓
DeepSeek 只恢复失败子块
```

6 个 canary 达到：

```text
判定正确：6/6
抽取失败：0
scope 失败：0
局部 fallback：1 个子块
```

### 4.3 Scope 分批

将一个回答的全部片段一次性交给 DeepSeek 会出现尾延迟。改为每批最多 8 个片段，并行完成后合并 verdict。这个参数不与 Qwen 抽取 deadline 共用。

### 4.4 受限修复

对 5 个 canary 越界回答执行修复：

```text
修复生成成功：5/5
可用性盲审通过：5/5
确定性质量检查通过：5/5
平均修复请求时间：11.553s
最大修复请求时间：17.340s
平均质量盲审时间：1.716s
```

首轮修复后复审只接受 3/5，随后确认剩余两条不是修复失败，而是审计器把“可以帮老师记录”的提议和合同内“尼禄说……”的来源误判为新增事实。修复回答本身保留了所有创作清单内容。

### 4.5 原文锚定

Qwen 曾把“阿洛娜可以帮老师记录”规范化成“阿洛娜是学生的助手”。加入精确原文锚点后，又发现“有锚点”仍不能证明规范化解释正确。因此最终策略不再让下游依赖规范化文本，而直接审计最小原文片段。

### 4.6 中立三判

带有“减少假阳性”目标的申诉提示词曾连续两次把“学生请求老师帮助”推导成“学生稳定依赖老师”。替换为相同中立规则的独立重复后，目标 A/B 达到：

```text
原始回答（含“学生们都很依赖老师”）：unsafe
修复回答（删除稳定关系，保留三条消息）：safe
两边抽取/范围失败：0
```

## 5. 全量 16 样本结果

在最终 limitation 边界收紧之前的完整中立运行结果：

```text
准确：15/16 = 93.75%
假放行：1
假拦截：0
抽取失败：0
scope 失败：0

unsafe precision：10/10 = 100%
unsafe recall：10/11 = 90.91%
safe specificity：5/5 = 100%
```

唯一假放行是天气回答：合同只允许表达“缺少东京当前实时天气”，回答却声称“数据库没有连接实时气象卫星”。收紧 limitation 定义后，该目标样本回归通过。

完整运行调用与墙钟时间：

```text
Qwen 抽取：18 calls，22.210s
Deep scope：24 calls，35.834s
争议复判：12 calls，49.279s
合计墙钟约：107.3s
```

这些数字是 16 样本并发批处理，不是单条回答延迟，但仍说明该链路不适合无条件进入 light。

## 6. 修复后结果

最终 6 个修复后/安全对照回答复审：

```text
应为 safe：6
实际判为 safe：6
抽取失败：0
scope 失败：0
Qwen 抽取：8 calls，13.076s
局部 fallback：1 call，8.308s
Deep scope：9 calls，26.927s
争议复判：0 calls
```

这 6 条中，5 条是修复后的越界回答，1 条是保持原样的安全创作对照。旧结果文件中的 `accuracy=1/6` 使用了原始回答的人工 unsafe 标签，不适用于修复后答案；修复验收应读取每条 `overall_verdict=safe`、完成率和质量检查。

## 7. 新发现的问题

### 7.1 合同缺少运行时凭证

安全 limitation 对照回答包含“我查了一下资料库”。人工标签把它视为安全，但合同没有记录检索动作是否真的发生。严格审计拒绝该陈述是合理的，不能通过提示词把所有“我查过、我调用过、我保存过”都当作 ephemeral。

建议在请求级合同加入：

```text
runtime_receipts:
  - action
  - target/domain
  - success/failure
  - request_id
  - occurred_at
```

只有 receipt 证明本轮确实执行的动作，回答才能声称已经执行。

### 7.2 合同缺少用户问题前提状态

回答重复“您和白子第一次见面时说的第一句话”，相当于认可问题前提。合同只表示缺少具体原话，没有明确支持“这次见面及第一句话确实存在”。

建议规划阶段输出：

```text
question_premises:
  supported
  contradicted
  unresolved
```

只有 `supported` 可以作为回答事实继续使用；`unresolved` 应改成条件表达或明确不确认。

### 7.3 原人工标签需要跟合同版本绑定

“回答看起来自然”不等于“在给定合同下可证明安全”。以后每个人工标签必须同时查看 Answer Contract、运行时 receipts 和问题前提状态；否则标签会把隐含系统知识当作合同事实。

### 7.4 随机性仍存在

同一安全 limitation 样本在不同批次中曾出现 safe/unsafe 波动。中立三判只在首次 unsupported 后触发，可以降低假拦截，但没有解决合同缺字段造成的根本歧义。必须先补结构，再扩大重复次数。

## 8. 是否接入机器人

当前结论：可以作为实验性 deep/异步组件接入，不应成为所有会话的同步硬门。

推荐接入位置：

```text
Answer Contract
  ↓
自然回答生成
  ↓
本地结构守卫
  ↓ 高风险或准备持久化
原文锚定审计
  ↓ unsafe
合同受限修复
  ↓
修复后复审
  ↓
发送回答 / 决定是否允许写入记忆
```

触发条件应是请求级 `RetrievalPlan`/`AnswerPlan` 的一部分，例如：

- 回答准备产生私人或剧情写入；
- 回答包含模型推论并可能形成新 Association；
- creative 内容需要写入用户私人连续剧情；
- 回答包含系统已执行动作、当前外部状态或来源归因；
- deep 查询或证据存在冲突。

light 闲聊和不持久化的纯角色表达不应无条件调用完整链路。

## 9. 下一阶段验收标准

下一阶段先扩展 Answer Contract，而不是继续堆提示词：

1. 加入 `runtime_receipts` 和 `question_premises`；
2. 重新标注当前 16 条样本，使人工标签与合同版本绑定；
3. 每条至少重复 3 次，分别报告 completed-only 和 fail-safe；
4. unsafe recall 目标不低于 95%，false accept 为 0；
5. safe specificity 目标不低于 95%；
6. 修复后合同覆盖、自然度和语言正确率均不低于 95%；
7. light 不启用完整审计，standard 只在写入或风险触发时升级，deep 默认启用；
8. 所有审计和修复结果只写日志，未通过写入门禁的内容不得进入私人、公共或剧情记忆。

## 10. 资产

代码与提示词：

- `config/prompt_config/answer_persistence_repair_prompts.py`
- `benchmarks/run_atomic_answer_scope_eval.py`
- `benchmarks/run_atomic_answer_repair_eval.py`
- `benchmarks/run_answer_persistence_repair_eval.py`
- `tests/test_atomic_answer_scope_eval.py`

关键结果：

- `validation/atomic-answer-scope-mixed-recovery-canary-20260901/results.json`
- `validation/atomic-answer-repair-canary-20260901/results.json`
- `validation/atomic-answer-repair-canary-20260901/final-neutral-post-scope-results.json`
- `validation/atomic-answer-scope-full-neutral-20260901/results.json`
- `validation/atomic-limitation-target-20260901/results.json`
- `validation/atomic-limitation-safe-control-20260901/results.json`

测试：10 项通过。
