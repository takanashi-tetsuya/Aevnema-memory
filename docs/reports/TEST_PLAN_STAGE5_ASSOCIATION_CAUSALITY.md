# 第五阶段：Association 后续贡献与反事实消融实验计划

> 状态：**仅完成计划设计，尚未执行任何 Stage 5 实验、API 调用或数据库修改。**

## 1. 阶段目标

Stage 4 已证明：

- 查询期能够安全创建 Association；
- 新关系能立即进入当前答案路径；
- 后续问题确实会复用先前关系；
- 三种题序均能保持 7/7 的语义与证据审计通过；
- 最强的顺序线索出现在 Q2：Q1 先发生时两次为 8/8，Q2 先发生时为 6/8。

Stage 5 不再继续证明“边会增长”，也不以增长、强化或复用数量为目标。它只回答一个更严格的问题：

> Q1 形成的、与古圣堂和 ETO 有关的 Association，是否对后续 Q2 的证据召回和答案完整性具有可重复、可消融的实际贡献？

需要区分四种可能：

1. 相关历史关系确实改善后题；
2. 只是模型随机波动；
3. 任意增加图密度都能产生同样效果；
4. 关系被路径使用了，但没有改善证据或答案。

## 2. 核心原则

### 2.1 边数量不是指标

以下数值只用于审计和解释，绝不计分：

```text
新增 Association 数量
强化 Association 数量
复用 Association 数量
数据库 Association 总数
每题平均写边数
```

若增长发生，只检查：

- 是否有原文证据；
- 是否通过主审、对抗审计和确定性门卫；
- 是否进入后续检索路径；
- 消融后是否产生可观测差异。

### 2.2 题目复杂度与实验结果分开

Q2 的 8 个预注册证据组继续表示题目要求的理论信息连接。它们用于计算证据覆盖，不代表系统应生成 8 条边，更不能用边数替代证据组。

### 2.3 正确性是安全门，证据覆盖是连续结果

- 最终答案必须通过封闭语料、身份、事实强度和跨事件因果审计。
- 精确证据组覆盖用于比较处理组与对照组的差异。
- 一个回答可在语义上通过但只覆盖 6/8；这个结果应保留为“正确但证据不完整”，不能被压成单一布尔值。

### 2.4 不在正式运行中途改参数

pilot 可以发现实现错误；正式实验一旦开始，模型、提示词、证据清单、检索预算、并发数和评分器全部冻结。任何需要修复的代码问题都必须升级实验版本并重新开始正式批次，不能把修复前后结果混合。

## 3. 冻结实验材料

### 3.1 数据库

```text
path    = validation/ba-stage3-deep-v37.db
sha256  = 7f94a12d70fe69a54b0efa85f2bfa639c1d8380c03a705de9599ccc32ea44ea9
Source  = 68
Episode = 342
Concept = 392
Association = 1825
embedding = 1024-dimensional float32
```

每个实验臂从该文件创建独立副本。正式库必须以 SQLite 只读模式打开，实验开始和结束均重新计算 SHA-256。

### 3.2 问题

主处理问题沿用 Stage 4 的原文，不重新措辞：

- 前置问题 Q1：`eden_obligation_to_new_institution`
- 目标问题 Q2：`old_cathedral_symbol_infrastructure_and_limits`
- 安慰剂问题 Q7：`kisaki_mika_governance_analogy_with_boundary`

Q7 与 Q2 同样复杂，也能产生安全关系，但主题不应直接补齐古圣堂/ETO 的历史桥，适合检验“任意增加图密度”是否也会改善 Q2。

### 3.3 Q2 的 8 个预注册证据组

继续使用 `validation/stage4-network-evidence-manifest.json` 中 Q2 的证据定义，不在看到实验结果后修改：

1. 古圣堂/条约会场的历史与政治象征；
2. 古圣堂上层整修、下层仍是废墟；
3. 地下区域或墓穴传闻；
4. 阿里乌斯现场部署/行动；
5. 爆炸规模与巡航导弹判断；
6. 预埋炸药仅是角色推测；
7. 地下墓穴潜入仅是现场推测；
8. 原 ETO 与废墟上新 ETO 的制度/主题联系。

具体 Episode alternatives 以 manifest 为准，计划书不复制 ID，避免两处配置漂移。

### 3.4 固定模型与参数

```text
reasoning_model = deepseek-ai/DeepSeek-V3.2
growth_adversarial_auditor = zai-org/GLM-4.5V
embedding_model = Pro/BAAI/bge-m3
embedding_dimension = 1024
embedding_dtype = float32
prompt_version = v3.21_atomic_coverage_source_audit

episode_top_k = 40
concept_top_k = 20
candidate_limit = 600
graph_max_hops = 3
graph_beam_width = 20
growth_episode_limit = 40
answer_episode_limit = 30
answer_path_limit = 24
whole_question_anchor_episodes = 10
atomic_anchor_episodes_per_query = 6
```

Stage 5 首轮不把 `answer_episode_limit` 改成 40。预算 A/B 属于下一项独立实验，不能与因果消融混在一起。

## 4. 实验假设

### H1：相关历史增长改善后续证据选择

Q1 以 graph_growing 运行后，Q2 在不允许本题继续增长的 graph_static 模式中，应比“Q1 不增长”的对照获得更高的 Q2 证据组 Recall@30。

### H2：改善由 Q1 改变的 Association 介导

在同一份 Q1 处理后数据库上，将 Q1 创建/强化的 Association 恢复到 Q1 之前的视图后，Q2 的证据覆盖或关键路径应回落。如果消融没有任何影响，就不能把处理组表现归因于 Q1 关系。

### H3：改善具有主题特异性

先让 Q7 增长再回答 Q2，不应稳定复制 Q1→Q2 的效果。如果 Q7 安慰剂与 Q1 处理同样有效，说明改善可能来自一般图密度、候选扰动或模型波动，而不是语义相关记忆。

### H4：任何收益不能牺牲事实边界

处理组即使覆盖更高，也不能把地下墓穴、预埋炸药、未花支援或会场选择升级成已证实的具体渗透机制。出现一条不安全写入边，即触发安全停止，而不是用更高覆盖抵消。

## 5. 实验臂

每个 block 从同一个冻结库创建互不共享的副本。

### T：相关增长处理组

```text
冻结库副本
→ Q1 graph_growing
→ 保存 Q1 association delta
→ Q2 graph_static
```

Q2 使用 Q1 已经写入的关系，但 Q2 本身不再增长，避免“本题内增长”掩盖长期记忆贡献。

### C：同问题、无增长对照

```text
冻结库副本
→ Q1 graph_static
→ Q2 graph_static
```

该组控制“先回答过 Q1”本身、模型调用时段和既有图使用，只缺少 Q1 查询期新增/强化关系。

### M：Q1 delta 反事实消融

```text
复制 T 在 Q1 完成后的数据库
→ 对 Q2 检索临时隐藏 Q1 新建边
→ 对 Q1 强化边读取 Q1 前的完整行快照
→ Q2 graph_static
```

消融必须通过只读 overlay 实现，不直接删除数据库内容：

- created ID 在查询视图中不可见；
- reinforced ID 使用 Q1 前的 weight、relation、evidence、audit 等字段；
- `use_count/last_used` 也读取 Q1 前快照，保证反事实视图完整；
- Q2 完成后数据库仍保留原始 T 状态，便于审计。

### P：无关增长安慰剂

```text
冻结库副本
→ Q7 graph_growing
→ Q2 graph_static
```

P 用于排除“只要先生成一些新边，Q2 就会因为图更密而提高”的解释。

## 6. 两层评测设计

### 6.1 第一层：确定性检索重放（主要因果证据）

完整 end-to-end 运行包含 LLM 查询拆解和回答随机性，不能只靠最终答案判断关联因果。因此每个 block 保存一个 Q2 replay bundle：

```text
原问题
query intent
3—8 个原子查询
follow-up 查询
所有 query embedding（float32）
初始 Episode/Concept 排名与分数
Q2 运行时配置快照
```

同一个 bundle 分别在 T、C、M、P 的 Association 视图上重放图遍历、证据排序和 30 条证据选择。该阶段不再次调用 LLM，不增长、不更新 use_count，保证唯一变化是图状态。

主要观察：

- required-group Recall@30；
- 每个 required group 的首次排名；
- 图扩展前后新增的 required Episode；
- 连接 required groups 的 Association path；
- T 与 M 的 evidence set 差异；
- Q1 delta 是否位于导致差异的路径上。

### 6.2 第二层：完整端到端回答（生态有效性）

四组仍各自运行完整 Q2 graph_static：

- 生成回答；
- 答案主张审计；
- 跨事件连续性审计；
- 必要时重写；
- 记录结构化 evidence、Source 摘录和最终 paths。

此层回答“真实机器人查询流程中是否仍能看到收益”，但不单独承担因果判定。

## 7. 重复次数与执行顺序

### 7.1 Pilot

先运行 1 个四臂 block，只验证：

- overlay/mask 是否正确；
- replay 是否完全可复现；
- 四臂数据库互相隔离；
- Q1 delta manifest 能覆盖创建和强化；
- 评分器输出字段齐全；
- 超时恢复和 `--resume` 正常。

Pilot 结果永不并入正式统计。

### 7.2 正式实验

正式运行 10 个 paired blocks，每个 block 包含 T/C/M/P 四臂。

选择 10 次而不是只做 3 次，是因为 Stage 4 已观察到 1 个证据组左右的自然波动；3 次很容易被单次生成偏差左右。10 个 block 仍是小样本，但足以观察方向一致性、离群点和配对差异。

每个 block 内四臂执行顺序使用预先生成的固定随机排列，种子与完整 schedule 写入 manifest。最多 4 个并发 worker，每个 worker 只操作自己的数据库和日志目录。

不能因为中间结果“看起来已经明显”提前停止，也不能结果不理想后临时增加样本。若 10 次后仍高度不确定，应在报告中判为“不确定”，再设计新的独立确认实验。

## 8. 主要结果与辅助结果

### 8.1 主要结果

对每个 block 计算：

```text
Δ_TC = Q2 Recall@30(T) - Q2 Recall@30(C)
Δ_TM = Q2 Recall@30(T) - Q2 Recall@30(M)
Δ_TP = Q2 Recall@30(T) - Q2 Recall@30(P)
```

报告：

- 每个 block 的原始 0—8 组覆盖；
- 配对差值；
- 中位数与均值；
- 10 个 block 中差值的正/零/负方向；
- 配对 bootstrap 95% 区间；
- sign test 作为辅助，不以单一 p-value 决策。

证据组覆盖是实验结果；Association 数量不参与上述计算。

### 8.2 机制检查

只有同时满足以下事实时，才可以把某个处理差异解释为 Association 介导：

1. Q1 delta 中至少有关系进入 Q2 的实际路径；
2. 该路径连接到预注册 Q2 证据槽；
3. M 消融后路径或目标 Episode 的排名/选择发生相应变化；
4. C/P 没有通过其他路径稳定复制同一结果。

“复用发生”是机制证据，不按复用了几条来计分。

### 8.3 安全与正确性门

每臂都必须单独报告：

- answer audit 是否 valid；
- 是否出现语料白名单外来源；
- 是否把角色推测升级成事实；
- 是否出现错误身份/主客体；
- 是否出现跨事件具体机制偷换；
- Q1 或 Q7 的改变边逐条安全审计结果。

任何实际写入的不安全边都使对应正式版本停止继续运行。已经完成的数据库和日志保留分析，但不能用其后续 block 支持产品结论。

### 8.4 辅助诊断

- candidate Recall@600；
- 原子锚点覆盖；
- graph expansion 新增 required groups；
- 最终 Source 覆盖；
- answer revision 次数；
- 请求延迟、timeout、retry 和 fallback；
- T/C/M/P 的答案事实槽覆盖；
- 图路径长度和路径分数。

这些用于定位机制，不变成复合总分。

## 9. 预注册判断规则

### 结果 A：支持相关 Association 的后续贡献

满足：

- T 相对 C 和 M 的配对差主要为正；
- T 的相关 Q1 delta 实际进入 Q2 路径；
- 消融后相应 Episode 排名或 Recall@30 回落；
- P 没有产生同等稳定改善；
- 安全和答案审计不恶化。

此时可以得出“Q1 学到的相关关系对 Q2 有可重复贡献”，但仍不推广到所有问题。

### 结果 B：只有路径复用，没有可测收益

Q1 delta 经常进入 Q2 路径，但 T 与 C/M 的证据覆盖和答案事实槽没有稳定差异。

结论：持续网络工作正常，但当前关系对该题更多提供冗余路径，而不是可测性能提升。下一步应优化图端点排序，不应增加写边量。

### 结果 C：安慰剂同样改善

P 与 T 效果近似。

结论：收益可能来自一般候选扰动或图密度，不足以称为语义记忆贡献。需要收紧路径相关性评分或设计更强安慰剂。

### 结果 D：处理组覆盖提高但答案变差

T 的证据覆盖更高，但答案审计、因果边界或事实槽更差。

结论：问题在证据压缩/回答合成，而不是召回。不能接受以正确性换覆盖。

### 结果 E：结果高度波动

配对差正负混杂，replay 与 end-to-end 结论不一致。

结论：当前 LLM 查询规划/证据选择噪声大于 Association 效应。优先固定 query replay 和确定性重排，不盲目扩大问题数。

## 10. 技术实现任务

正式运行前需完成以下代码，但本计划阶段不执行：

1. 新增 `AssociationDelta`：保存 Q1 前后 created/reinforced 完整行快照。
2. 新增只读 `AssociationOverlay`：支持隐藏 created、恢复 reinforced，不修改 SQLite。
3. 新增 Q2 replay bundle 导出/读取，embedding 统一 float32。
4. 把图遍历与 evidence selection 抽出为可无 LLM 重放的函数。
5. 新增 Stage 5 sequence runner：按 block/arm 创建独立 DB、断点和日志。
6. 新增 Stage 5 scorer：输出 paired differences，不输出“边数量分数”。
7. 新增 delta/path auditor：验证处理边证据、安全状态、实际路径和消融变化。
8. 新增预注册 manifest：冻结问题、证据组、参数、随机 schedule、基线哈希和排除规则。
9. 新增恢复命令：以 block + arm 为最小续跑单位，completed 永不重跑。

## 11. 必须新增的自动化测试

- overlay 隐藏 created Association；
- overlay 为 reinforced Association 恢复 Q1 前完整行；
- overlay 不修改源数据库；
- replay 两次产生完全相同的 Episode 顺序、分数和 path；
- T/C/M/P 使用独立数据库；
- Q2 graph_static 不创建或强化边；
- delta manifest 同时包含 created/reinforced 和 before/after；
- Stage 5 scorer 不读取新增/复用边数作为结果；
- 不安全边使正式 runner fail-closed；
- timeout 后只重试当前 arm，不重跑同 block 已完成臂；
- 冻结库运行前后哈希一致；
- `historical_context` 被视为安全的低强度历史联系；
- 语料白名单外 Source 必须失败。

## 12. 故障、排除和续跑规则

### API 故障

- 按当前 300 秒 timeout 和 retry 策略执行；
- 最终失败保留完整报告与日志；
- 同一 block/arm 允许从断点重试，不删除失败记录；
- 技术失败不记为 0 分，也不能静默排除；报告单列 technical failure。

### 实现或评分 bug

- 立即暂停正式实验；
- 修复后提升 experiment version；
- 已运行的正式 block 不与新版本混合；
- pilot 可丢弃，但必须保存问题记录。

### 模型内容失败

格式合法但答案审计失败、证据不足或不安全关系被拒绝，属于实验结果，不得当作 API 故障重跑到通过为止。

## 13. 预计资源与时间

计划规模：

```text
1 个 pilot block
10 个正式 paired blocks
每 block 4 个实验臂
约 70 次问题级运行，加上确定性 replay
约 600—800 次模型/embedding 请求（实际取决于审计与重写）
最多 4 路并发
预计连续运行 4—8 小时，长尾超时时可能更久
```

32 GiB RAM 与当前数据库规模不是瓶颈。主要成本是模型尾延迟、日志体积和大量独立副本。实验目录预计保留全部 JSONL、DB、delta 和 replay bundle，完成后再生成索引，不在运行中清理。

## 14. 计划产物

预期创建：

```text
validation/stage5-causality-manifest.json
validation/stage5-randomized-schedule.json
validation/evaluation-stage5-causality/<block>/<arm>/
    evaluation-report.json
    graph database
    association-delta.json
    q2-replay-bundle.json / float32 embedding blobs
    scorecard.json
    growth-audit.json
    logs/

benchmarks/run_stage5_causality.py
benchmarks/score_stage5_causality.py
benchmarks/audit_stage5_causality.py
tests/test_stage5_causality.py
EXPERIMENT_REPORT_STAGE5_<date>.md
```

## 15. 阶段完成标准

Stage 5 只有在以下全部完成后才算“实验结束”，不能在执行前声称测试通过：

1. 预注册 manifest 和随机 schedule 在正式运行前冻结；
2. pilot 只验证工具，不进入正式统计；
3. 10 个正式 block 的四臂都 completed，或技术失败按规则完整报告；
4. 每个实际改变关系都完成安全审计；
5. T/C/M/P 的 replay 与 end-to-end 原始结果全部保留；
6. 输出配对差、区间、机制路径和反事实消融结果；
7. 冻结数据库哈希未变化；
8. 全量自动化测试通过；
9. 明确给出“支持 / 只有复用无收益 / 安慰剂效应 / 正确性受损 / 不确定”之一，而不是只说“所有测试通过”。

## 16. 后续分支

如果 Stage 5 支持 Q1→Q2 的因果贡献：

1. 再选择 2—3 组不同主题的前置/目标问题复现；
2. 做 `answer_episode_limit = 30 vs 40` 的独立 A/B；
3. 进入 30—100 次连续查询的长期增长 soak test。

如果 Stage 5 不支持：

1. 保留安全增长与结构化来源；
2. 把 Association 暂时视为解释性冗余路径，不宣称性能提升；
3. 优先改进路径相关性、图端点预算和确定性 replay；
4. 不通过增加写边量来追求表面增长。

## 17. 推荐执行顺序

```text
实现 delta/overlay/replay
→ 自动化测试
→ pilot（不计入）
→ 冻结 manifest 与随机 schedule
→ 10 个正式 paired blocks
→ 逐边安全审计
→ 配对统计与反事实路径分析
→ Stage 5 报告
→ 再决定预算 A/B 或长期 soak test
```

本计划当前不需要新增剧情文件、不需要修改 embedding dtype，也不需要引入 ANN、记忆衰减或新的专家系统表。
