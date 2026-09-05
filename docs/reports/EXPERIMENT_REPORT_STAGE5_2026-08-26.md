# Stage 5 实验报告：Association 后续贡献与反事实消融

## 1. 结论

Stage 5 已实际完成，不是计划或模拟结果。

正式实验包含 10 个 paired blocks，每个 block 都运行：

- T：Q1 增长后回答 Q2；
- C：Q1 不增长后回答 Q2；
- M：从 T 的 Q1 后数据库派生，但在 Q2 临时隐藏 Q1 新边、恢复强化边旧行；
- P：无关 Q7 增长后回答 Q2。

主要结论是：

> Q1 创建的 Association 会被后续 Q2 路径复用，但在当前检索预算与证据分配算法下，没有提高 Q2 的预注册证据组 Recall@30。

因此本阶段属于“路径复用存在，但没有可测收益”，不支持“Q1 学到的关系已经改善了 Q2”的更强结论。

这不是 Association 完全无作用：10 个 blocks 中有 7 个处理组让 Q1 delta 进入 Q2 路径，4 个 block 的遮罩改变了具体 Episode 顺序或集合，4 个 block 的 delta 路径触及预注册证据 Episode。但 10 个 block 中没有一次改变 0—8 证据组覆盖。

## 2. 实验完整性

冻结库：

```text
validation/ba-stage3-deep-v37.db
SHA-256 before = 7f94a12d70fe69a54b0efa85f2bfa639c1d8380c03a705de9599ccc32ea44ea9
SHA-256 after  = 7f94a12d70fe69a54b0efa85f2bfa639c1d8380c03a705de9599ccc32ea44ea9
```

正式运行：

```text
experiment_version = stage5-causality-v1
analysis_revision  = stage5-analysis-v1.1
pilot              = 1 block，排除于正式统计
official           = 10 blocks × 4 arms
workers            = 4
开始               = 2026-08-26 08:44:26（Asia/Tokyo）
完成               = 2026-08-26 11:54:10（Asia/Tokyo）
连续墙钟时间       ≈ 3 小时 9 分 44 秒
```

完整性检查：

- 10/10 blocks completed；
- 40/40 Q2 末次证据审计 valid；
- T/P 的所有实际改变边都通过双模型审计和本地结构审计；
- M 没有暴露任何 Q1-created Association；
- replay 没有调用模型、没有更新 use_count、没有修改 Association；
- 所有 evidence Source 都在封闭语料白名单中；
- 冻结库前后哈希一致；
- 73 项自动化测试通过。

## 3. 确定性 replay 主结果

每个 block 只生成一次 Q2 query intent、原子查询、follow-up 和 float32 query embeddings，然后在 T/C/M/P 四个 Association 视图上重放同一检索输入。

| Block | T | C | M | P | T-C | T-M | T-P |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 2 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 3 | 6/8 | 6/8 | 6/8 | 6/8 | 0 | 0 | 0 |
| 4 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 5 | 6/8 | 6/8 | 6/8 | 6/8 | 0 | 0 | 0 |
| 6 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 7 | 6/8 | 6/8 | 6/8 | 6/8 | 0 | 0 | 0 |
| 8 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 9 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |
| 10 | 7/8 | 7/8 | 7/8 | 7/8 | 0 | 0 | 0 |

三项差值的结果完全相同：

```text
mean                  = 0
median                = 0
positive / zero / negative = 0 / 10 / 0
paired bootstrap 95%  = [0, 0]
sign test p           = 1
```

这排除了“正式结果只是不同查询规划随机性”的解释：在相同 query bundle 下，Q1 增长、Q1 不增长、反事实遮罩和无关增长没有产生证据组覆盖差异。

## 4. 完整端到端结果

| Block | T | C | M | P |
|---:|---:|---:|---:|---:|
| 1 | 7/8 | 7/8 | 6/8 | 6/8 |
| 2 | 7/8 | 7/8 | 7/8 | 6/8 |
| 3 | 6/8 | 7/8 | 6/8 | 7/8 |
| 4 | 6/8 | 7/8 | 6/8 | 6/8 |
| 5 | 7/8 | 7/8 | 7/8 | 6/8 |
| 6 | 7/8 | 7/8 | 7/8 | 6/8 |
| 7 | 6/8 | 7/8 | 7/8 | 7/8 |
| 8 | 6/8 | 7/8 | 7/8 | 7/8 |
| 9 | 7/8 | 7/8 | 6/8 | 6/8 |
| 10 | 7/8 | 7/8 | 7/8 | 7/8 |

配对 Recall 差：

| 对比 | mean | median | 正/零/负 | bootstrap 95% | sign test p |
|---|---:|---:|---:|---:|---:|
| T-C | -0.050 | 0 | 0/6/4 | [-0.0875, -0.0125] | 0.1250 |
| T-M | 0.000 | 0 | 2/6/2 | [-0.0500, 0.0500] | 1.0000 |
| T-P | 0.025 | 0.0625 | 5/2/3 | [-0.0500, 0.0875] | 0.7266 |

端到端层没有稳定处理收益：T-M 均值为零；T-C 甚至偏负；T-P 的小幅正均值跨零且方向混杂。结合确定性层全零，最合理解释是查询规划、答案证据选择与生成带来的自然波动，而不是 Association 因果收益。

## 5. Association 实际做了什么

关系数量只作审计，不计分：

```text
T created = 75, reinforced = 0
P created = 73, reinforced = 0
```

所有改变边均为：

```text
claim_level = supported_inference
audit_status = dual_accepted
```

T 的主要 relation_key 是：

```text
evidence_bridge      25
thematic_contrast    19
thematic_response    12
historical_context    5
role_evidence_bridge  4
```

Q2 实际复用的 T 边主要连接：

- Episode 274 ↔ 84：原 ETO 的目的与古圣堂成立仪式；
- Episode 123 ↔ 84：第一次公会议/戒律守护者历史与原 ETO；
- Episode 123 ↔ 198：古圣堂历史与废墟上的新 ETO；
- Episode 199 ↔ 198：老师代理发起人角色与新 ETO 宣告。

这些边语义正确，也能进入路径，但端点大多本来就是 Q2 的高相关证据。它们提供的是冗余解释桥和更可读的路径，而不是把未召回的证据带入 Top-30。

## 6. 真正的瓶颈：候选已经找到，证据分配丢掉了它

Q2 的 candidate Recall 在 10/10 blocks 都是 8/8，但最终 Top-30 只有 6/8 或 7/8。

最稳定的漏项是第 5 组：

```text
Episode alternatives = [139, 140]
含义 = 阿里乌斯现场部署/行动
candidate 命中 = 10/10 blocks
最终选中 = 0/10 blocks
candidate first rank = 13—33
```

其中 Episode 140 在多数 blocks 还是图扩展带入的节点。它不是“完全没找到”，而是在最终 evidence allocation 阶段被排除。

当前选择器把 `answer_episode_limit = 30` 的 3/4，即 22 个位置优先给 `preferred_episode_ids`。这些 preferred IDs 来自整题与原子查询 anchor；随后再为路径端点留位置。结果是：

- 第 5 组仅在 2 个 blocks 进入 atomic anchor 列表；
- 两次位置分别是第 26 和第 30，已经超过 22 个 anchor 的基础配额；
- 即使它在候选层排到第 13，也可能被前 22 个 anchor 加路径端点挤出。

所以现阶段不应继续增加写边量。优先级应是重新设计 Top-30 证据槽分配。

## 7. 查询规划合同存在一个实现不一致

Prompt 要求：

```text
search_queries 输出 3—8 个原子证据问题
follow-up 为未解决槽位生成 2—6 个查询
```

但代码实际：

```text
QueryIntent.from_dict: search_queries[:4]
QueryEngine._plan_followup_queries: followups[:4]
```

因此硬问题即使有 8 个证据槽，首轮最多保留 4 个原子查询；加整题后 replay 中观察到 5 个初始查询，再加 4 个 follow-up。

这可能使后半问题槽位缺少专属 anchor。直接把上限提高到 8/6 也不够，因为现有 evidence allocator 会让更多 anchors 争夺同一个 30 条预算。查询数量与全局槽位分配必须一起修改。

## 8. 答案质量与安全

安全结果：

```text
末次答案证据审计 valid = 40/40
语料白名单外 Source     = 0
不安全实际写入边         = 0
```

精确答案标签完整性：

```text
所有要求核心词出现       = 40/40
“推测/推论/推断”组出现   = 40/40
“事实/原文/文本明确”组   = 36/40
“未知/未明确/不能确定”组 = 36/40
三组全部满足             = 32/40
```

这 8 个漏项是答案格式/完整性问题，不是越界知识或错误事实。Stage 5 审计器最初把两者合并成一个失败标志；运行后修订为 `stage5-analysis-v1.1`，分别报告：

- evidence/corpus safety；
- exact semantic label completeness。

该修订没有改变任何原始运行、证据组、Recall、配对差、schedule、问题或配置。此偏差已写入 scorecard 和 summary，不能隐去。

## 9. API 与性能资产

正式运行日志：

```text
JSONL files       = 80
JSONL bytes       = 55,062,375
events            = 1,901
requests          = 651
missing outcomes  = 0
log parse errors  = 0
```

按模型：

```text
DeepSeek-V3.2 = 443
bge-m3        = 160
GLM-4.5V      = 48
```

故障：

```text
llm_error          = 12
read timeout       = 9
remote disconnect  = 3
retry              = 12
fallback           = 0
```

所有错误都得到日志结果，没有未闭合 request。Block 10 出现连续 300 秒审计超时；候选关系被安全拒绝，查询继续完成，证明 fail-closed 路径有效。

延迟摘要：

| 类别 | count | p50 | p95 | max |
|---|---:|---:|---:|---:|
| embedding | 160 | 0.844s | 1.442s | 3.848s |
| query intent | 80 | 15.704s | 27.322s | 52.844s |
| association growth | 40 | 44.578s | 111.784s | 141.945s |
| answer | 74 | 106.135s | 186.651s | 209.787s |
| other chat/audits | 291 | 21.914s | 126.809s | 300.467s |

结论：当前规模下，本地 float32 向量检索不是瓶颈。真正的运行成本是答案生成、双模型关系审计和远端长尾。

## 10. 当前语料与分片资产

冻结库：

```text
Source      = 68
Episode     = 342
Concept     = 392
Association = 1825（冻结基线）
embedding   = float32 × 1024，BLOB 长度统一 4096 bytes
```

分片配置：

```text
target_chars  = 6000
max_chars     = 8000
overlap_chars = 800
minimum_blocks = 2
```

实际分布：

| 数据 | min | p50 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|
| Source 字符数 | 1000 | 5909 | 7990 | 8059 | 5904 |
| Episode 字符数 | 11 | 71 | 160 | 284 | 78.5 |
| 每 Source 的 Episode | 2 | 5 | 9 | 12 | 5.03 |

这说明当前 Source 分片基本按目标工作；Episode 适合精确事件召回，但存在少量 11 字符极短项，后续可单独审计其实际价值，不应现在整体加长。

## 11. 当前有效提示词资产

Prompt 版本：

```text
v3.21_atomic_coverage_source_audit
```

核心思维约束：

1. 查询解析器：把复杂问题拆成原子证据问题，保留人物、组织、时间和因果限定。
2. 多跳规划器：只用问题和第一轮节点补齐未解决槽位，不把候选当最终答案。
3. Association growth：允许 evidence bridge、thematic response/contrast 和 historical context，但必须至少连接一个 Episode。
4. 跨 source_key 默认视为不同事件；长期支援不能升级成本次具体渗透机制。
5. 一次主审加一次 GLM 对抗审计；任何一方不接受就不写库。
6. 回答器只用 Episode、Source 摘录和 Association 路径，必须区分事实、角色说法、推论和未知。
7. 答案审计逐主张检查身份、主客体、时间、因果和 Source；必要时最多修订三次。
8. 跨事件连续性审计专门阻止把早期命令、防御缺口或爆破移植到另一事件。

这些提示词在 Stage 5 中成功守住了关系与答案安全边界，但还没有强制输出统一的“明确事实 / 角色推测 / 跨片段推论 / 未确定”结构，所以出现 8/40 的精确标签漏项。

## 12. 已实现的技术资产

新增：

```text
AssociationDelta
AssociationOverlay
QueryEngine.build_replay_bundle()
QueryEngine.replay_retrieval()
Stage 5 paired runner
Stage 5 scorer
Stage 5 isolation/safety auditor
Stage 5 unit tests
```

Overlay 能：

- 隐藏 created IDs；
- 为 reinforced IDs 恢复 Q1 前完整行；
- 过滤 neighbors/get/list_temporal；
- 禁止 mark_used/upsert/delete 写回；
- 不修改处理组 SQLite。

Replay bundle 保存：

- 原问题与 QueryIntent；
- 初始/后续查询；
- float32 query embeddings；
- Episode/Concept 排名；
- 固定 seed hits；
- atomic anchor IDs；
- 检索配置快照。

## 13. 下一阶段建议

### 13.1 先做完全离线的 evidence allocator 消融

复用现有 10 个 replay bundles，不调用模型、不改 Association，比较：

1. 当前算法：22/30 位置优先给 atomic anchors；
2. anchor 上限 15/30，其余先按全局 score 填充；
3. 每个原子查询最多保留 1—2 个独占 anchor，再做全局去重；
4. 为 query-relevant informative path 端点设上限，避免冗余历史桥占满余量；
5. 最后按全局分数补齐 30 条。

主要指标仍是 8 组 Recall@30，尤其要求 Episode 139/140 进入结果；不得使用边数评分。

### 13.2 再修查询规划上限

把原子查询上限从 4 调整到 8、follow-up 从 4 调整到 6，但必须与全局证据预算一起测试。新增测试保证 prompt 合同和数据类型上限一致。

### 13.3 做边的逐条边际贡献，而不是继续增边

对进入 Q2 路径的每条 Q1 delta 单独遮罩，记录：

- 哪个 Episode 排名变化；
- 是否只是替换同一证据组内的等价 Episode；
- 是否把 candidate-only Episode 推入 Top-30；
- 是否减少而非增加冗余路径。

优先保留能改变证据槽覆盖的关系；把只连接已高排端点的边视为解释性冗余，不提高其权重。

### 13.4 固定答案边界结构

回答输出强制包含：

```text
明确事实
角色推测
跨片段推论
仍未确定
```

这可用确定性结构检查完成，不需要为了标签再调用一次 LLM。

### 13.5 暂不做的事情

- 不增加 Association 写边目标；
- 不引入 ANN；当前语料与运行中向量计算不是瓶颈；
- 不回到 float8；当前全部 float32，且无内存瓶颈证据；
- 不先扩大 answer_episode_limit 到 40；先证明 30 条预算的分配算法是否能取回已在 candidate 中的第 5 组。

## 14. 最终判断

Stage 5 验证了三个独立事实：

1. 自主增长是安全且可审计的；
2. 新关系确实能在后续问题中被路径复用，并有时改变具体 Episode 集合；
3. 当前这种复用没有改善预注册证据组覆盖。

因此下一阶段的技术重点不应继续在“增长边和可信度之间徘徊”。可信度门已经工作；现在需要解决的是：

> 已经被候选召回的正确证据，如何在有限的 30 个槽位中稳定胜出，以及什么样的 Association 才能对这个选择产生边际贡献。
