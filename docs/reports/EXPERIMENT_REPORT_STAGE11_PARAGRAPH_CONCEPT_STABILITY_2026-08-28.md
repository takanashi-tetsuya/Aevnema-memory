# Stage 11：Paragraph 与积极 Concept 稳定收益实验报告

日期：2026-08-28  
状态：实验完成；当前策略未通过启用门槛，保持回退状态

## 1. 结论摘要

本轮实验没有证明现有 Paragraph 检索和 `fine_grained` Concept 创建策略能带来稳定收益。

- Paragraph 确实补回了极少量候选证据：网络题候选覆盖由 `167/168` 提升到 `168/168`。
- 但 Paragraph 同时改变了基础锚点顺序，并大量占据最终证据槽。完整重排后的网络题 Top-20 覆盖由 A 组的 `137/168` 降至 P 组的 `132/168`。
- 积极 Concept 在网络题中没有补回任何新的必要候选，候选覆盖仍为 `167/168`；完整重排后的 Top-20 覆盖由 `137/168` 降至 `131/168`。
- Paragraph 与积极 Concept 同时启用没有产生协同收益，PC 组 Top-20 同样为 `131/168`。
- 三次独立规划中，没有任何一个网络问题满足预注册的“稳定正收益”标准；相反，多项问题出现稳定负回归。

因此当前生产/主实验默认值保持：

```text
Paragraph retrieval: disabled
Concept extraction profile: conservative
Embedding dtype: float32
```

这里的“回退”不删除实现和实验资产。Paragraph 表、分段器、索引、回填能力及 `fine_grained` 提示词仍保留，便于下一轮修复后复测。

## 2. 专有名词与评价含义

### 2.1 Paragraph

Paragraph 是介于 Source 与 Episode 之间的原文检索单元。它按自然记录边界切分 Source，允许相邻段重叠，并单独生成 embedding。

它的目标不是替代 Episode，而是在 Episode 摘要遗漏原文细节时，先通过原文段落命中 Source，再把同一 Source 中的 Episode 送入候选集合。

### 2.2 积极 Concept 创建

`fine_grained` Concept 是比原保守档更积极的概念提取策略。它要求模型逐项检查人物、别名、组织、制度、命名事件、地点子区域、物品、持续心理状态和抽象主题，软目标为每个 Episode 提取 4—10 个 Concept。

“积极”只表示候选覆盖更广，不表示这些 Concept 自动更正确、更稳定或更适合长期存储。

### 2.3 候选覆盖与最终覆盖

- 候选覆盖：必要 Episode 是否进入供后续重排使用的候选池。
- 最终覆盖：必要 Episode 是否进入最终 Top-20 或 Top-12 证据列表。
- required Episode group：同一事实可能由多个等价 Episode 表达；任意命中组内一个 Episode 即算覆盖该事实组。

候选覆盖衡量“有没有被找回来”，最终覆盖衡量“有没有在有限证据预算中被保留下来”。

### 2.4 固定层与重排层

- fixed：不调用 LLM 重排，直接使用冻结查询计划和确定性检索/锚点选择，适合定位基础召回和融合问题。
- reranked：在候选池上继续执行证据重排、覆盖缺口审计和独立复核，更接近实际查询流程。

### 2.5 稳定收益

每个问题独立生成三次查询计划。同一处理相对 A 组至少两次改善、且一次都不退化，才算该问题的稳定正收益。

整体机制还必须同时满足：

1. 至少两个不同网络问题稳定改善；
2. 网络题平均差值大于 0；
3. 范围控制题没有稳定负回归。

## 3. 四臂因果设计

| 实验臂 | Paragraph | Concept 数据 |
|---|---:|---|
| A | 关闭 | 原保守 Concept |
| P | 开启 | 与 A 完全相同 |
| C | 关闭 | `fine_grained` augmentation |
| PC | 开启 | 与 C 完全相同 |

每个问题、每次重复只生成一次 QueryIntent、初始查询、后续查询及 float32 查询 embedding；四臂共享该冻结计划，然后分别重新执行自己的检索、图遍历和重排。

这避免了某一实验臂仅仅因为模型临时换了搜索措辞而获益。

## 4. 数据资产与不变量

冻结基线：

```text
Source       68
Episode      342
Concept      392
Paragraph      0
Association 1825
```

处理后：

| 实验臂 | Source | Episode | Concept | Paragraph | Association |
|---|---:|---:|---:|---:|---:|
| A | 68 | 342 | 392 | 0 | 1825 |
| P | 68 | 342 | 392 | 541 | 1825 |
| C | 68 | 342 | 554 | 0 | 2190 |
| PC | 68 | 342 | 554 | 541 | 2190 |

Paragraph 回填处理 68 个 Source，生成 541 个 Paragraph，失败为 0。

积极 Concept 共产生 1321 个候选，最终新增 162 个持久 Concept；Concept 总数增长约 41.3%。Association 净增 365 条，其中 277 条为 Episode→Concept `involves`，其余主要是新 Concept 与相似 Concept 的关系。

所有 embedding 在内存和 SQLite 中均为 float32。

### 4.1 排除旧边强化混杂

最初实现中，augmentation 再次遇到已有 Episode→Concept 边时会调用普通 `upsert`，从而提高旧边 weight 和 `evidence_count`。这样无法判断结果来自“新概念”还是“旧边被普遍强化”。

本轮正式资产已修复为：augmentation 可复用旧边，但不强化旧边；正常增量导入仍保留原强化语义。

正式资产验证：

- 冻结基线 SHA-256 前后一致；
- A/C 中原有 ID `1..1825` 的 Association 全字段摘要一致；
- `old_associations_unchanged = true`。

此前中断的 `assets` 和 `assets-v2` 只是诊断资产，没有进入任何正式结论。正式资产固定为 `assets-v3`。

## 5. 测试集与运行规模

测试集包括：

- 7 个网络级问题：每题需要覆盖 7—9 组跨文件 Episode 证据；
- 5 个范围控制问题：检验检索扩张是否把不属于问题范围的相似证据带入结果；
- 每题 3 次独立查询规划；
- Top-20 为主要证据预算；
- Top-12 作为更严格的前缀压力测试。

运行规模：

```text
36 个冻结查询计划
144 个四臂 fixed 检索
144 个四臂完整 rerank
每个 rerank 包含重排、覆盖审计、独立复核
```

增长边、Association cue 和查询期图增长全部关闭，本轮只测 Paragraph 与 Concept 数据变化。

## 6. 主要结果

### 6.1 网络题：绝对覆盖

分母 `168` 是 7 个问题 × 3 次重复所需证据组总数。

| 层级 | 预算 | A | P | C | PC |
|---|---:|---:|---:|---:|---:|
| fixed | Top-12 | 110/168（65.5%） | 102/168（60.7%） | 110/168（65.5%） | 102/168（60.7%） |
| fixed | Top-20 | 141/168（83.9%） | 140/168（83.3%） | 141/168（83.9%） | 140/168（83.3%） |
| reranked | Top-12 | 108/168（64.3%） | 102/168（60.7%） | 103/168（61.3%） | 97/168（57.7%） |
| reranked | Top-20 | 137/168（81.5%） | 132/168（78.6%） | 131/168（78.0%） | 131/168（78.0%） |

### 6.2 网络题：候选覆盖

| 实验臂 | reranked 候选覆盖 |
|---|---:|
| A | 167/168（99.4%） |
| P | 168/168（100%） |
| C | 167/168（99.4%） |
| PC | 168/168（100%） |

这解释了本轮为何很难得到正收益：成熟基线在进入最终筛选前已经覆盖 99.4% 的必要证据。Paragraph 最多只能补回一个证据组，积极 Concept 则没有补回新的必要证据组。

### 6.3 配对方向

网络题 reranked Top-20 的 21 个配对比较中：

| 处理 | 改善 | 持平 | 退化 | 平均证据组差值 |
|---|---:|---:|---:|---:|
| P − A | 3 | 13 | 5 | -0.238 |
| C − A | 0 | 18 | 3 | -0.286 |
| PC − A | 2 | 14 | 5 | -0.286 |

没有任何处理通过预注册的稳定收益门槛。

### 6.4 稳定负回归示例

- `hifumi_azusa_identity_coalition_and_declaration`：C 在 reranked Top-20 三次差值为 `[-4, 0, -1]`，构成稳定负回归；PC 相同。
- `arius_two_patrons_and_control_boundaries`：P 在 fixed 和 reranked Top-12 三次均为 `[-1, -1, -1]`。
- `old_cathedral_symbol_infrastructure_and_limits`：P 在 reranked Top-12 三次均为 `[-1, -1, -1]`。
- `mika_motive_epistemic_four_layers`：PC 在 reranked Top-12 为 `[-1, -1, 0]`。

范围控制题中出现一个 Paragraph 的局部稳定正例，但同时也出现 Concept 的稳定负例；它不能替代网络题收益，更不能满足整体门槛。

## 7. Paragraph 根因分析

Paragraph 长度统计：

```text
count   541
min     587 chars
median  1255 chars
mean    1265.4 chars
p95     1602 chars
max     2091 chars
```

最大值高于 1400 的软上限，是因为当前分段器优先保留完整剧情记录和重叠上下文，没有硬切断单条长记录。

### 7.1 辅助通道反客为主

P 组每次网络题最终 20 个证据中，有 16—20 个同时属于 Paragraph 展开结果，平均约 18.33 个。

当前原子锚点采用多通道 round-robin，Paragraph 展开与 Episode dense、Episode sparse、Source sparse expansion 处于同级通道。它因此不仅补充候选，还直接占据有限锚点名额，挤掉强基线证据。

### 7.2 embedding 输入含重复脚手架

Paragraph 文本重复包含：

```text
[source_key]
[segment_index]
[speaker_alias_legend]
[record]
[script_raw]
多语言平行文本
```

人物别名表在同一 Source 的多个 Paragraph 中反复出现。它对追溯有用，但作为 embedding 输入会让许多段落共享大量人物名和格式噪声，降低段落之间的区分度。

### 7.3 与现有 Source sparse retrieval 重叠

成熟基线已经对原始 Source 做稀疏检索并展开同 Source Episode。精确词语和文件内细节通常已经能通过 Source sparse 通道补回，因此 Paragraph 的真正边际价值只剩“原文语义改写命中、但字面词项不重合”的场景。现有网络题没有为该狭窄缺口提供足够机会。

## 8. 积极 Concept 根因分析

### 8.1 新 Concept 大量进入搜索但没有新增必要候选

所有 21 次网络题运行的 Concept 排名中都出现了新增 Concept；按查询列表计，共有 374 个列表包含 ID 大于 392 的新 Concept。

但是 C 的必要候选覆盖与 A 完全相同，仍为 `167/168`。也就是说，新 Concept 广泛参与了图扩张，却没有带来新的目标证据。

### 8.2 软目标 4—10 迫使模型填充低复用概念

新增 Concept 中既有有价值的锚点，例如：

```text
观看的义务
联邦学生会介入原则
方针变化
茶会主持人
红冬学园交流会
```

也有明显过细、过泛或上下文绑定的候选，例如：

```text
他人
职责
主持
任务
结局
品味很差
米卡（未武装）
女主角眼神迷离、流着口水祈求原谅的结局
```

大多数新增 Concept 只连接一个 Episode。它们更像局部标签，而不是可跨记忆复用的稳定节点。

### 8.3 新图节点改变候选顺序

C 与 A 的原子 Episode 锚点在 21/21 次网络运行中完全一致，但图候选顺序在 21/21 次中都发生变化；最终候选列表只有 9/21 次完全一致。

因此退化并非 Episode dense 基线被改写，而是新增 Concept 和关系改变了静态图遍历顺序及重排上下文。LLM 在更噪的候选组合中丢失了部分必要证据。

## 9. 工程修复与验证

本轮新增或完善：

1. 四臂资产生成器；
2. 冻结查询计划与按臂重算检索的 Stage 11 执行层；
3. 三次独立规划的配对评测器；
4. Paragraph provenance、Concept ranking 和候选/最终覆盖诊断；
5. Concept augmentation 并发提取与批量关系判断；
6. augmentation 复用旧边但不强化旧边的因果隔离；
7. 重排任务逐项检查点；
8. 恢复时直接复用固定层结果，避免重复本地检索。

完整单元测试：

```text
122 tests passed
```

新增测试覆盖：

- Paragraph 自然边界与重叠；
- Paragraph 检索可补回被 Episode 摘要遗漏的 Source 内 Episode；
- Paragraph 回填可安全重试；
- 保守/积极 Concept 提示词可逆；
- augmentation 复用旧边时不改变 weight、evidence_count 或更新时间；
- 冻结 float32 查询向量在 A/P 间保持相同，且仅 P 可触发 Paragraph 补召回。

## 10. 当前决策

### 10.1 不启用现有 Paragraph 融合

Paragraph 数据可以保留，但当前检索融合不能进入默认流程。特别是不应让 Paragraph 展开与基础 Episode 锚点等权轮转。

### 10.2 不启用当前 `fine_grained` Concept 持久化策略

积极抽取的原始输出继续保存在日志中，可用于分析；默认持久 Concept 仍采用保守档。

### 10.3 不删除实验实现

失败发生在“信号清洗、持久化准入和融合策略”，不是 Paragraph/Concept 机制在理论上无效。实现保留为可逆实验分支，避免之后重新开发基础设施。

## 11. 下一轮建议

### 11.1 Paragraph 改为 recall-only

Paragraph 只允许向候选池追加基线没有的 Episode，不得：

- 改变已有 Episode 的基础分数；
- 进入原子锚点 round-robin；
- 在没有新增候选时改变候选顺序。

这样可以建立“单调候选扩展”：Paragraph 最坏情况是没有帮助，而不是先打乱强基线。

### 11.2 清洗 Paragraph embedding 输入

数据库仍保存完整 Paragraph 原文用于追溯，但 embedding 输入应单独清洗：

- 去掉 `[source_key]`、`[segment_index]` 等格式标签；
- 不重复嵌入完整 `speaker_alias_legend`；
- 将 `script_raw` 与用户可读文本分离；
- 保留发言者和主要语言文本；
- 多语言仍可保存，但需要测试全部平行文本与主语言优先两种方案。

### 11.3 Concept 使用“积极发现、谨慎晋升”

不要再用每 Episode 4—10 个作为持久化目标。更合理的两阶段机制是：

1. LLM 积极产出候选并全部写日志；
2. 只有满足准入条件的候选晋升为持久 Concept。

建议准入信号：

- 明确命名的人物、组织、制度、事件、理论、地点、物品；
- 或在至少两个独立 Episode 中重复出现；
- 或后续用户对话明确确认其长期重要性；
- 一次性动作、泛词、整句改写和上下文状态不持久化；
- 低准入候选可以作为临时查询标签，但不进入长期 Association 图。

### 11.4 分开“机制挑战集”和“非回归集”

当前网络题候选覆盖为 99.4%，只能有效测非回归，不能充分测新召回通道。

下一轮应预注册两类题：

- Paragraph 机制题：答案细节存在于 Source 原文，但 Episode 摘要遗漏；问题使用语义改写，避免被 Source 稀疏词项直接命中。
- Concept 机制题：问题依赖一个保守图没有、但满足持久化准入的新稳定概念，并要求跨至少两个 Episode 复用。

现有 7 个网络题和 5 个范围控制题继续作为非回归集，不能因为机制题变容易而删除。

### 11.5 下一轮启用门槛

Paragraph 或积极 Concept 只有同时满足以下条件才进入默认流程：

1. 机制挑战集至少两个不同问题稳定增加候选覆盖；
2. 新增候选实际进入最终证据，而非只停留在候选池；
3. 现有网络题 fixed 与 reranked 均无稳定负回归；
4. 范围控制题无稳定负回归；
5. 三次独立规划的总体平均差值不小于 0；
6. 所有收益都能由 Paragraph provenance 或新增 Concept 路径解释。

## 12. 资产位置

- 正式资产报告：`validation/evaluation-stage11-paragraph-concepts/assets-v3/asset-report.json`
- 完整评测报告：`validation/evaluation-stage11-paragraph-concepts/pilot-v1/run-report.json`
- 实验预注册清单：`validation/stage11-paragraph-concept-manifest.json`
- Stage 11 冻结检索实现：`src/memory_demo/stage11.py`
- 四臂资产生成器：`benchmarks/prepare_stage11_paragraph_concepts.py`
- 四臂评测器：`benchmarks/run_stage11_paragraph_concept_eval.py`

`run-report.json` 包含逐问题、逐重复、逐实验臂的候选 ID、锚点 ID、最终 Episode ID、Paragraph 展开来源、Concept 排名、Association 路径和重排轨迹，可用于后续逐例复盘。
