# Stage 9：Association 作为一等检索入口的效益实验报告

日期：2026-08-27  
状态：实现、单次消融、五次独立重复与完整回归均已完成  
当前配置：Episode、Concept、Association cue、数据库向量和内存向量统一使用 float32；Paragraph 关闭；Concept 策略不变

## 1. 本阶段要回答的问题

Stage 8 已经证明：查询中自主生成并通过审计的 Association 有时能把缺失 Episode 带回答案，但收益稀疏。主要结构限制是，原系统只有在先召回某个端点后才能沿图遍历一条关系。如果问题使用新的抽象措辞，而且关系两端都没有进入基础 Top-K，那么关系即使已经存在，也不会被系统“想起”。

Stage 9 因此验证：

> 能否把经过审计的 `Association.relation_text` 直接做向量检索，让新问题先唤起一条旧联想，再回溯其 Episode/Concept 端点，并在冻结消融中产生可归因的证据召回收益？

本阶段不以新增边数量、关系被遍历次数或模型主观评价作为成功标准。成功必须同时满足：

1. 查询确实命中了前置问题生成的目标 Association。
2. 保留目标边时，比遮掉目标边时多覆盖至少一个必要 Episode 证据组。
3. 启用 Association cue 后不得低于普通图检索。
4. 最终事实仍回到 Episode/Source；关系文本本身只作为推论和检索线索。

## 2. 专有名词

### 2.1 Association-first retrieval

“Association-first retrieval”不是只搜索关系，也不是用关系替代 Episode。它表示查询可以直接命中关系文本，然后把关系端点加入正常候选池：

```text
新问题
  ↓
查询 Association.relation_text
  ↓
命中一条已审计关系
  ↓
取回关系的 Episode / Concept 端点
  ↓
与普通 Episode / Concept 召回合并
  ↓
rerank、coverage 保护、证据选择、回答
```

普通 Episode/Concept 检索仍然是主通道，Association 只是增加一条可回退的联想入口。

### 2.2 Association cue

Association cue 是被查询向量直接命中的关系。它只说明“这条旧联想可能与当前问题有关”，不自动证明 `relation_text` 中的推论为事实。回答仍需展示端点 Episode，并保留关系的 `claim_level`、`generation`、`audit_status` 和证据来源。

### 2.3 endpoint

endpoint 是 Association 的两端，例如：

```text
Episode 175 --thematic_response--> Episode 188
```

Episode 175 与 Episode 188 都是端点。命中关系后，系统可把两端加入检索种子。

### 2.4 transfer question

迁移问题不复述生成关系时的原问题，不直接列出所有人物、文件名或目标 Episode，而用新的抽象表达询问同一经验模式。它用于区分“记住一条可迁移联想”与“对原问题字符串过拟合”。

### 2.5 frozen replay

模型生成的意图、原子查询、追问和 rerank 计划具有随机性。冻结重放先保存一次完整计划，然后让各消融臂共享同一计划，只改变图和 Association cue 是否可见。这样差异才能归因于目标关系，而不是第二次模型调用恰好换了检索词。

### 2.6 masked ablation

遮罩消融不修改真实数据库。它在读取层隐藏前置问题创建的边，并把被强化的旧边恢复为强化前状态。处理组与遮罩组拥有完全相同的基础召回计划；唯一关键差异是目标增长边是否存在。

### 2.7 bridge slot

bridge slot 是最终证据列表中专门允许审计通过的学习关系补入端点的少量位置。当前最多补入 2 个 Episode。它的作用是让真正有用的关系有机会补全链条，同时防止稠密关系图任意挤掉可靠的基础 Top-K。

### 2.8 utility opportunity 与 ceiling

- **utility opportunity**：遮罩组尚未覆盖全部必要证据，因此处理组还有理论提升空间。
- **ceiling**：遮罩组已经满分，例如 5/5；此时处理组不可能再增加 required-group recall。满分持平不应计作机制失败。

### 2.9 non-regression

启用关系检索后的 S 组，其 required-group recall 不低于普通图 G 组。它只表示没有损害基础检索，不等同于关系已经产生净收益。

## 3. 实现设计

### 3.1 可索引关系范围

Stage 9 只索引同时满足以下条件的关系：

```text
audit_status = dual_accepted
relation_key != involves
relation_text 非空
created_reason 包含“查询中自主增长：”
```

这样排除了大量只表示 Episode 包含某个 Concept 的结构边，也排除了未完成双模型审计的推论。`generation` 没有被当作可信度扣分项；不同代数仍由原字段显式保存。负向关系仍可作为检索线索，但负极性路径不能占用正向 bridge slot。

### 3.2 独立的 float32 Association 索引

新增 `association_index`，复用现有 `EmbeddingIndex`：

```text
ids: int64 一维预分配数组
embeddings: float32 二维预分配数组
count / capacity
```

关系文本使用与 Episode/Concept 相同的 BGE-M3、1024 维 embedding。实验阶段不改 Association 表结构，不把关系向量写入 SQLite；启动时从合格 `relation_text` 重建 RAM 索引。默认配置关闭，因此旧查询行为不变。

实验参数：

```text
association_cue_top_k = 8
association_cue_min_similarity = 0.45
association_cue_rrf_weight = 0.80
dtype = float32
```

当前两个试验库分别索引 7 条与 5 条关系。由于 `EmbeddingIndex` 的最小预分配容量为 16，报告内存均为 65,664 bytes；这不是单条关系的真实线性成本。大规模下 1024 维 float32 约为每条关系 4 KiB，100 万条约 3.8 GiB，另加 ID、容量余量和其他索引，因此不能据当前小样推断规模成本已经解决。

### 3.3 查询流程

每个原始问题、意图原子查询和后续查询共享已生成的 query embedding，同时搜索：

1. Episode dense index。
2. Concept dense index。
3. 已启用的 sparse/source/paragraph 通道；本阶段 Paragraph 仍关闭。
4. Association cue index。

关系命中通过 cosine 门槛后，按 RRF 权重产生端点 seed。每次使用前还会回查当前 Association repository，重新验证关系仍存在且仍满足资格条件。因此遮罩层、删除或审计状态变化会立即生效，不会因为冻结向量或陈旧 RAM ID 绕过控制。

### 3.4 最终证据选择

关系向量召回不能无条件占据答案。最终策略为：

1. 先由向量 Top-K 与 cosine 门槛决定哪些关系有资格进入。
2. 在合格关系中，优先使用 `relation_text` 与当前问题词义更直接的路径；cue cosine 作为次级排序，关系权重与可信度再作为后续信号。
3. 仍只允许最多 2 个 bridge slots。
4. 保护 reranker 中已经独立覆盖必要语义槽的 Episode。
5. 对普通旧图边继续执行文本相关度、端点连接和重复度限制。
6. 对被当前问题直接向量命中的关系，允许绕过普通词面重复门槛，因为“关系被新措辞唤起”正是本阶段要测试的机制，但仍不能绕过槽位上限、审计、极性和证据保护。
7. 结果保存命中的 Association ID、cosine、端点、实际使用的 bridge path 和新增 Episode provenance。

### 3.5 为什么没有直接把 cosine 当最终排序

第一次修复曾把 cue cosine 放在所有选择信号之前。它修复了 Top-20，却让一个 Top-12 问题从有收益退化为持平：最高相似关系补入了与已选证据属于同一语义槽的 Episode，消耗了两个有限位置；稍低相似但能闭合“回应 → 恢复行动”的关系反而进不来。

这说明：

> 向量相似度适合高召回地找关系候选，不足以单独决定最终证据预算。

最终采用“向量资格过滤 + 当前问题词义优先 + cosine 次级排序 + 受限桥接槽”。修正后 Top-20 与 Top-12 的收益同时恢复，且没有非回归失败。

## 4. 实验臂

每个迁移问题、每次独立重复都比较四组：

- **A0**：`graph_max_hops=0`，不读取 Association，也不启用 cue。
- **G**：图深度为 3，只能从基础检索已召回端点遍历关系；cue 关闭。
- **S**：图深度为 3，并允许 query 直接向量检索 Association。
- **SM**：与 S 配置完全相同，但遮掉对应前置问题创建或强化的目标增长边。

主要因果比较是 S 与 SM。G 用于检查新的入口是否优于“只有端点遍历”的旧机制；A0 用于观察普通图是否已经产生变化。

## 5. 迁移问题设计

### 5.1 日奈—老师责任链

使用 Stage 8 已生成并双审计通过的 7 条 generation 1 边，设计两个不复述原增长问题的抽象迁移问法：

1. **成人责任不是强迫耗尽的学生继续承担任务**：要求找出学生耗尽、努力被承认、支持者接手后续、学生恢复行动的完整经历，并把“大人的义务”解释为主题推论。
2. **被看见的需要与恢复自主行动**：要求从跨片段记忆中重建“长期公共职责—情感缺口—非命令式支持—恢复自主性”的证据链。

必要 Episode 组覆盖退出/崩溃、渴望认可、老师回应、恢复行动和跨篇章责任背景。问题没有直接给出所有人物和文件名。

### 5.2 妃咲/弥奈—未花治理对比

使用 5 条已审计 generation 1 边，设计两个跨活动与主线的结构类比问题：

1. 把内部顾虑转化为制度连续性，或借外部势力打破连续性。
2. 面对内部反对和外来影响时，延续共同传统与利用外敌重组秩序的对比。

这一组主要用于非回归与天花板观察；基础检索在当前 342 个 Episode 小库上已经很强。

## 6. 单次双主题消融结果

单次冻结计划覆盖 2 个主题、4 个迁移问题、Top-20/Top-12 两种预算，共 8 个比较。

| 主题 | 问题 | 预算 | A0 | G | S | SM | S-SM | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 日奈 | 成人责任 | 20 | 4/5 | 4/5 | 5/5 | 4/5 | +1 | 严格因果收益 |
| 日奈 | 成人责任 | 12 | 3/5 | 3/5 | 4/5 | 3/5 | +1 | 严格因果收益 |
| 日奈 | 认可与自主 | 20 | 3/5 | 3/5 | 5/5 | 3/5 | +2 | 严格因果收益 |
| 日奈 | 认可与自主 | 12 | 2/5 | 2/5 | 3/5 | 2/5 | +1 | 严格因果收益 |
| 治理 | 吸收焦虑/打破连续性 | 20 | 5/5 | 5/5 | 5/5 | 5/5 | 0 | 天花板持平 |
| 治理 | 吸收焦虑/打破连续性 | 12 | 4/5 | 4/5 | 4/5 | 4/5 | 0 | 改变证据但无净增 |
| 治理 | 相同压力/不同外力策略 | 20 | 5/5 | 5/5 | 5/5 | 5/5 | 0 | 天花板持平 |
| 治理 | 相同压力/不同外力策略 | 12 | 5/5 | 5/5 | 5/5 | 5/5 | 0 | 天花板持平 |

总计：

- 严格因果收益：4/8，全部来自有辨识度的日奈链。
- S 不低于 G：8/8。
- 负收益：0/8。

治理组 Top-12 的一个问题中，S 选入 Episode 241、SM 选入 Episode 237；两者都属于不同的必要证据组，最终总覆盖仍为 4/5。这种“正确证据之间的替换”不计作 recall 增长。

## 7. 五次独立重复结果

由于治理组大多已经达到天花板，正式重复聚焦于两个日奈迁移问题。共执行 5 次独立模型规划；每次内部的 A0/G/S/SM 共享同一冻结计划。总计 20 个“问题 × 预算 × 重复”比较。

| 问题 | 预算 | 重复数 | 有提升空间 | 观察到因果收益 | S≥G | 平均 S-SM 证据组差值 |
|---|---:|---:|---:|---:|---:|---:|
| 成人责任 | 12 | 5 | 5 | 5 | 5 | +1.0 |
| 成人责任 | 20 | 5 | 2 | 2 | 5 | +0.4 |
| 认可与自主 | 12 | 5 | 5 | 5 | 5 | +1.6 |
| 认可与自主 | 20 | 5 | 5 | 5 | 5 | +1.8 |

合计：

- 完成比较：20。
- 失败或中断：0。
- S 不低于 G：20/20。
- 遮罩组尚未满分、存在提升空间：17/20。
- 在这些机会中观察到严格因果收益：17/17。
- 另外 3 次均为“成人责任 Top-20”的 SM 已经 5/5 满分，S 也为 5/5；不是失败。
- 全部 20 次的平均 `S-SM` 为 +1.2 个必要证据组；只看 17 次有效机会，平均约 +1.41 个组。

这个结果比 Stage 8 的“从端点开始遍历”明显更强。Stage 8 日奈链在 Top-20 的五次重复中没有 recall 增益；Stage 9 直接检索关系后，只要基础结果仍有缺槽，就在每次重复中补出了至少一个必要组。

## 8. 能够确认的结论

### 8.1 已确认

1. 通过审计的自然语言 `relation_text` 可以成为有效的独立向量检索对象。
2. 新的抽象措辞能够直接唤起以前问题形成的关系，而不要求先命中关系端点。
3. 命中关系后回溯端点，能产生可遮罩、可重复的 required-group recall 净收益。
4. 在 17 次仍有提升空间的重复比较中，17 次都产生严格收益；这已不是单次随机命中。
5. 受限 bridge slots 与基础证据保护在 20/20 重复比较、8/8 双主题单次比较中没有造成 recall 负回归。
6. `generation` 与 `confidence` 继续分离；本机制不需要通过降低高代关系可信度来控制检索，而是依靠审计、provenance、槽位和最终 Episode 核验。
7. 冻结 replay v4 能保存 cue ID、cosine、端点和基础 seed，并在遮罩 repository 下重新验证关系，因此消融没有写坏数据库。

### 8.2 尚未确认

1. 当前只有 12 条合格关系和两个核心增长主题，不能估计长期增长到数十万或百万关系后的平均精度。
2. `top_k=8`、cosine `0.45`、RRF `0.80` 只是本阶段可工作参数，尚未通过大规模正负样本 ROC、Recall@K 或延迟测试确定。
3. 关系索引目前启动时调用 embedding API 重建；尚未实现持久化、增量更新、删除槽位复用和原子热切换。
4. 治理组因基础检索天花板没有证明净收益，也没有证明该主题无效。需要新设计非天花板迁移问题，而不是继续缩小 Top-K 来人为制造差值。
5. 本阶段测的是 Episode 证据召回，不是长期答案质量。Stage 8 已显示在 Episode 完全一致时，额外高代关系文本未稳定改善最终答案，因此不应从本结果推断回答质量必然同比上升。
6. 小索引无法衡量纯 CPU brute-force 在百万 Association 上的吞吐。统一 float32 在 32 GB RAM 下可能需要只索引活跃/高价值关系、分块扫描或后续 ANN，而不是同时让所有 Episode、Concept、Association 都无限全量常驻。

## 9. 本阶段发现并排除的错误方向

1. 不能要求先召回端点再遍历关系；这使关系无法解决真正的基础召回盲区。
2. 不能把“命中 cue”本身当作效益；必须与遮罩组比较必要证据覆盖。
3. 不能把满分持平算作失败；必须单独统计 utility opportunity。
4. 不能仅按 cue cosine 消耗最终槽位；最高相似关系可能与现有证据语义重复。
5. 不能让所有旧关系直接挤进答案；只有合格 cue 使用极少 bridge slots，普通旧边继续受更严格限制。
6. 不能把 Association 推论当作新的直接事实。关系负责“想起去哪里看”，Episode/Source 负责回答“原文到底说了什么”。

## 10. 代码与实验资产

### 10.1 主要实现

- `src/memory_demo/config.py`：Association cue 开关、Top-K、cosine 门槛和 RRF 权重；默认关闭。
- `src/memory_demo/repositories/association.py`：列出满足审计条件的关系索引候选。
- `src/memory_demo/app.py`：构建独立 float32 `association_index`，并报告索引统计。
- `src/memory_demo/retrieval/engine.py`：关系向量召回、端点 seed、repository 重验、replay v4、cue provenance、受限 bridge slot 和最终选择排序。
- `src/memory_demo/stage9.py`：cue 开关冻结变体与 S/SM 因果效益诊断。
- `tests/test_stage9_association_cues.py`：端点召回、冻结配置、因果判定和 cue bridge slot 单元测试。

### 10.2 实验入口

- `benchmarks/run_stage9_association_cues.py`：双主题单次 A0/G/S/SM 消融，支持复用冻结 bundle。
- `benchmarks/run_stage9_association_cues_repeated.py`：多次独立计划、并行检查点和 utility opportunity 聚合。

### 10.3 清单与机器结果

- `validation/stage9-association-cue-manifest.json`
- `validation/evaluation-stage9-association-cues/transfer-pilot-v1/run-report.json`
- `validation/evaluation-stage9-association-cues/transfer-repeated-v1/run-report.json`

五次重复中的每个 family report、question report、cue replay bundle 和 planner log 均保留在对应 `repeat-01` 至 `repeat-05` 目录中，可逐条追溯。

### 10.4 回归

完整测试：115 项全部通过。  
重复实验：5/5 完成，0 失败。  
实验数据库只读回放，Association 快照前后相同。

## 11. 下一阶段建议

当前已经得到第一个较强的正结论：增长边不仅能够保存和被遍历，在拥有独立检索入口后，也能在未来不同措辞的问题中稳定补回缺失证据。下一阶段不应立即放宽审计或继续追求更多边，而应验证这种收益能否推广并控制规模风险。

建议按以下顺序推进：

1. 从其他剧情主题中再建立至少 5 条独立增长链，每条设计 2—3 个不复述原问题的迁移问题。
2. 预注册有缺槽但不靠任意缩小 Top-K 制造困难的测试；继续报告机会条件下收益、总体收益和非回归三类指标。
3. 为 Association cue 构造硬负样本：措辞相似但人物、时间、因果方向或极性不同，测试错误端点是否进入答案。
4. 扫描 `top_k`、cosine 门槛和 bridge slots，优化目标应是“机会条件下 recall 增益 + 全局零/低回归”，不是命中关系数量。
5. 扩大到数千、数万条合成和真实关系，实测 float32 brute-force 的 RAM、启动时间和 CPU 延迟，再决定是否持久化 embedding、分块扫描或引入 ANN。
6. 只有在多主题迁移和硬负样本仍成立后，才让 Association cue 默认开启，并实现新增/强化/删除关系后的增量索引一致性。

## 12. 最终结论

Stage 9 已经验证 Association 自主增长的关键闭环：

```text
过去的问题形成关系
  ↓
关系通过审计并长期保存
  ↓
未来问题使用不同的抽象措辞
  ↓
关系文本被直接向量唤起
  ↓
系统回溯原 Episode 端点
  ↓
补回基础检索遗漏的证据
```

在五次独立规划的日奈迁移实验中，所有 17 次存在提升空间的比较都出现严格、可遮罩的净收益，20 次比较全部无回归。这足以说明 association-first retrieval 是当前系统中值得继续发展的方向。

同时，这不是对长期可扩展性的证明：关系样本仍少，阈值未系统校准，治理主题存在天花板，float32 全量关系索引在百万规模下会带来数 GiB 额外内存。下一阶段的重点应从“机制是否可能有效”转向“在更多主题、硬负样本和更大规模下，是否仍保持准确、可控和可解释”。
