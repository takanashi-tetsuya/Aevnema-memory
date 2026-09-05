# Stage 8：增长 Association 的可消融效益实验报告

日期：2026-08-27  
状态：实验、实现和回归测试均已完成  
当前配置：Episode/Concept/数据库/内存向量统一使用 float32；Paragraph 关闭；Concept 保持保守策略

## 1. 本阶段要回答的问题

本阶段不再把“新增了多少条边”当作成功。核心问题是：

> 前一个问题自主生成的 Association，在后一个问题中是否产生了可隔离、可重复、且不损害基础检索的净效益？

“净效益”被拆成两类：

1. **证据召回效益**：遮掉新增边后，正确 Episode 证据是否真的少了。
2. **答案组织效益**：在 Episode 完全相同的情况下，保留关系文本是否让最终回答更完整、更连贯、更守事实边界。

这种拆分很重要。边可能把一个远处 Episode 带回来，却不改善已经有足够证据的答案；也可能不改变 Episode 集合，但把多段事实整理成可复用的解释。反过来，边也可能只是被遍历，却既不改变证据也不改变答案。

## 2. 专有名词与实验臂

### 2.1 Association

Association 是两个记忆节点之间的有方向联想。当前端点可以是 Episode 或 Concept，关系保存 `relation_type`、`relation_key`、固定自然语言 `relation_text`、`weight`、`confidence`、`claim_level`、`audit_status` 和 `generation`。

### 2.2 generation

`generation` 表示推论距直接经验的推导层数，不等同于可信度：

- generation 0：直接事实或直接结构关系。
- generation 1：直接以 Episode 等直接经验为前提形成的首层推论。
- generation 2：使用至少一条 generation 1 推论作为前提继续推导。
- generation 3：推导链中已经使用 generation 2 前提，依此类推。

本阶段没有按 generation 自动降低 confidence。generation 只负责让系统和回答模型知道“距离直接经验有多远”，事实可信度仍由证据和双模型审计决定。

### 2.3 冻结重放

一次真实查询会产生问题解析、原子检索问题、dense/sparse 候选、LLM 覆盖选择等带随机性和远程调用的结果。冻结重放把这些计划保存下来，然后在完全相同的候选和排序上只改变图或证据预算。这样 T 与 M 的差异才可以归因于 Association，而不是模型第二次恰好给了不同检索词。

### 2.4 A / B / O / T / M

- **A（无图基线）**：`graph_max_hops=0`，不读取 Association。
- **B（冻结旧图）**：读取增长前已有 Association。
- **O（Oracle）**：在副本中插入预注册的理想桥，只验证机制上限；不算系统学会了关系。
- **T（Treatment）**：先执行前置问题并允许自主增长，再重放目标问题。
- **M（Masked）**：数据库与 T 相同，但遮掉前置问题创建的边，并把被强化的旧边恢复到增长前状态。
- **T2 / M2**：同样的处理/遮罩设计，但只隔离第二轮产生的 generation 大于等于 2 的边。

### 2.5 hard group

预注册的、预计最难进入 Top-K 的证据组。它只用于定位机制，不是独立成功标准。真正的净效益必须提高完整 `required_episode_groups` 的覆盖；仅仅把一个正确 Episode 换成同组另一个正确 Episode，不算提高 recall。

### 2.6 bridge slot

为通过审计且与当前问题相关的查询增长边保留的少量证据入口。当前最多 2 个。它允许新边把缺失端点带进答案候选，但不能让稠密旧图任意挤掉基础 Top-K。

## 3. 首先发现并修复的基础问题

最初实现把大约四分之一答案槽位预留给图端点。Oracle 小样立即发现这会让旧的高权重结构边把基础检索中的正确 Episode 挤走：图虽然“参与了”，整体 recall 却下降。

最终选择器改为以下原则：

1. 先完整保留 reranker 的基础 Top-K。
2. 普通旧图路径只能作为解释，不能自动替换基础 Episode。
3. 只有当前查询新生成的边、实验 Oracle 边，或带查询增长来源且通过双审计的持久边，才可使用最多 2 个 bridge slots。
4. 持久边必须与问题达到最低文本相关度，而且至少一个端点已经被基础检索选中或优先命中。
5. 引入新端点时，优先替换没有独立证据槽价值且语义重复的尾部 Episode。
6. rerank coverage 中唯一代表某个证据槽的 Episode 受保护；`joint` 证据组的各个互补成员也受保护。

这次修复的意义不是“让图一定加分”，而是建立非回归底线：一条弱边最多不起作用，不能仅因存在就破坏可靠的基础 Top-K。

## 4. generation 1 桥接边实验

### 4.1 日奈—老师义务主题

前置问题自主生成 7 条 generation 1 关系，均通过来源审计和双模型审计。重要关系包括：

- 日奈表达情感需求，与老师非命令式回应之间的主题联系。
- 老师感谢、准许休息并主动承担后续，与日奈重新等待指示之间的变化链。
- 日奈从公开职责、个人极限到希望被认可之间的综合联系。

五次独立冻结计划重复结果：

| 预算 | T 平均 recall | M 平均 recall | T-M | 正/平/负 | 新边被使用 |
|---|---:|---:|---:|---:|---:|
| Top-20 | 0.9778 | 0.9778 | 0 | 0/5/0 | 5/5 |
| Top-12 | 0.8222 | 0.7111 | +0.1111 | 5/0/0 | 5/5 |

Top-12 的每一次重复都多覆盖 1/9 个必要证据组。Top-20 已接近天花板，因此没有额外 recall，但也没有负回归。

### 4.2 妃咲—未花治理对比

前置问题自主生成 5 条 generation 1 关系，均通过安全审计。代表性关系包括：

- 弥奈对外部商业化、传统责任的顾虑，与未花借阿里乌斯重组秩序之间的治理对比。
- 妃咲维持月影祭常态，与未花颠覆圣三一秩序之间的对比。

五次独立冻结计划重复结果：

| 预算 | T 平均 recall | M 平均 recall | T-M | 正/平/负 | 新边被使用 |
|---|---:|---:|---:|---:|---:|
| Top-20 | 0.975 | 0.950 | +0.025 | 1/4/0 | 5/5 |
| Top-12 | 0.850 | 0.825 | +0.025 | 1/4/0 | 5/5 |

第二次重复中，关系把 Episode 237 带入并替换了无关 Episode 84，使必要证据覆盖从 7/8 提升到 8/8。这是本阶段第一次在默认 Top-20 下观察到由学习边造成的严格净召回收益。

### 4.3 generation 1 总结

在默认 Top-20 的 10 个“主题 × 重复”配对中：

- 正收益：1 次。
- 持平：9 次。
- 负收益：0 次。

因此可以确认：

1. 新边不是只会增长，它已经能在真实目标问题中被遍历。
2. 在基础证据缺槽时，新边能把远端 Episode 带入，并产生严格的 recall 净收益。
3. 当前收益仍稀疏；它更像困难查询的保险，而不是每次查询都能加分的通用放大器。

## 5. 连续增长：generation 2/3 实验

实验采用 Q1 → Q2 → Q3：

1. Q1 已生成 generation 1 边。
2. Q2 明确要求优先引用可见 Association 前提，继续形成综合。
3. Q3 固定查询计划，只遮罩 Q2 中 generation 大于等于 2 的变化边。

结果：

- 日奈主题生成 6 条高代边：4 条 generation 2，2 条 generation 3。
- 妃咲/未花主题生成 3 条 generation 2。
- 合计 9 条；全部通过本地来源审计和双模型关系审计。
- Top-20 和 Top-12 中，这些边都被实际遍历并进入可见答案路径。
- T2 与 M2 的 Episode 列表完全相同，required-group recall 差值均为 0。

这说明连续推论机制成立，但“高代边产生净检索效益”尚未成立。

原因不是路径未被使用，而是二阶边连接的端点已经被基础检索或 generation 1 边选入。它们没有新的缺失端点可带回。

## 6. 固定 Episode 的答案质量消融

为了避免错过“边不改变 Episode，但帮助整理思路”的效益，又增加了答案层实验：

1. 每组 T2 和 M2 使用完全相同、顺序也相同的 20 个 Episode。
2. 唯一差异是 T2 能看到 generation 大于等于 2 的关系路径，M2 看不到。
3. 每个主题独立生成 3 对答案。
4. 所有答案继续通过系统自己的证据审计。
5. DeepSeek-V3.2 与 GLM-4.5V 作为两个盲评员，标签顺序打乱。
6. 评委只给证据完整性、推理连贯性、事实边界、不确定性校准四项 0—4 分；总分和赢家由本地程序重算。

第一次小样暴露了评测缺陷：DeepSeek 把回答中出现的内部 Association 编号误判为外部知识。规则修正为“不能因编号自动扣分，必须回到 Episode 核验主张”，并只重评已有答案，没有重新采样。

最终 6 对答案、12 个独立判断：

| 结果 | 次数 |
|---|---:|
| T2 胜 | 2 |
| M2 胜 | 2 |
| 持平 | 8 |

总平均 `T2-M2 = 0`。

分主题看：

- 日奈：6/6 全部持平。
- 妃咲/未花：2 次 T2 胜、2 次 M2 胜、2 次持平，平均为 0。

所有 12 份回答的事实安全审计都通过，答案修订次数均为 0。少数 `semantic_terms_passed=false` 来自字符串评测过于严格，例如答案表达了“不存在直接因果”，但没有命中清单中某个完全相同的固定短语；这不是事实安全失败。

答案层结论：在完整 Episode 已经可见、回答模型本身足够强的条件下，当前长篇二阶关系没有稳定改善答案。妃咲/未花组偶尔帮助形成更明确的“维持连续性/借外力颠覆”框架，也偶尔诱导更宏观、比原文更强的治理定性，两种效应相互抵消。

## 7. 当前能够确认的技术结论

### 已确认有效

1. `generation` 能正确记录连续推论距离；本次真实产生了 generation 2 和 generation 3。
2. 查询增长关系可以保留来源、证据、审计结论和前提关系，且能够在以后查询中参与遍历。
3. 遮罩实验可以同时隐藏新建边并恢复强化边的旧状态，不写坏数据库。
4. 受限 bridge slots、问题相关度门槛、coverage 保护和语义重复替换能避免旧图拖累基础 Top-K。
5. generation 1 边在证据预算受限时有稳定收益，并至少一次在默认 Top-20 下产生真实净收益。
6. 最终算法在所有重复中没有出现 T 低于 M 的召回回归。

### 尚未确认有效

1. generation 大于等于 2 的边尚未带来净 Episode recall。
2. 当 Episode 已齐全时，高代边尚未稳定提高最终答案质量。
3. 当前样本只有两个增长主题，不能据此估计长期平均收益或统计显著性。

### 已排除的错误方向

1. 不能按图路径数量预留固定比例答案槽位。
2. 不能把“边被遍历”当作效益。
3. 不能把 hard group 内的正确替换当作完整 recall 增长。
4. 不能因为 generation 高就自动降低可信度，也不能因为双审计通过就假定它必然有用。
5. 不能用增加边数作为评测指标；边数只能用于审计系统是否失控。

## 8. 为什么收益目前稀疏

当前 Association 主要是**从已找到的端点出发再遍历**。如果查询没有先召回任何相关端点，关系本身的自然语言没有独立的向量检索入口。

这会产生两个天花板：

1. 基础 Top-K 已经找到两端时，边无法再提高 Episode recall；强 LLM 也常能现场重建同样推论。
2. 基础 Top-K 两端都没有找到时，系统又没有机会看到这条边；在百万 Episode 规模下，这会比当前 342 个试验 Episode 更明显。

因此，单纯继续增长更多边不会解决问题。要让边成为真正的长期记忆，它必须既能被遍历，也能被问题直接唤起。

## 9. 下一阶段建议：Association 作为一等检索入口

下一阶段应验证 **association-first retrieval**，而不是继续放松审计。

建议先做可回退的实验索引，不立即迁移主数据库：

1. 只选 `relation_key != involves`、`audit_status=dual_accepted` 的信息型增长边。
2. 使用当前同一个 embedding 模型为 `relation_text` 生成 float32 向量。
3. 建立独立的 `association_index`，结构仍是预分配 `ids + embeddings + count + capacity`。
4. 查询时先按正常方式搜索 Episode/Concept，同时用同一 query embedding 搜索 Association。
5. Association Top-N 只作为候选入口，把端点加入 Episode 候选池，再交给现有 reranker 和 coverage 审计；不能直接占答案事实槽。
6. relation_text 仍是待验证推论，回答前必须回溯端点 Episode；高 generation 只展示距离，不自动扣分。
7. 第一版实验索引可以启动时从关系文本重建，验证有效后再决定是否把 embedding BLOB 写回 association 表。

预注册评测应使用“迁移问题”：问题用抽象或改写后的表达，不直接重复增长时的原句，也不把所有端点人物和文件名写进问题。比较三组：

- A0：基础 Episode/Concept 检索，不读取新边。
- G：只能从已经召回的端点做图遍历。
- S：额外允许 query 直接召回 Association，再引入端点。

主要成功标准仍然是默认 Top-20 的 required-group recall：

1. S 实际命中新增长边。
2. S 相对 G 带入至少一个原来缺失的必要证据组。
3. 所有测试中 S 不低于 G。
4. 答案质量只作第二指标；先证明它能找到新证据，再测它是否帮助组织答案。

建议先使用现有两条已审计链设计 6—10 个改写迁移问题，再增加至少 3 个不同剧情主题。若 Association 向量入口仍不能在 Top-20 产生稳定收益，就应考虑缩小自主增长范围，而不是继续扩大图。

## 10. 本阶段代码与资产

主要实现：

- `src/memory_demo/stage8.py`：冻结变体、hard group、净效益诊断、盲评归一化与汇总。
- `src/memory_demo/retrieval/engine.py`：非回归证据选择、查询增长 bridge slots、coverage 保护、路径 provenance。
- `src/memory_demo/repositories/association.py`：强化旧边时保留查询增长来源。
- `src/memory_demo/association_overlay.py`：修复同一查询中“先创建再强化”导致的 delta 错误。
- `src/memory_demo/config.py`：增长桥数量、相关度、重复度参数。

实验入口：

- `benchmarks/run_stage8_association_utility.py`
- `benchmarks/run_stage8_repeated_replay.py`
- `benchmarks/run_stage8_generation2_probe.py`
- `benchmarks/run_stage8_answer_utility.py`

实验清单：

- `validation/stage8-association-utility-manifest.json`
- `validation/stage8-generation2-manifest.json`
- `validation/stage8-answer-utility-manifest.json`

机器结果：

- `validation/evaluation-stage8-association-utility/pilot-v1/`
- `validation/evaluation-stage8-association-utility/repeated-v1/`
- `validation/evaluation-stage8-association-utility/generation2-pilot-v1/`
- `validation/evaluation-stage8-association-utility/answer-utility-v1/`

测试：111 项全部通过。

## 11. 最终结论

本阶段已经证明增长边可以带来真实效益，但效益目前集中在“基础 Top-K 缺少一个证据槽、而新边正好连接到该证据”的场景。generation 1 在 Top-12 中表现明显，在 Top-20 中观察到一次严格净收益；最终选择器没有发生负召回。

连续 generation 2/3 的自主增长已经技术成立，但还没有产生净召回或稳定答案质量收益。此时继续增加边或放松审计没有依据。下一步最合理的结构变化，是让审计通过的 Association 文本拥有独立的检索入口，使抽象问题可以先唤起关系，再回溯直接 Episode。只有这一步能够验证“已经形成的联想是否会在未来不同措辞的问题中被自主想起”。
