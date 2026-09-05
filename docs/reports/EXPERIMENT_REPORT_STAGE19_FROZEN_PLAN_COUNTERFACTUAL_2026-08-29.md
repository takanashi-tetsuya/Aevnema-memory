# Stage 19：冻结查询计划、内存增长与反事实效用报告

日期：2026-08-29（Asia/Tokyo）  
冻结基线：`validation/evaluation-stage17-full-rebuild/assets-v3-throughput/graph.db`  
在线实验版本：v3.32 / v3.33  
最终代码版本：`v3.34_scoped_identity_role_guard`  
主实验：`validation/evaluation-stage19a-frozen-plan-overlay-q3`  
复用诊断：`validation/evaluation-stage19b-role-guard-reuse`

## 1. 结论摘要

本阶段没有证明“新边提高了答案召回”，但完成了在继续研究增长收益之前必须具备的实验基础设施：

1. 静态图与增长图现在可以共享同一个带哈希的 `FrozenQueryPlan`，消除了两边分别调用 LLM 规划造成的混杂；
2. 查询期新边先进入 RAM overlay，不会在通过效用检验前写入 SQLite；
3. 新边用同一查询计划执行 treatment/masked 反事实重放；没有增加直接 Episode 证据的批次不会持久化；
4. Stage 19A 中静态图与增长图的最终 30 个 Episode 完全一致，两条临时边都被判定为零证据增益并丢弃；
5. 三个实验数据库都保持 8,904 条 generation=0 关系、0 条 generation>0 关系，冻结基线哈希没有改变；
6. 严格得分是 0/2，不是通过。两种模式都找到了问题锚点和袭击执行证据，但最终证据包漏掉
   `main/32170.json` 中的政变片段；
7. 关键片段 Episode 1167 已经进入原子候选，却在最终 30 条选择时被淘汰。这把瓶颈定位在“候选到最终证据”的
   覆盖分配，而不是基础 embedding 完全没有召回；
8. Stage 19B 的回答文本正确给出“阿里乌斯分校 + 未花”，但仍因缺少上述原始证据闭环而严格失败；
9. 与 Stage 18F 两边各自重新规划相比，Stage 19A 的请求数减少 43.6%，token 减少 45.5%，墙钟时间减少
   47.9%；
10. 当前同查询增长只能在已经可见的节点之间连边，无法凭空召回一个从未进入增长候选集的 Episode。要验证长期
    联想网络的价值，下一阶段应改成“先学习关系，后用不同措辞的问题复用关系”的跨查询实验。

因此，本阶段的准确结论是：**反事实零写与公平 A/B 已经可用，新边本身尚未产生可测收益；基础证据选择仍是当前
首要召回缺口。**

## 2. 本阶段要解决的实验问题

Stage 18 的静态和增长模式会各自执行一次 LLM 查询规划。即使数据库起点相同，它们也可能生成不同的原子问题、
不同的 embedding 查询和不同的 rerank 结果。此时静态失败、增长通过不能证明边有效；增长失败、静态通过也不能证明
边有害，因为两边的输入已经不同。

此外，Stage 18 的“新边进入最终路径便保留”只是使用门，不是效用门。一条边可能被回答引用，但屏蔽它以后最终证据
和答案没有变化。Stage 18E 提议 18 条边、保留 17 条，却没有一道题获得静态失败到增长通过的净增益，正是这个问题。

Stage 19 因而把问题改为：

```text
完全相同的查询计划与直接证据
              │
       ┌──────┴──────┐
       │             │
   屏蔽新边       启用新边
    masked        treatment
       │             │
       └──────┬──────┘
              ↓
       比较最终直接证据
              ↓
   只有可归因的新增证据才提交
```

## 3. 专有名词与设计定义

### 3.1 FrozenQueryPlan

`FrozenQueryPlan` 是一次问题规划和基础检索的可序列化快照，包含：

- 原始问题及解析后的 QueryIntent；
- 初始查询和后续原子查询；
- Dense/Sparse 等检索产生的 seed hits 与排名；
- Association cue 的当次候选快照；
- source_key cohort 选择和追踪信息；
- 增长前 rerank 的 Episode ID 与完整 trace；
- `plan_schema`、运行配置和内容 SHA-256 `plan_id`。

系统复用计划时会验证问题文本、schema、plan hash 和所有影响重放的配置。问题或配置漂移时直接拒绝，而不是悄悄生成
不可比较结果。

它解决的是实验中的**外生随机性**：LLM 如何拆问题、候选如何初排、reranker 如何选择，不能同时成为静态/增长 A/B
之间的差异来源。

### 3.2 staging overlay

`StagedAssociationOverlay` 是位于 RAM 中的临时 Association 仓库：

- 新关系使用负数临时 ID；
- 对已有边的强化只保存在 overlay 快照；
- 图遍历和回答可以像访问普通 Repository 一样读取临时关系；
- 查询异常、审计失败或效用不足时直接丢弃 overlay；
- 只有通过效用门后才执行 `commit()`，把临时 ID 映射为 SQLite ID，并同步修正 premise 引用。

SQLite 继续是 source of truth；overlay 是可丢弃的推理工作区。

### 3.3 treatment、masked 与反事实效用

- **treatment**：允许本批查询期新边参与图扩展和最终证据选择；
- **masked**：保持同一个 FrozenQueryPlan，但屏蔽本批新建或强化的边；
- **反事实效用**：比较 treatment 与 masked 的最终证据差异。

当前持久化条件是：

```text
新边进入了 treatment 的最终路径
并且
treatment 比 masked 增加了至少一个直接 Episode
```

这个条件有意偏保守。它不把“回答写得更顺”或“模型引用了这条边”当作足够证据，因为这两种现象可能只是语言模型
偏好，不能证明检索网络获得了新能力。

### 3.4 confidence、generation 与 utility

三者回答不同问题：

| 字段 | 回答的问题 | 示例 |
|---|---|---|
| confidence | 关系本身有多可信 | 文本是否充分支持“未花与阿里乌斯联手” |
| generation | 距离直接经验有几层推断 | 直接关系为 0，使用直接 Episode 推断为 1 |
| utility | 这条关系是否改善了检索/回答 | 屏蔽后是否丢失关键 Episode |

高 confidence、低 generation 不等于高 utility。一条完全正确的解释边，也可能对检索没有任何新增价值。

### 3.5 候选、最终证据与严格通过

- **候选**：检索系统曾经看见的 Episode；
- **最终证据**：回答模型实际收到的有限 Episode 包；
- **严格通过**：必需证据组、必需 source_key、答案术语和答案审计同时满足。

Stage 19 的关键 Episode 1167 在候选/原子锚点里，但不在最终证据里。因此不能把“曾经召回”写成“测试通过”。

## 4. 实现资产

### 4.1 共享与外部复用 QueryPlan

`RetrievalEngine.build_query_plan()` 生成 `frozen_query_plan_v1`；`query(..., frozen_plan=...)` 只重放计划，不再执行
意图规划、embedding 检索和 rerank。评测器按 graph hop 配置建立计划组，使 `graph_static` 和
`graph_growing` 自动引用相同的 `plan_id`。

CLI 新增：

```powershell
python src/main.py evaluate evaluation_questions.json replay-output `
  --modes graph_growing `
  --query-plans-from previous-evaluation
```

外部复用目录缺题、plan hash 不一致或配置不兼容时 fail closed。

### 4.2 RAM staging 与提交映射

增长查询会暂时把 AssociationRepository、图遍历器、增长引擎和时间查询切换到同一个 overlay。查询完成后：

1. 计算反事实效用；
2. 删除无效临时关系或恢复无效强化；
3. 仅提交剩余关系；
4. 将结果中的负数临时 ID 改写为 durable ID；
5. 无论成功或异常，都恢复原 Repository 对象。

这避免了“先写 SQLite、查询结束再删”的观察窗口，也避免程序在清理前崩溃留下脏边。

### 4.3 批次反事实效用

当前反事实以一次查询产生的整批 changed IDs 为单位：

- treatment 使用全部 changed IDs；
- masked 同时隐藏全部 changed IDs；
- 记录新增/被挤出 Episode、source_key、changed path 和可提交 ID；
- 不读取 evidence manifest，避免测试答案泄漏到生产逻辑。

它已经能证明“整批边没有收益”，但暂时不能区分同一批中哪一条边真正贡献了收益。逐边归因需要后续的 leave-one-out
消融或小规模 Shapley 近似。

### 4.4 角色身份完整性门

Stage 18 出现过“真琴（茶会高层，格黑娜万魔殿成员）”的错误关系，而且 DeepSeek 与 GLM 双审计都接受了它。
Stage 19 增加确定性身份门：显式的组织高层、成员、负责人、会长、主席、话事人身份，必须能从可见端点直接得到，或有
明确的接任/取代/下台等权力变更证据。

在线运行又暴露了两次过严：

1. `未花为了成为茶会的主持` 没有匹配最初只认识“主持人”的规则；
2. `正义实现部（Justice Task Force）成员` 的中英括注打断了中文精确匹配，而且普通叙事称谓不应被视为新的身份断言。

最终 v3.34 做了两项修复：

- 中英括注比较时忽略 ASCII gloss；
- 只对括号同位语和“是/作为/身为/担任/属于”等显式身份断言执行硬门，不审计普通句子主语。

离线回归同时覆盖：错误真琴身份被拒绝、未花主持证据被接受、普通正义实现部成员叙事被接受。

## 5. 实验设计

测试问题仍使用第 3 道深度题“古圣堂袭击因果链”，因为它同时要求：

1. 找到古圣堂爆炸和巡航导弹锚点；
2. 识别直接执行者为阿里乌斯；
3. 回溯未花与阿里乌斯合作/政变的早期证据；
4. 区分直接事实与“协助具体渗透方式”的因果推断。

### Stage 19A：公平成对实验

- 输入：同一题；
- 模式：graph_static、graph_growing；
- 起点：同一冻结数据库备份；
- QueryPlan：两模式共享同一个 plan；
- 增长：RAM staging + batch counterfactual gate；
- 评分：Stage 17 冻结 evidence manifest。

### Stage 19B：外部计划复用与角色规则诊断

- 只运行 graph_growing；
- 直接读取 Stage 19A 的 QueryPlan；
- 不重新调用 planner、embedding 或 reranker；
- 观察 LLM 增长输出的随机变化、身份门行为和零写结果。

## 6. 实验结果

### 6.1 Stage 19A 的公平性

两种模式共享：

```text
plan_id = c57fd0fa5e401a6fa968e3a2ce04a5a88271c2761f9d5b8f8a3cdf8e181ebf62
```

静态与增长最终 Episode ID 列表逐项相同，共 30 条。这是第一次能确认两边的基础证据完全一致，因而增长差异不再由
查询规划噪声解释。

### 6.2 严格得分

| 模式 | 问题锚点 | 袭击执行证据 | 政变/未花早期证据 | 必需 source | 严格结果 |
|---|---:|---:|---:|---:|---:|
| graph_static | 命中 | 命中 | 未进入最终证据 | 缺 `main/32170.json` | 失败 |
| graph_growing | 命中 | 命中 | 未进入最终证据 | 缺 `main/32170.json` | 失败 |

合计为 **0/2**。两边答案包含“阿里乌斯”和“未花”，答案审计也有效，但证据闭环不完整，不能判通过。

旧 Stage 3 scorer 还要求 growing 模式必须有 changed edge，因此安全的零写会被标记为 `mode_behavior_passed=false`。
这条旧规则已经不符合新的反事实语义，后续评测器应允许“没有收益所以零写”的安全通过。不过本题即使忽略该旧规则，
仍会因缺关键 Episode/source 而失败。

### 6.3 召回断点

FrozenQueryPlan 的 `atomic_anchor_episode_ids` 已包含 Episode 1167，候选集合也包含它；但增长前
`reranked_episode_ids` 的 30 条预算没有 1167，最终回答证据同样没有它。

因此实际断点为：

```text
基础检索：命中 1167
      ↓
原子候选：仍保留 1167
      ↓
rerank / 最终覆盖分配：淘汰 1167
      ↓
缺 main/32170.json，严格失败
```

下一步不应继续无目的地增大 top-k；应为问题中的独立证据槽分配最终预算，并确保每个高风险槽至少有一条候选进入
答案证据。

### 6.4 新边反事实结果

Stage 19A 增长模式两轮共在 overlay 建立 2 条临时边。反事实结果为：

- treatment Episode 与 masked Episode 完全相同；
- 新增 Episode：0；
- 新增 source_key：0；
- 可归因证据增益：0；
- 可持久化边：0；
- overlay committed：false。

两条临时边全部删除，没有触碰持久数据库。Stage 19B 随机生成了另一组两条临时边，其中一条进入 treatment 路径，
但 treatment/masked 的 Episode 仍完全相同，因此同样全部删除。这说明“进入路径”确实不等于“有检索收益”。

### 6.5 数据库完整性

| 资产 | generation=0 | generation>0 |
|---|---:|---:|
| Stage 19A static DB | 8,904 | 0 |
| Stage 19A growing DB | 8,904 | 0 |
| Stage 19B growing DB | 8,904 | 0 |

冻结基线 SHA-256：

```text
864096431E0BAB1D187473D6D261DB5CCC40AA0E824F8EC14CC1BF67DD10FD11
```

基线没有变化。

## 7. token 与时间

### 7.1 Stage 19A

| 部分 | 请求 | tokens | 墙钟时间 |
|---|---:|---:|---:|
| 共享 QueryPlan | 7 | 51,567 | 235.544 秒 |
| graph_static | 3 | 95,794 | 148.513 秒 |
| graph_growing | 12 | 263,057 | 558.110 秒 |
| 合计 | 22 | 410,418 | 942.167 秒（约 15.7 分钟） |

### 7.2 与 Stage 18F 比较

| 实验 | 请求 | tokens | 墙钟时间 |
|---|---:|---:|---:|
| Stage 18F，各自规划 | 39 | 752,869 | 1,809.094 秒 |
| Stage 19A，共享计划 | 22 | 410,418 | 942.167 秒 |
| 减少 | 43.6% | 45.5% | 47.9% |

减少量包含共享规划、调用路径变化和回答是否触发修订等共同影响，不能全部归因于一个函数；但在同一道题上，规划只执行
一次且两模式完全复用，是确定的结构性节省。

Stage 19B 外部复用计划后没有 planner 请求，增长与回答仍使用 9 次请求、166,569 tokens、325.230 秒。当前剩余成本
主要来自长证据回答、双模型关系审计和答案审计，而不是 embedding。

## 8. 为什么本轮新边无法带来当前查询收益

增长引擎目前只在已经选入增长上下文的节点之间提出边。若目标 Episode 根本没有进入该节点集合，新边没有一个可指向的
未知端点，因而不可能把它召回。

即使某个 Episode 位于较宽的候选集合，只要它没有进入增长节点或图端点，当前查询中新建的边也只能重新描述已知节点之间
的关系。其可能改善解释文字，却不会增加直接证据。Stage 19 的 treatment/masked 完全相同正符合这个结构限制。

这不表示 Association 永远无用，而是说明要区分两种目标：

1. **同查询即时增长**：在已经看见的证据间形成解释边，主要改善组织和推理；
2. **跨查询长期学习**：第一次查询把共同出现但措辞不同的节点连起来，后续查询从一个端点沿旧边找到另一个端点。

用户要验证的“神经网络自主增长”更接近第二种。继续用同一问题内“建边后立刻看是否召回未知 Episode”作为主要测试，
会系统性低估长期边的价值。

## 9. 尚未解决的问题

### 9.1 最终证据槽覆盖

1167 已被召回但被最终预算淘汰。当前原子保底按查询顺序和固定数量轮询，不能保证“未花的早期政变证据”这一语义槽
一定获得席位。需要把 QueryIntent 中的目标实体、因果约束和 requested_relation 转换为可追踪的证据槽，并记录每个槽：

```text
是否有候选
候选 ID
最终是否保留
被谁替换
替换原因
```

### 9.2 FrozenQueryPlan 冻结边界

当前计划连当次 Association cue entries 也冻结。这适合 Stage 19 的“同查询增长”公平性实验，却不适合测量一条已学习边
在后续问题中通过关系向量 cue 带来的召回收益，因为 treatment 的新 cue 会被旧快照屏蔽。

下一阶段应拆成：

- `FrozenBaseRetrievalPlan`：冻结问题拆解、query embedding、Dense/Sparse 排名和基础 rerank；
- **动态图层**：在 treatment 与 masked 图上分别执行 association cue 和 graph traversal。

这样外生检索相同，而由图变化产生的差异被允许出现，才是正确的因果设计。

### 9.3 批次效用不能逐边归因

整批有收益时，当前实现可能把同批所有 changed path 边一起保留。下一步至少执行逐边 leave-one-out：每次只屏蔽一条边，
观察新增证据是否消失。边数较多时再使用分组消融或二分定位，避免指数级组合。

### 9.4 效用定义偏保守

目前只认可“新增直接 Episode”。它会拒绝以下潜在收益：

- 同样 Episode 下，答案因果边界更准确；
- 正确关系减少了答案审计/修订次数；
- 对证据排序、来源多样性或不确定性表达有稳定改善。

在基础 recall 稳定前维持这一保守门是合理的。以后可以增加独立的 answer utility 指标，但不能与证据增益混成一个分数。

### 9.5 LLM 双审计不是身份数据库

两种模型可能共享相同的语言偏见，因此“双模型都接受”不代表独立事实校验。确定性规则适合拦截清晰的身份升级，但不应
扩展成大型专家系统。当前 v3.34 只守显式身份断言，其余仍交给证据、generation 和反事实效用共同约束。

## 10. 下一阶段实验路线

### Phase A：先修最终证据槽，恢复基础可靠性

1. 从 QueryIntent 建立独立 evidence slots；
2. 用现有 FrozenQueryPlan 离线重放，不调用 API；
3. 确保 Episode 1167/1168 这类已经召回的独立槽不会被全局相关项挤掉；
4. 在 Stage 17 的 23 题上重新计算最终 Recall@20，不只看 Candidate@100；
5. 不使用评测 manifest 参与运行时选择，只用问题本身产生的槽。

验收：不降低已有题，且第 3 道深度题最终证据包含 `main/32170.json`；23 题 Final Recall@20 的提升有逐题明细。

### Phase B：建立跨查询学习—复用实验

为每个关系设计成对问题：

```text
学习问题 Q1：两端证据都容易被直接召回，允许建立关系 E
复用问题 Q2：措辞只明显指向端点 A，基础检索较难找到端点 B
```

每个样本建立三条分支：

- control：没有 Q1 新边；
- treatment：Q1 产生并通过审计的边存在；
- masked：同一 treatment DB，但查询 Q2 时屏蔽目标边。

Q2 共享 `FrozenBaseRetrievalPlan`，动态图 cue/traversal 分别读取各自图状态。只有 treatment 新增正确直接 Episode，且
masked 丢失该增益，才算 Association 的因果收益。

### Phase C：逐边效用与持久化

1. 对有批次收益的 changed edges 做 leave-one-out；
2. 记录每条边带来的 Episode/source 增量、使用次数和负面挤出；
3. 只有至少一次跨查询复用成功的边升级为 durable；
4. 尚未证明复用价值的边可保留在短期 probation overlay，而不是立即进入长期图；
5. generation 继续记录推断距离，confidence 继续记录可信度，utility 单独累计实际收益。

### Phase D：扩大样本

至少覆盖三类联想：

- 同一人物跨时间/跨文件的事件桥；
- 化名、别名、组织身份的实体桥；
- 情绪、创伤、动机到具体事件的语义桥。

报告必须同时给出：基础召回、图增量召回、错误端点侵入、数据库写入数、每个有效增益的边级证据和 token 成本。

## 11. 验证与资产清单

### 11.1 离线验证

- 146/146 项确定性测试通过；
- 包含共享计划、外部计划复用、配置漂移拒绝、内存 staging、无收益零写、强化恢复、角色误判和合法角色证据测试；
- 默认 embedding 存储与内存索引继续统一使用 float32。

### 11.2 代码资产

- `src/memory_demo/retrieval/engine.py`：计划生成/验证/重放、反事实效用、staging 查询包装；
- `src/memory_demo/evaluation.py`：共享计划组和外部计划复用；
- `src/memory_demo/association_overlay.py`：RAM staging overlay；
- `src/memory_demo/associations/growth.py`：显式角色身份完整性门；
- `src/memory_demo/cli.py`：`--query-plans-from`；
- `.env.example`：增长反事实和 staging 开关。

### 11.3 实验资产

- Stage 19A evaluation report、严格 scorecard、共享 QueryPlan、三个日志摘要和两种模式数据库；
- Stage 19B evaluation report、严格 scorecard、复用 QueryPlan、日志摘要和模式数据库；
- Stage 18 错误“真琴是茶会高层”关系的确定性回归；
- 本报告和 README 的当前状态说明。

## 12. 最终判断

本阶段最有价值的不是得分，而是把“边是否可信”和“边是否有用”真正分开了：

- 双审计、确定性身份门、confidence 和 generation 负责安全与证据距离；
- treatment/masked replay 负责测量边的实际收益；
- RAM overlay 保证收益未证明前不污染长期记忆。

本轮所有新边都未带来新增直接证据，所以全部不写入数据库，这是正确结果。接下来应先修复已经召回却被最终选择漏掉的
证据槽，再把增长收益实验从“同查询即时自证”改为“Q1 学习、Q2 复用”的跨查询因果测试。只有后者成功，才能支持
“Association 的自主增长使系统获得了可复用联想能力”这一核心主张。
