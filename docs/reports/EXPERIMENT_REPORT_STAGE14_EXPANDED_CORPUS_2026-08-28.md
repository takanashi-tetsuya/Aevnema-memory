# Stage 14 实验报告：扩大语料后的基础召回与辅助机制验证

日期：2026-08-28  
状态：扩大语料导入、数据库审计、基础检索、Paragraph 对照和细粒度 Concept 对照均已完成。  
数值类型：SQLite、NumPy 内存索引及全部 embedding 统一使用 `float32`。

## 1. 结论摘要

本阶段已经真实执行，不是测试计划或结果预测。

系统从 342 个 Episode 的旧基线扩展到 3,582 个 Episode，并在冻结的 23 道题上执行了无增长边检索。扩大后的数据库结构和向量数据通过完整审计，但基础 Top-K 策略还不能称为可靠完成：

- 全集普通运行的 Candidate Recall@100 平均为 97.34%，最低为 75.0%。
- Selected Recall@20 平均为 92.15%，最低为 50.0%。
- 7 道高难网络题的 Candidate Recall@100 平均只有 91.27%，Selected Recall@20 平均只有 78.97%。
- 8 道 Paragraph 细节题的两项指标均为 100%。
- 8 道 Concept 桥题的 Candidate Recall@100 为 100%，Selected Recall@20 平均为 95.83%。

所以“总体平均 Candidate Recall@100 超过 95%”是真的，但它被 16 道较容易的机制题抬高；“每道题都超过 95%”和“复杂问题可以可靠闭合证据链”都是假的。

固定相同查询意图后的因果重放显示，扩大语料本身并没有使粗召回变差：它为 3 个原有 Candidate 缺口中的 2 个各补入了一个必需证据组，另外一个不变。普通运行中表面新增的 Candidate 回退，经固定意图后全部消失，主要来自 LLM 查询拆解波动。不过，候选集合扩大后，LLM 精排仍会把已经召回的必需证据挤出 Top-20，因此当前更急迫的问题已经从“只提高 Candidate 数量”转向“按证据槽稳定压缩候选”。

Paragraph 在两次固定意图重放中都没有补入任何缺失证据，默认仍应关闭。

细粒度 Concept 同样没有补入任何 Candidate 缺失证据，但在“妃咲/未花治理类比”题上三轮都改善了 Selected 覆盖，三轮平均提升 20.83 个百分点。三道缺口题的平均 Selected 增量在三轮中均为正；代价是阿里乌斯题在第三轮下降 12.5 个百分点。额外的 8 道 Concept 桥题安全筛查全部保持满分。因此它表现出窄而可重复的精排收益，但还不足以全局默认启用，应保留为实验性处理臂并扩大困难题样本。

## 2. 本阶段验证什么

### 2.1 核心问题

Stage 13 的数据库只有 342 个 Episode，Candidate@100 相当于把约 29% 的全库送入精排。Paragraph 和 Concept 没有收益，可能只是因为基线候选过宽、正确证据早已全部进入 Candidate。

Stage 14 因此验证三件事：

1. 当 Episode 增长到数千条时，Dense + Sparse + Concept 基础召回是否仍能保持高覆盖。
2. 当真实 Candidate 缺口出现后，Paragraph 或更积极的 Concept 是否能稳定补入缺失证据。
3. 扩大候选空间后，LLM 能否把完整的多跳证据链压缩到最终 Top-20。

增长边数量不是评测指标。增长边只影响问题设计所需的联想路径长度；真正的成功标准始终是必需证据是否被召回、是否进入最终证据集，以及最终回答能否区分事实、推论和未知。

### 2.2 本阶段没有验证什么

- 基础评测把 `graph_max_hops` 和 `growth_max_rounds` 都设为 0，所以不验证查询时增长边收益。
- 大批量导入为节省数小时关系判断调用，延后了大部分 Episode–Episode / Concept–Concept 推断关系生成；本数据库是基础检索压力资产，不是完整增长图资产。
- 新增语料主要充当规模压力和困难干扰项。23 道冻结问题的正证据仍集中在先前核验过的证据文件，所以本阶段不等同于“从新增文件学习全新事实”的泛化评测。
- `--retrieval-only` 跳过最终自然语言答案生成，因此本阶段评的是证据链可用性，不是答案措辞质量。

## 3. 专有名词定义

### 3.1 Source、Episode、Concept、Association

- **Source**：长度受控的原始文本片段，是证据原文的持久化载体。
- **Episode**：由 Source 提取出的自包含事件或状态摘要，是当前主要检索和回答证据单位。
- **Concept**：可跨记忆复用的语义锚点，包括人物、组织、地点、命名物品、持续心理、关系主题等。
- **Association**：Episode 与 Concept 或其他记忆节点之间的有向关系。`generation=0` 表示直接经验关系，`generation>=1` 表示至少使用过一次推断前提。

### 3.2 Candidate Recall@100

每道题预先冻结若干“必需证据组”。同一组内可以有多个等价 Episode，只要命中其中一个就算该组被覆盖。Candidate Recall@100 是前 100 个粗召回候选覆盖的必需证据组比例。

它回答的是：“正确证据有没有机会交给后续模型判断？”

### 3.3 Selected Recall@20

LLM 精排把 Candidate 压缩到 20 个 Episode。Selected Recall@20 是这 20 个结果覆盖的必需证据组比例。

它回答的是：“在有限回答上下文里，证据链还剩多少？”

### 3.4 普通运行与固定意图重放

- **普通运行**：每次由 LLM 重新解析问题、生成原子查询和追问，代表系统日常真实行为，但会包含查询改写随机性。
- **固定意图重放**：复用中间层基线保存的同一个 `QueryIntent` 和同一组追问，只更换数据库或处理机制。它用于隔离语料、Paragraph、Concept 本身的因果影响。

### 3.5 Paragraph

Paragraph 是 Source 的确定性原文小分块，不是新的事实节点。本阶段 P 处理臂把 Paragraph embedding 作为候选种子，测试被 Episode 摘要省略的原文是否能帮助找回对应 Episode。

### 3.6 细粒度 Concept 与晋升门

细粒度提取比保守策略更积极，软目标为每个普通 Episode 提取 4—10 个跨记忆锚点。候选先在全库统计复现情况，再由 LLM 判定：

- 复用已有 Concept；
- 晋升为新的持久 Concept；
- 仅作为 transient 候选，不写入图。

新的 Concept 至少需要覆盖 2 个不同 Episode；这样可以减少只出现一次的临时名词污染长期图。

## 4. 数据集与扩展策略

完整语料目录共有 564 个 JSON 文件，约 35 MB：

| 模块 | 文件数 | 说明 |
|---|---:|---|
| `main` | 77 | 全部纳入 |
| `event` | 7 | 全部纳入 |
| `favor` | 480 | 每个一级角色目录选择一个中位编号文件，并强制保留旧证据文件 |

最终冻结选择 208 个文件：全部主线、全部活动、123 个好感角色目录的分层样本和已有基线证据。选择清单在导入前计算 SHA-256 并冻结，防止根据结果事后换样本。

完整 564 文件预检得到：

- 38,514 个可读取剧情块；
- 按当前参数预计产生 1,598 个 Source；
- JSON 解析失败 0；
- 空文本失败 0。

本阶段实际导入的 208 文件产生 735 个 Source。预估为 736，差一条来自旧基线资产的历史分片边界，不影响文件覆盖；审计以实际 Source、任务和 Episode 闭包为准。

## 5. 导入与提取参数

### 5.1 Source 分片

当前统一兼容层先把 JSON 转为有自然边界的剧情块，再按以下参数组装 Source：

| 参数 | 值 |
|---|---:|
| 目标长度 | 6,000 字符 |
| 硬上限 | 8,000 字符 |
| 相邻重叠 | 800 字符 |
| 边界偏好 | 对话/剧情块自然边界优先 |

这组参数让 Source 足以保留人物标记和上下文，又不至于让 Episode 只能回指一段过长原文。

### 5.2 Episode 提取

- 推理模型：`deepseek-ai/DeepSeek-V3.2`。
- prompt 版本：`v3.23_association_generation`。
- 第一遍提取自包含 Episode，第二遍检查粒度、代词、遗漏和需要拆分的复合事件。
- 同一 Source 允许重叠 Episode。
- 原人物标记和多语言别名保留。
- 失败最多重试 6 次；本轮没有文件级 fallback。

### 5.3 Embedding

- 模型：`Pro/BAAI/bge-m3`。
- 维度：1,024。
- SQLite BLOB：`float32`。
- RAM 索引：`float32`。
- 写入前 L2 normalization；检索时单位向量点积等价于 cosine similarity。

### 5.4 并发与断点

- 文件并发：4。
- 单文件 Source 准备并发：4。
- 进度账本逐文件记录 `running/completed`，可恢复中断运行。
- SQLite 先提交，内存索引随后更新；SQLite 始终是 source of truth。

该进度账本只保证本次冻结导入的可恢复性，不重新引入已经取消的“整个 demo 必须通用幂等”要求。

## 6. 实验资产

| 资产 | Source | Episode | Concept | Paragraph | Association |
|---|---:|---:|---:|---:|---:|
| 旧基线 A | 68 | 342 | 392 | 0 | 1,825 |
| 主线+活动 ME | 432 | 1,960 | 1,294 | 0 | 6,130 |
| 全扩展 MEF | 735 | 3,582 | 2,883 | 0 | 9,610 |
| Paragraph 对照 P | 432 | 1,960 | 1,294 | 2,930 | 6,130 |
| 细粒度 Concept 对照 C | 432 | 1,960 | 1,311 | 0 | 8,161 |

全扩展资产的 Episode 分类：

| 模块 | Episode |
|---|---:|
| `main` | 1,739 |
| `event` | 212 |
| `favor` | 1,631 |

全扩展资产的 Association：

| relation_type | 数量 |
|---|---:|
| semantic | 9,032 |
| temporal | 425 |
| identity | 100 |
| causal | 31 |
| recall_trigger | 9 |
| co_occurrence | 7 |
| interpersonal | 6 |

其中 `generation=0` 为 8,808 条，`generation=1` 为 802 条。

## 7. 数据库审计

全扩展快照通过以下检查：

- 冻结清单 208/208 文件在进度账本中均为 `completed`。
- 208/208 文件均有提取任务和 Episode。
- `PRAGMA integrity_check = ok`。
- 外键违规 0。
- 多态 Association 非法类型 0、悬空端点 0。
- 孤立 Source 0。
- Episode 3,582/3,582 的 embedding 均为 4,096 字节。
- Concept 2,883/2,883 的 embedding 均为 4,096 字节。
- 非有限向量 0。
- Episode norm 范围约为 0.99999994—1.00000012。
- Concept norm 范围约为 0.99999988—1.00000012。

数据库中保留一条 `partial` 中间任务：`main/33235.json` 第一遍因粒度审计未返回有效 Episode，但同一 Source 的第二遍任务已经 `completed` 并产出 Episode。它是可追溯的已恢复错误，不是最终资产失败。

## 8. 基础检索结果

### 8.1 协议

- 问题：23 道。
- 高难网络题：7 道。
- Paragraph 机制题：8 道。
- Concept 桥题：8 道。
- Dense + Sparse + Concept 联合候选。
- Candidate 上限：100。
- LLM 最终选择：20。
- Paragraph：关闭。
- 图遍历：0 跳。
- 查询增长：0 轮。
- 最终答案生成：关闭。

### 8.2 普通运行总结果

| 数据层 | Episode | Candidate@100 平均 | Candidate 最低 | Candidate 完美题 | Selected@20 平均 | Selected 最低 | Selected 完美题 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ME | 1,960 | 97.83% | 75.0% | 20/23 | 94.69% | 62.5% | 18/23 |
| MEF | 3,582 | 97.34% | 75.0% | 19/23 | 92.15% | 50.0% | 16/23 |

普通运行从 ME 到 MEF 的 Candidate 平均下降 0.48 个百分点，Selected 平均下降 2.54 个百分点。不过这组差值混合了数据库变化与 LLM 重新拆解查询的随机性，不能直接解释为扩容因果效应。

### 8.3 MEF 分题型结果

| 题型 | 题数 | Candidate@100 平均 | Candidate 最低 | Selected@20 平均 | Selected 最低 |
|---|---:|---:|---:|---:|---:|
| 高难网络题 | 7 | 91.27% | 75.0% | 78.97% | 50.0% |
| Paragraph 细节题 | 8 | 100% | 100% | 100% | 100% |
| Concept 桥题 | 8 | 100% | 100% | 95.83% | 66.67% |

这证明只看总体均值会高估可靠性。当前基础策略对单槽细节题足够强，但对需要 7—9 个独立证据槽的网络题不可靠。

### 8.4 固定意图的扩容因果重放

| 问题 | ME Candidate | MEF Candidate | ME Selected | MEF Selected | 解释 |
|---|---:|---:|---:|---:|---|
| 古圣堂结构与袭击链 | 87.5% | 100% | 75.0% | 87.5% | 补入阿里乌斯作战部署，最终位于 Candidate 82 / Selected 20 |
| 阿里乌斯两名资助者 | 75.0% | 87.5% | 75.0% | 62.5% | 补入真琴被骗，Candidate 36，但精排未保留 |
| 妃咲/未花治理类比 | 87.5% | 87.5% | 62.5% | 75.0% | Candidate 不变，精排覆盖重排 |
| 日富美/梓身份链 | 100% | 100% | 100% | 100% | 普通运行中的回退消失 |
| 日奈职责与疲惫 | 100% | 100% | 77.78% | 66.67% | Candidate 稳定，精排真实回退 |
| 未花动机四层 | 100% | 100% | 87.5% | 87.5% | 普通运行中的回退消失 |
| 未花背叛三视角 | 100% | 100% | 100% | 66.67% | Candidate 稳定，精排真实回退 |

固定结果支持以下判断：

1. 扩大语料对基础 Candidate 的净信号是正向的：2 个原缺口改善，其他受控题不变，没有受控 Candidate 回退。
2. 普通运行出现的 Candidate 波动主要由查询意图和追问变化造成。
3. Candidate 改善不保证 Selected 改善。Top-20 精排可能保留新证据，也可能为它牺牲另一个证据槽。
4. 当前精排器仍以“整体看起来相关”为主，没有可靠执行每个独立证据槽至少保留一个候选的约束。

## 9. Paragraph 与细粒度 Concept

### 9.1 Paragraph

P 资产从 425 个 Source 建立 2,930 个 Paragraph，失败 0；全部 embedding 为 1,024 维 float32。

普通 P 运行表面上把古圣堂题 Candidate 从 87.5% 提升到 100%，但该运行重新生成了查询意图。复用完全相同的基线意图后：

- 第一次：3 道真实 Candidate 缺口题均为 0 正 / 3 平 / 0 负。
- 第二次：仍为 0 正 / 3 平 / 0 负。
- Selected 第一次为 3 平；第二次古圣堂题反而下降 12.5 个百分点。
- 两轮的候选 ID/顺序和最终 ID 存在重排，说明模型/远程 embedding 排序并非逐 ID 稳定，但目标证据组覆盖结论一致。

结论：Paragraph seed 没有稳定补入缺失证据，当前默认继续关闭。它仍可能在最终答案需要精确引用原文时有价值，但那应使用答案级事实准确率评测，而不是把它当作已经证明有效的 Episode 召回器。

### 9.2 细粒度 Concept

C 资产对 425 个有 Episode 的 Source 重新执行细粒度提取：

- 产生 7,751 次 Concept 候选出现。
- 归并为 2,079 个候选组。
- 1,055 组通过明确名称/别名直接复用已有 Concept。
- 剩余 1,024 组进入 LLM 准入；其中 179 组复用相似已有节点，17 组晋升为新 Concept，828 组作为 transient 丢弃。
- Concept 从 1,294 增至 1,311，Association 从 6,130 增至 8,161。
- 失败 Source 0，准入格式错误 0；所有 Concept embedding 都是 4,096 字节 float32。

为了分离“多连旧节点”和“真正增加细节点”，在新候选尚未晋升时冻结了 `C-reuse` 中间臂：Concept 保持 1,294，Association 已增至 8,032。

三轮固定意图结果：

| 处理臂 | Candidate 三轮平均增量 | Selected 三轮平均增量 | 稳定信号 |
|---|---|---|---|
| C-reuse | 0 / 0 / 0 | 0 / 0 / +4.17 pp | 无稳定总体收益 |
| 完整 C | 0 / 0 / 0 | +8.33 / +4.17 / +4.17 pp | 3/3 轮总体为正 |

逐题看：

- 古圣堂题：完整 C 三轮都是 0。
- 阿里乌斯两名资助者题：前两轮为 0，第三轮 Selected -12.5 pp。
- 妃咲/未花治理类比题：Selected 分别 +25、+12.5、+25 pp，三轮平均 +20.83 pp；稳定补回的核心证据是未花利用茶会命令控制校内力量并自认叛徒的政治行动组。
- 8 道 Concept 桥题安全筛查：Candidate 8/8 不变，Selected 8/8 不变，全部保持 100%。

解释：积极 Concept 没有解决粗召回缺口，但改变了候选构成和精排输入，使一条跨学校治理类比证据链更容易保留。完整 C 比只复用旧节点的方向更稳定，说明 LLM 相似复用和少量新晋升节点可能提供了增量；不过结果无法证明 17 个新节点本身就是唯一原因，因为处理臂同时增加了 179 组语义复用关系。

决策：保留 C 资产和细粒度策略作为实验性精排辅助，不全局替换保守 Concept。升级为默认前至少需要在更多真实 Candidate 缺口和新文件正证据题上复现收益，并消除阿里乌斯题的偶发回退。

## 10. 运行成本与故障信息

208 文件的完整导入阶段：

- 墙钟时间约 13,185 秒，即 3 小时 39 分。
- JSONL 日志约 327 MB，共 28,185 个事件，解析错误 0。
- 模型请求 4,667 次：推理 2,633 次、embedding 2,034 次。
- 记录到 62 次 HTTP 429、1 次 HTTP 400、8 次非 HTTP 错误。
- 重试等待 71 次；重试时间中位数约 11.47 秒，P95 约 35.05 秒。
- 文件级失败 0，fallback 0。

主要延迟：

| 阶段 | 样本数 | P50 | P95 |
|---|---:|---:|---:|
| Episode 提取 | 686 | 30.77 s | 49.02 s |
| Concept 提取 | 1,060 | 41.82 s | 86.98 s |
| 第二遍理解 | 356 | 16.26 s | 40.90 s |
| Embedding | 2,034 | 0.94 s | 1.65 s |

因此当前批量导入的耗时瓶颈不是 SQLite 或 float32，而是 Concept/ Episode 的推理调用和供应商限流。下一次应优先优化批大小、准入并发、日志裁剪和 429 自适应，而不是过早更换向量索引。

本轮还发现细粒度 Concept 的“发现”阶段使用了 8 并发，但 1,024 个新候选的 52 批准入判断仍串行。代码已修复为准入请求并发、决定按原批次顺序应用；当前 C 资产为避免中途改变实验条件，仍按旧串行路径完成。新增并发路径已纳入 128/128 单元测试。

## 11. 当前技术问题

### 11.1 查询拆解不是稳定控制变量

同一个问题在不同运行中会生成不同原子查询和追问，足以让 Candidate 证据组出现 ±12.5 个百分点变化。普通在线结果仍有意义，但机制消融必须复用同一 QueryIntent；否则很容易制造伪收益和伪回退。

### 11.2 Top-20 精排没有证据槽约束

复杂问题要求多个互不替代的事实。例如“谁资助、谁执行、谁被骗、谁失去控制”是四种角色，不应由多个泛相关的阿里乌斯片段占满。当前精排器虽然有覆盖审计，但最终仍能丢掉已进入 Candidate 的必要槽。

### 11.3 Candidate@100 平均值掩盖长尾失败

总体 97.34% 看似达到先前的 95% 目标，但高难网络题只有 91.27%，最低 75%。后续门槛必须同时报告：

- 全题均值；
- 高难题均值；
- 逐题最低值；
- 完美闭合题数。

### 11.4 新增事实泛化尚未被覆盖

新增 197 个非基线文件显著增加了干扰项，却没有成为大多数题目的正证据。下一阶段需要从这些新增文件中独立抽样并冻结新问题，才能验证系统是否不仅“抗干扰”，还真正“学会新内容”。

### 11.5 图资产与检索资产需要分开

本轮为了完成数千 Episode 的基础召回验证，批量延后了推断关系。该决策对 `graph_max_hops=0` 的主实验有效，但后续重新测试增长边前，必须从 MEF 复制单独资产并补建推断 Association，不能直接把本轮关系数当作完整神经图规模。

## 12. 下一阶段建议

优先顺序如下：

1. **先修精排覆盖策略。** 把问题分成明确证据槽，每槽保留最低配额，再在剩余槽位做全局相关度排序；避免 Top-20 被同主题近重复片段占满。
2. **固定查询计划做回归。** 为 7 道高难题保存 QueryIntent、原子查询、追问和输入哈希；每次代码或资产变化先跑确定性重放，再跑普通在线问题。
3. **把可靠性门改成逐题门。** 候选阶段建议要求高难题每题 Recall@100 ≥95%，而不是只看总体平均；Selected 阶段至少先恢复到每题 ≥87.5%，再逐步提高。
4. **建立新增文件正证据集。** 从此次新增的 main/event/favor 文件中抽取新的跨文件问题，确保训练/设计阶段没有读过这些答案，并冻结 Episode 证据组。
5. **在独立数据库补建推断关系。** 基础检索达到门槛后，再恢复 Association 多跳和自主增长实验；继续使用 `generation` 区分直接经验与推论距离。
6. **Paragraph 保持可回退关闭。** 只有在答案级原文准确率实验出现稳定正差值时再启用。
7. **Concept 进入扩大样本确认，而非直接默认启用。** 当前已出现窄而可重复的 Selected 收益，但 Candidate 无收益且有偶发回退；下一轮应扩大困难题并做按证据槽的配对审计。

## 13. 可复现实验资产

- 完整语料预检：`validation/stage14-full-corpus-preflight.json`
- 预注册：`validation/stage14-expanded-corpus-preregistration.json`
- 冻结选样：`validation/evaluation-stage14-expanded-corpus/assets-v1/selection-manifest.json`
- 导入进度账本：`validation/evaluation-stage14-expanded-corpus/assets-v1/import-progress.json`
- 全扩展数据库：`validation/evaluation-stage14-expanded-corpus/tier-mef/graph.db`
- 全扩展审计：`validation/evaluation-stage14-expanded-corpus/tier-mef/database-audit.json`
- 23 题：`validation/evaluation-stage14-expanded-corpus/stage14-questions.json`
- 证据清单：`validation/evaluation-stage14-expanded-corpus/stage14-evidence-manifest.json`
- ME 基础结果：`validation/evaluation-stage14-expanded-corpus/tier-me/base-retrieval.json`
- MEF 基础结果：`validation/evaluation-stage14-expanded-corpus/tier-mef/base-retrieval.json`
- 扩容固定重放：`validation/evaluation-stage14-expanded-corpus/tier-mef/exact-gap-replay-vs-me.json`
- 新 Candidate 回退重放：`validation/evaluation-stage14-expanded-corpus/tier-mef/exact-new-regressions-vs-me.json`
- Selected 回退重放：`validation/evaluation-stage14-expanded-corpus/tier-mef/exact-selected-regressions-vs-me.json`
- Paragraph 资产报告：`validation/evaluation-stage14-expanded-corpus/tier-me-P/graph.report.json`
- Paragraph 两轮重放：`validation/evaluation-stage14-expanded-corpus/tier-me-P/exact-gap-replay.json`、`exact-gap-replay-r2.json`
- 旧 Concept 复用中间臂：`validation/evaluation-stage14-expanded-corpus/tier-me-C-reuse/graph.db`
- 旧 Concept 复用三轮重放：`validation/evaluation-stage14-expanded-corpus/tier-me-C-reuse/exact-gap-replay.json`、`exact-gap-replay-r2.json`、`exact-gap-replay-r3.json`
- Concept 资产报告：`validation/evaluation-stage14-expanded-corpus/tier-me-C/graph.report.json`
- 完整 Concept 三轮重放：`validation/evaluation-stage14-expanded-corpus/tier-me-C/exact-gap-replay.json`、`exact-gap-replay-r2.json`、`exact-gap-replay-r3.json`
- Concept 桥题安全筛查：`validation/evaluation-stage14-expanded-corpus/tier-me-C/exact-concept-safety-screen.json`
- 综合分析：`validation/evaluation-stage14-expanded-corpus/analysis-report.json`
- 导入模型日志汇总：`validation/evaluation-stage14-expanded-corpus/assets-v1/model-log-summary.json`

## 14. 验证状态

- 完整语料预检：通过。
- 208 文件导入：208/208 完成。
- 数据库完整性与向量 dtype：通过。
- 23 题 MEF 基础检索：已执行完成。
- 扩容固定意图归因：已执行完成。
- Paragraph 两轮固定意图对照：已执行完成，无 Candidate 收益。
- 细粒度 Concept 三轮对照与 8 题安全筛查：已执行完成。
- Python 全量语法编译：通过。
- 标准库 `unittest` 全量测试：128/128 通过；其中 Concept 晋升门测试已改为 2 个准备线程，覆盖新的并发准入路径。
