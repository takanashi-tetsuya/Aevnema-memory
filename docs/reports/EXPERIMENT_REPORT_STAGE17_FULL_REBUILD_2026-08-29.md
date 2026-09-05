# Stage 17：完整语料重建、基础召回与增长边效益报告

日期：2026-08-29（Asia/Tokyo）  
提示版本：`v3.25_balanced_orchestration_guarded`  
最终数据库：`validation/evaluation-stage17-full-rebuild/assets-v3-throughput/graph.db`

## 1. 本阶段要回答的问题

Stage 16 只在 48 个冻结高风险案例上验证了导入编排优化。本阶段用蔚蓝档案测试集的 208 个文件完成一次
全量重建，并依次回答：

1. `balanced` 导入流程在完整语料上是否能稳定完成、是否真的减少 token；
2. 不依赖增长边时，Dense + Sparse + LLM rerank 的基础检索能否稳定达到 Recall@20 > 95%；
3. 直接关系图是否比纯向量更有用；
4. 查询期自主增长的新 Association 是否能补齐多跳证据，并被后续回答实际使用；
5. `confidence` 与 `generation` 分离后，当前主要瓶颈究竟是可信度审计，还是候选、路径和证据预算。

结论先行：

- 完整导入成功，208/208 文件完成，数据库审计通过，0 个失败任务；
- 新流程总 token 比旧完整导入日志少 7.64%，按实际记录的 Source 归一化后约少 15.05%；
- 基础 Candidate@100 平均 98.43%，Top-20 平均 94.37%，比旧基线分别提高 1.63 和 4.63 个百分点；
- 因为仍有 5/23 题的 Top-20 没有超过 95%，不能宣称基础策略已稳定达到 Recall@20 > 95%；
- 三道深度题严格通过数为：纯向量 1/3、静态图 2/3、增长图 0/3；
- 增长模式写入 11 条 generation=1 边，但没有产生净收益。当前瓶颈是新增路径挤占原始证据、无效边写入过多、
  以及增长后未保护原子查询锚点，不是 generation 或可信度过严；
- 下一阶段不应继续放宽审计。应先把原始锚点预算与图路径预算分离，并在持久化前验证新边的边际检索效益。

## 2. 术语与计分定义

### 2.1 Source、Episode、Concept、Association

- **Source**：长度适中的原始证据分片。文本保持原样，`source_key` 指向语料中的文件名或相对键。
- **Episode**：可独立理解、通常围绕一个主要事件或状态的事实单元。Episode 通过 `source_id` 回到原始 Source。
- **Concept**：人物、组织、地点、物品、抽象主题、情绪或其他可产生联想的自然语言概念。
- **Association**：Episode/Concept 之间的有方向关系。`relation_text` 保存关系的自然语言含义，机器字段只保存
  必要的类型、权重、可信度、极性和推断距离。

### 2.2 confidence 与 generation

- **confidence**：当前证据对关系成立的支持程度。它回答“这条关系有多可信”。
- **generation**：关系离直接经验有多少层推断。generation=0 表示直接事实关系；只使用直接 Episode/Concept
  得出的第一层综合推论为 generation=1；若使用已有 generation=n 的推论继续推断，则新关系取
  `max(前提 generation) + 1`。

两者不能互相替代：高 confidence 的 generation=3 仍然离原文较远；低 confidence 的 generation=0 仍然是
直接观察，只是证据本身不充分。本阶段新增关系全部是 generation=1，没有出现远层推断，因此增长失败不能归因于
generation 限制。

### 2.3 Candidate@100 与 Recall@20

- **Candidate@100**：Dense、Sparse 和多原子查询合并后，前 100 个候选覆盖了多少必需证据组。
- **Recall@20**：LLM rerank 后最终 20 个 Episode 覆盖了多少必需证据组。历史字段仍叫 `recall_at_30`，
  但当前配置和报告均使用 20 个结果；本报告按真实配置称 Recall@20。
- **证据组**：同一事实可能由多个等价或重提取后的 Episode 表达，任意命中组内一个即可覆盖该事实。
- **严格通过**：必需 Episode 组、必需 Source、答案关键词和答案审计都通过；图模式还必须满足模式行为。
  增长模式要求本题改变的关系能被回答路径使用，不能只凭“写了边”计成功。

### 2.4 静态图与增长图

- **纯向量（vector_only）**：不遍历 Association，也不新增关系。
- **静态图（graph_static）**：遍历已有关系，但查询时不写入新关系。
- **增长图（graph_growing）**：检索后提出新关系，经主模型和对抗审计共同接受后写入，再重新遍历并生成答案。

最终导入刻意延后了推断关系：冻结数据库只有 8,904 条 `episode -> concept` 的 `involves` 直接边，全部
generation=0。因此静态图测的是直接结构化连接，增长图才测查询期推断，避免把预先写好的答案误算成自主增长收益。

## 3. 导入路线与断点决策

### 3.1 串行即时推断试跑

第一条路线每次导入同时判断推断关系。7 个完整文件、31 个 Source、168 个 Episode 已耗时约 47 分钟和
478,422 tokens，其中关系判断 105,267 tokens。线性外推整库约需 10–12M tokens，墙钟可能超过 20 小时。

该路线被主动停止，数据库和日志保留为成本对照。停止原因不是余额，而是它违背“缩短流程、减少 token”，
并把昂贵推断花在尚未被查询证明有用的关系上。

### 3.2 保守并发试跑

第二条路线使用 4 个文件 worker、每文件 1 个 Source worker，并延后推断关系。13 个文件、57 个 Source、
306 个 Episode 在 2,059 秒内完成，646,094 tokens，0 错误，峰值 4 个在途请求。吞吐约 1.7 Source/分钟，
稳定但仍偏慢。

### 3.3 最终全量路线

最终使用 4 个文件 worker、每文件最多 4 个 Source worker、`balanced` 编排、Paragraph 关闭、Concept
保守模式、推断关系延后。这样保留直接证据边，把推断成本移动到实际查询出现之后。

结果：

| 项目 | 结果 |
|---|---:|
| 输入文件 | 208/208 完成 |
| 分类 | main 77、favor 124、event 7 |
| Source | 736 |
| Episode | 4,073 |
| Concept | 2,970 |
| Concept alias | 4,232 |
| Association | 8,904 |
| Paragraph | 0 |
| 提取任务 | 1,756 |
| 失败任务 | 0 |
| 墙钟时间 | 11,411.5 秒，约 3 小时 10 分 |
| 模型请求 | 4,601 |
| 总 token | 8,060,618 |

模型调用遇到 HTTP 500 三次、HTTP 400 一次和一次非 HTTP 连接错误，全部通过既有重试恢复；0 fallback、
0 validation failure。数据库 `integrity_check` 为 `ok`，外键违规、悬空 Association、未完成中间任务均为 0。

Episode 和 Concept embedding 均为 1,024 维 float32：每条 4,096 bytes，所有值有限，L2 norm 在
0.99999988—1.00000012 之间。磁盘和内存继续统一 float32，本阶段没有重新引入 float8。

## 4. 导入质量与成本

### 4.1 与旧完整资产比较

旧日志只记录了 677 个本轮处理的 Source，另有 59 个 Source 在旧数据库中预先存在；新流程从头处理全部
736 个 Source。因此既报告整轮总量，也报告按日志内实际处理 Source 归一化的成本。

| 指标 | 旧流程 | 新流程 | 变化 |
|---|---:|---:|---:|
| 总 token | 8,727,786 | 8,060,618 | -7.64% |
| 每个实际处理 Source | 12,891.9 | 10,951.9 | -15.05% |
| 请求数 | 4,667 | 4,601 | -66 |
| 日志时间跨度 | 13,184.7 s | 11,411.5 s | -13.45% |
| Episode | 3,582 | 4,073 | +13.71% |
| Concept | 2,883 | 2,970 | +3.02% |

新流程在生成更多 Episode 和 Concept、并从头处理更多 Source 的情况下仍减少总 token，说明 Stage 16 的编排
优化在整库上成立。但整轮只减少 7.64%，小于冻结案例中最昂贵两个环节的 52.0%，原因是 Episode 初提取、
Concept 提取和 embedding 等未优化阶段仍占大量成本。

阶段变化：

- 二次理解从 2,139,111 降至 1,477,412 tokens，减少 30.93%；
- 旧时间审计 + 粒度审计共 1,819,467 tokens，新合并审计为 1,561,626，减少约 14.17%；
- 导入期关系判断从 85,730 降为 0；
- Episode 初提取增加 210,432，Concept 提取增加 254,352，与新增 Episode/Concept 数量一致；
- embedding 减少 127,996 tokens。

### 4.2 二次理解负载

4,073 条 Episode 中有 1,020 条被二次理解修订，修订率 25.04%。这说明二次理解不是罕见 fallback，仍是
主流程的重要组成部分。下一轮若继续压缩 token，应优先对高修订率文件研究更好的首轮提取，而不是简单删除
二次理解。

### 4.3 Concept 质量风险

活跃 Concept 没有规范名完全重复组，但存在 138 个归一化别名同时指向多个活跃 Concept。典型情况包括：

- `未花 / Mika / 미카` 被分成普通、未武装状态、不同译名等多个 Concept；
- `日富美 / Hifumi`、`小春 / Koharu` 有多语言形式分裂；
- `???` 同时成为多个未知人物概念的别名；
- “阿露”的数据库别名包含阿露、アル、Aru、아루等，但用户常用译名“阿鲁”缺失。

这不是 embedding 存储问题，而是 Concept 归一化和多语言实体消歧问题。暂不自动批量合并，因为服装形态、未知
人物和同名角色有可能确实需要区分；应先加入别名冲突审计和人工审查工具。

## 5. 基础检索：不依赖增长边

### 5.1 方法

使用 23 道扩展问题、82 个必需证据组。旧数据库 Episode ID 不能直接用于新提取结果，因此映射严格限制在
相同 `source_key` 内，对旧 Episode 与新 Episode 做 float32 cosine 比较，每条旧 Episode 最多保留 3 个、
相似度至少 0.55 的新候选。这样允许一条旧 Episode 被新提取器拆成两到三条，又不会跨文件扩大证据范围。

映射结果无空组、无缺失 Source、无缺失旧 ID；最佳映射相似度平均 0.8930、最低 0.6656。映射不是人工改答案，
但仍是自动近似，因此本报告同时保留完整映射报告供抽查。

检索配置为 Paragraph off、增长 off、图遍历 off、Dense + Sparse、adaptive rerank、Candidate@100、最终 Top-20。

### 5.2 结果

| 指标 | 旧基线 | 新完整重建 | 变化 |
|---|---:|---:|---:|
| Candidate@100 平均 | 96.80% | 98.43% | +1.63 pp |
| Candidate@100 最低 | 75.00% | 75.00% | 0 |
| 近完整候选题数 | 18/23 | 21/23 | +3 |
| Recall@20 平均 | 89.73% | 94.37% | +4.63 pp |
| Recall@20 最低 | 0% | 62.50% | +62.50 pp |
| Recall@20 >95% 的题 | 16/23 | 18/23 | +2 |

新流程明显改善基础检索，但不能回答“已经稳定 >95%”。如果标准是 23 道题的宏平均，94.37% 仍低于 95%；
如果标准是每题都超过 95%，只有 18/23 达标。

低分题分为两类：

1. Candidate 阶段已漏证据：`kisaki_mika_governance_analogy_with_boundary` 为 75.0%/62.5%，
   `hina_public_duty_personal_exhaustion_and_teacher_response` 为 88.89%/88.89%。应改原子查询、别名和多主题候选生成。
2. Candidate 完整但 rerank 挤出证据：两道 concept bridge 题从 100% 降到 66.67%，
   `eden_obligation_to_new_institution` 从 100% 降到 85.71%。应改按原子槽位分配 Top-20，而不是继续增加 Candidate K。

本轮基础评测使用 124 次模型请求、732,082 tokens、约 549.8 秒，无 API 错误。它说明当前高成本主要来自证据
覆盖审查和 rerank，而不是 NumPy float32 全量扫描。

## 6. 三道深度题的图对照

### 6.1 证据与严格条件

三道题共有 9 个证据组，重新映射后无空组，最佳映射相似度平均 0.8768、最低 0.7358。严格通过要求：

- 所有指定 Episode 事实组进入最终证据；
- 所有目标 Source 被引用；
- 答案包含必要实体/事实词；
- 最终答案审计为 valid；
- 模式行为正确。增长模式还检查本题改变的 Association 是否进入回答路径。

所有 9 次查询的最终答案审计都为 valid，但严格总通过只有 3/9。这证明答案审计主要保证“已写出的断言受当前
证据支持”，不能替代完整性评测：一个谨慎但漏掉半条因果链的答案仍可能是 valid。

### 6.2 严格结果

| 问题 | 纯向量 | 静态图 | 增长图 |
|---|---:|---:|---:|
| 补习部表象与真相 | 失败 | 失败 | 失败 |
| 乐园悖论与两位高层 | 通过 | 通过 | 失败 |
| 古圣堂袭击因果链 | 失败 | 通过 | 失败 |
| 合计 | 1/3 | **2/3** | 0/3 |

具体解释：

- **补习部题**：纯向量和静态图都找到了 `31090` 的退学、叛徒和伊甸条约事实，但漏掉 `31060` 的表面成立
  背景。增长模式建立了 8 条关系，只把其中 1 条放入回答路径，最终还丢失全部三个指定证据组。
- **乐园悖论题**：纯向量和静态图完整覆盖三组证据。增长模式新建 3 条关系且 3/3 都进入路径，但“乐园悖论”
  原始 Episode 被图路径挤出，剩下渚与未花两组，因此失败。
- **古圣堂题**：静态图成功补齐爆炸现场、阿里乌斯执行、未花此前协助三段闭环；纯向量缺两组。增长模式没有
  新边通过写入，也没有复用前两题留下的关系补回 `32170` 前置证据，因而失败。

### 6.3 增长边内容

增长数据库新增 11 条边，全部满足：

- generation=1；
- `claim_level=supported_inference`；
- `audit_status=dual_accepted`；
- confidence 为 0.75—0.90；
- 没有 generation=2 或更远推断。

第一题的 8 条边多数在重复表达“补习部表面目的—退学真实目的—叛徒嫌疑—梓实例”，只有 1 条进入最终路径。
第二题的 3 条边连接未花支援阿里乌斯与政变目标、渚的叛徒怀疑与集中嫌疑人、未花的具体执行手段，三条都被
使用，但它们占用了原始哲学锚点的预算。

因此问题不是边的文本完全错误，也不是审计拒绝所有推论，而是：

1. 在写入前没有要求新边提供不可替代的边际证据；
2. 一题可以写入多条高度冗余边；
3. 重遍历后图路径端点会与原始检索锚点竞争同一 30 条 Episode 预算；
4. 系统保护了部分 atomic anchor，但没有对每个原子证据槽做不可驱逐保留；
5. 新边持久化与“当前题实际产生收益”仍然是两个分离步骤。

### 6.4 模式成本

| 模式 | 请求 | token | 时间 | 严格通过 |
|---|---:|---:|---:|---:|
| 纯向量 | 28 | 400,074 | 978.6 s | 1/3 |
| 静态图 | 33 | 595,457 | 1,175.3 s | 2/3 |
| 增长图 | 39 | 609,357 | 1,280.6 s | 0/3 |

静态图比纯向量多 48.84% token、20.11% 时间，换来一题严格收益；增长图比静态图再多 2.33% token、8.95%
时间，却从 2/3 降到 0/3。样本只有三题，不能估计总体收益率，但已经足以否定“当前增长策略稳定有益”。

## 7. 当前架构判断

### 7.1 已得到验证的部分

1. float32 SQLite BLOB、float32 RAM index、启动 rebuild、L2 normalization 均工作正常。
2. 4 文件并发 + 每文件多 Source 并发能在约 3 小时完成 208 文件，且没有 429 风暴。
3. 延后推断关系显著缩短导入路径；SQLite 保持 source of truth，评测副本没有污染正式数据库。
4. 新提取器让基础 Candidate 和 Top-20 都比旧完整资产更好。
5. 静态直接关系图能在至少一个严格多跳问题上补齐纯向量缺失的证据。
6. 双模型增长审计没有写入 generation>1 的远层推论，也会拒绝第三题中不够充分的新边。

### 7.2 尚未通过的部分

1. 基础 Recall@20 尚未稳定超过 95%。
2. 查询期新边没有在本轮产生净收益。
3. “写入多少边”不能作为增长指标；必须测同一题或后续题的证据增益和答案增益。
4. Concept 多语言归一化仍存在明显分裂和别名冲突。
5. 直接 `involves` 图会显著增加图模式上下文，关系信息量门控仍不足。
6. 自动证据映射是可靠的工程近似，不是人工金标；未来大规模评测需要人工抽检映射边界案例。

## 8. 下一阶段修改顺序

### P0：先修证据预算，不再放宽审计

把最终 Episode 预算分成互不挤占的三部分：

1. **原子槽锚点区**：每个 query slot 至少保留一个通过覆盖审查的 Episode；
2. **直接检索区**：保留全问题 Dense/Sparse 的高分事实；
3. **图增量区**：只容纳 Association 路径新引入的端点。

图端点只有在模型明确判断与已保护 Episode 语义等价时才能替换它，不能仅凭路径分更高驱逐原文锚点。

### P0：增长边先放临时 overlay，再决定持久化

新关系先进入本题临时图，执行一次“有边/无边”重放：

- 是否新增必需证据组；
- 是否让某个原子槽从未覆盖变为覆盖；
- 是否使答案完整性提高；
- 是否挤掉已经覆盖的原始事实。

只有产生正边际效益、或被后续独立文本再次确认的关系才写回 SQLite。这样把“关系语义可信”与“关系检索有用”
分开：双模型审计负责前者，overlay delta 负责后者。

### P1：限制同题冗余边

对端点、`relation_key` 和 relation embedding 聚类；同一问题每个证据槽最多持久化 1—2 条新增边。优先保留：

- 连接当前未覆盖 Source 的边；
- 跨文件但有明确前提的边；
- 能在移除后造成 recall 下降的边。

第一题的 8 条边应被压缩为少量“表面目的 ↔ 真实目的”和“总体嫌疑 ↔ 具体身份”桥。

### P1：基础 Top-20 按槽位 round-robin

Candidate@100 已达 98.43%，所以不先扩大到 200。先让 rerank 输出每个 atomic slot 的支持 Episode，并在
Top-20 中轮询保留；只有剩余名额再按全局相关度填充。两道 Candidate=100% 而 Top-20=66.67% 的题是直接回归样本。

### P1：Association cue 的复用测试

第三题应能通过“未花—阿里乌斯—政变”检索到第二题新建的关系。需要记录每条新边在后续问题中的：

- cue 是否被召回；
- 图路径排名；
- 端点是否进入候选、rerank 和答案；
- 被丢弃的具体阶段与原因。

没有这条 trace，无法区分“新边没被检索”与“检索到后被预算挤掉”。

### P2：Concept 别名冲突审计

先增加只读报告和人工修改入口，不自动全库合并。优先处理真实人物多语言别名和用户常用译名，保留服装状态、
未知人物等可能需要区分的 Concept。Paragraph 继续关闭；之前实验未证明它有稳定收益，本轮基础召回也表明首要问题
是证据配额，而不是缺少更长文本向量。

## 9. 复现资产

主要结果：

- `validation/evaluation-stage17-full-rebuild/assets-v3-throughput/database-audit.json`
- `validation/evaluation-stage17-full-rebuild/assets-v3-throughput/model-log-summary.json`
- `validation/evaluation-stage17-full-rebuild/analysis-report.json`
- `validation/evaluation-stage17-full-rebuild/evidence-manifest-remapped.json`
- `validation/evaluation-stage17-full-rebuild/evidence-remap-report.json`
- `validation/evaluation-stage17-full-rebuild/base-retrieval-adaptive.json`
- `validation/evaluation-stage17-full-rebuild/base-retrieval-log-summary.json`
- `validation/evaluation-stage17-full-rebuild/deep-evidence-manifest-remapped.json`
- `validation/evaluation-stage17-full-rebuild/deep-evaluation/evaluation-report.json`
- `validation/evaluation-stage17-full-rebuild/deep-scorecard.json`

新增可复现脚本：

- `benchmarks/remap_evidence_manifest.py`：同 Source 内把旧 Episode 证据组映射到新提取结果；
- `benchmarks/analyze_stage17_rebuild.py`：比较数据库、导入 token、检索结果和 Concept 别名质量；
- `benchmarks/summarize_jsonl_logs.py`：按阶段和模型汇总真实 provider usage；
- `benchmarks/score_stage3_evaluation.py --manifest ...`：使用重映射证据严格评分。

正式数据库评测前后 SHA-256 均为：

`864096431E0BAB1D187473D6D261DB5CCC40AA0E824F8EC14CC1BF67DD10FD11`

三模式使用 SQLite backup 的独立副本，11 条增长边只存在于
`deep-evaluation/graph_growing.db`，没有写回正式资产。

最终确定性测试：133/133 通过，不调用外部 API。

## 10. 最终判断

Stage 17 证明了完整导入和基础检索优化是有效的，也第一次在整库上明确量化了图的真实代价：静态图有一次严格
收益，而当前查询期增长边没有净收益。这个负结果不是“联想网络方向错误”，而是说明系统已经从“能不能生成边”
进入了更具体的工程阶段：**如何让少量新边补充、而不是替换直接经验。**

因此下一轮的中心不再是提高写边率，也不是继续放松可信度；而是建立原子证据不可驱逐规则、临时 overlay 的
边际效益门卫，以及新边从 cue 到答案的全链路 trace。完成这三项后，再用同三题和 23 题扩展集做冻结重放，
才能判断持续增长是否开始产生稳定收益。
