# Stage 12 实验报告：清洗后的 Paragraph 与门控细粒度 Concept

日期：2026-08-28  
状态：已完成；336/336 个正式精排样本全部通过完整性审计。

## 1. 本阶段要回答的问题

Stage 11 已经证明，直接加入 Paragraph 和积极创建 Concept 并不会自然产生稳定收益：Paragraph 容易被文件元数据与脚本标记污染，细粒度 Concept 则会制造大量低价值节点。本阶段不再单纯增加数量，而是分别修复两个机制：

1. Paragraph 只作为补充召回通道，不能抬高或重排基线已经找到的 Episode。
2. 细粒度 Concept 先被发现，再经过持久化准入判断；一次性、过细或缺乏复用价值的候选不写入长期图。

核心假设是：

- 清洗后的 Paragraph 可以补回 Episode 摘要没有保留的原始细节，同时不破坏成熟基线。
- 更积极的 Concept 发现可以提高联想入口密度，但只有少量适合长期复用的概念应晋升为正式节点。
- P、C 各自必须证明独立收益；PC 的组合收益不能替代任一单项机制的因果证据。

## 2. 专有名词与指标定义

### 2.1 Episode

从 Source 中提取的、自包含的事件或状态摘要，是当前系统主要的检索与回答证据单位。本轮冻结资产包含 342 个 Episode。

### 2.2 Paragraph

由 Source 中相邻原始记录组成的较小窗口。它保留 Episode 摘要可能省略的原话、语气和局部细节，但本身不是新的事实判断。

本轮的 `recall-only Paragraph` 指：

- Paragraph 可以通过 embedding 找到其所属 Source，再把该 Source 的 Episode 加入候选池；
- Paragraph 不参与基线 Episode/Concept 融合分数；
- Paragraph 不得提升基线已经找到的 Episode 排名；
- 只有 A 中完全缺失的 Episode 才能由 Paragraph 作为辅助种子补入。

### 2.3 Paragraph embedding 清洗

数据库仍保存完整 Paragraph，便于审计和回看；发送给 embedding 模型的文本会删除：

- `source_key`、`segment_index` 等索引元数据；
- `speaker_alias_legend` 及其内容；
- `record`、`script_raw` 及跨行脚本标记；
- `#na`、`#fontsize` 一类游戏脚本控制信息。

人物标记与中、日、英、繁中、泰文等可读对白保留。这样把“审计文本”和“检索文本”分开：前者完整，后者只保留语义内容。

### 2.4 Concept discovery 与 promotion

`discovery` 是从每个 Episode 中尽量积极地提出概念候选；`promotion` 是决定该候选是否进入长期数据库。二者分离后，积极发现不再等于积极污染数据库。

本轮准入动作有三种：

- `promote`：创建新的持久 Concept；
- `reuse`：候选实际对应已有 Concept，只增加别名或建立 Episode 关联；
- `transient`：记录到日志，但不创建长期节点和边。

若 LLM 准入返回缺失或无效，只有在多个 Episode 中重复出现的候选才可按保守回退规则晋升；一次性候选保持 transient。

### 2.5 四个实验臂

| 实验臂 | Paragraph | Concept 数据 |
|---|---:|---|
| A | 关闭 | 原保守 Concept，基线 |
| P | 开启清洗后的 recall-only Paragraph | 与 A 完全相同 |
| C | 关闭 | 加入通过准入门的细粒度 Concept |
| PC | 与 P 相同 | 与 C 相同 |

同一问题、同一次重复只生成一份查询意图、原子查询、后续查询及 float32 embedding，四臂共享这份冻结计划。这样避免把查询规划的随机变化误认为处理收益。

### 2.6 Candidate@100

图展开和多路召回后交给证据精排器的前 100 个 Episode。候选集包含正确证据，只代表检索上限足够，不代表最终回答真的会选中它。

### 2.7 Selected Recall@20 / @12

LLM 精排后最终保留的前 20 或前 12 个 Episode 覆盖了多少个预先标注的必要证据组。一个证据组可包含多个等价 Episode，命中任一即可。

- Top-20 是主要结果；
- Top-12 是更窄答案上下文下的压力测试；
- 新增候选数量、Paragraph 命中数量、Concept 数量和边数量都不是收益指标。

### 2.8 稳定正向与稳定负向

每题独立生成三份冻结查询计划。某臂相对 A：

- 至少 2/3 次差值大于 0，且没有一次小于 0，称为稳定正向；
- 至少 2/3 次差值小于 0，且没有一次大于 0，称为稳定负向。

Paragraph 或 Concept 要通过机制门槛，必须在自己的机制题中至少有两道不同问题稳定正向、总体平均差值大于 0，并且旧网络题与范围控制题不得稳定退化。

## 3. 实现内容

### 3.1 Paragraph 修复

实现了独立的 embedding 文本清洗，并修复了 `script_raw` 跨多行时只删除首行的缺陷。数据库中的原文不变，只有 embedding 输入被清洗。

检索侧将 Paragraph 分数从基础融合分数中隔离。处理顺序变为：

1. Episode、Concept、稀疏检索先完成基线召回和排序；
2. Paragraph 独立搜索并映射到 Source 内 Episode；
3. 只把基线未出现的 Episode 作为辅助种子加入；
4. 已有锚点的分数与顺序不变。

### 3.2 Concept 修复

实现了批量 Concept admission prompt 和严格返回校验。导入时先完成候选发现和归组，不立即写库；随后向准入模型提供候选名称、解释、出现 Episode、出现次数和已有相似 Concept，再执行 promote/reuse/transient。

`reuse` 只能指向当前批次明确提供给模型的相似 Concept ID，防止模型凭空引用数据库节点。所有决定、验证错误和回退理由进入 JSONL 日志。

### 3.3 断点与评测可靠性

正式 runner 会缓存查询计划、固定检索和每个精排任务。新增的缓存完整性检查只复用完成了初次精排、覆盖审计和最终审计的结果；含 `error` 或审计错误的缓存会自动重跑。

高并发诊断轮发现 32 路请求触发 provider TPM 限流：336 个任务中只有 29 个完整通过，307 个发生降级。该轮保留为诊断资产，不用于正式结论。正式轮改为 6 路并发，只重跑失败样本。

## 4. 冻结资产

正式资产目录为 `validation/evaluation-stage12-clean-paragraph-gated-concept/assets-v2`。

| 实验臂 | Source | Episode | Concept | Paragraph | Association |
|---|---:|---:|---:|---:|---:|
| A | 68 | 342 | 392 | 0 | 1,825 |
| P | 68 | 342 | 392 | 541 | 1,825 |
| C | 68 | 342 | 407 | 0 | 1,985 |
| PC | 68 | 342 | 407 | 541 | 1,985 |

细粒度 Concept 发现阶段共得到 1,343 个候选，归成 468 组：

- 321 组精确复用已有 Concept；
- 35 组由 LLM 判断为复用已有 Concept；
- 15 组晋升为新 Concept；
- 97 组保持 transient；
- 2 次准入校验错误按保守规则处理；
- 0 个 Source 导入失败。

相比 Stage 11 直接增加 162 个新节点，本轮只新增 15 个，说明持久化污染被显著压缩。但节点数量减少不等于查询相关性已经解决，仍需检索实验验证。

## 5. 问题集与预注册

所有机制题都在 A/P/C/PC 检索之前生成并冻结，正式结果不得用于修改题目。

- 8 道 Paragraph 机制题：从与同 Source Episode embedding 最大相似度最低的 Paragraph 中选择，考察摘要可能遗漏的对白、语气或局部细节；
- 8 道 Concept 机制题：从本轮新晋升 Concept 的直接证据 Episode 构造；
- 7 道冻结的 Stage 4 网络题：检查已有跨片段联想能力；
- 5 道冻结的 Stage 10 范围控制题：检查新入口是否把相关但不该进入答案的主题挤入窄窗口。

共 28 道题，每题 3 次独立查询规划、4 个实验臂，共 84 份冻结计划和 336 个精排结果。使用 float32、Dense+Sparse、最多 3 跳静态图、Top-100 候选和 Top-20/Top-12 最终证据窗口；本轮关闭动态增长和 Association cue，避免额外机制混入。

## 6. 结构性审计（不依赖精排结论）

在 84 份冻结计划上：

- A 与 P 的原子 Episode 锚点 84/84 完全一致；
- P 在 84/84 中产生 Paragraph 展开；
- P 在 30/84 中实际改变候选列表；
- Paragraph 机制题中有 19/24 次改变候选列表；
- C 的新晋升 Concept 在 84/84 份计划中都至少进入一个原子查询的 Concept Top-20 ranking。

前四项证明 recall-only 隔离按预期工作：Paragraph 能增加召回入口，又不改变基线锚点。最后一项暴露了新的风险：即使持久节点只有 15 个，它们仍可能过于普遍地参与所有查询，因此“创建时准入”还不能替代“查询时相关性门控”。

## 7. 正式结果

### 7.1 运行质量

高并发诊断轮只有 29/336 个结果完成全部审计，不能用于结论。低并发正式轮首先修复到 333/336，随后以 2 路并发补跑最后 3 个错误样本，最终达到：

- 精排结果：336；
- 完整无错误：336；
- 降级或审计错误：0；
- 查询计划：84/84 已冻结；
- 正式结论有效：是。

### 7.2 绝对召回

下表是三次重复合并后的证据组微平均召回。Candidate 是交给精排器的 Top-100 上限，Selected 是最终答案上下文。

| 题集 | 指标 | A | P | C | PC |
|---|---|---:|---:|---:|---:|
| Paragraph 机制题 | Candidate@100 | 100.00% | 100.00% | 100.00% | 100.00% |
| Paragraph 机制题 | Selected@20 | 100.00% | 100.00% | 100.00% | 100.00% |
| Paragraph 机制题 | Selected@12 | 100.00% | 100.00% | 100.00% | 100.00% |
| Concept 机制题 | Candidate@100 | 100.00% | 100.00% | 100.00% | 100.00% |
| Concept 机制题 | Selected@20 | 100.00% | 100.00% | 100.00% | 100.00% |
| Concept 机制题 | Selected@12 | 100.00% | 100.00% | 100.00% | 100.00% |
| 网络题 | Candidate@100 | 99.40% | 99.40% | 99.40% | 99.40% |
| 网络题 | Selected@20 | 98.81% | 98.21% | 97.62% | 97.02% |
| 网络题 | Selected@12 | 73.81% | 72.62% | 73.81% | 75.60% |
| 范围控制题 | Candidate@100 | 98.15% | 98.15% | 98.15% | 98.15% |
| 范围控制题 | Selected@20 | 98.15% | 96.30% | 98.15% | 96.30% |
| 范围控制题 | Selected@12 | 75.93% | 75.93% | 72.22% | 75.93% |

最重要的观察是：四臂在每个题集上的 Candidate@100 必要证据召回完全相同。Paragraph 和新 Concept 都没有增加任何新的必要证据组。机制题上的 A 已经在 Top-12 达到 100%，所以 P/C 不存在获得正差值的空间。

### 7.3 配对差值与预注册判定

主指标为 reranked Selected@20。下表为每个观测中“处理臂命中证据组数减去 A”的平均值。

| 题集 | P − A | C − A | PC − A |
|---|---:|---:|---:|
| Paragraph 机制题 | 0.000 | 0.000 | 0.000 |
| Concept 机制题 | 0.000 | 0.000 | 0.000 |
| 网络题 | -0.048 | -0.095 | -0.143 |
| 范围控制题 | -0.067 | 0.000 | -0.067 |

预注册结果：

- Paragraph：0 道机制题稳定正向，机制题平均增益 0，未通过；
- Concept：0 道机制题稳定正向，机制题平均增益 0，未通过；
- 两个单项臂在 Top-20 没有某一道旧题达到“稳定负向”，但旧题总体平均值未满足非负要求，因此严格非退化门槛也未通过；
- PC 不可代替单项判定，并且在 `kisaki_mika_governance_analogy_with_boundary` 的 Top-20 出现稳定负向（-1、0、-1）。

### 7.4 Top-12 压力结果

Top-12 显示出 Concept 入口的双向扰动：

- C 在 `old_cathedral_symbol_infrastructure_and_limits` 稳定正向（+3、+1、0）；
- C 在 `eden_obligation_to_new_institution` 稳定负向（0、-1、-1）；
- C 在范围控制题 `paradise_paradox_without_mika_psychology` 稳定负向（-1、0、-1）；
- P 在 `hifumi_azusa_identity_coalition_and_declaration` 稳定正向（0、+1、+1）；
- PC 在三道网络题稳定正向，但这些组合结果不能证明 P 或 C 单独有效。

无 LLM 精排的固定层更能直接显示 Concept 的检索效应：C 在 `mika_motive_epistemic_four_layers` Top-12 稳定正向（0、+1、+1），同时在 `paradise_paradox_without_mika_psychology` 稳定负向（-1、0、-1）。也就是说，新入口能把某些相关材料推进窄窗口，但也会把问题明确要求的哲学证据挤出窗口。

### 7.5 Paragraph 归因

P 在 30/84 次改变了检索候选列表，其中 Paragraph 机制题为 19/24；但所有题集的 Candidate 必要证据组召回与 A 完全相同，固定排序也没有任何证据组正负差异。

P 在正式 Top-20 出现两次单次负差值：网络题一次、范围控制题一次。两次 A/P 交给 reranker 的 100 个 Episode ID 及顺序都完全相同，因此差异来自对同一输入进行两次独立 LLM 调用的随机性，不是 Paragraph 候选造成的。这揭示了 runner 的一个评测方法问题：以后若候选输入哈希相同，应复用同一精排结果，避免把 LLM 波动算成处理效应。

因此对 Paragraph 的因果解释是：

- 清洗和 recall-only 隔离成功；
- 它能补充候选，但没有补充本题集所缺的必要证据；
- 没有证据证明其产生稳定收益；
- 也没有观察到由 Paragraph 新候选直接造成的 Top-20 伤害。

### 7.6 Concept 归因

15 个晋升 Concept 在 84/84 份计划的某个原子查询 Top-20 Concept ranking 中出现。出现最频繁的包括：

| Concept | 出现计划数 | 原子查询排名出现次数 | 平均名次 | 最好名次 |
|---|---:|---:|---:|---:|
| 保护欲 | 21 | 118 | 11.76 | 3 |
| 陪伴的执念 | 21 | 112 | 9.86 | 1 |
| 公会议 | 32 | 87 | 13.51 | 3 |
| 后悔/遗憾 | 19 | 87 | 10.43 | 2 |
| 校规 | 14 | 55 | 7.80 | 1 |
| 联邦学生会的介入原则 | 10 | 55 | 4.20 | 1 |
| 联邦学生会记者招待会 | 10 | 55 | 4.00 | 1 |

进一步检查图结构发现：每个新 Concept 都只直接连接一个 Episode。只有 `圣娅的变故`、`奖赏的重要性`、`联邦学生会的介入原则`、`陪伴的执念` 额外拥有 Concept–Concept 边；其余 11 个节点度数为 1。

这解释了没有收益的根因：这些节点主要是“给一个已有 Episode 增加新的语义标签”，不是“多个 Episode 共享的抽象入口”。它们可能让某个 Episode 更容易被 query embedding 命中，却几乎不能形成新的跨 Episode 桥。基线已经召回这些 Episode 后，新标签只会改变候选构成和排序压力。

C 的两个 Top-20 负差值都伴随 rerank Candidate 顺序或构成变化，和 P 的同输入随机波动不同，属于 Concept 通道可能造成的真实上下文扰动。固定 Top-12 的一正一负也进一步证明：当前 Concept 策略扩大了入口，但没有控制入口的答案边界。

## 8. 问题解释

### 8.1 机制题存在基线饱和风险

正式结果中，16 道新机制题的 A 在 Candidate@100、Top-20 和 Top-12 全部达到 100%，导致 P/C 即使补入证据也无法获得正差值。这不是机制成功，也不必然说明机制在所有语料上无效；它说明“Paragraph 与同 Source Episode 的 embedding 相似度低”不足以保证“成熟 Dense+Sparse+图基线检索不到”。

下一轮机制题必须先在独立开发语料上建立难度生成规则，再冻结到独立测试语料，或改测目标证据的排名改善和上下文压缩。不能根据本次正式 A 的结果事后删除饱和题，否则会引入选择偏差。

### 8.2 Concept 有两道门

创建时门控回答“这个概念值不值得长期存在”；查询时门控回答“这个概念与当前问题是否足够相关，值得占用种子和最终上下文”。本轮只系统修复了第一道门。15 个新节点在所有计划中都至少进入一个原子查询的 Concept Top-20，说明第二道门仍然偏松。

此外，长期图节点的价值不仅取决于名称是否合理，还取决于预计图效用。只连接一个 Episode 的新 Concept 通常只是别名式检索标签；要支持联想增长，更值得晋升的是跨多个独立 Episode 重复出现、能统一多个别名、或能建立明确 Concept–Concept 桥的节点。

### 8.3 API 并发不能只按余额决定

余额充足不代表没有吞吐限制。此次 32 路并发触发的是 token-per-minute 限制，短时间内发送多个包含 100 个候选的大 prompt 会使主模型和备用模型同时被拒绝。正式 benchmark 应记录完整/降级样本数，并把“全部审计成功”作为结果有效性的前置条件。

## 9. 决策与下一步

正式决定：本阶段两个机制都不设为默认。保持：

- Paragraph 默认关闭；
- Concept 使用原保守配置；
- Stage 12 两项机制作为可回退实验开关保留；
- 所有 embedding 与缓存继续统一使用 float32。

下一阶段建议按以下顺序进行：

1. 修复评测配对：对 question、atomic queries、Candidate ID/顺序和候选文本完全相同的精排输入复用同一 LLM 输出；不同输入才分别精排。
2. 建立独立开发集：寻找“Episode 摘要确实丢失、A 的 Candidate@100 未命中、但 Paragraph 命中原文”的样本，再冻结到未参与规则开发的测试集。
3. 测试 Paragraph 的另一种用途：Paragraph 不作为图种子，而是在 Episode 已召回后补充该 Source 的原始局部文本，供 reranker 判断细节；这更符合本轮观察到的“摘要遗漏补充”价值。
4. 给 Concept promotion 增加图效用条件：优先晋升跨至少两个独立 Episode 重复出现、统一多个别名、或能形成明确跨节点桥的候选；单 Episode 概念默认 transient 或作为非图检索标签。
5. 增加查询时 Concept gate：限制每个原子查询可进入图扩展的新 Concept 数量；对度数为 1 的新节点只允许直接召回其 Episode，不允许继续图扩展；同时加入最小查询相关性与独立证据槽覆盖要求。
6. API 调度按估算输入 token 加权限速，并把“无错误完成所有审计”作为 benchmark 的硬前置条件。

本轮最可靠的结论不是“Paragraph/Concept 永远无效”，而是：在当前已经接近 Candidate 饱和的 342-Episode 数据集上，这两种实现没有提供新的必要证据；Paragraph 修复后表现安全但冗余，门控 Concept 则减少了持久化污染，却仍制造了过于普遍、低图度的查询入口。

## 10. 可复现资产

- 预注册：`validation/stage12-mechanism-preregistration.json`
- 正式清单：`validation/stage12-clean-paragraph-gated-concept-manifest.json`
- 正式资产：`validation/evaluation-stage12-clean-paragraph-gated-concept/assets-v2/asset-report.json`
- Paragraph 机制题：`validation/evaluation-questions-stage12-v2-paragraph.json`
- Paragraph 证据：`validation/stage12-v2-paragraph-evidence-manifest.json`
- Concept 机制题：`validation/evaluation-questions-stage12-v2-concept.json`
- Concept 证据：`validation/stage12-v2-concept-evidence-manifest.json`
- 高并发诊断：`validation/evaluation-stage12-clean-paragraph-gated-concept/pilot-v1`
- 正式低并发运行：`validation/evaluation-stage12-clean-paragraph-gated-concept/pilot-v2-throttled`
- 结果分析器：`benchmarks/analyze_stage12_results.py`
