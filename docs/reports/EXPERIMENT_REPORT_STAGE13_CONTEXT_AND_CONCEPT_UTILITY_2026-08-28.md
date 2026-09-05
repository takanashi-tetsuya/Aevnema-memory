# Stage 13 实验报告：Paragraph 原文证据与 Concept 图效用门

日期：2026-08-28  
状态：机制方向性筛查完成；因无收益触发停止条件，未升级为三轮三审计正式确认。  
数值类型：SQLite、NumPy 索引和所有 embedding 均为 `float32`。

## 1. 结论

本阶段没有得到足以启用 Paragraph 或更积极 Concept 的证据。

- Paragraph：8 道目标题在 Top-20、Top-12、Top-8 均为 `0 正 / 8 平 / 0 负`。
- Concept：8 道跨 Episode 桥题在三个预算下同样均为 `0 正 / 8 平 / 0 负`。
- Paragraph 确实进入了 8/8 道题的精排提示，且没有改变任何图候选；但最终 Episode 选择也 8/8 不变。
- Concept 处理臂在 8/8 道题改变了候选图，5/8 道题的最终路径包含新增加的“复用既有 Concept”边，仍未增加任何必需证据组。

因此默认配置保持：

- Paragraph 检索关闭。
- Concept 使用原保守提取策略。
- Stage 13 新机制与资产保留为可回退实验分支，不删除。

## 2. 本阶段要区分的概念

### 2.1 Paragraph

Paragraph 是 Source 的确定性小分块，保存原文而不是 LLM 摘要。它不是新的事实层，也不自动等于 Episode。本阶段不再让 Paragraph 扩张图种子，而是让精排器在候选 Episode 已经来自同一 Source 时，额外读取查询命中的原文片段。

### 2.2 Source 级证据

Paragraph 只能证明“这段内容出现在该 Source 中”。同一 Source 可能对应多个 Episode，所以不能把 Paragraph 中的每句话自动归到某一个 Episode。提示中使用 `source_level_raw_paragraph` 标记，并由候选的 `source_context_refs` 引用。

### 2.3 Concept 复用边

细粒度提取器发现一个候选 Concept 后，如果它与已有 Concept 是同一个语义对象，就不创建新节点，而是把当前 Episode 连接到已有 Concept。这种边可能立即把新 Episode 接入已有记忆簇，因此理论上比单 Episode 新标签更有图价值。

### 2.4 图效用门

图效用门要求 Concept 经由直接 Episode 边或一跳 Concept 邻居至少能到达两个不同 Episode，才允许作为普通向量种子。精确别名命中绕过该门，以免单次出现但被用户明确点名的人物或实体失去召回能力。

### 2.5 Candidate Recall 与 Selected Recall

- Candidate Recall：必需证据是否进入精排候选集合。
- Selected Recall：必需证据是否进入最终 Top-K。

本阶段两类机制题的 Candidate Recall@100 都已经是 100%。所以实验只能证明压缩排序是否变好，不能再证明粗召回是否扩大。

### 2.6 单次方向性筛查

为了避免在完全没有收益信号时消耗三轮、每轮三次审计调用，本阶段先执行一轮单次覆盖精排。它可以证明当前样本上是否出现方向性变化，但不能证明跨三轮稳定性。预注册的完整正式门要求至少 2/3 次正差值；由于第一轮 16/16 机制题在所有预算上全部为零，实验按无效性停止，没有声称“正式通过”。

## 3. 实现修复

### 3.1 相同精排输入共享结果

精排输入现在计算 SHA-256，覆盖：问题、意图、原子查询、候选 ID/顺序/文本/得分、Source 上下文、预算、模型、提示版本和审计开关。只有完全相同的输入才复用整次精排结果；失败或缺审计结果不会进入缓存。

这修复了 Stage 12 中“候选完全相同，却因重复询问模型而出现伪正差或伪负差”的方法学问题。

### 3.2 Paragraph 不再偷偷改变候选

P 臂设置：

- `paragraph_seed_enabled = false`
- `paragraph_rerank_context_enabled = true`

固定计划审计中，A/P 的图候选不一致数为 `0/84`。机制筛查中，A/P 的精排 Candidate ID 不一致数为 `0/8`。因此观察到的差异只能来自原文上下文，而不是候选集合变化。

### 3.3 Paragraph 上下文目录去重

初版把同一段 Source 原文复制到该 Source 的每个候选 Episode 中，导致提示膨胀和 TPM 限流。修复后，每个 Paragraph 原文只在目录中传一次，Episode 只保存引用 ID。

诊断样本的首轮精排提示大小：

- 重复展开：平均 121,401 字符（n=6）。
- 目录引用：平均 64,721 字符（n=8）。
- 平均减少约 46.7%。

### 3.4 Concept 准入与查询门

Stage 12 的 15 个新 Concept 全部只有一个直接 Episode，因此 Stage 13 将它们从处理资产中移除；保留 140 条 LLM 已确认的“新 Episode → 既有 Concept”复用边。

查询时，Concept 向量命中必须至少可达两个 Episode。本轮 1,900 次原始 Concept 命中中：

- 1,072 次通过；
- 828 次被拒绝；
- 拒绝率约 43.6%。

### 3.5 Concept 可达性性能

最初的单条 CTE/OR Join 在 336 次评测重放中造成明显 CPU 放大。现改为：

1. 使用已有 `from`/`to` 索引分两段读取邻接边；
2. 在 AssociationRepository 中缓存派生的可达 Episode 数；
3. Association 新增或删除时清空缓存。

对当前 392 个 Concept：

- 冷加载全部可达数约 9.33 ms；
- 缓存后 100 次全量读取约 4.95 ms；
- 平均约 0.050 ms/次。

## 4. 实验资产

四臂资产：

| 臂 | Paragraph | Concept | Association |
|---|---:|---:|---:|
| A | 0 | 392 | 1,825 |
| P | 541 | 392 | 1,825 |
| C | 0 | 392 | 1,965 |
| PC | 541 | 392 | 1,965 |

所有臂共有：68 个 Source、342 个 Episode。

机制题：

- Paragraph 原文缺口题 8 道。
- Concept 桥题 8 道，包括地下墓穴传闻到潜入推测、世界真相到日富美反驳、未花的背叛三视角、憎恨的政治链、乐园悖论到信任回应等。
- 证据清单在处理臂检索前冻结。

## 5. 结果

### 5.1 Paragraph

| 预算 | A Selected Recall | P Selected Recall | 配对结果 |
|---|---:|---:|---:|
| Top-20 | 100.0% | 100.0% | 0 正 / 8 平 / 0 负 |
| Top-12 | 100.0% | 100.0% | 0 正 / 8 平 / 0 负 |
| Top-8 | 87.5% | 87.5% | 0 正 / 8 平 / 0 负 |

附加审计：

- 8/8 题有 Paragraph 上下文。
- 共使用 180 个去重后的 Paragraph 引用（按题累计）。
- A/P 候选完全相同。
- A/P 输入哈希 0/8 相同，证明模型确实看到了不同输入。
- A/P 最终 Episode ID 8/8 完全相同。

解释：当前 Episode、稀疏 Source 检索和 Source excerpt 已经足以让正确 Source 进入并保留在答案预算内。Paragraph 原文虽然进入精排器，但没有改变哪一个 Episode 应占槽。

### 5.2 Concept

| 预算 | A Selected Recall | C Selected Recall | 配对结果 |
|---|---:|---:|---:|
| Top-20 | 100.0% | 100.0% | 0 正 / 8 平 / 0 负 |
| Top-12 | 83.3% | 83.3% | 0 正 / 8 平 / 0 负 |
| Top-8 | 83.3% | 83.3% | 0 正 / 8 平 / 0 负 |

附加审计：

- 8/8 题的图候选发生变化。
- 5/8 题的答案路径包含新增复用边。
- 只有 3/8 题 A/C 最终 Episode ID 完全相同，说明机制确实改变了选择内容。
- 但所有变化都只是在同等证据覆盖内换候选，没有补入任何缺失证据组。

解释：复用边不是“没有工作”，而是“工作后没有信息增益”。在只有 342 个 Episode、Candidate@100 占全库约 29.2% 的环境中，成熟的 dense+sparse 基线已召回全部目标证据，Concept 图只能在已召回内容内重新洗牌。

## 6. 为什么没有继续完整三轮

完整计划原本要求：3 轮 × 28 题 × 4 臂，每个结果包含覆盖侦察、独立缺口审计和最终压缩。供应商对大候选提示实施 TPM 限流；并发 6 会大量触发 429，并发 2 稳定但完整运行成本很高。

在 16/16 机制题的一轮完整方向性筛查中，两个机制在三个预算均没有一个正差值。继续运行控制题或额外审计不能把“机制没有增加目标证据”改成“机制产生收益”。因此按无效性停止，保留已经完成的 64/64 个无错误机制结果，并明确不把本轮称为三轮稳定性验证。

## 7. 当前真正的问题

### 7.1 语料规模使 Candidate@100 过宽

当前实验库只有 342 个 Episode，Candidate@100 相当于扫描后把约三分之一全库交给精排器。两类机制题 Candidate Recall 都是 100%，新增召回通道没有发挥空间。

### 7.2 Paragraph 的评测目标仍偏向 Episode 选择

Paragraph 的潜在价值是保留 Episode 摘要省略的精确原文，但当前主要指标只检查“正确 Source 下的 Episode 是否被选中”。只要基线已经选中同一 Source，指标就无法判断原文细节是否让最终答案更准确。

### 7.3 高度通用的 Concept 容易造成候选 churn

“痛苦”“憎恨”“震惊”“背叛”等节点可以连接多个 Episode，却不一定形成问题所需的事实桥。`可达 Episode >= 2` 只能排除单点标签，不能衡量关系的特异性、方向性或答案槽价值。

### 7.4 图变化不等于答案增益

C 臂已经证明：改变候选、改变最终 ID、让新增 Association 进入路径，都可能不增加 required evidence。后续仍应把“补到此前缺失的证据”作为收益标准，而不是边数、路径数或候选变化量。

## 8. 下一阶段建议

优先顺序：

1. 扩大语料规模到至少数千 Episode，再重复 Candidate@100 实验；当前 342 条不足以检验辅助召回。
2. 为 Paragraph 建立答案级评测：固定同一批已选 Episode，比较是否能准确回答 Source 中被摘要省略的原文细节，并审计引用的 Paragraph ID/Source ID。
3. Concept 门从“节点度数”升级为“查询相关的桥效用”：要求该 Concept 实际连接至少两个不同答案槽，且每端都有高相关 Episode；通用情绪节点需要更高门槛或每 Concept 限额。
4. 在完整语料上先寻找基线 Candidate@100 的真实缺口，再冻结独立测试集；没有候选缺口时，不继续做召回机制实验。
5. 保留精排输入哈希与共享机制，后续所有消融都用它消除相同输入的模型随机性。

## 9. 可复现实验文件

- 正式预注册：`validation/stage13-mechanism-preregistration.json`
- 四臂资产报告：`validation/evaluation-stage13-evidence-context-concept-utility/assets-v1/asset-report.json`
- Concept 桥问题：`validation/evaluation-questions-stage13-concept.json`
- Concept 证据清单：`validation/stage13-concept-evidence-manifest.json`
- 固定计划完整结果：`validation/evaluation-stage13-evidence-context-concept-utility/pilot-v1/run-report.json`
- 方向性筛查分析：`validation/evaluation-stage13-evidence-context-concept-utility/screening-r1-single-pass/analysis-report.json`
- Stage 13 资产构建脚本：`benchmarks/prepare_stage13_assets.py`
- Stage 13 筛查分析脚本：`benchmarks/analyze_stage13_screening.py`

## 10. 验证状态

- 全量单元测试：128/128 通过。
- 机制筛查结果：64/64 无错误或降级字段。
- A/P 候选不变量：通过。
- float32 不变量：通过。
- 三轮稳定性正式通过：未执行，原因是首轮完整机制集触发无效性停止。

