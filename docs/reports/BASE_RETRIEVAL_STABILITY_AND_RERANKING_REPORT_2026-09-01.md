# 基础检索稳定性与重排实验报告

日期：2026-09-01  
范围：无 Association 增长、无 Paragraph、冻结剧情数据库、7 个 Stage 4 多跳问题  
状态：Candidate@100 验收通过；最终 Recall@20 尚未达到稳定大于 95% 的目标

## 1. 实验目的

本阶段只回答一个问题：在完全不依赖增长边的情况下，基础检索能否稳定地把回答所需证据送入最终
Top-20。

这需要把两个容易混淆的能力拆开：

1. **候选召回能力**：Dense、Sparse、Concept 等通道能否把证据放进 Candidate@100。
2. **证据压缩能力**：重排器能否从 100 条候选中稳定选出覆盖全部事实槽的 20 条。

如果 Candidate@100 已包含全部证据，而 Top-20 丢失证据，继续调 embedding、扩大向量 Top-K 或增加
Association 都不会解决根因。相反，错误的增长边可能暂时掩盖选择器缺陷。

## 2. 专有名词定义

### 2.1 Candidate@100

基础 Dense/Sparse/Concept 检索及候选合并后，交给重排器的最多 100 个 Episode。它是高召回集合，
不等于最终回答证据。

### 2.2 Top-20 / Recall@20

重排器最终交给回答模型的 20 个 Episode。一个问题预先登记若干必需证据组；命中某组任意一个等价
Episode 即视为覆盖该组。

```
Recall@20 = 已覆盖必需证据组数 / 必需证据组总数
```

本报告中的“严格 Recall@20”仍按 Episode ID 清单计算。它会低估未登记的语义重复 Episode，相关限制
见第 8 节。

### 2.3 Evidence slot（证据槽）

问题中一个不可由其他事实替代的回答义务。例如“谁直接执行”“谁提供协助”“该说法只是推测还是
事实”是三个不同槽。主题相近不代表槽位相同。

### 2.4 Hard floor（硬性保底）

由便宜的 Dense/Sparse/原子查询排序确定、无论 LLM 如何选择都尝试放入 Top-20 的 Episode。它用于
防止明显锚点被随机漏掉，但排序精度低于完整证据审计，因此预算必须受限。

### 2.5 Coverage scout（覆盖侦察器）

只负责从候选中为各事实槽找证据的 LLM，不负责回答问题。严格模式运行两个相互独立的侦察器，以
降低单次注意力漂移。

### 2.6 Compressor（槽位压缩器）

读取覆盖结果，把候选证据压缩到固定 Top-K。它应删除重复改写和背景，但不能删除某槽唯一证据。

### 2.7 Cross-Encoder reranker

本实验使用 `Pro/BAAI/bge-reranker-v2-m3`。它直接对“一个查询—一条候选文本”计算相关性，通常比
生成式 LLM 更快、更稳定；但单一相关性分数不会自动保证多事实槽、时间两端和因果链均被覆盖。

### 2.8 Semantic fingerprint（语义指纹）

冻结评测不能只检查 Episode 整数 ID。数据库重建后相同 ID 可能指向另一段文本。本阶段用评测报告中
保存的 Episode 文本与数据库当前文本共同校验，防止在错误数据库上得到看似合法的分数。

## 3. 冻结实验环境

- 数据库：`validation/rerank-stability-stage22-frozen-20260901.db`
- 问题：`validation/evaluation-questions-stage4-network.json`
- 证据清单：`validation/stage4-network-evidence-manifest.json`
- Candidate 数：每题 100
- 最终证据数：每题 20
- Association graph hops：0
- Association growth rounds：0
- Paragraph：关闭
- embedding 与磁盘/内存向量：float32
- LLM 严格重排：初次覆盖、独立覆盖、压缩三个阶段

共扫描 235 个历史数据库快照，159 个通过全部冻结 Episode ID + 文本语义指纹。实验选择其中一个复制
为统一基线，避免 Stage 17 重建数据库复用 ID 所造成的假失败。

## 4. 已新增的评测和审计工具

### 4.1 重复稳定性评测

`benchmarks/run_rerank_stability_eval.py`

- 冻结 Candidate@100，不重复做基础检索；
- 同一问题重复执行当前重排；
- 统计每个证据槽的命中频率；
- 记录两名 coverage scout 的分歧；
- 支持断点续跑、多 worker、替换 query-input 和关闭 hard floor；
- 启动前执行数据库语义指纹校验。

### 4.2 数据库匹配

`benchmarks/find_frozen_candidate_database.py`

用持久化 Episode 文本查找与旧报告真正相符的 SQLite 快照。它解决“ID 存在，但文本已不是原文本”的
评测污染。

### 4.3 Hard-floor 离线反事实

`benchmarks/audit_evidence_floor_policies.py`

复用已经完成的 LLM coverage/compressor 输出，只改变 floor 来源、预算和执行顺序，不再产生模型费用。

### 4.4 逐槽 BGE 对照

`benchmarks/run_atomic_cross_encoder_coverage_eval.py`

对同一 Candidate@100 比较：

1. 整题一次 BGE Top-20；
2. 每个原子查询分别 BGE 后轮询选 20 条；
3. 把候选视为可覆盖多个查询的集合，做贪心覆盖；
4. 将逐槽结果扩到 30/40/50 条，判断其是否适合做 LLM 前置压缩。

## 5. 实验结果

### 5.1 历史选择器重复稳定性

结果：`validation/rerank-stability-all7-r5-20260901.json`

- 7 题 × 5 次，共 35 次严格三阶段重排；
- Candidate@100：平均 100%，最低 100%；
- Top-20：平均 98.61%，最低 87.5%；
- 31/35 次大于 95%，通过率 88.57%；
- 56 个必需槽中有 3 个槽出现至少一次不稳定；
- 19 次双侦察器意见不一致；
- 平均纯重排耗时 134.20 秒，P95 172.44 秒；
- 105 次模型调用共使用 1,882,288 tokens，无模型错误、无 fallback。

这说明历史版本平均分很高，但还不能声称“稳定 Recall@20 > 95%”：同一输入仍会偶发漏掉关键槽。

### 5.2 当前代码修改前的静态基线

结果：`validation/current-base-retrieval-all7-strict-20260901.json`

- Candidate@100：7/7 均为 100%；
- Top-20 平均：72.42%；
- 最低：44.44%；
- 仅 1/7 题大于 95%。

追踪显示每题存在 20—24 个 hard-floor ID，而答案预算只有 20。程序几乎把 Top-20 全部交给低精度
floor，LLM coverage 的判断空间被清空。

### 5.3 限制 floor 后的 v2 基线

结果：`validation/current-base-retrieval-v2-all7-strict-20260901.json`

- Candidate@100：平均/最低均为 100%；
- Top-20 平均：87.90%；
- 最低：77.78%；
- 1/7 题大于 95%；
- 平均端到端检索耗时：200.29 秒。

相对 72.42% 有明显恢复，证明 floor 过载是实质性缺陷，但不是唯一缺陷。

### 5.4 Floor 与 coverage 的执行顺序反事实

结果：`validation/current-base-floor-policy-audit-v2-20260901.json`

代码曾按以下顺序执行：

```
LLM compressor
→ coverage 补槽
→ hard floor 再替换
```

因此 coverage 已找到的唯一证据仍可能被低精度 floor 删除。修复为：

```
LLM compressor
→ hard floor 补底
→ coverage 最终裁决
```

在同一批已完成模型响应上离线重放后，平均严格 Recall@20 从 87.90% 提升到 89.48%，最低从 77.78%
提升到 87.5%。该修复已保留，并有独立回归测试。

### 5.5 强制 coverage-slot 标签实验（已回退）

结果：`validation/current-base-retrieval-v3-coverage-slots-all7-strict-20260901.json`

- 平均：89.48%；
- 最低：75%；
- 2/7 题大于 95%；
- 平均耗时：214.98 秒。

该方案把 query planner 的所有搜索问法标成显式必答槽。平均值略升，但最差题下降，且同义查询争抢
注意力。它没有稳定收益，相关代码和提示词已回退，仅保留实验资产。

### 5.6 整题 BGE Cross-Encoder

生产路径结果：`validation/current-base-cross-encoder-all7-20260901.json`

- 平均：53.80%；
- 最低：44.44%；
- 0/7 题大于 95%；
- 包含 LLM 查询规划的端到端平均耗时：70.21 秒。

纯冻结候选对照中，去掉生产 hard floor 后整题 BGE 平均为 68%。两者均证明：整题相关性 Top-20 不
适合多槽证据链，不能作为 deep 路径的默认最终裁决器。

### 5.7 逐槽 BGE

结果：

- `validation/atomic-cross-encoder-coverage-all7-20260901.json`
- `validation/atomic-cross-encoder-coverage-all7-full-plan-20260901.json`

使用初始查询计划时：

- 整题 Top-20：平均 68%；
- 逐槽轮询 Top-20：平均 84%，最低 75%；
- 逐槽贪心 Top-20：平均 70%；
- 每题执行 12—16 个 BGE 查询，平均 29.75 秒。

逐槽轮询扩成候选压缩池后：

- Top-30：平均 91%，最低 75%；
- Top-40：平均 98%，最低 87.5%，6/7 题完整；
- Top-50：平均 98%，最低 87.5%，没有继续改善。

加入全部后续查询后，Top-20 反而降到平均 81%，说明更多查询不等于更好的槽位计划；Top-40 平均
96%，但仍漏掉阿里乌斯题的推理型后续证据。

## 6. 已确认的根因

### 6.1 候选召回不是当前主要瓶颈

所有正式 7 题实验的 Candidate@100 均为 100%。当前不应通过增加 ANN、扩大 embedding Top-K 或提前
依赖增长边来解决最终 Top-20 丢失。

### 6.2 低精度保障层曾覆盖高精度审计层

hard floor 的职责是补底，不是最终裁决。执行顺序错误会把侦察器已经找到的证据再次删除。

### 6.3 Query plan 同时承担“搜索提示”和“回答合同”

当前 search_queries 中既有真正独立事实槽，也有同义改写、实体解析问法和后续探索提示。把它们全部
视为必答槽会过度分解；完全不标注又可能漏掉后段槽。需要单独、受预算约束的 EvidencePlan，而不是
继续加长字符串列表。

### 6.4 单次 100 候选 LLM 存在注意力尾部问题

某些关键证据位于 Candidate@100 的后部。两个侦察器读取同一长提示时会产生相关性错误，不能把它们
当作完全独立的随机试验。

### 6.5 BGE 不能独立完成推理型槽位

“后续独立行动说明控制边界”需要先理解行动，再把它投影到控制关系。即使查询明确写出“控制边界”，
Cross-Encoder 仍可能把直接行动证据排在 60 名之后。它适合相关性缩池，不适合作为所有推理槽的裁判。

### 6.6 评测清单存在 Episode-ID 等价类不完整

例如某题要求新 ETO 证据 `[197, 199]`，系统选择的 Episode 198 实际包含更完整的同一宣言，却仍按
未命中计算。类似重叠 Episode 会使严格 ID Recall 低估语义覆盖。不能为了分数在生产检索器中写入
剧情 ID 规则；应修复评测表示。

## 7. 本轮保留的代码改动

1. `QueryIntent.from_dict` 将 `None`、`null`、无明确约束等空语义归一为空值。
2. 默认原子查询与 floor 预算缩小，hard floor 不再吞掉整个 Top-20。
3. 否定、禁止认定或明确未经证明的引号不再自动成为答案 hard floor。
4. 整题和结构化槽不再被残余中文分隔规则重复拆解。
5. 全问题 Sparse 结果保留为候选召回通道，不再全部强制进入最终答案。
6. final selection 改为 floor 先补底、coverage 最终裁决。
7. 冻结候选评测增加 Episode 文本语义指纹校验。
8. 增加重复稳定性、数据库定位、floor 反事实和逐槽 BGE benchmark。

所有改动均为领域无关的检索与证据不变量。生产 prompt 中没有加入“老师、会长、阿里乌斯”等当前
剧情专用规则。

完整测试：303 项通过。

## 8. 当前尚未解决的问题

### 8.1 最终 Recall@20 尚未稳定大于 95%

当前可证明的是 Candidate@100 稳定 100%。最终选择层的最好历史平均值为 98.61%，但最差运行仍为
87.5%；当前通用代码的可重放结果约为 89.48%。因此本阶段不能标记最终 Top-20 验收通过。

### 8.2 三阶段 LLM 延迟过高

当前 v2 平均约 200 秒/题。它适合离线深度验证，不适合普通聊天。light/standard/deep 仍必须使用请求级
RetrievalPlan，把查询规划、候选预算、BGE、覆盖审计和回答审计分别控制。

### 8.3 证据槽计划没有独立数据模型

目前槽位隐含在自然语言 search_queries 中。下一版本至少需要请求级、不可变的结构：

```text
EvidenceSlot
  id
  claim/question
  subject constraints
  relation constraints
  temporal/modal constraints
  required or optional
  retrieval hints
```

这不是知识领域专家规则，而是任何多事实问答都需要的执行合同。

### 8.4 ID 清单应升级为证据等价类

评测清单应优先登记 Source 证据 span、语义指纹或人工确认的 Episode 等价类，而不是假定某个整数 ID
永远是唯一正确摘要。

## 9. 下一阶段推荐实验：分块 LLM Map-Reduce

### 9.1 假设

Candidate@100 已完整，但单个长提示存在注意力尾部遗漏。把候选分成较小、互斥的块，让每块独立找证据，
再统一压缩，可能在不增加剧情规则的情况下提升稳定性。

### 9.2 实验流程

```text
冻结 Candidate@100
        ↓
按全局排名交错分为 4 × 25
        ↓
4 个并行 map coverage scout
每块输出 8—12 条及明确 missing slots
        ↓
合并、语义去重，形成不超过 48 条证据池
        ↓
一个 reduce coverage + compressor
        ↓
floor 补底
        ↓
coverage 最终裁决 Top-20
```

使用交错分块而非连续 `[1:25]...[76:100]`，可以降低某块全是同一事件重复摘要的风险。

### 9.3 对照组

1. 当前 v2 单提示三阶段 LLM；
2. 逐槽 BGE Top-40 + 单次 LLM reducer；
3. 4×25 LLM map + LLM reducer；
4. 逐槽 BGE Top-40 与 LLM map 结果并集 + reducer。

### 9.4 预注册指标

- Candidate@100 继续保持 100%；
- map 合并池 Candidate Recall 不低于 99%；
- 7 题 × 5 次最终平均 Recall@20 不低于 95%；
- 任一必需证据槽命中频率不低于 95%；
- 不允许通过剧情专用 ID、人物或事件规则提升分数；
- 记录请求数、prompt/completion tokens、端到端延迟、P95 与失败重试；
- 与当前约 200 秒/题严格 LLM 路径比较延迟，不只比较质量。

### 9.5 停止条件

如果 map 合并池仍不能稳定覆盖 Candidate@100 中的尾部证据，则问题在 EvidencePlan 或问题分解，而不
在压缩器；停止继续增加审计轮数，转向结构化 EvidenceSlot 生成与槽位级人工审计。

## 10. 阶段结论

本轮最重要的结论不是某个 Top-20 数字，而是完成了责任分层：

```text
基础检索 Candidate@100：可靠，7/7 为 100%
最终证据选择：尚不稳定
hard floor：只能补底，不能最终裁决
整题 BGE：不适合 deep 多跳问题
逐槽 BGE：适合作为候选压缩辅助手段，但不能替代推理
下一主要实验：分块 LLM map-reduce + 结构化 EvidencePlan
```

在这条基线稳定以前，不应把 Association 增长收益作为主要优化目标；否则会把“基础选择器漏证据”误判
为“需要更多增长边”。
