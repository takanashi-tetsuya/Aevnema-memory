# 基础检索可靠性实验报告

**日期：** 2026-08-26  
**实验阶段：** Association 增长重新启用前的基础召回加固  
**语料：** 已导入的《蔚蓝档案》剧情 Episode 集  
**结论性质：** 已执行实验结果，不是计划或理论推测

---

## 1. 执行摘要

本阶段回答的问题是：

> 在完全不依赖查询期增长边和 Association 图遍历的情况下，基础 Top-K 检索能否达到 `Recall@20 > 95%`？

当前答案需要分成总体指标和逐题指标：

- **Candidate Recall@100 已稳定达到 100%。** 最近多轮完整 7 题实验的平均值和最低值都为 100%。
- **总体 Selected Recall@20 已连续超过 95%。** v18、v21、v22 分别为 96.43%、96.63%、98.41%。
- **逐题最低 Recall@20 尚未稳定超过 95%。** v22 中 7 题有 6 题为 100%，日奈题因漏掉 Episode 7 而为 88.89%。
- 因此可以确认“总体 Recall@20 > 95%”，但不能宣称“每一道复杂问题都 >95%”。

这一区分非常重要。基础召回层已经不是主要瓶颈；当前剩余问题集中在：

```text
Candidate@100 已包含正确证据
        ↓
如何在 20 个最终证据槽中稳定保留所有必要角色
```

本阶段全程固定：

```text
Paragraph              关闭
Concept                 保守模式
Association 图遍历     关闭（graph_max_hops = 0）
查询期增长              关闭（growth_max_rounds = 0）
最终答案生成            跳过（retrieval-only）
Embedding               float32 / 1024 维
```

因此本报告中的收益不能归因于增长边。

---

## 2. 指标和术语定义

### 2.1 required Episode group

人工证据清单不是要求命中某一个绝对唯一的 Episode，而是按“可替代证据组”定义：

```text
[197, 199]
```

表示命中其中任意一个即可覆盖该事实槽。这样可以容纳相邻 Episode、不同摘要粒度和语义等价证据。

### 2.2 Candidate Recall@100

系统完成 Dense、Sparse、Source 映射和多查询融合后，送入证据重排器的前 100 个 Episode 中，覆盖了多少人工 required Episode group。

它衡量的是：

> 正确证据是否已经被基础检索找到。

若 Candidate@100 漏证据，后面的 LLM 无论多聪明都无法补回。

### 2.3 Selected Recall@20

系统把 100 个候选压缩到最终 20 个 Episode 后，仍覆盖多少 required Episode group。

它衡量的是：

> 已经找到的正确证据能否在有限上下文预算中被保留下来。

### 2.4 原子证据槽

原子证据槽是一条 Episode 可以独立证明或否定的单一主张。例如：

```text
梓刚转学
梓学习跟不上
```

是两个槽，不能合并成“刚转学、学习跟不上”后只保留一条证据。

类似地，下列两端也必须分开：

```text
故事开端 / 故事结尾
最初提出 / 后来应用
否定旧命题 / 提出正面替代主张
人物本人陈述 / 人物代组织成员汇报
控制者的计划 / 被控制组织自己的行动和目标
```

### 2.5 coverage scout

覆盖侦察器不回答问题。它读取 Candidate@100，逐原子槽标出直接相关 Episode。

当前运行两遍相互独立的侦察，以降低单次模型注意力漂移。两遍都不能使用作品常识。

### 2.6 alternatives 与 joint

- `alternatives`：多个 Episode 各自足以证明同一主张，最终保留一个即可。
- `joint`：多个 Episode 分别证明合取条件，需要共同保留。

两名侦察器若对同名槽的模式判断不一致，系统降级为 `alternatives`，避免错误的 `joint` 把大量相邻 Episode 全部锁进 Top-20。

### 2.7 Shortlist@48

两遍 coverage 合并后，先形成 48 条短名单，再由槽位压缩器选择最终 20 条。

48 是召回缓冲区，不是最终答案上下文。

---

## 3. 当前基础检索数据流

```text
用户问题
   ↓
QueryIntent：人物、关系、时间、因果、4—12 条检索问题
   ↓
确定性结构槽补充
   ├─ 控制边界：组织自身行动 / 自述目标
   ├─ 开端与结尾：两个叙述端点
   ├─ 比较题：各剧情对象保持自己的作用域
   ├─ 反驳题：否定旧命题 / 正面替代主张
   └─ 中文列举与成对主语拆分
   ↓
Dense Episode + Dense Concept
Sparse Episode trigram + bigram
Sparse Source trigram + bigram → Source 内 Episode
   ↓
RRF 与多通道原子锚点融合
   ↓
Candidate@100
   ↓
独立 coverage scout A
+ 独立 coverage scout B
   ↓
同名槽归一化、ID 合并、模式冲突降级
   ↓
Shortlist@48
   ↓
LLM 槽位压缩
   ↓
Selected@20
```

这里的 Concept 搜索仍会参与基础候选生成，但本阶段不通过 Association 图向外扩展。

---

## 4. 已实现资产

### 4.1 Sparse 检索

SQLite schema 已升级到 v6，包含：

- Episode FTS5 trigram；
- Source FTS5 trigram；
- Python 生成的 CJK bigram companion index；
- Episode 与 Source 的插入、更新、删除同步触发器；
- Source 命中后，用本地 dense 分数映射回相关 Episode。

bigram 通道解决了“乐园悖论”等较短中文锚点在 trigram 中不稳定的问题。多个 Sparse 通道取最大值而非直接相加，避免同一词因两种分词方式重复投票。

### 4.2 Dense + Sparse 多通道融合

每个检索问题都同时产生：

- Episode dense 排名；
- Concept dense 排名；
- Episode sparse 排名；
- Source sparse 排名及 Source→Episode 展开；
- 融合排名；
- 分通道原子锚点排名。

Candidate@100 采用全局候选与原子锚点交错，避免一个强主题占满全部候选。

### 4.3 查询拆解

- QueryIntent 上限从 8 个检索问题提高到 12 个；
- follow-up 上限提高到 8 个；
- 重排层最多保留 40 个原子槽；
- 中文长清单按句拆分，再按顿号拆分；
- 包含“还是”的互斥分类题不按合取条件误拆；
- `A 与 B 的顾虑/行动` 会拆成人物本人和另一主体两个证据归属槽。

### 4.4 两阶段证据压缩

第一版是一轮 LLM 直接从 100 条选 20 条，容易出现“修好一个槽、丢掉另一个槽”。当前改为：

1. 两名独立 coverage scout 分别从 100 条候选建立证据覆盖；
2. 同名 query 去除括号补充后合并；
3. 同槽证据 ID 求并集；
4. `joint/alternatives` 分歧时使用较保守的 `alternatives`；
5. 生成 48 条短名单；
6. 最终压缩到 20 条。

### 4.5 中断恢复

评测器现在每完成一题就写入 `checkpoint=true` 的部分报告，整批结束后改为 `checkpoint=false`。

另有恢复工具可以从完整 `retrieval_completed` JSONL 日志和独立题目结果中重建被中断的总报告：

```text
benchmarks/recover_base_retrieval_eval.py
```

查询日志文件名加入 UUID 后缀，避免多线程同微秒创建日志时发生冲突。

### 4.6 自动测试

当前自动测试：

```text
100 tests passed
```

覆盖内容包括 Sparse 索引、查询槽上限、结构查询、合取拆分、主语归属、coverage 合并、模式冲突、槽位保护、数据库与既有增长审计逻辑。

---

## 5. 完整实验结果

| 运行 | Candidate@100 平均 | Candidate 最低 | Selected@20 平均 | Selected 最低 | 逐题 >95% |
|---|---:|---:|---:|---:|---:|
| v4 bigram | 100% | 100% | 89.48% | 75.00% | 2/7 |
| v5 coverage→shortlist | 100% | 100% | 92.80% | 85.71% | 3/7 |
| v9 atomic coverage | 100% | 100% | 98.21% | 87.50% | 6/7 |
| v12 joint + atomic | 100% | 100% | 96.63% | 87.50% | 5/7 |
| v15 structural slots | 100% | 100% | 94.84% | 87.50% | 4/7 |
| v18 dual coverage | 100% | 100% | 96.43% | 87.50% | 5/7 |
| v21 independent merged | 100% | 100% | 96.63% | 87.50% | 5/7 |
| v22 stability repeat | 100% | 100% | **98.41%** | 88.89% | **6/7** |

Candidate@100 从 v4 开始连续完整运行均为 100%。这说明当前 Dense + Sparse 基础召回已经足以覆盖这 7 道题的人工证据组。

Selected@20 的波动证明：提示词变强并不保证每题单调改善。不同运行会在同一个 20 槽预算中漏掉不同的叙事角色，因此必须把“总体均值”和“逐题稳定性”分别报告。

### 5.1 v22 逐题结果

| 问题 | Candidate@100 | Selected@20 |
|---|---:|---:|
| 老师义务→新 ETO | 100% | 100% |
| 古圣堂象征/基础设施/机制边界 | 100% | 100% |
| 阿里乌斯两个资助者与控制边界 | 100% | 100% |
| 未花动机四层认识论 | 100% | 100% |
| 梓→日富美→新 ETO 链 | 100% | 100% |
| 日奈职责、疲惫与老师回应 | 100% | 88.89% |
| 山海经妃咲与圣三一未花类比 | 100% | 100% |

日奈题唯一遗漏组为 Episode 7：圣娅在 `main/31010.json` 中要求老师不要移开视线，并把看到故事最后称为老师做出选择后的义务。

该 Episode 在 Candidate@100 中存在，但 coverage 模型有时会把后期 `main/33190.json` 中“帮助、相信并支持学生梦想”的回忆 Episode 205 同时当作开端和结尾证据。这是叙述角色标注问题，不是向量漏召回。

---

## 6. 已证实的技术结论

### 6.1 基础 Candidate@100 足够可靠

在当前语料、问题和 embedding 模型上，基础检索不依赖增长边即可达到 Candidate@100 100%。

这意味着下一阶段评估 Association 增长收益时，可以把“基础检索本来漏了正确证据”与“增长边没有帮助”分开。

### 6.2 纯 Dense Top-K 不够

短中文锚点、文件名相关线索、跨语言别名和 Source 上下文需要 Sparse 与 Source→Episode 展开。bigram 对短 CJK 查询尤其重要。

### 6.3 最终证据选择不是普通相关性排序

复杂问题需要的是覆盖多个不同角色，而不是选 20 条最相似 Episode：

```text
制度定义
历史起点
后续应用
人物行动
人物动机
反证
不确定性边界
结果
```

因此最终选择更接近“受预算约束的证据集合覆盖”，不是单一相似度 Top-K。

### 6.4 文件名可以辅助叙述顺序，但不能代替故事时间

`main/31010.json` 与 `main/33190.json` 的编号可以帮助判断故事开端和后期叙事位置；但倒叙和回忆意味着文件顺序不能直接当作事件实际发生时间。

### 6.5 独立重复比依赖式自我审计更有效，但更慢

让第二个模型阅读第一遍 coverage 后“找缺口”，会继承第一遍的注意力框架。让第二遍从零独立侦察更容易产生互补证据。

代价是每题需要：

```text
QueryIntent / follow-up
+ coverage scout A
+ coverage scout B
+ Top-20 compressor
```

模型延迟和 token 使用明显增加。当前 demo 不考核延迟，所以先换可靠性；正式系统应按问题复杂度选择是否启用第二遍。

---

## 7. 尚未解决的问题

### 7.1 逐题 Recall@20 仍不稳定

总体已经超过 95%，但日奈题等高槽位问题仍可能漏一个组。不能把 v22 写成“所有测试全部通过”。

### 7.2 coverage 语义仍会误标

模型可能：

- 把后期回忆标为故事开端；
- 把人物本人观点与代他人汇报合并；
- 把多个互补事实错标为 alternatives；
- 把一组大量相邻 Episode 错标为 joint；
- 在跨剧情比较中把人物套进另一剧情的事件范围。

代码已对常见错误做结构修正，但没有完全消除。

### 7.3 coverage 超预算时缺少正式优化器

两遍侦察合并后可能得到超过 20 个 coverage 项。目前主要依赖 LLM 压缩和有限的槽位保护。

更可靠的方案应把问题建模为：

```text
Query slot ↔ Episode
二部图最大覆盖 / 加权集合覆盖
```

在 20 个 Episode 预算内最大化独立证据槽覆盖，同时惩罚同场景改写和重复事实。

### 7.4 当前评测集仍然很小

7 道题都来自同一批人工设计的高难网络问题。它们适合诊断，但不足以证明对任意剧情、用户对话或百万 Episode 都能维持同样指标。

### 7.5 本阶段没有评估最终答案

本轮使用 `--retrieval-only`，没有验证：

- 回答是否正确引用证据；
- 时间线是否被写反；
- 是否把推论写成事实；
- 答案审计是否稳定；
- Association 增长是否带来收益。

---

## 8. 下一阶段建议

### 8.1 冻结基础候选层

暂时冻结 Dense/Sparse 参数，不再为 Selected@20 的问题调整 Candidate 融合权重。Candidate@100 已经稳定 100%，继续调权重容易破坏已通过的基础召回。

### 8.2 实现显式证据集合覆盖器

优先实现 Query-slot/Episode 二部图和预算 20 的加权集合覆盖：

- LLM 只判断 Episode 支持哪些原子槽；
- 程序负责在预算内选择集合；
- `joint` 条件拆成多个子槽，不让模型用一个布尔标签锁死 9 条证据；
- 对相邻改写、同 source 重复和多语言同义 Episode 加冗余惩罚；
- 把 source_key 叙述位置作为低权重特征，而非事实裁决。

这是进一步提高逐题最低值最值得做的技术工作。

### 8.3 建立重复运行统计

同一固定 Candidate@100 至少重复 5 次重排，报告：

- 每题平均 Recall@20；
- 每题最低值；
- 每个 required group 的命中频率；
- coverage 模式分歧率；
- 两遍侦察的互补增益；
- 延迟和 token 成本。

### 8.4 再恢复 Association 增长 A/B

基础层冻结后再比较：

```text
A：无图、无增长
B：只读既有图
C：允许 generation=0 直接关系增长
D：允许 generation>0 推断链增长
```

主要指标应是同一证据槽覆盖的边际增益和答案质量，不是新增边数量。

---

## 9. 关键文件

### 实现

```text
src/memory_demo/retrieval/sparse.py
src/memory_demo/retrieval/engine.py
src/memory_demo/llm/prompts.py
src/memory_demo/config.py
src/memory_demo/database.py
src/memory_demo/schema.sql
src/memory_demo/types.py
```

### 评测工具

```text
benchmarks/run_base_retrieval_eval.py
benchmarks/rerank_saved_candidates.py
benchmarks/recover_base_retrieval_eval.py
benchmarks/audit_sparse_replay.py
```

### 证据与结果

```text
validation/evaluation-questions-stage4-network.json
validation/stage4-network-evidence-manifest.json
validation/base-retrieval-online-all7-v21-independent-merged.json
validation/base-retrieval-online-all7-v22-stability-repeat.json
validation/base-retrieval-online-logs/
```

### 数据库

```text
冻结基线：validation/ba-stage3-deep-v37.db
本阶段评测副本：validation/evaluation-stage7-generation/pilot/block-001/C/graph.db
```

---

## 10. 最终判断

当前基础检索可以作出以下有证据支持的判断：

> 在这 7 道已登记证据清单的高难剧情问题上，不使用 Paragraph、Association 图遍历或增长边时，Candidate@100 已稳定达到 100%；总体 Selected Recall@20 已连续超过 95%，最终重复达到 98.41%。

同时必须保留以下限制：

> 最终 20 条的逐题最低召回仍未超过 95%，主要风险是 LLM 对叙事端点、主语证据归属和多条件槽的随机压缩。下一步应实现显式集合覆盖器，而不是继续增加提示词特例或提前重新启用增长边。
