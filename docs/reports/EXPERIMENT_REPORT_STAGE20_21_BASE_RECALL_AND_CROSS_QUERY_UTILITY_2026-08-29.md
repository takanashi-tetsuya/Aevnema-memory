# Stage 20/21：基础证据槽与跨问题 Association 收益报告

日期：2026-08-29（Asia/Tokyo）  
冻结基线：`validation/evaluation-stage17-full-rebuild/assets-v3-throughput/graph.db`  
冻结基线 SHA-256：`864096431e0bab1d187473d6d261db5ccc40aa0e824f8ec14cc1bf67dd10fd11`  
最终提示词版本：`v3.36_dual_sparse_evidence_slots`  
确定性测试：157 项

## 1. 结论摘要

本阶段得到两个不同层次的结论。

第一，基础检索缺口已经被精确修复。Stage 20 在古圣堂袭击题中同时保护“谁直接执行袭击”和“哪段前文能
支持高层协助”两个独立证据槽。静态图与增长图共享同一冻结计划，最终都严格通过；增长提出的临时关系没有
增加直接 Episode，因此全部丢弃，冻结数据库保持不变。

第二，当前完整系统第一次得到可消融的跨问题 Association 正收益。Stage 21 让 Q1 在隔离数据库中学习
“日奈在签约前履行公开职责”与“灾难后暴露私人疲惫”之间的关系，再用不同的 Q2 检验这些关系能否带回
原本只在候选层出现、却没有进入最终证据包的公开职责 Episode 1513。

结果如下：

| Q2 最终证据预算 | Treatment | Masked | 净变化 |
|---:|---:|---:|---:|
| 30 | 9/9，Recall=1.0000 | 8/9，Recall=0.8889 | +1 证据组，+0.1111 |
| 20 | 9/9，Recall=1.0000 | 8/9，Recall=0.8889 | +1 证据组，+0.1111 |

两种预算下，Treatment 都净带回 Episode 1513；Masked 都缺少 `main/33060.json`，Treatment 的 required
source closure 则通过。这个差异来自 Q1 新关系的 cue：两边使用相同的 Q2 解析、query embedding、Episode
候选和 rerank，Masked 只遮掉 Q1 的关系变化。

Q1 一共产生 8 条双模型审计通过的候选边。数量只用于审计，不是成绩。逐边消融显示：

- 8907、8908、8912 各自单独存在时都能恢复 Episode 1513；
- 三条边互为替代，因此逐一 leave-one-out 时没有任何一条是不可替代的必要边；
- 最小晋级策略只建议保留质量最高的 8912，拒绝另外 7 条冗余或无效边。

因此当前最准确的表述是：**已经证明“经 Q1 学习的关系文本可在 Q2 中产生可归因的直接证据收益”，但只完成
一个有设计针对性的正向 pilot，尚不能宣称收益跨题型稳定，也还没有把观察期机制打开到生产默认路径。**

## 2. 专有名词定义

### 2.1 证据槽（evidence slot）

一条复杂问题通常包含多个彼此独立的事实要求。每个要求是一个证据槽。例如古圣堂题至少有：

1. 爆炸/巡航导弹的场景锚点；
2. 直接执行袭击的分校；
3. 较早政治合作或政变证据；
4. 事实与推断的边界。

如果只按全题相似度取 Top-K，多个高度相似的爆炸片段会占满预算，而较早的政治证据被挤掉。证据槽的作用是给
不同子问题各留至少一个进入 rerank 的机会。

### 2.2 constraint slot 与 answer slot

- `constraint slot`：保护问题中的限制条件或因果前提，例如“成功渗透是因为哪位高层暗中协助”；
- `answer slot`：保护所问直接答案，例如“古圣堂袭击由哪个分校直接执行”。

两者都通过确定性的结构化查询加 Sparse Top-1 实现，不调用额外 LLM，也不读取测试题答案 manifest。

### 2.3 FrozenQueryPlan

一次 Q2 的完整基础检索快照，冻结以下内容：

- QueryIntent 和原子问题；
- query embedding；
- Dense、Sparse 与 Source cohort 排名；
- Episode 候选和 rerank；
- 配置和内容哈希。

Treatment 与 Masked 使用同一快照，所以二者差异不能由 LLM 随机拆题或 embedding 调用波动解释。

### 2.4 观察期边（probation Association）

观察期边是 Q1 已通过证据审计、但尚未取得长期存储资格的候选记忆：

- 只存在于隔离实验数据库；
- 允许在后续 Q2 中参与检索；
- Q2 屏蔽实验没有证明效用时全部丢弃；
- 不修改冻结基线，也不改变生产默认策略。

它解决了同查询效用门的时间尺度问题：一条为未来联想而建立的边，可能对创建它的 Q1 没有新增价值，却可能对
之后不同措辞的 Q2 有价值。

### 2.5 Association cue

普通图遍历只有在某个端点已经成为高分 seed 时才能沿边扩展。Association cue 则把已审计的
`relation_text` 也生成 float32 embedding。查询时可以直接检索“关系本身”，命中后再把两个 Episode 端点带回。

Stage 21 没有重新生成 Q2 embedding，而是用 FrozenQueryPlan 中同一批 float32 query embedding 分别检索
Treatment 和 Masked 的关系视图。关系索引只常驻 RAM，SQLite 仍保存关系文本和证据。

### 2.6 Treatment、Masked、single-edge 与 leave-one-out

- **Treatment（T）**：Q1 候选边全部可见；
- **Masked（M）**：同一 Q2 计划中遮掉 Q1 的全部变化边；
- **single-edge**：一次只保留一条 Q1 边，检验它是否单独足够；
- **leave-one-out**：保留其余边，只遮掉当前一条，检验它是否不可替代。

“单独足够”和“不可替代”不是同一概念。三条功能重复的边可以各自单独足够，但任意删掉一条都不会破坏整批收益。

### 2.7 confidence、generation 与 utility

| 字段 | 含义 | 本实验中的作用 |
|---|---|---|
| confidence | 当前证据对关系断言的支持程度 | 用于同等效用候选间的质量排序 |
| generation | 距离直接 Episode 经过多少层推断 | 8 条候选均为 generation=1 |
| utility | 关系是否改善后续检索 | 由 Q2 Treatment/Masked 和逐边消融测量 |

高 confidence 不自动等于高 utility。8 条关系都通过了双模型证据审计，但只有 3 条能单独恢复缺失证据。

## 3. Stage 20：先修基础证据覆盖

### 3.1 问题

Stage 19 的古圣堂题已经在候选阶段找到 Episode 1167/1168，却在最终 30 条证据中丢失
`main/32170.json`；第一次单约束槽修复又挤掉了 Episode 1544，出现“修复政变证据、损失直接执行者证据”的
跷跷板。

### 3.2 实现

`_structural_queries()` 现在识别两类查询：

```text
__constraint_slot__ 成功渗透，因为茶会高层暗中协助；与政变有关的前文直接证据是什么
__answer_slot__ 古圣堂哪个分校直接执行袭击
```

证据保底顺序为：

```text
constraint slot Sparse Top-1
→ answer slot Sparse Top-1
→ 全问题 Sparse
→ 原子问题保底
```

系统记录 `evidence_slot_trace`，区分计划保护、进入 rerank、最终进入回答以及在哪一步丢失。新增配置：

- `rerank_constraint_floor_per_query = 1`
- `rerank_constraint_floor_total_limit = 3`

### 3.3 结果

最终 Stage 20B：

- plan ID：`019d58b26a6313e835b10cb5548f382b81c29d0656e2be8f7f51e4a876a53af9`；
- constraint slot Sparse Top-1：Episode 1168；
- answer slot Sparse Top-1：Episode 1544；
- 静态图严格通过 1/1；
- 增长图严格通过 1/1；
- 两种模式都命中 Episode 1481、1544、1167/1168 和两个 required source；
- 增长图提出 4 条临时关系，但 Treatment 与 Masked 的最终 30 个 Episode 完全相同；
- 4 条关系全部被反事实门删除；
- 静态库、增长库和冻结基线均保持 8,904 条 generation=0、0 条 generation>0；
- 冻结基线哈希未改变。

Stage 20B 成本：

| 阶段 | 请求 | tokens | 墙钟时间 |
|---|---:|---:|---:|
| 共享 Q2 计划 | 6 | 48,269 | 201.869 秒 |
| graph_static 回答/审计 | 6 | 191,936 | 286.727 秒 |
| graph_growing 增长/回答/审计 | 12 | 263,590 | 497.744 秒 |

这一步证明基础证据闭环已经恢复，也再次证明同查询临时边在没有 Episode 增益时会安全零写。

## 4. Stage 21：Q1 学习、Q2 复用

### 4.1 假设

旧策略只问：“新边是否帮助创建它的当前问题？”Stage 21 改问：

> 一条在 Q1 中形成、对 Q1 本身并不新增证据的关系，能否在之后的 Q2 中把原本被最终预算淘汰的直接 Episode
> 带回来？

### 4.2 Q1 与 Q2

Q1 要求比较：

- `main/33060.json`：日奈坚持风纪委员会不会解散、以 ETO 约束万魔殿，并先去履行签约仪式职责；
- `main/33190.json`：日奈承认自己早已撑不住，希望得到关心和夸奖，老师先感谢她，她后来重新等待指示。

Q2 使用已有长链题 `hina_public_duty_personal_exhaustion_and_teacher_response`，要求同时解释公共职责、爆炸后的战斗、
星野创伤比较、情绪坦白、老师回应、恢复行动以及老师义务。

预注册 hard group 是 Episode 1441/1512/1513/1514。基础 Q2 候选能看见 1513、1512、1514，但最终证据会把
这一组全部淘汰。

### 4.3 实验流程

```text
冻结 Q2 解析、query embedding、Episode 检索与 rerank
                         │
                 Q1 隔离观察期增长
                         │
              8 条 generation=1 候选边
                         │
           relation_text → float32 RAM cue index
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   Treatment                          Masked
   候选边可见                         候选边全遮罩
        │                                 │
        └──────────── 同一 Q2 计划 ───────┘
                         │
               比较最终直接 Episode
                         │
              逐边 single / LOO 消融
```

Q1 的同查询效用门只在这个隔离副本中临时关闭；双模型审计、generation、证据快照和来源审计没有关闭。正式默认配置
仍使用同查询反事实门。

### 4.4 一次边界审计误报及修复

第一次 Q1 已成功产生关系，但外层审计把 `main/33040.json`、`main/32070.json` 判为越界。原因不是这些文件在
用户语料之外，而是 remapped evidence manifest 没有全局 `corpus_boundary`，旧 runner 错误退化为只允许 Q2 的
`required_sources`。

修复后，本实验显式冻结 6 个允许来源：

```text
main/31010.json
main/32070.json
main/33040.json
main/33060.json
main/33070.json
main/33190.json
```

runner 在 resume 时会使用当前边界重新审计现有 delta；失败审计不能因 resume 被跳过。没有重复调用 Q1。

## 5. Stage 21 结果

### 5.1 批次级因果结果

30 条预算：

- Treatment：9/9，Recall=1.0，required source closure=true；
- Masked：8/9，Recall=0.8889，required source closure=false；
- Treatment-only：Episode 1513；
- Masked-only：Episode 175。

20 条预算：

- Treatment：9/9，Recall=1.0，required source closure=true；
- Masked：8/9，Recall=0.8889，required source closure=false；
- Treatment-only：Episode 1513、112；
- Masked-only：Episode 104、140。

这不是“边被答案路径引用”这种弱指标，而是遮掉 Q1 边以后，预注册证据组和 `main/33060.json` 同时从最终证据中
消失。

### 5.2 新关系 cue

Q1 产生 8 条 generation=1、`dual_accepted` 关系；8 条 relation embedding 全部为 float32，RAM 索引实际占用
65,664 bytes。Q2 对关系文本的最高 cosine 包括：

- 8906：0.7753；
- 8909：0.7182；
- 8910：0.7177；
- 8907：0.6992；
- 8912：0.6950。

数量不参与评分。批次级结果只证明“这批边至少包含有用成分”，不能证明 8 条都值得保存。

### 5.3 逐边归因

30 和 20 两种预算得到相同结论：

| Association | 端点 | 单独足够 | LOO 必要 | 处理 |
|---:|---|---|---|---|
| 8907 | 1513 ↔ 110 | 是 | 否 | 冗余拒绝 |
| 8908 | 1513 ↔ 143 | 是 | 否 | 冗余拒绝 |
| 8912 | 1513 ↔ 101 | 是 | 否 | 建议晋级 |
| 其余 5 条 | 不同端点 | 否 | 否 | 无效拒绝 |

三条单独足够边都会净带回 Episode 1513。它们同时存在时互为替代，所以没有单条边是不可替代的。最小晋级策略按：

1. 较小 generation；
2. 较高 confidence；
3. 较高 weight；
4. 稳定 ID；

只选择 8912。其关系为：

```text
Episode 1513：签约前坚持组织存续、政治目标和立即履责
Episode 101：灾难后承认已经不行、疲惫并想引退
relation_key：thematic_contrast
confidence：0.85
weight：0.855
generation：1
```

关系文本明确标为跨场景主题对比，不声称前一场景直接导致后一场景。

### 5.4 成本

Q2 冻结计划生成一次：

- 7 个请求；
- 50,021 tokens；
- 218.208 秒。

Q1 增长、双审计和 8 条关系 cue embedding：

- 14 个请求；
- 121,026 tokens；
- 日志跨度 443.451 秒，其中 Q1 主查询约 361.046 秒，cue 索引约 1.201 秒。

所有 T/M、single-edge、leave-one-out 重放均不调用模型。Stage 21 总计 21 个模型/API 请求、171,047 tokens；
最昂贵的是 Q1 的生成与审计，不是 cue embedding 或因果消融。

## 6. 负向诊断为什么也重要

Stage 21 先做了三个不成功的 oracle pilot：

1. 第一条理想桥指向圣娅义务，但当前基础检索已经保住该证据，没有增益空间；
2. 改为公共职责 hard group 后，理想边端点不是图 beam 的有效起点，没有被遍历；
3. 换成 beam 内端点后，Episode 1513 本来已可经 generation=0 Concept 图到达，新边没有改变最终证据分配。

这些失败说明“图里存在一条正确边”并不够。当前瓶颈不是 reachability，而是如何让后续问题直接召回关系语义，并给
关系端点保留最终证据槽。Association cue 才是本次正收益的实际机制。

## 7. 当前架构决策

### 7.1 保持生产默认不变

生产查询仍执行：

```text
RAM staging
→ 同查询 Treatment/Masked
→ 没有直接 Episode 增益则零写
```

Stage 21 的观察期策略只存在于 benchmark。单个 pilot 不足以允许所有“未来可能有用”的边进入正式 SQLite。

### 7.2 下一步应实现的最小生产形态

若扩大实验仍稳定，建议增加会话级 `ProbationAssociationStore`，而不是放宽正式 Association 表：

- Q1 审计通过但同查询无收益的边进入有限大小的观察区；
- 记录创建问题、过期时间、generation、confidence 和完整证据；
- 后续问题只对观察区 relation text 做 float32 cue 检索；
- Q2 出现可归因 Episode 增益后，执行逐边最小化并晋级；
- 超时、从未使用、产生回归或只与冗余边共同有效的候选删除；
- 正式 SQLite 仍只接收已经证明效用的最小边集。

这不会把系统变成专家规则库；它只是把“可信”和“有用”分成两个独立门。

## 8. 已知限制与下一阶段验收

本轮仍有以下限制：

1. 只有一个 Q1/Q2 家族和一次随机增长输出，不能宣称稳定性；
2. Q1 明确列出文件和两端主题，比自然用户对话更受引导；
3. Q2 是已知长链题，Q1 与 Q2 主题相关，仍需更隐式的迁移措辞；
4. cue semantic gate 在本实验中关闭，尚未测量更大候选池中的错边侵入；
5. 当前完成的是检索证据因果验证，尚未对 Treatment/Masked 最终自然语言答案做盲评；
6. 8912 是质量启发式下的最小选择，不是统计意义上的唯一真边；
7. 最终证据仍包含一些非必需 Episode，基础精度与预算利用率还有优化空间；
8. 观察期跨进程生命周期、并发、过期和用户确认尚未进入生产实现。

下一阶段建议至少包含：

- 3 个不同主题家族，每个 5 次独立 Q1 增长；
- 每个家族 2 个不复述 Q1 的迁移 Q2；
- 正样本、无关问题 placebo、相似实体 hard negative；
- 30 条默认预算为主，20 条为压力诊断；
- Treatment 必须比 Masked 增加预注册证据组并通过 source closure；
- 单边或最小子集消融必须复现收益；
- 错误身份、越界来源、推测升级为事实均为 0；
- 最终答案再做双模型盲评，确认新增 Episode 真正改善回答而非只改善 ID 指标。

## 9. 资产索引

核心代码：

- `src/memory_demo/retrieval/engine.py`：双证据槽、FrozenQueryPlan、cue attachment；
- `src/memory_demo/association_overlay.py`：staging 与 masked overlay；
- `src/memory_demo/stage8.py`：跨问题效用和最小晋级决策；
- `src/memory_demo/stage9.py`：Association cue 重放与诊断；
- `benchmarks/run_stage8_association_utility.py`：观察期 Q1/Q2 编排；
- `benchmarks/audit_stage21_probation_attribution.py`：逐边 single-edge/LOO 消融。

Stage 20：

- `validation/evaluation-stage20b-dual-slot-q3/deep-scorecard.json`；
- `validation/evaluation-stage20b-dual-slot-q3/deep-evaluation/evaluation-report.json`；
- `validation/evaluation-stage20-constraint-slots/offline-slot-replay-v3.json`。

Stage 21：

- `validation/stage21-cross-query-probation-manifest.json`；
- `validation/evaluation-stage21-cross-query-probation/pilot-v4-learned/seia_duty_to_hina_care_cross_query/family-report.json`；
- `validation/evaluation-stage21-cross-query-probation/pilot-v4-learned/seia_duty_to_hina_care_cross_query/learned-association-delta.json`；
- `validation/evaluation-stage21-cross-query-probation/pilot-v4-learned/seia_duty_to_hina_care_cross_query/probation-cue-replay-bundle.json`；
- `validation/evaluation-stage21-cross-query-probation/pilot-v4-learned/seia_duty_to_hina_care_cross_query/probation-attribution-report.json`；
- `validation/evaluation-stage21-cross-query-probation/pilot-v4-learned/seia_duty_to_hina_care_cross_query/model-log-summary.json`。

负向诊断保留在 `pilot-v1`、`pilot-v2`、`pilot-v3`，用于说明为什么普通图边不等于可测的跨问题收益。

## 10. 可复现实验命令

```powershell
$env:PYTHONPATH = "src;tests;benchmarks;."
$env:PYTHONUTF8 = "1"

python benchmarks/run_stage8_association_utility.py `
  --manifest validation/stage21-cross-query-probation-manifest.json `
  --output validation/evaluation-stage21-cross-query-probation/pilot-v4-learned `
  --phase learned `
  --workers 1 `
  --resume

python benchmarks/audit_stage21_probation_attribution.py
python -m unittest discover -s tests -v
```
