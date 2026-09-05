# Stage 7：Association Generation 与放宽审计实验报告

**日期：** 2026-08-26  
**实现版本：** `v3.23_association_generation`  
**数据库 schema：** v4  
**语料基线：** `validation/ba-stage3-deep-v37.db`  
**Paragraph：** 关闭  
**Concept：** 保持原有 conservative 提取结果，不做 augmentation

## 1. 本阶段要解决的问题

旧结构把两个不同问题混在了 `confidence` 里：

- 这条关系在现有证据下有多可信；
- 这条关系离 Source/Episode 直接经验隔了多少次推断。

一条经过多次推断的关系仍可能具有较高可信度；一条直接记录的角色猜测也可能可信度较低。因此，不能用一个浮点数同时表示两者。本阶段给 Association 增加 `generation INTEGER NOT NULL DEFAULT 0`，不修改 Paragraph 和 Concept 数据结构。

## 2. 专有名词与精确定义

### 2.1 confidence

`confidence ∈ [0, 1]`，表示当前证据对关系内容的支持强度。它回答：“如果只看目前掌握的前提，我有多相信这条关系？”

### 2.2 generation

`generation` 是非负整数，表示关系距离直接经验的**最长推断链深度**：

- `0`：直接经验关系，例如 Episode 明确涉及某 Concept、人工确认的关系；
- `1`：只使用 Episode/Concept 直接内容得到的首层综合推论；
- `n + 1`：新推论使用了最高 generation 为 `n` 的 Association 作为前提。

若同时使用 generation 1、2、4 三条前提，新关系是 generation 5，而不是 8。求和会把“并列证据的数量”误当成“离直接经验的距离”。

### 2.3 premise_association_ids

查询增长模型如果使用已有 Association 作为推断前提，必须显式返回所有 `premise_association_ids`。本地程序只接受当前提示中可见的 Association ID；引用不可见前提时整条候选关系被拒绝。前提 Association 的 ID、generation、relation type/key/text 会被写入 `evidence_json`。

### 2.4 强化时的 generation 规则

同一 durable Association 可能通过多条路径再次得到支持。数据库保存已知推导中的**最小 generation**：

```text
stored_generation = min(old_generation, new_generation)
```

因此，更远的推论可以提高 weight、evidence_count 或 confidence，但不能把已有直接关系伪装成更远或更近；后来若找到更接近直接经验的证明，generation 可以下降。所有不同证据仍保存在合并后的 provenance JSON 中。

## 3. 审计放宽了什么，保留了什么

系统仍保留 DeepSeek 主审计与 GLM 对抗复核的双接受写入门。放宽的是审计判据：

- 不再因为结论不是原文逐字陈述就自动拒绝；
- 允许由端点事实或显式 premise Association 合理推出、可供未来复用的关系；
- 允许跨 Source 的 `political_precondition`，前提是以“查询综合推论”标注，并明确保留“可能、未证明具体机制”等限制。

仍然拒绝：

- 无 Source/Episode/可见 Association 支撑的内容；
- 把问题措辞当证据；
- 人物、职位、组织身份的主客体转移；
- 把另一场事件的路线、权限、炸药或命令移植到当前事件；
- 把角色的“推测/怀疑”升级成已确认事实；
- 依赖已有关系却漏报或伪造 premise ID。

这不是取消审计，而是把“允许推论”与“推论必须可追溯”同时落实。

## 4. 实现范围

本阶段完成了：

1. schema v3 → v4 原地迁移；
2. legacy 边的保守回填：已知 LLM 推论回填为 generation 1，无法恢复的历史深度不猜成 2+；
3. AssociationDraft、Repository insert/reinforce、CLI 与路径输出增加 generation；
4. 导入期 Episode↔Episode、Concept↔Concept 的 LLM 关系标为 generation 1；
5. Episode↔Concept 直接提取关系保持 generation 0；
6. 查询增长返回和验证 premise Association；
7. generation 按最长前提链计算，并写入 evidence/audit JSON；
8. 回答模型能够看到 confidence 与 generation 的不同含义；
9. Paragraph 默认关闭，Concept 默认恢复 conservative，功能代码保留供以后消融。

## 5. 本地回归

最终回归包含 schema 迁移、generation 独立性、强化取最小值、负数拒绝、generation 3 前提传播、不可见 premise 拒绝、受限跨 Source 推论接受、身份幻觉拒绝和原有检索/导入行为。

```text
81 tests passed
```

## 6. 在线成对试验设计

沿用 Stage 5 的冻结基线、前置问题 Q1、目标问题 Q2 和四臂设计：

- `T`：相关前置问题，允许增长；
- `C`：相同相关前置问题，不允许增长；
- `M`：复制 T 的数据库，但在 Q2 时遮掉 T 新增/强化的边；
- `P`：无关主题前置问题，允许增长。

目标问题仍是古圣堂的象征、地下结构、袭击与新 ETO 八组证据闭环。边数量只用于审计枚举，不作为性能指标。Paragraph 明确关闭，Concept 不做新提取。

## 7. 在线 pilot 结果

运行时间约 17 分钟。T 创建 8 条边，P 创建 7 条并强化 1 条；T 的 8 条变化边全部是 generation 1，没有出现伪造的 generation 2+。

### 7.1 固定 replay bundle

本次 pilot 的固定 replay bundle 四臂都是 8/8，已经达到天花板，因此无法显示正向边际收益。

### 7.2 完整 Q2 运行

| Arm | 命中证据组 | Recall@30 | Source 闭环 | 最终答案审计 |
|---|---:|---:|---|---|
| T | 6/8 | 0.750 | 否 | 通过 |
| C | 7/8 | 0.875 | 是 | 通过 |
| M | 6/8 | 0.750 | 否 | 通过 |
| P | 7/8 | 0.875 | 是 | 通过 |

关键因果比较是 T 对 M，而不是 T 对 C。T 与 M 都是 6/8，说明遮掉新边没有改变 Recall。T 相比 C 少 1 组不能归因于新增边，更可能来自完整查询中 LLM 规划/选择的随机波动。

T 的后续答案路径实际使用了新边 #1828。它把 Episode #84 与 #123 连接起来，带入了 #123；但预注册 Q2 证据组没有把 #123 计为目标，且它占用了有限答案槽位。结果是排序发生改变，Recall 没有提高。

## 8. 十个冻结 replay bundle 的离线反事实复核

为排除单次 planner 随机性，把新 T 图和遮罩图放到上一阶段保存的 10 个 Q2 replay bundle 上重放：

```text
T - M Recall@30 平均差：0.000
正 / 零 / 负：0 / 10 / 0
10 个 bundle 中 9 个实际走过新边
10 个 bundle 中 7 个在遮罩后发生排名或入选变化
```

这进一步确认：新边能被遍历并改变候选构成，但没有改变八组目标证据覆盖。

## 9. 单边端点可达性探针

为了区分“边完全无效”和“边有效但未对当前评测目标产生收益”，又对 8 条变化边做双向端点探针，共 16 次：只把一端作为 seed，观察另一端在 T 与 M 中是否成为候选/最终证据。

```text
16/16：处理图实际使用目标新边
4/16：只有处理图能把另一端加入 candidate
5/16：只有处理图能把另一端加入最终选择
```

其中 #1828（Episode #84 ↔ #123）和 #1829（Episode #199 ↔ #275）在两个方向都产生了新的候选与选择可达性。这证明增长边具有局部检索作用；但这仍只是机制诊断，不等于在真实未来问题上获得稳定净收益。

## 10. 当前结论

### 已经验证

- confidence 与推断距离已在数据模型中独立；
- generation 能正确迁移、写入、强化和沿显式 premise 传播；
- 审计放宽后能接受更多证据约束的综合推论；
- 新边会被未来检索实际遍历；
- 某些新边确实增加端点 Episode 的可达性；
- 答案安全审计没有因放宽而失败。

### 尚未验证

- 增长边能稳定提高真实后续问题的预注册 Episode Recall；
- generation 2+ 能在真实连续对话中自然出现并带来收益；
- 放宽审计在长时间增长中不会造成高 generation 边爆炸；
- 高 generation 是否需要参与检索排序或仅用于答案措辞。

因此，不能把本阶段表述为“增长收益已经通过”。准确说法是：

> generation 与可追溯推论机制通过；增长边的局部可达性通过；真实问题的稳定净收益仍是负结果。

## 11. 下一步实验建议

1. 预注册一个新的 Q1→Q2 问题族，让 Q1 能形成明确桥，而 Q2 的目标证据在无桥时确实容易漏掉；不能用本次结果事后修改旧 Q2 的证据组。
2. 同时报告默认宽预算和受限检索预算。当前默认 40 个 Episode/原子查询、最终 30 个 Episode 容易让 replay 达到 8/8 天花板；受限预算用于观察边际作用，默认预算用于验证实际系统收益。
3. 把“新边带入了哪个新 Episode、该 Episode 是否属于预注册目标、挤掉了谁”做成正式诊断表。
4. 设计连续三问实验：第一问形成 generation 1，第二问必须显式使用它形成 generation 2，第三问比较保留/遮罩 generation 2 的差异。
5. 暂不自动按 generation 降权。generation 是距离，不是可信度；是否参与排序必须由新实验决定。
6. Paragraph 继续关闭，Concept 保持 conservative，避免同时改变三项变量。

## 12. 资产

```text
validation/evaluation-stage7-generation/pilot/run-report.json
validation/evaluation-stage7-generation/pilot/block-001/block-report.json
validation/evaluation-stage7-generation/pilot/block-001/T/association-delta.json
validation/stage7-generation-replay-probe.json
benchmarks/replay_generation_probe.py
```

