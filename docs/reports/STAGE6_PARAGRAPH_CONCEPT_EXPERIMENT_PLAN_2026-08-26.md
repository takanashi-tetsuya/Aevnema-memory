# Stage 6 实验计划：Paragraph 原文召回与细粒度 Concept 增强

## 1. 实验目的

Stage 5 已证明 Q2 的 8 组正确 Episode 每次都进入 600 条候选，但某些证据没有稳定进入最终 30 条。与此同时，Episode 是 LLM 摘要，可能省略原文中的专有词、具体部署、语气和局部细节。

Stage 6 检验两个独立假设：

1. **Paragraph 假设：** 在 Source 与 Episode 之间增加自然记录对齐的原文 Paragraph embedding，可以用原文措辞找回被 Episode 摘要弱化的 Source，再提升该 Source 内相关 Episode。
2. **细粒度 Concept 假设：** 更积极地保存人物、化名、命名制度/事件/地点子区域、关键物品、持续心理和抽象主题，可以增加跨 Episode 的稳定检索锚点与图路径。

两个变化都必须可独立关闭。没有正向证据时，不替换 Stage 5 基线。

## 2. Paragraph 的定义与边界

Paragraph 是一个 Source 内、按完整 dialogue record 边界组成的中等长度原文块：

```text
Source：约 6000 字，负责证据回查
Paragraph：约 900—1600 字，负责原文级向量召回
Episode：中位约 71 字，负责事件语义、时间线和 Association
```

Paragraph：

- 保存原文，不由 LLM 总结；
- 相邻块保留完整 record 级重叠；
- 使用 1024 维 float32 embedding；
- 有独立 SQLite 表和 RAM index；
- 不成为 Association 节点；
- 命中后只弱加权扩展到同一 Source 中最相关的 Episode；
- 不直接占用 atomic anchor 配额。

这样可以提高原文措辞召回，而不把现有图从两种节点扩成三种节点。

## 3. 当前参数与只读规模估算

```text
enabled        = true
target_chars   = 900
max_chars      = 1400（record body 软上限）
overlap_chars  = 200
minimum_chars  = 200
top_k          = 16
每个命中 Source 最多扩展 Episode = 6
Paragraph RRF 权重             = 0.35
```

对冻结库 68 个 Source 的只读切分估算：

| 指标 | 结果 |
|---|---:|
| Paragraph 数量 | 541 |
| 每 Source min / p50 / p95 / max | 1 / 8 / 12.65 / 13 |
| 字符数 min / p50 / p95 / max | 587 / 1255 / 1602 / 2091 |
| 平均字符数 | 1265.42 |
| float32 embedding 矩阵 | 2,215,936 bytes，约 2.11 MiB |

完整 Paragraph 长度包含重复的人物别名和 Source 上下文前缀，因此会高于 record body 的 1400 字软上限。当前长度仍远低于模型上下文限制，不构成内存问题。

## 4. Paragraph 的查询融合

每个 query 同时搜索：

```text
Episode Top 40
Concept Top 20
Paragraph Top 16
```

Paragraph 不直接进入图，而执行：

```text
命中 Paragraph
→ 找到 paragraph.source_id
→ 取得该 Source 的 Episode IDs
→ 用同一 query 向量在这些 Episode 中局部精排
→ 每个 Source 最多保留 6 个 Episode
→ 以 0.35 × RRF 贡献加入 Episode seed score
→ 进入现有 Association 图遍历
```

同一 Source 的多个 Paragraph 只取最佳 Paragraph 排名，避免一个长 Source 因切块数量多而重复加分。

## 5. 细粒度 Concept 策略

旧 `conservative` 档通常保留 2—6 个高复用 Concept。新的 `fine_grained` 档软目标为 4—10 个，并逐项检查：

- 每个明确参与者；
- 化名、代号、称号和多语言 alias；
- 命名组织、制度、协议、理论、计划和事件；
- 地点及承担独立剧情作用的子区域；
- 关键物品、技术和吉祥物；
- 持续动机、信念、创伤、偏执和关系主题；
- 可在其他 Episode 中再次被询问的抽象概念。

仍然禁止：

- 把一次性动词全部建成 Concept；
- 把完整事实句包装成复合节点；
- 同一对象的多语言名称重复建点；
- 用模型外部知识补充身份；
- 为达到数量目标制造泛词。

相同名称先复用 alias；相似但不确定的 Concept 仍按原流程建立 Association，等待后续证据或用户确认，不自动强制合并。

## 6. 四臂消融设计

每个实验臂都从冻结数据库的独立副本开始，禁止修改 `ba-stage3-deep-v37.db`。

| Arm | Paragraph | Concept profile | 目的 |
|---|---|---|---|
| A | off | conservative | 原基线 |
| B | on | conservative | Paragraph 单独贡献 |
| C | off | fine_grained augmentation | 细粒度 Concept 单独贡献 |
| D | on | fine_grained augmentation | 联合贡献与交互 |

Paragraph backfill 只补缺失 Source，重复运行不会重复插入。Concept augmentation 必须只在一次性数据库副本上运行；它会新增/复用 Concept 并建立 `Episode→Concept` Association，不在冻结库上重跑。

## 7. 固定输入和评测层

### 7.1 第一层：固定查询规划

为每道题只生成一次：

- QueryIntent；
- initial atomic queries；
- follow-up queries；
- 所有 query 的 float32 embedding。

四臂使用相同规划和向量，只在各自 Episode/Concept/Paragraph index 与 Association 图上重新排名。不能复用 Stage 5 已冻结的 final seed hits，因为那会绕过新 index。

### 7.2 第二层：完整端到端

对在固定层显示正向信号的配置，再运行独立 LLM 规划、回答和答案审计，确认实际回答没有因为更多 Paragraph/Concept 变得冗长、混乱或过度推论。

## 8. 测试问题

第一批继续使用 Stage 4 七题与 Stage 5 Q2，因为已有预注册证据组，可直接比较。

应额外加入更偏向原文措辞和 Concept 链的题：

1. 原文出现关键代号、Episode 摘要可能只写人物本名；
2. 地点子区域决定行动路径；
3. 物品或吉祥物连接人物身份与跨文件行动；
4. 情绪/创伤词触发远距离回忆；
5. 命名哲学概念连接多个政治行为，但必须保持推论边界。

所有答案仍只能来自当前 11 个允许文件或用户已提供的剧情文件，不能联网补充。

## 9. 指标

### 9.1 主要指标

- required evidence group `Recall@30`；
- A→B、A→C、A→D 的配对差值；
- 原本 candidate-only 的必需 Episode 是否进入 Top-30。

### 9.2 Paragraph 机制指标

- 关键 Paragraph 是否命中正确 `source_id`；
- Paragraph 扩展是否带来直接 Episode Top-K 未包含的 Episode；
- 这些 Episode 是否进入 600 候选和最终 30；
- 是否只增加同 Source 冗余 Episode；
- 关闭 Paragraph 后变化是否完全消失。

### 9.3 Concept 机制指标

- 新增 Concept 数量与类别分布只做诊断，不计分；
- 新 Concept 是否成为 query seed；
- 是否沿 `Concept→Episode` 路径带来新证据组；
- 是否产生多语言 alias 命中；
- 是否导致重复 Concept、泛节点或错误合并；
- 逐条遮罩新增 Concept/边后的边际贡献。

### 9.4 安全指标

- 白名单外 Source 必须为 0；
- Paragraph 原文不得被当作新的客观推论；
- Concept 描述不得含外部知识；
- 身份、因果和现场机制仍受双审计约束；
- 冻结数据库哈希保持不变。

## 10. 启用和回退门槛

### Paragraph 默认保留的条件

- 固定层平均 `Recall@30` 高于 A；
- 至少在多道不同类型问题上把缺失组推进 Top-30，而不只是增加候选数量；
- 没有稳定负向题；
- Source 冗余和延迟增量可接受；
- 机制日志能说明提升来自哪个 Paragraph 和 source_id。

如果没有收益或出现稳定负向，设置：

```text
MEMORY_PARAGRAPH_ENABLED=0
```

查询立即恢复旧路径，Paragraph 表可保留供以后实验，不必删除。

### fine_grained Concept 默认保留的条件

- 新 Concept 在不同题上产生非冗余路径或提高证据覆盖；
- 重复/泛化 Concept 比例可控；
- 没有使错误身份、强因果或无证据主题增加；
- 增长和回答 prompt 没有因 Concept 数暴涨而显著退化。

如果没有收益，设置：

```text
MEMORY_CONCEPT_PROFILE=conservative
```

并继续使用未做 augmentation 的基线数据库副本。

## 11. 执行顺序

1. 完成 schema v3、Paragraph segmenter/repository/index 和回退开关；
2. 完成 fine_grained/conservative 双档；
3. 运行全部本地单元测试；
4. 复制冻结库，建立 A/B/C/D 四个数据库；
5. B/D 回填 Paragraph；C/D 执行一次 Concept augmentation；
6. 生成固定查询规划与 embedding；
7. 运行固定层，先检查候选、Top-30 和逐通道机制；
8. 只有正向配置进入完整端到端评测；
9. 输出 paired scorecard、安全审计、错误和耗时报告；
10. 根据预先定义的门槛决定默认启用或回退。

## 12. 当前完成状态

已完成：

- schema v3 Paragraph 表和 v1/v2 迁移；
- record 对齐、重叠 Paragraph 分片；
- float32 Paragraph SQLite/RAM index；
- Source→Episode 局部精排融合；
- Paragraph 回填命令与可重试保护；
- fine_grained/conservative Concept prompt；
- 现有 Episode 的一次性 Concept augmentation 入口；
- 环境变量回退开关；
- Paragraph 召回、关闭回退、分片、回填和 Concept 档位测试；
- 全套 77 项单元测试通过。

尚未完成：

- A/B/C/D 的正式 API backfill/augmentation；
- 固定规划的 Stage 6 runner 和 scorecard；
- 正式剧情问题结果和启用/回退结论。

因此目前只能说“可逆实现已经就绪”，不能说 Paragraph 或更积极 Concept 已被实验证明有效。
