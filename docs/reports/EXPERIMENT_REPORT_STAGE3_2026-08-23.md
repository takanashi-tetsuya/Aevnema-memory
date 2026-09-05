# 第三阶段深度联想实验报告（2026-08-23）

## 1. 本阶段目标与状态

本阶段切换为三个已由本地原始文件确证的深度问题：

1. `main/31060.json → main/31090.json`：从补习部的成绩表象追踪到渚集中嫌疑人、寻找阻止伊甸条约的圣三一叛徒。
2. `main/33030.json → main/31090.json → main/32170.json`：把乐园悖论、渚的猜疑行动、未花承认的政变及阿里乌斯工具链串起来。
3. `main/33070.json → main/32170.json`：从古圣堂爆炸和阿里乌斯突袭回溯未花的秘密支援与政变前因。

截至本报告：四个缺失文件已全部自动导入，证据与持久 Association 已完成只读审计，代码修复通过 35 项测试。正式的 `vector_only / graph_static / graph_growing` 九次 API 问答尚未运行完成：沙箱网络策略拒绝 SiliconFlow 请求，而外部执行审批因 Codex 当前 usage limit 被拒绝。该阻塞与 SiliconFlow 余额无关，数据库与已导入资产不受影响。

## 2. v3.5 修复内容

当前提示词/行为版本为 `v3.5_reinforcement_and_path_rerank`。

### 2.1 强化关系成为一等结果

- `AssociationGrowthEngine.grow()` 不再只返回新建边 ID，而是返回 `GrowthOutcome(created_ids, reinforced_ids)`。
- 已有关系被再次确认时，强化 ID 会触发重新遍历，并与新建 ID 一同进入最终答案的优先路径集合。
- 查询结果新增 `reinforced_association_ids`，可以审计“复用并强化了旧联想”而不是只看新边。

### 2.2 避免反向重复 semantic 边

- 对相同端点、相同稳定 `relation_key`、相同 polarity 的 semantic 关系，端点反转也视为同一无向联想。
- 模型第二次把 `A→B` 写成 `B→A` 时强化原边，不再制造镜像副本。
- 不同 `relation_key` 仍可并存，避免把主题回应、证据桥和因果解释粗暴合并。

### 2.3 路径去重、相关性重排与多样性预算

- `candidate_limit` 从 120 提高到 320，减少有价值旧边在全局候选截断时被提前淘汰。
- 同一 Association 经不同路径被遍历多次时只保留最高 `path_score` 的一条。
- 对问题与 `relation_key + relation_text` 做轻量多语言词项/中文二三元组重合度计算，得到 `query_relevance`。
- 最终 24 条路径中，75% 预算优先给非 `involves` 的信息性关系，25% 留给 Episode→Concept 结构边。
- 被选路径的 Episode 端点仍强制进入 Episode 证据预算，保持可审计性。

### 2.4 评测失败状态持久化

- 每个问题先写入 `running`，成功后改为 `completed`。
- 任一问题异常会把该问题和总报告改为 `failed`，记录 `error_type/error/failed_at`，然后重新抛出异常。
- 不再出现进程失败但 `evaluation-report.json` 永久停留在 `running` 的假象。

## 3. 数据与分片资产

### 3.1 分片配置

| 参数 | 当前值 |
|---|---:|
| `target_chars` | 6000 |
| `max_chars` | 8000 |
| `overlap_chars` | 800 |
| `minimum_blocks` | 2 |
| 准备协程 | 8 |
| relation batch size | 24 |
| Episode 关系候选 K | 4 |

本轮预检：

| 文件 | 对话块 | Source 分片 | 分片长度 |
|---|---:|---:|---|
| `main/31060.json` | 62 | 4 | 7777, 3231, 6527, 5470 |
| `main/31090.json` | 74 | 5 | 8027, 4714, 7991, 6128, 1000 |
| `main/32170.json` | 89 | 5 | 6260, 5624, 8032, 7897, 5385 |
| `main/33070.json` | 112 | 7 | 4357, 6412, 3670, 5776, 6179, 5974, 4185 |

共 21 个 Source 分片。自然边界优先于严格字符数，所以有两个分片略高于 8000；当前适配器会尽量保留完整记录而不从一条记录中间硬切。

### 3.2 导入结果

| 文件 | Source | Episode | 二次修订 Episode | 失败任务 |
|---|---:|---:|---:|---:|
| `main/31060.json` | 4 | 19 | 1 | 0 |
| `main/31090.json` | 5 | 21 | 7 | 0 |
| `main/32170.json` | 5 | 27 | 6 | 0 |
| `main/33070.json` | 7 | 30 | 2 | 0 |
| 合计 | 21 | 97 | 16 | 0 |

扩展数据库从 `validation/ba-stage2-direct-k4.db` 的独立副本开始。导入后总量：

- Source：68
- Episode：343
- Concept：397
- Association：1846
- Association 类型：semantic 1410、identity 78、temporal 330、interpersonal 3、recall_trigger 9、causal 11、co_occurrence 5
- 负向 Association：4
- 所有 Episode/Concept embedding BLOB 均为 4096 字节，即 1024 维 little-endian float32
- pass1 完成 68 项；pass2 完成 72 项；主时间线与两个独立活动/好感 scope 均无时间环

## 4. 模型调用与性能信息

本轮导入耗时 2147.796 秒（约 35 分 48 秒），结构化日志 13,059,907 字节、1104 个事件、0 个 JSON 解析错误。

共 146 个模型请求，146 个都有响应或错误终态；实际全部成功，无 HTTP 错误、无 fallback、无验证失败：

- DeepSeek-V3.2：88 次
- BGE-M3 embedding：58 次
- Episode extraction：21 次，p50 28.222 秒，p95 47.853 秒，最大 83.619 秒
- Concept extraction：29 次，p50 75.582 秒，p95 115.234 秒，最大 292.758 秒
- Relationship judgment：20 次，p50 21.476 秒，p95 178.879 秒，最大 185.210 秒
- Embedding：58 次，p50 1.323 秒，p95 2.533 秒，最大 3.408 秒
- Second pass：8 次，p50 11.807 秒，p95 28.238 秒

最明确的性能结论是：当前小规模导入的墙钟瓶颈不是 float32 embedding 或 SQLite，而是 Concept 提取和关系判断的长尾 LLM 延迟。关系判断提示最大约 15,835 字符，二次理解提示最大约 18,012 字符；未来扩大全量语料时，应优先优化候选关系质量、批次大小与提示长度，而不是先换 ANN。

## 5. 固定证据审计结果

### 5.1 深度测试 1：补习部的真正目的

关键 Episode：

- #256 / `main/31060.json`：小春说明自己参加高年级考试才不及格，保留表面成绩设定。
- #269 / `main/31090.json`：渚说明不及格、退学规则和补习部的特殊权限。
- #271 / `main/31090.json`：直接写明补习部原本为让学生退学而建立；成员中混有圣三一叛徒，叛徒目标是阻止伊甸条约签订。
- #275 / `main/31090.json`：渚不知道破坏者身份，所以把所有嫌疑人集中在一处。
- #276 / `main/31090.json`：明确“箱子”就是补习部，必要时可将嫌疑人一同抛弃。
- #278 / `main/31090.json`：渚请求老师找出藏在补习部的叛徒。

持久图中 #271→#278 已有 `temporal/before`，#275→#276 已有 `temporal/before`，#280→#276 已有 `semantic/elaborates_on`。但是 `main/31060.json ↔ main/31090.json` 的 Episode 跨 Source Association 数为 0。

结论：事实提取已经足够直接回答问题，但静态图尚未显式保存“表面成绩原因 vs 真实政治目的”的跨文件解释边。增长模式应新增一条证据约束的 semantic `evidence_bridge` 或 `surface_vs_hidden_purpose`；若只靠 Episode #271 单点回答正确，只能算事实命中，不算三跳联想完全通过。

### 5.2 深度测试 2：乐园悖论的政治投影

关键 Episode：

- #62 / `main/33030.json`：把未花真实意图不可知引向“到达乐园者能否被证明存在”的公案。
- #63 / `main/33030.json`：抵达乐园者无法在乐园之外被观测；进一步对应“如何证明他人的真诚”。
- #271、#275、#276、#278 / `main/31090.json`：渚因叛徒猜疑建立补习部、集中嫌疑人并准备整体处理。
- #304 / `main/32170.json`：据对话回忆，未花向阿里乌斯提议联手，并一直暗中支援阿里乌斯。
- #305 / `main/32170.json`：未花承认行动可称为政变，目标是让渚下台并成为茶会主持人。

指定三文件之间已有 17 条 Episode 跨 Source 边，其中有 #305→#30 `semantic/confirms_analysis_in`，可连接未花的承认与前文分析。但没有边直接把 #62/#63 的“悖论/真诚不可证明”连接到渚的猜疑与未花的政变。

结论：事件层图已有不少跨文件连接，核心哲学投影仍需查询增长创建解释性 `thematic_projection`/`evidence_bridge`。这条边必须以“查询综合推论：”开头，不能把“悖论隐喻政治危机”冒充角色原话。

### 5.3 深度测试 3：古圣堂袭击因果链

关键 Episode：

- #304 / `main/32170.json`：未花与阿里乌斯交易并暗中支援。
- #305 / `main/32170.json`：未花承认政变目标。
- #319 / `main/33070.json`：单枚巡航导弹不足以造成爆炸规模，怀疑古圣堂预埋炸药；保持为未知人物的现场推测，没有升级成确定事实。
- #334 / `main/33070.json`：阿里乌斯各小队进入古圣堂并交战，纱织以阿里乌斯之名宣告审判。
- #341 / `main/33070.json`：阿里乌斯兵力从古圣堂地下墓穴出现，莲见质问当前状况是否为阿里乌斯一手造成。

两个指定文件之间已有 6 条 Episode 跨 Source 边，但没有正确的 #304→#334/#341 因果或使能边。已有 #341→#334 `causal/result_of` 只连接袭击内部阶段。

发现一条明确噪声边 #1792：它把 `main/33070.json` #326 的真琴与阿里乌斯串通，和 `main/32170.json` #304 的未花与阿里乌斯交易写成 `identity/same_as_candidate`，理由是“同一核心事实”。这是错误 identity：两者是不同人物、不同事件，只能至多形成 semantic 类比或共同借用阿里乌斯的模式，不能视为同一事件。

结论：增长模式应在 #304 与 #334/#341 之间建立证据约束的 `semantic/evidence_bridge` 或低置信度 `causal/enabled_by`，明确区分：阿里乌斯直接执行袭击是文本事实；未花早先支援为其进入圣三一危机提供前置条件是跨文本因果推断；预埋炸药的具体责任仍只是现场推测。#1792 不应进入最终路径，后续还应增加“不同具名行动者/不同事件禁止 Episode identity”的更强确定性校验或将其人工删除。

## 6. 当前提示词资产

所有正式提示词保存在 `src/memory_demo/llm/prompts.py`，关键设计如下：

### Episode

- “一个有意义的事件阶段”，不按逐句对白切分。
- 当前互动与回忆/倒叙必须分开，过去事件注明是谁的说法。
- 自包含、消除代词、保留原因与人物关系。
- 中文通常约 80—300 字；连续 15—30 条短对白通常约 3—6 个 Episode，均为软约束。
- 未标注说话者保持 `???/未标注发言者`，禁止按模型常识猜身份。
- 韩文原文与翻译冲突时保留差异并降低置信度。

### Concept

- 提取可跨记忆复用的稳定锚点，普通 Episode 通常 2—6 个，允许 0 个。
- 人物、组织、地点、命名物品、持续情绪/创伤/关系/主题可以进入；一次性动作和泛称通常不进入。
- 多语言别名合并到同一 Concept；临时人际事实优先用 Association 表达。

### 导入期 Relation

- 类型限制为 temporal、causal、identity、semantic、co_occurrence、recall_trigger、interpersonal。
- Episode identity 只允许同一事件/同一场景；共享人物、称号、面具不能作为 identity。
- temporal 必须按实际故事时间而非文件叙述顺序。
- 无有用关系返回空数组。

### 查询期 Growth

- 每条新关系至少连接一个 Episode，只能依据当前问题、候选节点和已有边。
- 允许证据支持的解释性 semantic 边，但关系文本必须以“查询综合推论：”开头。
- 每轮优先 1—5 条跨场景、可复用且当前图未完整表达的关系。
- 问题中的假设不能当证据；核心事实缺失时必须返回空边。
- `recall_trigger` 只允许原文明确描述某刺激/情绪使角色想起经历。
- 查询增长最多 2 轮。

### Answer

- 只能使用给定 Episode、Source 摘要和 Association 路径。
- 必须区分明确事实、角色说法、推测和未知。
- 输出直接结论、过程、证据、路径、其他解释、排除理由和不确定性。
- 不同 `timeline_scope` 不强行排序。

## 7. 正式三模式测试的通过口径

### vector_only

- 每题所需 `required_sources` 必须进入 `evidence_episodes`。
- 回答中的关键事实必须能回到正确 Episode；不得用模型自身剧情知识补洞。
- 该模式不读取 Association，主要验证分片、抽取和 embedding 召回。

### graph_static

- 相比向量模式，应至少使用一条与问题相关的非 `involves` 路径。
- `association_ids` 必须包含该路径，且路径 Episode 端点必须出现在最终证据。
- 需要检查现有跨文件边是否真正提高证据覆盖，而不是只被遍历后淘汰。

### graph_growing

- 测试 1 应补“成绩表象 ↔ 叛徒排查真实目的”。
- 测试 2 应补“乐园悖论/真诚不可证明 ↔ 渚猜疑与未花政变”的解释性桥。
- 测试 3 应补“未花暗中支援 ↔ 阿里乌斯直接袭击”的证据桥，同时保留预埋炸药的不确定性。
- 新边或强化边必须进入 `association_ids` 与 `association_paths`，并在答案中被实际消费。
- 重复运行同一题时，应减少新建边并出现 `reinforced_association_ids`；对应旧边 `use_count` 应增长。
- 错误 #1792 不得进入答案路径；若进入并影响回答，判定失败。

## 8. 当前资产索引

- 扩展数据库：`validation/ba-stage3-deep-v35.db`
- 固定问题：`validation/evaluation-questions-stage3-deep.json`
- 原文证据清单：`validation/stage3-deep-evidence-manifest.json`
- 导入账本：`validation/stage3-deep-import-progress.json`
- 导入完整日志：`validation/stage3-deep-import-logs/`
- 导入日志摘要：`validation/stage3-deep-import-log-summary.json`
- 数据库摘要审计：`validation/ba-stage3-deep-v35-audit-summary.json`
- 深度证据与跨 Source 审计：`validation/stage3-deep-evidence-audit.json`
- 可复用审计程序：`benchmarks/audit_stage3_deep.py`
- 网络受阻的未完成评测记录：`validation/evaluation-stage3-deep-v35-vector/evaluation-report.json`

## 9. 下一步恢复命令

网络外部执行恢复后，运行：

```powershell
$env:MEMORY_DB_PATH = (Resolve-Path 'validation/ba-stage3-deep-v35.db').Path
python -m memory_demo.cli evaluate `
  validation/evaluation-questions-stage3-deep.json `
  validation/evaluation-stage3-deep-v35 `
  --modes vector_only graph_static graph_growing
```

完成后应继续生成三模式数据库审计、日志摘要和逐题 scorecard，并把本报告中“尚未正式运行”的状态替换为最终结论。
