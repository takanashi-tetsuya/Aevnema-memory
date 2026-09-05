# Associative Memory Demo 阶段实验与资产报告

日期：2026-08-22  
实验语料：Blue Archive `favor / main / event`  
当前代码提示词版本：`v2.9_growth_evidence_guard`  
主要实验数据库：`validation/ba_full_v7.db`

## 1. 阶段结论

当前 Demo 已经跑通以下闭环：

```text
Blue Archive JSON / TXT
→ 多语言清洗与自然边界分片
→ Episode 自动提取、时间审计、粒度审计
→ Concept 自动提取与别名解析
→ float32 embedding 持久化与 RAM 索引
→ Association 抽取、规范化与强化
→ Episode/Concept 联合检索
→ 图遍历、查询期增长、证据约束回答
→ A/B/C 对照评测与 JSONL 全量日志
```

已经验证成立的工程结论：

1. 在本机 32 GB RAM、8 CPU 核心、无 GPU 的条件下，1024 维 float32 NumPy brute-force 足以继续 Demo；现在不需要 ANN。
2. `6000 / 8000 / 800` 的 Source 分片参数能够覆盖完整 564 文件语料，解析失败为 0；但 `max_chars` 目前是软上限，26 个分片因完整记录、header 或别名表略超 8000 字符。
3. Episode 粒度审计有效：`33190.json` 从旧版 90 条、41 条短于 40 字，收敛到 58 条、6 条短于 40 字；相似重复为 0。
4. “保留未知说话者”和“人物不等于面具/服装/称号”不能只靠提示词，必须有确定性写入守卫。
5. temporal 关系必须统一成 `earlier --before--> later`，否则 `A before B` 与 `B after A` 会成为两条边，并放大冲突和路径数量。
6. 查询期自主增长确实能够马上建立并使用新 Association，但 v2.8 过度积极：即使核心证据缺失也会制造大量 recall_trigger，并能重新制造已被导入审计消除的时间环。
7. 图检索当前没有稳定胜过纯向量检索。路径上限 100 会使每个问题都打满 100 条路径、携带 267—370 个 Association，增加噪声、延迟和错误推断。
8. 多语言原文优先仅写在大提示词里仍不可靠。`33200.json` 的韩文原文是“上次差点交手”，中/日译文是“之前见过面”；Episode 62 最终错误采用译文并给出 1.0 confidence。需要独立的语言冲突审计阶段。

因此当前系统已证明“可增长联想网络可以运转”，但尚未证明“增长一定改善回答”。下一阶段重点应从增加边，转向证据门控、关系守卫、路径压缩和多语言冲突审计。

## 2. 当前资产

### 2.1 核心代码

| 资产 | 位置 | 作用 |
|---|---|---|
| 应用装配 | `src/memory_demo/app.py` | 初始化 SQLite、Repository、两个 EmbeddingIndex、查询引擎 |
| 数据库工厂 | `src/memory_demo/database.py` | 每次操作独立 connection；WAL、foreign key、busy timeout、事务 |
| Schema | `src/memory_demo/schema.sql` | Source、Episode、Concept、alias、Association、提取任务与日志元数据 |
| BA 适配器 | `src/memory_demo/adapters/blue_archive.py` | JSON、多语言、ScriptKr、speaker marker、人物别名 |
| TXT 适配器 | `src/memory_demo/adapters/text.py` | 通用文本兼容入口 |
| Source 分片 | `src/memory_demo/ingestion/segmenter.py` | 自然边界、完整 record、重叠上下文、alias legend |
| 提取器 | `src/memory_demo/ingestion/extractor.py` | Episode、Concept、时间审计、粒度审计、未知身份清理 |
| 导入管线 | `src/memory_demo/ingestion/pipeline.py` | 两遍提取、并发准备、串行落库、批量关系判断 |
| Concept 解析 | `src/memory_demo/concepts/resolver.py` | alias fast path、向量候选、LLM 判断、延迟批关系 |
| Association 构建 | `src/memory_demo/associations/builder.py` | 权重、候选关系、identity/temporal 写入守卫与 canonical 化 |
| 查询增长 | `src/memory_demo/associations/growth.py` | 查询期新边；v2.9 加入证据/类型/时间守卫与同查询去重 |
| 图遍历 | `src/memory_demo/associations/traversal.py` | 多跳 Association 扩展 |
| 查询引擎 | `src/memory_demo/retrieval/engine.py` | intent、双索引召回、图扩展、增长、时间排序、回答 |
| 时间服务 | `src/memory_demo/chronology/service.py` | before/after 拓扑排序、时间环检测、人工校正 |
| 提示词 | `src/memory_demo/llm/prompts.py` | 所有提取、审计、关系、增长和回答 system/user prompt |
| 模型客户端 | `src/memory_demo/llm/client.py` | OpenAI-compatible HTTP、300 秒 timeout、重试、fallback、请求关联日志 |
| Telegram | `src/memory_demo/interfaces/telegram_bot.py` | Demo 查询入口，不写入用户对话记忆 |
| A/B/C 评测 | `src/memory_demo/evaluation.py` | SQLite backup 后运行 vector/static/growing 三模式 |

### 2.2 工具与验证资产

| 资产 | 位置 | 说明 |
|---|---|---|
| 完整语料预检 | `benchmarks/preflight_corpus.py` | 解析文件、语言、分片长度、oversize 统计 |
| 可恢复导入器 | `benchmarks/import_corpus.py` | 按文件 ledger 恢复、优先文件、workers、关系 batch |
| 数据库审计 | `benchmarks/audit_database.py` | BLOB、任务、重复 Episode、关系类型、时间环 |
| 数据修复器 | `benchmarks/repair_database.py` | 未知身份文本、错误 identity、temporal canonical repair；默认 dry-run |
| float32 基准 | `benchmarks/benchmark_float32_search.py` | RAM 与 Top-K CPU 延迟 |
| 六问题集 | `evaluation_questions.json` | 第一次见面、人物经历、死因不确定性、创伤联想、多跳因果、跨剧情返回 |
| 测试 | `tests/` | 当前 25 项，包括 float32、并发、删除、时间、增长守卫、端到端查询 |

### 2.3 关键持久化产物

| 产物 | 位置 | 用途 |
|---|---|---|
| 当前主实验库 | `validation/ba_full_v7.db` | 4 个优先剧情文件，已做 v2.7/v2.8 repair |
| v2.7 修复前备份 | `validation/ba_full_v7-before-v2.7-repair.db` | 复现未知身份与“浮士德”身份污染 |
| v2.8 修复前备份 | `validation/ba_full_v7-before-v2.8-temporal-repair.db` | 复现时间环与 before/after 重复 |
| 当前全库审计 | `validation/ba_full_v7-audit-priority-v2.8.json` | 83 Episode / 121 Concept / 463 Association 的行级审计 |
| 完整语料预检 | `validation/corpus-preflight-20260822.json` | 564 文件、1593 分片统计 |
| float32 benchmark | `validation/float32-benchmark-20260822.json` | 10 万与 100 万向量 CPU 实测 |
| 文件恢复 ledger | `validation/full-import-progress-v7.json` | 已完成文件、prompt 版本、Source/Episode/失败数 |
| 导入日志 | `validation/full-logs-v7/` | 全部请求、响应、修订、关系与 repair 记录 |
| A/B/C 结果目录 | `validation/evaluation-v8-priority/` | 三份数据库快照、每模式 JSONL、最终 report |
| v2.9 缺证据烟雾库 | `validation/ba-v29-yume-smoke.db` | 在主库副本上复测“梦前辈死因”，未产生新关系 |
| v2.9 烟雾审计 | `validation/ba-v29-yume-smoke-audit.json` | 83 / 121 / 463，主线与好感时间图均无环 |
| v2.9 烟雾日志 | `validation/ba-v29-yume-smoke-logs/` | 4 次真实 LLM 请求/响应、0 错误、最终证据与回答 |

旧的 v3—v6 数据库和日志仍保留在 `validation/`，用于比较串行、超时、并发、粒度变化，不应视作当前生产数据。

## 3. 固定数据与模型配置

### 3.1 模型

```text
Embedding provider protocol: OpenAI-compatible
Embedding model: Pro/BAAI/bge-m3
Embedding dimension: 1024
Reasoning model: deepseek-ai/DeepSeek-V3.2
Fallback model: zai-org/GLM-4.5V
Timeout: 300 seconds
Retries: 2
```

### 3.2 Embedding 格式

硬盘与 RAM 统一 float32，不再包含 float8/float16 路径：

```text
API vector
→ np.float32
→ 维度/NaN/Inf/零向量检查
→ L2 normalize
→ little-endian <f4 BLOB
→ SQLite
→ rebuild 时再次检查和 normalize
→ RAM float32 matrix
```

1024 维的硬盘 BLOB 必须严格等于 `1024 × 4 = 4096 bytes`。当前审计中 Episode 和 Concept 的 BLOB 长度集合都只有 `[4096]`。

### 3.3 内存索引

Episode 与 Concept 使用两个独立 `EmbeddingIndex`：

```text
ids: int64[capacity]
embeddings: float32[capacity, 1024]
positions: dict[database_id, array_position]
count / capacity
```

关键实现：

- 2 倍扩容；
- 搜索只读取 `[:count]`；
- 删除用最后一行移动到空槽，复杂度 O(dimension)；
- rebuild 按 SQLite `fetchmany(2000)` 流式读取，预留约 10% capacity；
- Condition 实现 writer preference；多个 search 可并行，但写操作等待所有 reader；
- 当前 reader lock 覆盖完整矩阵乘法，不是“只取数组引用后立即释放”的 snapshot 方案。Demo 可接受，未来高并发要重新 benchmark。

### 3.4 SQLite 生命周期

当前没有共享全局 connection/cursor：

- `Database.connect()` 每次创建 connection；
- `PRAGMA foreign_keys=ON`；
- `PRAGMA journal_mode=WAL`；
- `PRAGMA busy_timeout=5000`；
- 写操作使用 `BEGIN IMMEDIATE`；
- Repository 每个方法自行进入 connection/transaction context；
- 因此 Source 提取可并发，SQLite 写入仍由 pipeline 串行安排。

## 4. Source 分片参数与全语料统计

当前参数：

```python
SegmentConfig(
    target_chars=6000,
    max_chars=8000,
    overlap_chars=800,
    minimum_blocks=2,
)
```

分片优先级：

1. 永远保持完整 dialogue record；
2. 优先寻找 `boundary_score >= 2` 的自然边界；
3. 达到约 target_chars 后在最近自然边界切分；
4. 重叠至少保留一个完整 record；
5. speaker alias 在每个 Source 顶部压缩成一次 legend，不在每句重复；
6. Source 永久保存完整清洗文本，LLM reasoning view 再选择 zh-CN / ja / en / ko 与 fallback。

完整语料预检：

| 指标 | 结果 |
|---|---:|
| 文件 | 564 |
| 解析成功 | 564 |
| 空文件 | 0 |
| 解析失败 | 0 |
| 原始 dialogue blocks | 38,514 |
| Source segments | 1,593 |
| 总字节 | 36,613,023 |
| segment 最小 | 226 chars |
| segment 平均 | 4,946 chars |
| segment P50 | 5,211 chars |
| segment P95 | 7,816 chars |
| segment P99 | 8,026 chars |
| segment 最大 | 8,071 chars |
| 超 8,000 的 segment | 26 |

分类：

| 分类 | 文件 | blocks | segments |
|---|---:|---:|---:|
| favor | 480 | 31,408 | 1,172 |
| main | 77 | 6,197 | 384 |
| event | 7 | 909 | 37 |

判断：当前分片适合继续实验，不需要为了 26 个轻微 oversize 立即缩短内容。若模型输入成本成为瓶颈，可把 alias legend/header 的长度纳入硬限制，而不是截断完整 record。

## 5. Episode 粒度和二次理解

### 5.1 当前 Episode 原则

- 一个有意义的事件阶段，而不是逐句对白；
- 同一时间、地点、目标与连续冲突合并；
- 实际故事时间改变必须拆分；
- 角色在当前说起过去，当前言语事件与过去事件分开；
- 过去事实注明“据某角色说法/回忆”；
- 自包含、消除代词、保留必要原因和人物关系；
- 未标注 speaker 保持 `??? / 未标注发言者 / [USERNAME]`；
- Source 未明示时不写“某人不在场”；
- 允许同一 Source 产生语义重叠 Episode；
- 摘要与推论分离。

### 5.2 软长度经验

提取 prompt 当前建议中文约 80—300 字、1—3 句；连续 15—30 条短对白通常约 3—6 个 Episode，但不设硬数量。

四个优先文件合并库：

```text
Source: 16
Episode: 83
每 Source Episode: min 2 / mean 5.19 / max 9
Episode length: min 17 / median 68 / mean 73.10 / max 244
短于 40 字: 15
短于 20 字: 3
cosine >= 0.92 的重复 Episode: 0
```

仍有改进空间：`33210.json` 的“准备出发/宣布出发”等 4 条短 Episode 没触发粒度审计，因为旧规则要求候选数至少 10。下一轮可改为：4—9 条候选时，若短/微事件占比超过 50%，也运行粒度审计。

### 5.3 两类审计

Temporal audit：

- 检测“上次、过去、曾经、回忆、previously、かつて、과거”等；
- 强制拆开当前谈话与被回忆事件；
- `story_time_text` 描述 Episode 主事件时间；
- 不把角色说法升级成无条件事实。

Granularity audit：

- 合并同场景沉默、点头、呼唤、回答等微片段；
- 删除纯章节标题；
- 恢复未经 Source 证实的身份为未知标记；
- 不重新合并已经分开的倒叙和当前事件。

### 5.4 第二遍理解

第一遍完成后，仅对未知标记或低 confidence Episode 做第二遍：

```text
目标 Episode
+ 同文件前后约 8 条 Episode 纲要
+ 对应短 Source
→ 修订 Episode
→ 重新 embedding
→ 重新提 Concept
→ 重新建立 Association
```

v2.7 后触发标记包含 `???`、`[USERNAME]`、`未标注发言者`、`未知发言者`。即使 LLM 再猜身份，确定性 regex 也会移除“根据上下文应为某人/可能是某人”。

## 6. Concept 与 Association

### 6.1 Concept

Concept 允许人物、实体、地点、命名物品、持续状态、情绪、创伤、人物关系和抽象主题；不为每个普通名词和一次性动作建节点。

去重流程：

```text
exact alias / canonical fast path
→ Concept embedding Top-K
→ similarity candidate
→ LLM 判断 identity / semantic / not_same_as
→ 暂不自动物理 merge
→ 后续用户确认可 merge/unmerge
```

当前阈值 `concept_relation_min_similarity=0.72`，LLM 权重远高于 embedding。

### 6.2 Association 类型

```text
temporal
causal
identity
semantic
co_occurrence
recall_trigger
interpersonal
```

当前主库在 v2.8 repair 后：

| 类型 | 数量 |
|---|---:|
| semantic | 332 |
| temporal | 82 |
| recall_trigger | 12 |
| identity | 15 |
| causal | 8 |
| interpersonal | 6 |
| co_occurrence | 8 |
| 合计 | 463 |

权重公式：

```text
LLM 0.80
embedding 0.10
structural 0.05
evidence base 0.05
reinforcement learning_rate 0.20
```

这符合“主要由 LLM 判断，其余参数低权重加入”的目标。

### 6.3 写入守卫

已从真实失败中加入：

1. Episode 正向 identity 只表示同一事件/场景，不能表示两个 Episode 中的人物是同一人；
2. 人物与面具、服装、状态、职位、组织、物品不是 identity；
3. 未知 speaker 不能通过 Episode identity 被猜成具名人物；
4. current-event temporal 与同一 Source 的 segment/Episode 顺序明显矛盾时拒绝；
5. `main/数字.json` 在双方无显式 story_time 时可作为低权重顺序约束；
6. positive temporal 统一写为 `earlier --before--> later`；
7. v2.9 把同样规则接入 Query Growth，并拒绝 temporal Episode→Concept、identity 跨节点类型；
8. 同一次查询的等价 growth fingerprint 只写一次，不在三轮中重复强化同一证据。

## 7. 关键 System Prompt 设计

完整、可执行版本以 `src/memory_demo/llm/prompts.py` 为准。以下是对后续实验最有参考价值的约束。

### 7.1 Episode 提取

核心角色：长期记忆事实提取器。关键指令：

```text
只能依据 Source，不能使用外部知识。
Episode 是一个有意义的事件阶段，不是逐句对白摘要。
同一连续互动合并；实际故事时间变化必须拆分。
当前说起过去时，当前言语与过去事件分开。
未知人物标记原样保留，不能猜测。
韩文原文与译文实质冲突时，保留原文细节与差异并降低 confidence。
不要把推论或角色假说写成事实。
```

有效的 prompt 经验：给出“当前增援 + 口述上次夺回大楼”必须拆成两个 Episode 的正反例，比单纯描述“注意倒叙”更可靠。

已知不足：原文优先规则在 `붙을 뻔했잖아?` 案例中仍失败，因此不能只保留在 Episode 大 prompt 中。

### 7.2 Temporal audit

```text
唯一重点是区分当前谈话/反应和谈话中提到的过去事件。
任何“X 说上次 Y”都必须产生独立的过去 Episode。
过去条目写“据 X 的说法，Y 曾经发生”。
不能只修改 story_time_text 而仍把两个实际时间混在同一 text。
```

### 7.3 Granularity audit

```text
把逐句对白碎片恢复为可独立回想的事件阶段。
合并同场景微动作，删除纯标题。
移除“可能是老师/推测为白子”等身份猜测。
不得为了减少数量而丢失关键行动、因果、情绪、关系或译文冲突。
```

### 7.4 Concept

```text
提取能跨 Episode 复用的检索锚点。
优先人物、组织、地点、命名物、持续状态、情绪、创伤和主题。
不要为大家/西边/出发/闲聊等临时词制造节点。
临时人际事实由 Association 表达，不制造复合 Concept。
```

### 7.5 Relation

```text
只能依据候选节点。
current 在 candidate 之后用 after，之前用 before。
Episode identity 只表示同一事件。
共享人物、称号、面具或角色身份不能使不同 Episode identity。
```

实践结论：关系 prompt 后仍必须做 canonical normalization 与 provenance guard；LLM 经常让 relation_key 与 relation_text 相反。

### 7.6 Growth（v2.9）

```text
问题中的假设和措辞不是事实证据。
不能为了填补答案缺口制造关系。
核心事实、实体或 requested relation 缺失时返回空 relationships。
recall_trigger 只在 Episode 明确描述刺激/情绪触发回忆时建立；
不能因为同一人物、情绪或主题就建立。
temporal 只能 Episode→Episode；identity 只能相同节点类型。
```

### 7.7 Answer

```text
只能使用 Episode、Source 摘要和 Association 路径。
区分明确事实、角色说法、推测和未知。
给出直接结论、实际时间、关键证据、联想路径、替代解释、排除理由和不确定性。
资料不足时直接说明。
```

## 8. 导入实验结果

| 文件 | Source | Episode | pass2 修订 | 失败 | 用时 |
|---|---:|---:|---:|---:|---:|
| `main/33190.json` | 12 | 58 | 13 | 0 | 22m49s |
| `main/33200.json` | 1 | 5 | 3 | 0 | 8m03s |
| `main/33210.json` | 1 | 4 | 2 | 0 | 4m20s |
| `favor/10005/1000515.json` | 2 | 16 | 2 | 0 | 7m24s |
| 合计 | 16 | 83 | 20 | 0 | 42m36s |

导入执行参数：

```text
prepare_workers = 3
relation_batch_size = 24
SQLite writes = serialized
API timeout = 300 s
per-file resumable ledger = enabled
```

### 8.1 v2.6 → v2.7 修复

真实错误：

- Episode 文本写入“未标注发言者（根据上下文，应为日富美）”；
- 白子和日富美因共享“浮士德”称号被判为同一人物；
- 未标注发言者通过 Episode identity 被判为日富美。

修复后：推测身份文本 0；Episode 正向 identity 断言人物相同 0。

### 8.2 v2.7 → v2.8 修复

修复前：

```text
main temporal edges = 81（参与拓扑）
main has_cycle = true
```

原因：关系模型同时产生相反方向，且 after/before 等价边未 canonical。

修复操作：

- 拒绝 16 条与 current-event Source 顺序明确相反的边；
- 77 组 temporal 规范成 before；
- 等价边合并 evidence_count；
- temporal 总数从 101 收敛到 82。

修复后：

```text
main: 67 Episodes / 63 temporal edges / no cycle
favor:10005: 16 Episodes / 19 temporal edges / no cycle
```

## 9. float32 与 ANN 决策

本机实测：

| 向量数 | 维度 | Matrix RAM | Top-100 平均延迟 |
|---:|---:|---:|---:|
| 100,000 | 1024 | 0.381 GiB | 8.44 ms |
| 1,000,000 | 1024 | 3.815 GiB | 86.31 ms |

Episode 100 万 + Concept 100 万的纯矩阵约 7.63 GiB，加上 capacity headroom、IDs、Python/SQLite 和查询内存后仍适合 32 GB 机器做单用户实验。

当前结论：

- 不引入 float8；CPU/NumPy 的实际支持与转换成本不值得；
- 不引入二阶段 float8→float16 rerank；统一 float32 更简单且本机足够快；
- 暂不引入 ANN；先解决 Episode/Association 语义质量；
- 只有在实际库达到数十万—百万且多查询延迟成为瓶颈时，再比较 HNSW/FAISS/SQLite vector extension。

## 10. A/B/C 评测

评测起点：同一个 v2.8 无时间环数据库备份。v2.8 参数是：

```text
episode_top_k = 40
concept_top_k = 20
graph_beam_width = 30
graph_max_hops = 4
growth_max_rounds = 3
candidate_limit = 120
answer_episode_limit = 30
answer_concept_limit = 30
answer_path_limit = 100
```

三模式：

```text
vector_only   = embedding only
graph_static  = embedding + persisted Association
graph_growing = static graph + up to 3 query-time growth rounds
```

### 10.1 定量结果

六问共完成 18 次真实回答，三种模式均为 6/6 成功，API 错误为 0。

| 模式 | 总用时 | LLM 请求/响应 | 平均回答字符 | 平均送答 Association | 平均路径 | 查询结果中新边合计 |
|---|---:|---:|---:|---:|---:|---:|
| vector_only | 6m22s | 18 / 18 | 1557.00 | 0 | 0 | 0 |
| graph_static | 8m46s | 18 / 18 | 1772.00 | 328.50 | 100 | 0 |
| graph_growing | 31m51s | 36 / 36 | 1871.67 | 388.33 | 100 | 127 |

`graph_growing` 的 127 是六次查询各自返回的 `new_association_ids` 数量之和；日志中全局去重后的新 Association ID 是 109。差异来自后续查询再次命中或 upsert 已有 ID，不能把 127 解释为数据库净增长 127 行。

| 问题 | static Association | growing Association / 新边 | 人工证据判定 |
|---|---:|---:|---|
| 日奈与星野第一次见面 | 295 | 312 / 17 | 三模式均受错误 Episode 62 污染；growing 进一步把无日奈出场的蒙面场景说成首次身体相遇，失败 |
| 星野个人经历时间线 | 325 | 363 / 27 | 只能部分回答；vector 最谨慎，图模式擅自排列“很久以前买枕头”和发现遗体的先后，未获证据支持 |
| 梦前辈死因 | 366 | 423 / 21 | vector/static 正确拒答；growing 虽最终称未知，却增加“可能死于冲突、地点可能在阿拜多斯”等无证据猜测 |
| 白子为何回忆被前辈救助 | 348 | 423 / 30 | 三模式最终都正确指出核心场景缺失；growing 仍写入大量 recall_trigger，持久化精度失败 |
| 日奈为何阻止星野去沙漠 | 267 | 352 / 18 | 三模式均正确指出当前库没有“阻止/沙漠”；现有内容只支持日奈敬佩星野、双方曾协作 |
| 莉音被驱逐后是否回来 | 370 | 457 / 14 | 三模式均正确指出莉音、千禧年和驱逐事件完全缺失；增长出的边对答案没有帮助 |

### 10.2 语义结论

1. **纯向量是当前最稳健基线。** 它会召回无关片段，但回答模型多数能明确拒绝缺证据问题，且不会污染持久化图。
2. **静态图没有在这份 83 Episode 样本上提高正确率。** 六问全部打满 100 路径，回答更长、用时更高；能提供较丰富的邻接背景，但也诱发无依据时间排序。
3. **v2.8 增长图证明了“可以自主增长”，没有证明“增长有益”。** 六问新增结果 ID 合计 127，耗时约为纯向量的 5 倍；日志记录 161 个 growth event、109 个全局唯一新 ID 和 4 次 timeline conflict。
4. **Association 可达性不等于证据。** “同人物、同地点、同情绪”适合做召回线索，却不能直接升级成时间、因果、身份或回忆触发事实。
5. **答案正确与写库正确必须分开评分。** 白子、日奈和莉音问题的最终回答可以正确拒答，但查询过程中新增的持久化边仍可能是错误资产。

### 10.3 v2.9 真实烟雾复测

在 `ba_full_v7.db` 的独立副本上，以当前 `v2.9_growth_evidence_guard` 再次查询“梦前辈最终怎么死”，结果为：

```text
Episode sent to answer = 20
Concept sent to answer = 20
Association paths = 23（上限 24）
new_association_ids = 0
database associations = 463 → 463
LLM requests/responses = 4 / 4
API errors = 0
wall time ≈ 63 s
main temporal cycle = false
favor temporal cycle = false
```

回答明确说明：只知道会长死亡、遗体由星野发现；没有证据支持事故、疾病、冲突或其他死因猜测。相比 v2.8 growing，这一条真实样本同时改善了答案和写库行为。

但这只是单问题烟雾测试。v2.9 目前主要依赖 prompt 做 evidence sufficiency，类型、时间方向和同查询重复才是确定性守卫；仍需六问全量复测和可执行的证据充分性规则。

## 11. 已证实的失败模式

### 11.1 多语言事实冲突被翻译多数覆盖

原始记录：

```text
ScriptKr: 저번에 한 번 붙을 뻔했잖아?
含义: 上次差点交手/差点打起来
TextCn/TextJp: 不久前见过面
```

Episode 62 最终写成“不久前刚见过面”，confidence=1.0。第一次见面问题的三种模式都会在错误事实起点上继续推断。

建议 v2.10：在 Episode 提取前增加独立 conflict manifest：逐 record 比较 `ko` 与翻译，只输出事实差异清单；Episode/Temporal/Granularity audit 都必须携带该清单，最终 validator 检查冲突是否被保留。

### 11.2 图路径爆炸

v2.8 static-graph 六问全部达到 100 path 上限；每问涉及 267—370 个 Association。后果：

- 回答输入巨大，单次调用曾接近 5 分钟；
- 与问题弱相关的节点进入答案；
- 模型把“图中可达”误读为“因果或实际先后”；
- 输出 association_ids 与日志变得难审阅。

v2.9 已把默认值改为：beam 20、hops 3、growth rounds 2、answer Episode 20、Concept 20、path 24；只有进入回答的 path 才记为 used。

### 11.3 无证据仍增长（v2.8；v2.9 单例已改善）

“梦前辈死因”和“白子为何回忆被救助”在当前四文件库缺少核心事件，但 v2.8 GrowthEngine 仍建立大量 recall_trigger。这些边不是新证据，只是围绕问题词汇的语义联想。

建议：

1. intent 后增加 evidence sufficiency 判定；
2. requested relation 不存在时允许 answer，但禁止 growth；
3. recall_trigger 必须有显式“想起/回忆/触发”文本；
4. 将“基于问题临时推理”与“可持久化事实关系”分开；前者不写库。

v2.9 的“梦前辈死因”真实烟雾复测已做到 0 新边，并正确拒绝无依据猜测；这一结果支持上述方向，但不能据单例判定问题已经完全解决。

### 11.4 Query Growth 绕过导入守卫

v2.8 导入库审计无环，但 growing 模式第 2 个问题后出现 timeline_conflict。日志中存在：

- `67 before 64`，与同一主线文件当前事件顺序相反；
- `79 before 81` 与 `81 before 79` 同时存在；
- relation_text 说 79 在前，但 endpoints/key 写成 81 before 79。

v2.9 已把 temporal canonical/provenance guard 和 identity 类型守卫接入 GrowthEngine，并增加同一查询 fingerprint 去重。

### 11.5 小文件仍可能过度切碎

粒度审计旧触发器偏向 `>=10` 条候选；只有 4—8 条但大部分很短的 Source 可能跳过审计。需要加入小样本比例触发，而不是单纯降低全局 Episode 长度。

## 12. 推荐的下一轮顺序

### P0：在继续 564 文件全量导入前

1. 实现独立多语言 conflict manifest，并只重跑 `33200.json` 验证“差点交手”能保留；
2. 用 v2.9 对同一份 83 Episode 库重新跑六问；当前“梦前辈死因”单例已通过，但其余五问和连续增长路径尚未复测；
3. 为 Growth 增加确定性的 evidence sufficiency gate，不能只依赖 v2.9 prompt；
4. 把 evaluation report 改为每问增量落盘，避免长评测仅在最后写总报告；
5. 增加自动质量指标：unsupported claim、source conflict coverage、new-edge precision、cycle delta、path utilization。

### P1：扩大到更多相关文件

优先寻找能覆盖六个问题的证据，而不是立即导入全部 564 文件：

- 星野/白子过去与阿比多斯主线；
- 日奈阻止星野去沙漠的直接事件及老师中间关系；
- 莉音驱逐与活动回归；
- 梦前辈相关文件。当前本地 favor/main/event 扫描未发现“梦/ユメ/유메/Yume”名字，因此该问题应继续输出证据不足。

### P2：全量导入

质量门通过后，继续使用 v7 ledger 的剩余 560 文件。按当前优先样本速度，完整导入会持续多小时甚至更久，应：

- 保持 per-file resume；
- 每 20—50 文件做 summary audit；
- 定期备份 SQLite；
- 统计 API p50/p95、失败类型、Episode/Source 密度和 Relation/ Episode 比例；
- 不在导入过程中改变 prompt 版本；新版本使用新数据库或明确 migration/repair。

### P3：规模与性能

当 Episode 达到 10 万以上再重新 benchmark：

- RAM headroom 与 rebuild 峰值；
- reader lock 覆盖矩阵乘法时的写入等待；
- 分块 dot product 是否更稳定；
- ANN 与 brute-force 的 recall@K、延迟和维护复杂度。

## 13. 可复现实验命令

完整语料预检：

```powershell
python benchmarks/preflight_corpus.py ./story `
  --output validation/corpus-preflight.json
```

可恢复导入：

```powershell
$env:MEMORY_DB_PATH = "validation/ba_full_v7.db"
$env:MEMORY_LOG_DIR = "validation/full-logs-v7"
python benchmarks/import_corpus.py ./story `
  --ledger validation/full-import-progress-v7.json `
  --workers 3 `
  --relation-batch-size 24
```

只读审计：

```powershell
python benchmarks/audit_database.py validation/ba_full_v7.db `
  --duplicate-threshold 0.92 `
  --output validation/audit.json
```

修复预览与应用：

```powershell
python benchmarks/repair_database.py `
  --database validation/ba_full_v7.db `
  --log-dir validation/full-logs-v7

# 先备份，再显式应用
python benchmarks/repair_database.py `
  --database validation/ba_full_v7.db `
  --log-dir validation/full-logs-v7 `
  --apply
```

A/B/C：

```powershell
$env:MEMORY_DB_PATH = "validation/ba_full_v7.db"
python src/main.py evaluate evaluation_questions.json validation/evaluation-output
```

测试：

```powershell
python -m unittest discover -s tests -v
```

## 14. 决策记录

当前保留：

- SQLite + Python sqlite3；
- Repository 风格类名暂不重构；
- Source 只负责原始证据，不增加专家系统式结构；
- Episode/Concept 分离索引；
- Association 保存自然语言 relation_text 与少量机器字段；
- 允许模糊、弱关系和否定关系；
- 所有硬盘/RAM embedding 为 float32；
- 查询期可以增长，但必须通过证据与完整性守卫。

当前明确不做：

- float8/float16；
- ANN/vector database；
- 联网补充作品知识；
- Telegram 对话自动写入记忆；
- 自动物理合并相似 Concept；
- 为每个推理问题增加专家系统专用字段。

本报告的核心用途不是宣称 Demo 已成功，而是保留哪些设计已被实测支持、哪些仅是目标、哪些已经被真实语料证伪。后续每轮应继续复用同一六问题集、数据库审计和失败样本，避免只看更流畅的答案而误判联想能力提高。
