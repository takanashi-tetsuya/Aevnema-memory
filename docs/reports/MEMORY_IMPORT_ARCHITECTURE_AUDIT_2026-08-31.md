# 记忆系统与导入机制架构审计

日期：2026-08-31  
范围：Associative Memory 核心、批量文档导入、chatbot 前后台记忆接入

## 1. 总结

`Source / Episode / Concept / Association` 四层语义模型目前足够，不需要为了优化导入再增加新的语义节点。
当前复杂度主要不在记忆理论，而在运行编排：一次 Source 的内容写入、可重建索引、Concept 更新和推断
Association 被放在同一条流程中，却分别使用许多独立事务。一旦中途失败，数据库可能已经包含部分结果，
任务状态却只是 `failed/partial`，恢复程序无法知道应该从哪一阶段继续。

最值得实施的目标架构是：

```text
文件发现与解析（无模型、无数据库写入）
        ↓
Source 分片与结构化证据包
        ↓
Episode / Concept 生成与确定性校验（仍不写数据库）
        ↓
统一 float32 embedding 微批处理（固定 BGE-M3，仅同模型重试）
        ↓
一个 Source 的基础内容一次事务提交
        ↓
提交成功后更新 RAM index
        ↓
Paragraph、推断 Association、Concept 丰富化进入可恢复派生任务
```

这条路线不会把系统改成专家系统。新增内容应当是运行元数据和证据指针，而不是新的知识类型。

建议优先级：

1. 持久化 Episode 的精确证据范围；
2. 建立 Source 级原子提交和 run 内 exactly-once commit；
3. 将基础内容完成与派生关系完成拆成两个状态；
4. 建立全局有界准备队列、embedding 微批处理器和单一 SQLite writer；
5. 缩短 Concept 解析锁并消除逐 alias/逐 Association 的连接与事务风暴；
6. 在百万级之前实现 `search_many()`，再依据实测决定是否引入 ANN；
7. 将实验抽取档移出生产主路径，建立逐阶段 prompt/version lineage。

## 2. 当前架构中已经正确的部分

### 2.1 语义层保持简单

- Source 是可回读证据；
- Episode 是检索与联想的主要事件单位；
- Concept 保留多语言 alias，且不局限传统实体；
- Association 同时记录方向、关系文本、强度、可信度、极性和 generation；
- `confidence` 与 `generation` 已经正交，不再用一个分数同时表示可信程度和推断距离。

无需增加 Paragraph 以外的新知识节点，也不建议加入规则型“事件模板”“角色职责表”等专家系统结构。

### 2.2 embedding 边界正确

- SQLite 与 RAM 统一 float32；
- 写入前做 L2 normalization；
- embedding 模型固定为 BGE-M3；
- embedding 失败只允许同模型重试，不会 fallback 到其他模型；
- Episode、Concept、Paragraph 使用独立索引，ID 空间不混淆；
- 删除使用尾部槽位搬移，扩容采用倍增策略。

### 2.3 读取并发方向正确

`Database` 每次操作创建独立 sqlite3 connection，没有跨线程共享 cursor。chatbot 前台为每个请求复制配置并
创建 query-local `QueryEngine`，公共 RAM index 与 Repository 可以共享；会写 Association 的增长被移到后台
服务。这已经消除了早期“临时修改共享检索配置”的主要风险。

### 2.4 批量导入已有安全门

- 适配器严格处理编码、重复 JSON key、非有限数值和二进制控制字符；
- 单文件失败不会阻止后续文件；
- LLM 重阶段在 Source 持久化前完成，完全失败的初次提取不会留下空 Source；
- 推断关系写入保持单 writer 顺序；
- 中断会关闭 `extraction_run` 和仍处于 running 的 task；
- 候选数据库、严格审计、SQLite backup 后切换正式库的做法适合当前批量剧情导入。

## 3. 主要问题

## 3.1 Episode 的精确证据在落库时丢失

新版 `EpisodeDraft` 已携带 `evidence_quotes` 和 `evidence_spans`，人物 grounding、别名检查和蕴含审计都
依赖这些字段。但 `episode` 表及 `EpisodeRepository.insert()` 不保存它们；代码注释明确说明它们仅存在于
导入期和日志中。

后果：

- 数据库只能回溯到整个 Source，不能回到创建该 Episode 的具体行；
- 二次修订、人工修改、未来模型升级后无法执行同样的证据范围审计；
- 日志压缩、移动或缺失后，精确证据链永久丢失；
- 查询回答只能展示 Source 级摘录，无法稳定给出 Episode 级证据；
- 相邻 Source overlap 产生近重复 Episode 时，无法用证据 record/span 做确定性识别。

建议新增 Episode 字段：

```sql
evidence_spans_json TEXT NOT NULL DEFAULT '[]'
evidence_quotes_json TEXT NOT NULL DEFAULT '[]'
extraction_task_id INTEGER
```

为了保持 Demo 简单，可以先使用两个 JSON 字段，不必立即建立独立 `episode_evidence` 表。quote 应保存模型
当时看到的规范化证据文本；span 用于回读，quote/hash 用于检测 Source 或 parser 变化。

这是证据 provenance，不是新的记忆节点。

## 3.2 一个 Source 没有原子提交边界

当前成功顺序大致是：

```text
Source INSERT（事务 1）
Paragraph INSERT（事务 2，可失败）
每条 Episode INSERT（每条各一个事务）
RAM Episode upsert
每个 Concept 查找/INSERT/UPDATE（多次连接和事务）
每条 Episode→Concept Association upsert（每条各一个事务）
Episode/Concept 推断关系（另一个异步批次）
```

`_insert_episode_batch()` 名字是 batch，但 Episode 实际逐条调用 Repository，每条都会 `BEGIN IMMEDIATE`。
direct `involves` Association 还会先打开连接读取 Episode，再打开另一个事务 upsert。

如果 Source 已写入后发生以下任一问题：

- Paragraph embedding 失败；
- Episode embedding 成功一部分后 RAM 更新失败；
- Concept placeholder 更新中的 LLM/embedding 失败；
- Concept INSERT、alias INSERT 或 direct Association 写入失败；

任务会被标记失败，但此前事务不会回滚。现有外部 ledger 也明确要求重新尝试 interrupted 文件前人工清理
partial database run。这是当前恢复复杂度的根源。

建议新增 `ImportWriter.commit_prepared_segment()`，在同一 SQLite transaction 中完成：

- Source；
- Episode 及其 evidence；
- 本 Source 新建的 Concept/alias；
- Episode→Concept 的直接 `involves` 边；
- 可选 Paragraph 行（若 Paragraph 已完成 embedding）。

所有远程模型和 embedding 调用必须在取得 `BEGIN IMMEDIATE` 之前结束。事务只做短时间 SQL 写入。

事务提交后才更新 RAM index。若 RAM 更新失败，设置 `index_dirty` 并在下一次查询前从 SQLite rebuild；不要把
已经可靠落库的 Source 标记为内容提取失败。

## 3.3 “基础内容完成”和“派生关系完成”混在一个状态里

Paragraph、Episode→Episode 推断边、Concept→Concept 推断边都可以从 SQLite 基础内容重建，但当前任一
关系尾批失败都会把文件从 completed 降成 partial。这样会产生两个问题：

1. 操作者不知道 partial 表示 Episode 缺失，还是仅仅一次可重跑的关系判断失败；
2. 为了让文件最终 completed，主导入流程必须一直维护跨文件 relation queue、future tail 和 provisional status。

建议状态拆分：

```text
content_status:
    prepared / committed / failed

derived_status:
    pending / running / ready / partial / failed
```

基础查询可以在 `content_status=committed` 后使用 Episode、Concept 和 direct `involves`。推断 Association 进入
持久化 outbox/job，失败后只重跑对应 job。批量候选库是否切换为正式库，可由验收策略决定是否要求
`derived_status=ready`。

这个拆分还能删除 `_import_files()` 中大部分 `relations_pending / provisional_status / corpus_tail` 控制流。

## 3.4 run/task 记录不足以安全恢复

`extraction_task` 目前记录 source key、segment、stage、状态、配置模型和 retry_count，但缺少：

- 输入内容 hash；
- adapter/parser 版本；
- 实际 prompt ID 与 prompt hash；
- 实际使用的模型（包括 reasoning fallback）；
- 已通过校验的准备产物位置；
- durable commit token；
- 各阶段独立状态。

外部 `benchmarks/import_corpus.py` ledger 以文件为单位，`failed/partial` 默认都是终态；interrupted 只有在人工
清理数据库后才允许重试。这是实验工具能接受的策略，不应成为长期生产导入器的恢复模型。

需要的是 **run 内 exactly-once commit**，不是全局内容幂等：

```text
UNIQUE(run_id, source_key, segment_index, stage)
```

同一个文件以后仍可被用户明确再次导入；但同一 run 因超时或进程中断恢复时，已提交 segment 不得重复创建
Source/Episode。准备产物可按以下 fingerprint 缓存：

```text
source_text_hash
+ adapter_version
+ extraction_profile
+ prompt_hash
+ reasoning_model
```

这不会改变用户此前“不做全局重复导入检测”的决定，只解决一次执行自身的崩溃恢复。

## 3.5 Source 的运行 provenance 被重复存放在下游表

`source` 仍只有 `id/raw_text`，而 `source_key/segment_index` 重复出现在 Episode、Paragraph 和 task 中。
Paragraph backfill 甚至必须通过已有 Episode 反向恢复 Source 的 source key。这会造成：

- Source 没有 Episode 时无法知道它来自哪个文件和分片；
- Episode 与 Paragraph 的 source key 有发生不一致的可能；
- parser 或分片参数升级后无法区分旧 Source；
- 无法可靠判断同一逻辑文件的 segment 完整性。

为了保持 Source 概念类简单，建议增加一张一对一的运行元数据表，而不是扩张 Source 的语义：

```sql
source_provenance(
    source_id INTEGER PRIMARY KEY,
    source_key TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    first_record INTEGER,
    last_record INTEGER,
    content_sha256 TEXT NOT NULL,
    adapter_id TEXT NOT NULL,
    adapter_version TEXT NOT NULL,
    extraction_run_id INTEGER NOT NULL
)
```

这张表是导入 manifest，不参与知识推理。

## 3.6 Concept 解析锁覆盖远程调用和大量 N+1 查询

`resolve_many_deferred()` 在 `_resolve_lock` 内：

- 为每个 Concept 和每个 alias 分别打开 SQLite connection 查询；
- 对未解析项调用远程 embedding；
- placeholder description 更新时可能调用 reasoning LLM，再调用 embedding；
- 逐个 INSERT/UPDATE Concept。

锁最初用于防止并发文件创建重复 Concept，这个目标正确，但锁范围过大。当前主 writer 串行掩盖了影响；一旦
扩大文件级并发，它就会成为全局停顿点。

建议改成三阶段：

1. 一次批量 SQL 读取所有 normalized alias，得到候选快照；
2. 在锁外批量 embedding 未解析 Concept，并把 placeholder 丰富化放入独立维护任务；
3. 单 writer 重新检查 alias 后，在 Source 基础事务中 insert/reuse。

不要用“先查、解锁、直接插入”替代，因为仍会有 TOCTOU 重复。单 writer 或数据库级唯一 canonical fingerprint
才是最终仲裁者。

## 3.7 并发粒度仍以单文件为边界

当前每个文件都会创建和销毁一个 `source-prepare` ThreadPoolExecutor。文件只有一个 Source 时，无法利用
Source 并发；外部工具再增加 `file_workers` 后，又形成文件池与 Source 池两层并发。关系推断未关闭时还会
再创建 relation pool。

建议改成进程级有界流水线：

```text
parser queue       多文件并行、无模型
prepare queue      固定 N 个 reasoning worker
embedding queue    一个微批处理器，按 items/token/deadline 合批
commit queue       一个 SQLite writer
derived queue      独立关系/Paragraph/丰富化 worker
```

队列本身提供 backpressure，避免一次把整个语料的 Source 和 prompt 放进 RAM。无需为了“异步”立刻重写所有
HTTP 客户端；现阶段保留阻塞 ModelClient 加固定线程池即可，先消除嵌套 executor。

## 3.8 embedding 调用已经批量，但批次只局限在一个 Source

历史完整导入中，735 个 Source 产生 2,034 次 embedding 请求；最近 production_kb 的 63 个 Source 仍有
132 次 embedding 请求。请求内容通常分别是 Paragraph、Episode、未解析 Concept，单批较小。

由于三类都使用同一个 BGE-M3 和 1024 维 float32，可以由一个微批处理器合并请求并保留 item type/owner 映射：

```text
EmbeddingItem(kind, owner_key, text)
        ↓
最多 64/128 item 或等待 20–50 ms
        ↓
一次 BGE-M3 请求
        ↓
按 owner_key 分发 float32 向量
```

embedding 模型固定与不 fallback 的规则完全不变。失败时整批使用相同模型重试；若仍失败，可二分批次定位单条
超限输入，但不能换 embedding 模型。

## 3.9 重试存在多层预算叠加

一次提取可能同时经历：

- ModelClient HTTP retry；
- JSON repair 调用；
- 同模型 validation correction；
- reasoning fallback；
- `task_max_attempts` 重跑整个 Episode + Concept 准备阶段。

外层 task retry 不区分 transport、JSON schema、证据拒绝和永久配置错误，因此可能把已经成功的 Episode 提取、
审计和部分 Concept 生成全部重做。

建议统一失败分类：

```text
TransportRetryable       同模型、同请求重试
ProviderPermanent        立即失败，不重试
OutputSyntaxInvalid      只做一次 JSON repair
OutputSchemaInvalid      只重做当前 stage/当前 batch
EvidenceRejected         使用校验错误做一次受控 correction
EmbeddingFailed          固定模型重试；失败则 Source 不提交
CommitConflict           writer 重新读取 alias/commit token 后决定
DerivedJobFailed         只重跑派生任务
```

每个 Source 应有总请求预算，避免多层重试相乘。

## 3.10 prompt 与实际模型 lineage 仍主要依赖 JSONL 猜测

`extraction_run.prompt_versions` 当前只保存 `{"all": config.prompt_version}`。不同的 Episode、Concept、审计、
关系和修复提示没有独立 ID/version。`extraction_task.model` 保存配置的主模型，而不是某次 fallback 后实际接受的
模型。ModelClient 日志有 request_id，但没有自动绑定 run_id、task_id、source_key 和 stage；并发日志分析只能
根据 system prompt 内容推测请求属于哪个任务。

建议：

- 使用 `PromptSpec(prompt_id, version, schema_version, content_hash)` registry；
- logger 支持 `bind(run_id, task_id, source_key, segment_index, stage)`；
- ModelClient 每条 request/response 自动带绑定上下文；
- task 保存最终 accepted request_id、actual_model 和 output artifact hash；
- 重复 Source/prompt 正文在日志中使用 content-addressed artifact，只保存一次全文，其余事件记录 hash。

这样既满足“所有输出保留”，又能显著减少 JSONL 体积和分析 token。

## 3.11 实验 profile 已侵入生产主控制流

Episode 提取目前包含 `legacy`、两个 single-pass、两个 document-map 和 adaptive anchor 六个分支。实验已经证明
全文地图不是默认路径，但相关 map 构建、校验、prompt 和 `_import_files()` 分支仍位于生产核心文件。

建议：

```text
ProductionEpisodeExtractor
    当前被正式选定并通过冻结回归的一个实现

ExperimentalExtractorRegistry
    single-pass 对照、document map、anchor map
```

不要立即删除实验代码；把它移到 `ingestion/experiments/` 并通过显式研究 CLI 选择。生产 README 只展示一个
稳定 profile。当前报告推荐 `single_pass_audited`，而 `IngestionConfig` 默认值仍是 `legacy`，说明“研究推荐”
与“生产默认”尚未经过正式晋升流程。应先完成更大样本问答回归，再明确切换或保留 legacy，避免文档和运行值
长期分叉。

## 3.12 核心类过大，阶段边界难以测试

当前文件规模：

- `QueryEngine`：约 5,681 行、80 个函数；
- `MemoryExtractor`：约 2,608 行、51 个函数；
- `ImportPipeline`：约 2,186 行；
- prompt 模块：约 1,423 行。

`_import_files()` 约 500 行，`_query_impl()` 约 700 行。问题不是单纯代码行数，而是一次函数同时负责状态机、
线程池、模型调用、数据库写入、索引更新、统计和日志。

建议按已经存在的真实阶段拆分，而不是按抽象设计模式拆分：

```text
ingestion/
    planner.py       文件发现、adapter、segment manifest
    preparer.py      Episode/Concept 纯生成与校验
    embed_batcher.py 固定模型向量微批
    writer.py        一个 Source 的事务提交
    derived.py       Paragraph、关系、Concept 丰富化任务
    recovery.py      run/task 恢复与 commit token

retrieval/
    planning.py
    recall.py
    coverage.py
    growth_runtime.py
    answer.py
```

每个阶段输入输出使用 slots/frozen dataclass，避免继续传递无结构 dict。

## 4. 检索与 RAM 索引的进一步空间

## 4.1 先实现批量 brute-force，再决定 ANN

当前一次问题会生成多个 search query。embedding 已一次批量生成，但随后每个 query 分别调用
`EmbeddingIndex.search()`，因此每个 query 都会完整扫描 Episode、Concept，启用 Paragraph 时再扫描
Paragraph。

历史基准在 100 万 × 1024 float32 上单次搜索约 86 ms；如果一次 deep 查询生成 40 个向量，仅 Episode 理论上
就会读取同一约 3.8 GiB 矩阵 40 次。

建议增加：

```python
search_many(query_matrix, top_k, block_rows=32768)
```

按 embedding block 做 `block @ query_matrix.T`，为每个 query 维护 Top-K。这样只顺序读取一次矩阵，并更充分
利用 BLAS。该改动保持精确 brute-force，不引入 recall 损失，应当先于 ANN。

ANN 的触发条件不应只看 Episode 数，而应同时满足以下任一实测门槛：

- vector retrieval p95 超过会话预算；
- `N × active_queries` 的扫描耗时已成为主要阶段；
- float32 主索引及扩容峰值接近可用 RAM 的安全上限；
- 并发查询使内存带宽饱和。

即使以后使用 ANN，SQLite float32 embedding 仍应保留为 source of truth 和精排依据。

## 4.2 搜索锁目前覆盖整个矩阵乘法

`EmbeddingIndex.search()` 在 read lock 内完成矩阵乘法和 Top-K；writer preference 使一个等待中的 writer 阻止新
reader。百万级矩阵扫描时，这会放大前台查询与后台追加的互相阻塞。

短期先用 `search_many()` 降低持锁次数。中期可引入 index snapshot/version：

- append 写到当前 count 之后，最后发布新 count；
- 扩容构造新数组后原子发布新 snapshot；
- existing-row update/delete 使用 copy-on-write snapshot 或 tombstone；
- reader 只在短锁内取得 snapshot 引用和 count，随后无锁计算。

不要在当前有 update/remove 的实现上直接释放锁读取可变数组，否则会引入数据竞争。

## 4.3 RAM index 需要显式健康状态

SQLite commit 后、RAM upsert 前崩溃本来可以靠 rebuild 恢复，但当前进程继续运行时没有统一的 `dirty/version`
状态。建议每个索引记录：

```text
db_generation
index_generation
dirty
last_rebuild_at
```

写入后 upsert 失败就标记 dirty；查询入口遇到 dirty 时 rebuild 或拒绝使用不完整 index。这样“SQLite 是真实
数据、RAM 可重建”的原则不仅是约定，也成为可检查的不变量。

## 5. 推荐目标数据流

### 5.1 准备产物

```python
PreparedSegment(
    run_id,
    task_id,
    source_key,
    segment_index,
    source_text,
    source_sha256,
    adapter_id,
    adapter_version,
    episodes,
    episode_evidence,
    concepts_by_episode,
    paragraphs,
    embedding_items,
)
```

这个对象只包含已通过确定性校验的候选，不拥有数据库 ID，不修改 RAM。

### 5.2 基础提交

单 writer 为 `PreparedSegment` 完成 alias 二次检查，分配 Source/Episode/Concept ID，写 direct `involves`，更新
task commit token。整个操作只有一个事务。

返回：

```python
CommittedSegment(
    source_id,
    episode_ids,
    concept_ids,
    paragraph_ids,
    derived_job_ids,
)
```

### 5.3 派生任务

- Episode→Episode LLM 关系判断；
- Concept→Concept LLM 关系判断；
- placeholder Concept 描述丰富；
- 可选 Paragraph backfill；
- chronology/retrieval 辅助索引。

派生任务必须保存输入端点和 generation/evidence 快照。失败不回滚基础内容，只改变 derived status。

## 6. Prompt 与抽取流程的简化方向

### 6.1 不再把全文派生结论喂给局部提取

最新 A/B 已证明自由摘要、结构地图和逐字 Anchor Map 都没有默认净收益。全文层最多用于查询阶段 Source 路由，
或只向确实证据不足的片段提供位置/时间模式；不得提供全局人物名单、主题或因果结论。

### 6.2 保持 Episode 与抽象 Concept 两次调用，不盲目合并

把 Episode、Concept、审计、关系一次性塞入同一 prompt 会增加输出 schema 和注意力竞争，目前没有证据说明
它能降低总成本而不损害质量。更安全的缩短方式是减少 Concept 模型必须处理的内容：

- participant 及 adapter 已确认的多语言 speaker alias 直接确定性创建/reuse Concept；
- Episode→participant Concept 的 `involves` 直接建立；
- LLM Concept 提取只负责组织、地点、物品、情绪、主题和抽象状态；
- 只对未被确定性 lane 覆盖的 Episode 调用抽象 Concept 提取。

这既符合“更积极创建 Concept”，又减少模型重复发现已经在 participants 中明确给出的名字。

### 6.3 结构化元数据不要伪装成 Source 正文

当前规范化 Source 同时包含 `[source_key]`、record、speaker、alias legend、script 和多语言文本。它有利于保留
信息，但迫使 LLM 从一大段脚手架中再次解析结构。

建议 `SegmentEnvelope` 同时保留：

- 用于持久化和人工回读的完整 normalized source；
- 用于模型的 line-numbered evidence view；
- 结构化 speaker alias/evidence annotation；
- 原始 record ID 到 evidence line 的映射。

模型只看到任务所需的紧凑证据 view；确定性代码直接读取结构化 alias 和来源标注。不能为了省 token 删除原文，
而是避免同一元数据以文本形式在多个 prompt 中反复出现。

## 7. 分阶段实施计划

## Phase A：证据与可恢复性，不改变抽取结果

1. schema v9 增加 Episode evidence JSON、source_provenance 和 task commit token；
2. 新数据写入 evidence，旧数据默认空 evidence 并继续兼容；
3. logger.bind 自动关联 run/task/request；
4. 增加 Source 事务故障注入测试：在 Paragraph、Episode、Concept、direct edge、RAM upsert 各阶段抛错；
5. 验证失败前后数据库只允许“零提交”或“完整基础提交”，不得出现中间态。

验收：不调用额外模型；相同冻结输入产生相同 Episode/Concept/Association；277 项引擎测试和 chatbot 测试通过。

## Phase B：基础提交与派生任务解耦

1. 引入 `PreparedSegment / ImportWriter`；
2. direct `involves` 放入基础事务；
3. inference relation 改为持久化派生 job；
4. 文件报告分别输出 content/derived 状态；
5. interrupted run 自动跳过已提交 commit token，不再要求人工清理已完成 segment。

验收：随机终止进程并恢复 20 次，最终节点/边集合与不中断运行一致；同一 run 不重复提交。

## Phase C：并发与调用成本

1. 全局 prepare queue 替换每文件 executor；
2. embedding 微批处理；
3. Concept alias 批量查询和短锁二次检查；
4. direct Association 批量写入；
5. WAL 只在数据库初始化时设置，不在每次 connect 重复执行。

验收：用冻结 63 Source production_kb 和更大的 735 Source 样本对照：

- Episode/Concept 语义覆盖不下降；
- embedding 模型始终只有 BGE-M3；
- embedding 请求数显著下降；
- SQLite busy/locked 为零；
- 总耗时下降且峰值 RAM 有界。

## Phase D：检索批量扫描

1. 为 Episode/Concept/Paragraph index 实现 blockwise `search_many()`；
2. 对 10 万、100 万向量和 1/8/20/40 query 做基准；
3. 在现有 Recall@20 冻结题集验证结果与逐 query search 完全一致；
4. 根据向量阶段 p95 决定是否继续 ANN 实验。

## Phase E：生产/实验代码分层

1. 将 document-map/anchor-map 移入 experiments；
2. 建立 PromptSpec registry；
3. 拆分 ImportPipeline 与 QueryEngine 大函数；
4. 完成 single_pass_audited 与 legacy 的扩大问答 A/B，再明确生产默认。

## 8. 暂时不建议做的事情

- 不增加新的知识图谱实体类型来表示 import task、文件阶段或证据槽；它们属于运行元数据。
- 不恢复 float8/float16 两阶段方案；当前统一 float32 更简单，且已有真实 CPU 基准。
- 不因为代码很长就全面重写；先抽出事务 writer、派生 job 和 embedding batcher 三个真实边界。
- 不把全文总结作为默认导入步骤。
- 不把所有抽取/审计合成一个超大 prompt。
- 不立即引入 ANN；先消除一次问题对同一矩阵的重复扫描。
- 不做全局“相同内容不允许再次导入”；只保证同一 run 崩溃恢复不会重复提交。

## 9. 最值得先验证的三个假设

1. **一个 Source 一次事务能否显著降低恢复复杂度，同时不让事务持有期间发生任何远程调用。**
2. **participant Concept 确定性 lane + 抽象 Concept LLM lane，能否降低 Concept token，同时提高人物 alias 覆盖。**
3. **blockwise `search_many()` 能否在 20–40 个 query 时明显降低百万向量检索延迟，并保持精确 Top-K 完全一致。**

这三个实验都不需要改变记忆理论，也不依赖针对蔚蓝档案写死的规则，适合成为下一阶段架构优化的主线。
