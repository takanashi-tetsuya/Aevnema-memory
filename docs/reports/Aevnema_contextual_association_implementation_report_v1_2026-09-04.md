# Aevnema「双键条件联想」实现报告 v1

日期：2026-09-04  
对应计划：`Aevnema_contextual_association_development_experiment_plan_v1.md`

## 1. 范围与结论

本次工作把附件计划书中的技术方案落到了 Aevnema 核心记忆库和 chatbot 接入层。附件是实现依据，不是对话指令；本次用户指令是“按计划实现，并记录改动”。实现保持现有普通检索、证据审计和知识增长路径可回退，新的条件联想默认关闭、默认 shadow（只观测不改变答案），因此升级后不会自动改变已有回答。

当前结论：双键联想的存储、索引、检索、效用更新和 chatbot 侧接线已经具备可运行的第一版；真正的线上收益需要在启用开关后用真实多轮对话测量。它还不是完整的因果推理器：条件边只改变 Episode 的候选召回，不会冒充剧情事实、摘要或答案证据。

## 2. 设计落地

### 2.1 双键模型

一条 contextual association 由两条独立线索组成：

* `context_cue_id`：当前已知的上下文线索；
* `need_cue_id`：用户当前问题、目标或需求的线索。

只有两个线索都命中时才召回目标 Episode。分数默认采用两个相似度的乘积，可通过配置改成 `min` 或 `geomean`。这避免了“只像问题、但没有当前上下文”的单键误召回。

条件边使用 `association_mode=contextual_recall`、`claim_level=retrieval_only`。锚点与目标均必须是同一知识域中的 generation-0 Episode；跨域 cue、推论 Episode、Fact/Capsule 均被拒绝。

### 2.2 数据库和迁移

数据库 schema 已升至 v13。

`association` 新增：

* `association_mode`、`context_cue_id`、`need_cue_id`；
* `utility_weight`；
* `success_count`、`noop_count`、`harm_count`、`distinct_query_count`；
* `lifecycle`、`expires_at`、`last_evaluated_at`；
* `source_request_hash`、`utility_query_hashes`。

新增 `association_cue_prototype` 表，保存域、cue kind（`context`/`need`）、模型标识、维度、`float32` 向量、文本/查询 hash、可读文本和源请求 hash。向量持久化和内存索引统一使用 float32，SQLite 仍是 source of truth。

v10→v11、v11→v12、v12→v13 迁移均已实现。v13 会安全重建 association 的约束，以纳入 `retrieval_only`，并保留旧行和既有字段。新增索引在迁移完成后创建，避免旧数据库在新列创建前执行索引而启动失败。

### 2.3 内存索引和请求级 embedding

`EmbeddingIndex` 增加了 `search_many`，按块进行精确扫描，只读取有效的 `[:count]` 区域；并保留预分配数组、扩容、快照和 id→槽位映射。暂不引入 ANN，因为 demo 的目标是先验证正确性。

`EmbeddingCoordinator` 对一轮请求的 whole、atomic、follow-up 查询做去重后一次 batch embedding，并把 query id、角色、文本和向量封装成 `QueryVectorBundle`。context/need 两套索引只读取已保存的 cue prototype，启动 rebuild 不调用网络。

### 2.4 检索流程

1. chatbot 先按原有 light/standard/deep 路由决定基础检索计划。
2. contextual 开启时，知识域一次生成 whole/private/public/knowledge 查询 bundle；各域复用同一 bundle，不为每条边单独请求模型。
3. 每个域的 `ContextualAssociationMatcher` 在本地分别搜索 context 和 need prototype，执行双门槛、分数组合、目标端点上限和生命周期过滤。
4. 非 shadow 模式下，命中的目标 Episode 作为零分数的候选种子加入原有图检索；随后仍由原有 graph traversal、coverage selector、reranker 和证据审计处理。
5. contextual 边不能成为回答事实来源：答案路径排序、LoreFactContract 和事实证据构造都会过滤 `contextual_recall`、`retrieval_only` 及 `relation_key=contextual_recall`。
6. 可见回答通过 guard 后，程序从本轮检索轨迹派生候选边和效用观察；不读取回答 prose 来“猜”边，也不向模型追加每条边的判断请求。
7. 后台增长队列把同一请求的 bundle 作为进程内临时句柄传给知识域。队列重启或向量不可用时安全跳过 contextual 写入，不影响普通记忆增长。
8. cue prototype 与边先写 SQLite，再刷新内存索引；崩溃后可由 SQLite rebuild。

## 3. 效用学习和生命周期

新边初始为 `probation`。本地 `record_utility` 支持 `sufficient`、`redundant/no_op`、`harm` 三类结果：

* 同一 query hash 重复上报只计一次；
* 达到配置的不同 query 成功数、且没有 harm，才晋升 `active`；
* no-op 按衰减因子降低 utility，harm 按更强倍率惩罚；
* 过期 probation/active 边可本地 prune，保留 retired 记录用于审计。

当前 chatbot 的 Treatment/Masked receipt 是保守版本：它使用已完成的 evidence-slot trace 判断“目标是否贡献了新槽位”，尚未实现完整的逐边 LOO 重放。因此它适合第一轮塑性实验，不应被解释为严格的因果证明。

## 4. 配置和运维入口

所有新开关均默认关闭，可在 `.env` 或等价配置中设置：

* `MEMORY_CONTEXTUAL_ASSOCIATION_ENABLED`；
* `MEMORY_CONTEXTUAL_ASSOCIATION_SHADOW`；
* context/need/edge top-k、相似度阈值、组合方式；
* light/standard/deep 的端点上限；
* probation TTL、晋升所需成功数、no-op/harm 衰减；
* `MEMORY_CONTEXTUAL_ASSOCIATION_ALLOW_NETWORK`（实现强制拒绝 `true`，维护命令保持离线）。

核心 CLI 增加：

```text
memory contextual-cues stats
memory contextual-cues audit
memory contextual-cues prune --dry-run
memory contextual-cues rebuild-local-index --offline
memory contextual-cues migrate-legacy --offline
```

这些命令不调用模型，`prune` 默认要求显式 `--dry-run` 才只预览，实际删除需明确执行。

## 5. 代码变更清单

核心记忆库（`src/memory_demo`）：

* `types.py`：查询 bundle、cue prototype、contextual hit、utility observation、plasticity event 和 `retrieval_only` 类型；
* `embeddings/index.py`、`embeddings/coordinator.py`：批量查询、float32 索引和请求级 embedding；
* `retrieval/contextual_association.py`：纯本地双键匹配器；
* `retrieval/coverage.py`：证据槽 greedy set-cover 与 Treatment/Masked 辅助；
* `associations/plasticity.py`：候选派生、事件分类和本地效用应用；
* `repositories/association.py`：cue、条件边、生命周期、效用、迁移兼容；
* `association_overlay.py`：事务内隐藏 cue/edge、回滚和测试隔离；
* `database.py`、`schema.sql`：v13 schema、迁移、存储审计；
* `app.py`、`retrieval/engine.py`：索引重建、请求级 contextual recall、证据隔离和运行时 receipt；
* `config.py`、`.env.example`、`cli.py`：配置校验和维护命令。

chatbot（`src/memory`、`src/bot`）：

* `contracts.py`：检索质量 receipt、contextual trace、事实合同过滤；
* `config.py`：域级配置和 light/standard/deep 端点预算；
* `domain.py`、`system.py`：bundle 复用、知识域写入、塑性和效用更新；
* `answer_consolidator.py`：从检索轨迹派生候选/观察，加入异常输入容错；
* `growth.py`：进程内 bundle 句柄和后台任务安全交接；
* `chat_service.py`、`__init__.py`：守卫后接入增长及公开接口。
* `src/cli/import_data.py`：回滚数据库使用可关闭的事务上下文，修复 Windows 临时 SQLite 文件句柄泄漏。

## 6. 验证结果

已完成的验证：

* 核心 unittest：`383 tests, OK`；
* 核心 pytest 回归：`390 passed`；
* contextual smoke：覆盖 `search_many`、请求 bundle 去重、双键 AND 命中、未来成功晋升、overlay masking，全部通过；
* 核心与 chatbot 共 111 个 Python 文件完成 AST 解析；
* 现有数据库可从旧 schema 启动并迁移到 v13；当前数据库 SHA-256：`9b05ea3d15e3b7efe250e5dbfb38fc311c5423865e5ec8ed8ce7b8481cea4cea`；
* 当前数据库 contextual storage audit：`ok=true`，孤立 cue、非法向量、跨域边、非法目标和非法 claim level 均为 0；
* chatbot 既有回归记录：`173 passed, 13 skipped, 1 warning`；本轮用项目自带虚拟环境实际运行 `186 tests`，全部通过。此前 Windows 临时目录清理阶段的 3 个 `WinError 32` 已通过显式关闭 SQLite 连接修复；仍有日志文件权限警告，但不影响测试结果。

## 7. 已知限制和回退方式

1. 默认 `enabled=false`、`shadow=true`；关闭开关即可完全回到旧路径。
2. shadow 模式只写 trace，不把 contextual 目标加入答案候选，也不会产生新边。
3. contextual 边不能直接支持事实主张；必须由普通 Episode 证据支持答案。
4. 自动学习依赖 guard 通过、知识域写入策略允许、bundle 尚在当前进程；服务重启时未持久化 bundle 的任务安全跳过。
5. 当前是 float32 brute-force，未验证 ANN；大规模数据需另做分块/ANN 基准。
6. 当前 utility receipt 不是严格 LOO 因果实验，不能单凭一次成功就提升权重。
7. `migrate-legacy --offline` 只迁移可识别的旧结构，不会凭空生成缺失向量；无法识别的行会记录审计结果。
8. 计划中的 E0–E8 真实语料实验尚未宣称通过；本报告只确认代码和最小回归闭环已经完成。

## 8. 后续建议

先保持 shadow，采集一批真实问答的 contextual trace，比较“普通检索候选”和“双键候选”的 slot coverage、答案修改率、重复问题延迟与误召回率；确认收益后再关闭 shadow。之后再实现严格的 Treatment/Masked LOO 重放和可取消的并行批处理，而不是继续向 prompt 中堆叠规则。
