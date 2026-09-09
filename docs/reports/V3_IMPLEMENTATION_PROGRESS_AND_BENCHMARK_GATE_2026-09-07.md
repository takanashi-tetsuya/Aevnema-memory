# Aevnema-memory v3 执行进展与 Benchmark 门禁报告

**记录时间：** 2026-09-07（Asia/Tokyo）  
**对应计划：** `Aevnema_memory_v3_plan_bundle_2026-09-05.zip`  
**性质：** 当前双仓库脏工作区的实现/离线验证记录；不是正式 benchmark、语料导入或线上模型实验报告。

## 结论

v3 的核心检索、一次学习、前置再回想、trace/replay、请求截止时间和聊天层交付收尾已形成可本地回归的实现链路。最近完成的重点是四类此前仍缺少硬门禁或存在崩溃窗口的边界：

1. 核心查询在模型返回之后、SQLite 提交之前再次核对请求 deadline，迟到结果不能再写入 Association、使用计数或学习收尾。
2. 稳定平台消息在发送前可加密暂存一份与回执绑定的对话计划；计划替换必须同时持有文件级跨进程排他锁、仅在持锁期有效且绑定锁域摘要的进程内 permit，以及每次获取的新 SQLite fencing token。发送权与 HMAC、锁域摘要在同一事务边界内确认，避免旧计划已取得发送权却被另一进程或另一 journal 根目录改写。
3. 只有 adapter 已持久化确认发送成功、且四个摘要绑定完全匹配时，才允许计划成为可导入的对话 Source；长时间文件导入会续租，失去租约的 worker 不得写 receipt、移动 Source 或完成 claim。进程中断后只恢复确定性的 Source，不重发消息，也不重放会话、授权、增长或命令副作用。
4. 正式 recall preflight 现在仅接受原始 bytes 绑定的 spec、gold、split 与冻结 Source manifest；它逐一重算所有冻结来源文件及 evidence span 的哈希，并要求 approved gold 在顶层、family、claim group、atom 都有明确的审核/可评分标记和 source-disjoint holdout。草稿或内存对象入口只能诊断，永远不能产出可执行的正式评分预检。

正式 benchmark 仍未执行。这不是性能或质量失败：当前 Source gold 是明确标注为待人工 source-span 审核的草稿，不能作为正式分数分母；本轮通过的是“防止错误启动 benchmark”的离线门禁，而不是实际质量分数。

## 本轮实现范围

### 核心记忆引擎

当前工作区已沿 v3 路线加入或扩展以下能力：

- 不可变需求/向量 bundle、slot 级贡献、可审计 retrieval trace 与只读 replay/preflight；
- anchor-first contextual matcher、贡献级遮罩、联合/替代覆盖及 harm 优先的效用判断；
- recovery/reuse 两类 contextual Association 创建、原子 receipt/索引更新，以及精确和受限改写的前置 revisit；
- Source 版本、来源闭包、受限 rewrite 和 residual-gap repair 的回退边界；
- provider 的共享并发门、绝对 deadline、排队/重试/解析阶段的剩余预算与迟到响应隔离；
- SQLite 事务级 liveness hook：请求被取消或 deadline 到期后，提交前会回滚，不把迟到查询的 `mark_used`、staged Association 或 contextual 学习写入数据库。

这些实现位于 `src/memory_demo/`、`benchmarks/`、`tests/` 的当前未提交工作区中；没有把草稿 gold 注入生产检索路径。

### Source-gold 与正式 benchmark 门禁（T04）

本轮新增的 `benchmarks/validate_source_gold.py` 是 evaluator-only 的本地校验器。它不导入应用、不打开知识库、不调用模型，也不会写语料；它的正式入口仅接受 gold、split、冻结 Source manifest 的原始 bytes，并从同一份 bytes 计算绑定哈希。

- approved gold 必须使用正式 schema，且顶层、family、claim group、atom 都显式标记为已审核且 `usable_for_scoring=true`；仅修改草稿的 `status` 或 flag 不会晋级。
- 每个 atom 必须重算并匹配 `source_key`、来源文件哈希、JSON Pointer/Blue Archive record locator、Unicode code-point span 和 raw-span 哈希；所有 manifest 来源（包括未被 atom 引用者）都会被读取、严格解析、哈希核对并纳入最终快照稳定性检查。
- split 必须与 gold 的 family/source closure 精确一致，且 approved 集必须至少拥有一个与其余 family 来源物理隔离的 holdout。硬链接、Windows 路径别名、大小写/保留名/遍历拼写和无法获得稳定文件 identity 的场景均 fail-closed。
- 正式 `run_recall_v3.py` CLI 只经 `preflight_formal_run_bytes()` 进入；spec、gold、split 和冻结清单都从原始 bytes 解析。保留的对象形式只用于诊断，必定返回不可执行，避免把构造的内存对象伪装成已冻结工件。
- 两个 CLI 都会先将 artifact、`source_root` 和文件输出固定为直接、本地、绝对路径；UNC、远程映射盘、无法确认的 Windows 盘类型、symlink/junction 及其父路径都会拒绝。报告输出不得覆盖输入、输入硬链接别名或来源树，也不得创建缺失的父目录；相对路径在预检后不会因 CWD 变化重新绑定。
- 正式报告只保留固定枚举、计数、哈希、索引和受限错误代码；不回显 family ID、arm ID、状态原文、Source 正文或 evidence span。

这不是对敌对并发文件系统替换的绝对证明。路径式 Python 校验能检测常规前后变化，但正式运行仍要求 `source_root` 是受控、静止、不可变的本地快照；该运维前提已在校验器说明中明确。

### 聊天层交付与学习收尾（T19 边界）

聊天仓库新增的是一个窄且可恢复的 journal 边界，而不是“重放整个 finalizer”：

```text
生成确定性回复
  → AES-GCM 加密暂存 user/reply + 4 个摘要绑定
  → SQLite 仅保存 delivery key + plan HMAC + 状态
  → （如需替换未发送计划）取得文件级排他锁 → 发放该锁域的临时 permit + 新 fencing token，并原子重绑 HMAC
  → 独占发送权 → adapter send
  → adapter_send_succeeded
  → 匹配四个摘要的 journal finalizer lease
  → 生成确定性 .ready.txt Source
  → outbox=journal_ready → 删除加密暂存
```

- 加密暂存使用 `ADAPTER_FINALIZATION_KEY`（32-byte base64url）；未配置时保持兼容路径，绝不退回写明文预发送计划。
- receipt 数据库没有用户文本、回复、identity 或 API key；它只保存 SHA-256/HMAC 和状态。
- 任意重绑 outbox HMAC 的操作必须持有同一 delivery key 的 OS 文件锁、该锁范围内有效的进程内 permit 与 per-acquisition staging fencing token；permit 在退出锁范围前撤销，不能以过期或构造的值继续重绑。旧 token 一旦被接管即不可继续重绑。
- SQLite 还持久化规范化 journal 根目录的 SHA-256 锁域摘要；所有可变 outbox 操作、发送权、finalizer 与恢复路径都要求它精确匹配。因此不同 journal 根目录即使意外共用了 receipt 数据库，也不能接管彼此的暂存计划。
- `acquire_delivery_send_owner()` 与 journal finalizer 对任何非空 staging token（包括已过期但尚未安全修复的 token）均拒绝放行，且这个规则覆盖普通回复和命令回复；不会把发送/发布权授予正在替换或遗留不确定性的文件版本。取消中的异步锁获取在后台完成后会同步释放晚到的 OS 锁，避免取消造成永久占锁。
- 文件锁序列化加密计划的删除、重写和 HMAC 重绑；若运行中断而留下 token，发送会保守停止，接管者必须重新取得锁、验证当前 HMAC 与锁域并完成安全重绑后才会解除阻塞。升级前遗留的空锁域记录明确 fail-closed，不会被自动发送、重绑或完成 finalizer，需在受控迁移/人工处置后恢复。
- 第二个 runtime 不能与现有 runtime 同时发布同一 `.ready.txt`：SQLite finalizer lease 负责跨进程仲裁。
- finalizer lease 必须同时验证 `input_batch`、`input_order`、`reply`、`prompt_ledger` 四个摘要，并且只接受 `adapter_send_succeeded`、`finalization_failed` 或 `finalized` 的匹配回执；`prepared`、`send_started`、`uncertain` 均不会写日志、更不会自动重发。
- 恢复不调用 `ChatReply.finalize()`，因此不会重新执行 session 更新、preflight authorization、growth 或命令副作用；`finalization_failed` 也不会被伪装升级为完整 `finalized`。
- 同一 `.ready.txt` 的 importer claim 使用 owner token、续租和延迟重扫。活跃的慢导入不会因初始租约到期而被第二个 runtime 并发调用；失权 worker 只能保留文件等待后续安全处理。
- 若上游 importer 已成功并先写入本地 `.imported.json`/`.partial.json` receipt，随后在改名为 Source 前中断，接管 worker 只完成本地改名，不会再次调用 importer。
- `requirements.txt` 已声明 `cryptography>=42.0.0`，`.env.example` 给出非秘密配置槽位。

## 离线验证

| 范围 | 结果 | 含义 |
|---|---:|---|
| 核心仓库完整本地 unittest | 695：693 passed、2 skipped、0 failed/error（53.549 秒） | 覆盖 68 个 `tests/test_*.py` 模块；含 deadline 前提交回滚、查询学习收尾、Source-gold/formal preflight 的全量哈希/跨度/路径/CWD 门禁。 |
| Source-gold/formal preflight 定向回归 | 51：50 passed、1 skipped、0 failed/error（3.925 秒） | 覆盖草稿不可评分、raw-bytes binding、approved review gate、全量 frozen manifest、span/hash、holdout 隔离、JSON 溢出/深度、文本脱敏、UNC/远程盘/链接拒绝、输出防覆盖和绝对路径绑定。跳过项仅为当前 Windows 权限不允许创建 symlink 的可选场景。 |
| 聊天仓库完整本地 unittest | 251 passed、0 failed/error（7.033 秒） | 包含加密计划、篡改拒绝、HMAC/outbox、跨 store finalizer lease、确认后恢复、文件锁/permit 与过期 token fencing、跨 journal 根目录隔离、取消锁获取清理、暂存重绑/发送权竞争（含命令回复）、空锁域旧记录在 no-HMAC 发送路径的拒绝、长导入续租与失权隔离、receipt 后无重导入收尾、正常发送顺序和 coordinator 绑定测试。 |
| `git diff --check` | 两仓库通过 | 未发现 diff 空白错误；聊天仓库仅有既有/工作树换行提示。 |

完整命令、时间和边界保存在：

- `validation/v3-baseline-20260906/test-baseline-v22-t18-liveness-t19-receipt-20260906.json`

测试中的异常日志是刻意注入的发送失败、取消、增长失败和 deadline 场景；测试命令最终返回成功。验证未连接模型服务、未导入剧情语料、未运行正式 benchmark；Source-gold 门禁使用的仅是临时合成来源文件，不是用户的真实剧情语料。

## Benchmark 当前门禁

`benchmarks/manifests/aevnema_v3_source_gold_draft/` 当前状态如下：

| 项目 | 当前值 |
|---|---:|
| Gold 状态 | `draft_pending_source_span_review` |
| Family / claim group / atom 数 | 3 / 9 / 19 |
| 可评分 atom 数 | 0 |
| 待 source-span 审核 atom 数 | 19 |
| Split assignment / 全部 source key 数 | 3 / 5 |
| holdout assignment 数 | 0 |

新的正式门禁会正确拒绝这份草稿；本轮没有通过改写 `status`、flag 或内存对象把它人为晋级，也没有打开真实剧情语料来重锚定 source span。因此，任何“原 benchmark 与新知识库 benchmark 对比”都会缺少独立评分标准和 holdout 分母。现在运行只会得到不可解释的诊断输出，不能当作正式结果报告。

正式执行前需要依序完成：

1. 人工审核并冻结 Source span gold，建立至少一个 source-disjoint、已审核 holdout；
2. 为旧库/全量库建立可校验、受控且静止的本地只读 Source 快照及语料处置 manifest；
3. 确认原 benchmark 的 delivered-budget、题集和评分口径；
4. 在用户授权的模型与语料范围内执行原口径和 v3 口径，并保存 trace、调用观测和失败分母。

## 本次继续执行：真实 Source 审核包、Q1→Q2 pilot 与规则冻结

### T04 真实 Source 审核包（仍待人工签核）

已读取用户授权的真实剧情根目录
`C:\Users\Admin\Documents\train_assets\blue-archive\apps\blue-archive-story-viewer\public\story`，并以第一份冻结中的只读历史 SQLite 快照恢复候选窗口。交付物位于：

- `validation/v3-t04-real-source-review-20260906T233016Z/source_gold_review.full_local.json`
- `validation/v3-t04-real-source-review-20260906T233016Z/source_gold_review.md`
- `validation/v3-t04-real-source-review-20260906T233016Z/frozen_source_manifest.candidate.json`

该包逐项覆盖当前 19 个 logical atom、5 个真实 Source 文件；每项都含未清洗 `TextCn`、`TextJp`、`ScriptKr`、相邻 record 上下文、`/content/<n>/TextCn` locator、Unicode code-point span、raw span/source-file hash、历史 Episode 线索、主张边界与争议说明。所有 19 项仍为 `pending_source_span_review`，包的顶层也明确标记为不可评分、禁止 promotion；它没有把旧 `any_of` 候选关系代签为 Source claim，也没有修改 draft gold。

审计确认旧库可恢复的 156 条 record `TextCn` 与当前真实 JSON 经原适配器规范化后逐条一致，但当前分段器的 rendered segment hash 与历史 hash 不同。因此旧 `segment_index` 只作为候选窗口线索，不能充当新 Source gold 锚点。人工审核仍需对跨 record 摘要拆分/收窄，并保留角色推测、通讯报告和跨文件推论的认识论边界。

### 隔离真实 pilot 数据准备

为避免把未审核 gold 或旧库无 span 的 Episode 混入学习，新增了非评分输入清单：

- `benchmarks/manifests/v3_q1_q2_diagnostic_source_slice.json`

它只含同文 Q1/Q2、domain/scope 与 `main/32170.json` 的 SHA-256
`2078eb5900fa83d30191e3d5d67ee4e308b5893e9eaee7fc579207484cac00c8`；不含 answer、need、selected ID、gold 或评分标签。实际 Source 导入使用 DeepSeek-V3.2 主模型、GLM-4.5V transport fallback、GLM-4.5-Air 对抗审核和 BGE embedding，且延后推断关系。

第一次新目标库导入在任何模型请求前暴露了初始化顺序问题：recovery 在新 SQLite 尚无 `extraction_run` 表时运行。`benchmarks/import_corpus.py` 现仅在 recovery 前对**配置的可写目标库**执行幂等 schema 初始化；不触及 Source、冻结库或已有语料。首次默认沙箱模型连接则以 Windows `10013` 被拒，留下 `interrupted` ledger、0 Source、0 Episode；该失败目录未覆盖。取得用户授权的模型网络后，在新的隔离目录完成一次真实导入：

- `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/source_import.ledger.json`
- 1 文件、14 Source segment、45 Episode、45 literal evidence quote/span、0 partial/failed task、推断关系 deferred。

完成后从导入库只读 backup 生成 `source_bound_slice.static-v2.sqlite`，其 SHA-256 为
`04fa505c91d7ce4c1a7c8619e68194eddaa0b76717a13c0836d3cd96340db1b0`，journal 为 `DELETE`，无 WAL/SHM、0 foreign-key error。由于 V3 DDL 可引用应用自定义 SQLite 函数，bare SQLite 的 `quick_check` 不能作为所有 clone 的可靠验证；pilot clone 工具改为 foreign-key check 加 `source`/`episode` 可读性验证。它只验证新 clone，不改变原 SQLite。

### 真实 Q1→Q2 一次性诊断结果

新增的 evaluator-only runner 为每个实例创建独立 SQLite clone，并将 Q1 的**原 application finalizer**包在 pre-commit gate 前：只有 Q1 已实际生成可提交的学习计划时，才会在提交 edge 前运行 immediate Q2。它不会手工调用 `finalize_contextual_creation`、预置 edge、缓存 answer/plan/need/selected ID，也不会读取 Q2 gold。所有 arm 固定 `growth_max_rounds=0`、关闭 association cue/fast path、关闭 promotion 与 restricted rewrite；mask 仅隐藏实际 Q1 edge，同时保留独立基础检索。

本次真实 case 使用上述静止 Source-bound 切片和同文问题 `圣园未花说自己一直在暗中支援哪个组织？`。case 原始轨迹和脱敏报告为：

- `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/real_q1_q2_case/q1_q2_case.full_local.json`
- `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/real_q1_q2_case/q1_q2_case.report.md`
- `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/real_q1_q2_case/q1_q2_case.report.v2.md`

该一次 trial 的 Q2 embedding warmup 在 Q1 前完成，且只保留原问题向量；但 Q1 的实际 live planner 仍遵循冻结配置的 `followup_planning_mode=always`。后续 live rerank 失败，answer provider 的响应在 120 秒绝对 deadline 后被标为 `late_discarded`，Q1 以 `ModelDeadlineExceeded` 终止。它从未到达 pre-commit gate，因此没有新 receipt、association 或 runtime projection；immediate、ready、masked、ordinary-cache、restart 五个 Q2 arm 都以 `not_run` 明确保留，而非用假边、假缓存或重跑填补。整个 case wall time 为 181.094 秒（含等待 pre-commit gate 的安全终止窗口），case SHA-256 为 `d4450095c3ae3fb47a7782bcdbb07dd1b09d06f2c4f5445074cd242453ff9434`。这不是正式质量分数，也不支持任何收益/无收益结论。

原始 `full_local` 的逐条 provider observation 完整保存；复核后发现其聚合器把 ModelClient 的 `succeeded` 状态误作失败。runner 已改正并以本地回归覆盖，原始 case 不重写，新增 v2 脱敏报告从既有 observation 重算：Q1 为 6 次 HTTP（4 succeeded、1 timeout、1 late-discarded），Q2 embedding warmup 为 1 次 succeeded；Q2 arm 没有调用。该修正不改变 Q1 的 deadline 失败、无 edge/receipt 或任何实验状态。

### Pilot 后规则冻结与定向回归

已创建新的 post-pilot freeze：

- `validation/v3-postpilot-freeze-20260907T022000Z/`

它记录当前双仓库工作树身份、脱敏配置、静止 Source-bound pilot snapshot、Source selection、import ledger、T04 审核包以及 real pilot 的 `full_local`、v1/v2 可读报告和本报告哈希。它不覆盖最初的双仓库三 SQLite baseline freeze；此前的 `...T020600Z` 与 `...T021500Z` 也保留为聚合修正前/报告索引更新前的中间冻结。

本轮新增/定向测试共 12 项，12 passed、0 failed、0 skipped：Source 审核包、edge-only overlay、真实 Q1 pre-commit/finalizer/遮罩/cache/restart local fixture、fresh importer target initialization、UDF-independent clone validation 和 provider `succeeded` 计数均已覆盖。真实模型调用只发生在上文明确的 single-file data preparation 与 single Q1 trial；未执行正式 benchmark、正式 promotion 或 canary。

## 仍然明确不宣称的能力

- 不宣称近 100% 召回、加速比例、线上稳定性或正式 benchmark 成绩；
- 不宣称临时合成 fixture 已验证真实剧情 Source 的语义、span 锚定或人工审核结论；
- 不宣称路径式校验可抵抗受控快照之外持续、敌对的并发文件系统替换；
- 不宣称远端 provider 的取消必然停止计费，或跨进程 provider 限流已经通过线上压力验证；
- 不把 journal Source 的恢复说成完整聊天 finalizer 恢复；
- 不把 importer 的跨进程 lease 说成对任意外部写入的绝对 exactly-once 保证：若进程在上游 importer 已产生副作用、但尚未来得及写本地 receipt 前硬退出，仍需下游按 deterministic Source key 幂等；
- 不把进程内 permit 说成对不受信任同进程代码的安全隔离边界；它是 runtime 内部的生命周期/一致性约束。迁移前空锁域 outbox 也不会被自动修复或继续发布；
- 不自动使用未授权模型，也没有在本轮调用任何模型服务；
- 不把当前极大的未提交双仓库 diff 视作一个干净、可发布的提交。

## 下一步

已完成的 pilot 规则被冻结，但本次真实 Q1 因 deadline 失败，尚未产出可比较的 edge-ready case。下一步首先需要人工 source-span 审核并形成 approved、source-disjoint holdout；在不改动正式门禁的前提下，可继续准备全量静止 Source snapshot。只有在独立 holdout 获批且至少一个完整 pilot 闭环真实结束后，才扩展拟人记忆测试与旧库—全量库的正式对比。届时应单独产出正式的质量、延迟、调用量、失败分类和可回放包，不与本报告的离线回归、未审核诊断或本次超时 case 混用。

## 追加：v10 独立 Q1→Q2 诊断闭环（不改变正式门禁）

首次真实 trial、v2 至 v9 的失败目录均原样保留。随后只对已经定位的编排问题做了最小修正，并创建了新的 v10 目录；没有重写失败 case、没有预置边、没有用答案或 gold 缓存替代边复用，也没有扩展通用 deadline、journal、promotion 或 Source-gold 范围。

v10 使用同一静止 Source SQLite、同一 Q1/Q2 问题、120 秒总 Q1 budget 和零重试。单 HTTP 上限由 40 秒降至 25 秒，目的是给已授权的 GLM-4.5V transport fallback 与后续 answer audit 留出总 deadline 内的窗口；实际模型仍仅为 DeepSeek-V3.2、GLM-4.5V 和 BGE-M3，未使用 GLM-4.5-Air 或任何未授权模型。

Q1 在原应用公开 finalizer 路径完成：receipt `1` 与 edge `121` 均实际创建，learning 状态为 `ready`。六个同题 Q2 臂（pre-Q1-state clone、发送后立即、learning-ready、edge mask、ordinary embedding cache、restart）均只执行一次并保存完整轨迹。每臂为 45 candidate、1 selected、episode `32` delivered；mask 后也相同。随后对同一 receipt/edge 的公开 Q2 preflight 确认：receipt 没有 V16 contract 或 V17 runtime manifest，exact-revisit 未进入；普通 selector 因 `no_unresolved_slots` 在 matcher 前短路。因此该非评分单例只证明基础检索独立覆盖，不能宣称边参与后的收益或无收益。详细逐阶段证据、失败分类、调用量、耗时与可审核交付物见 `docs/reports/V3_Q1_Q2_DIAGNOSTIC_CLOSURE_2026-09-07.md`。
