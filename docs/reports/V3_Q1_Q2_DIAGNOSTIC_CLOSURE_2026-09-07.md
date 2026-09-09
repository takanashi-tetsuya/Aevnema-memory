# V3 Q1→Q2 诊断闭环与交付说明（2026-09-07）

## 结论

本轮以**新的、独立目录**完成了一个真实 Source 的 Q1→Q2 闭环。Q1 通过原应用的公开 finalizer 自动提交了一个 ready receipt（`1`）和一条关联边（`121`）；六个 Q2 状态/对照臂都各执行一次并保留了原始轨迹。

这不是正式 benchmark：本例没有加载或使用 gold，19 个 Source atom 仍待人工 source-span 审核，正式 promotion/canary 保持关闭。普通基础检索确实独立找到并交付 episode `32`，但后续公开入口复核表明 receipt `1` 没有 V16 contract 或 V17 runtime manifest，Q2 exact-revisit 没有进入；普通 contextual selector 也因 `no_unresolved_slots` 在 matcher 前短路。因此这些臂不能称为“边正常参与后的无收益”验证；原始结果仍保留，且不会以重跑替换。

## 范围、冻结输入与模型

- 静态 Source SQLite：`validation/v3-q1-q2-real-source-pilot-20260907T001700Z/source_bound_slice.static-v2.sqlite`
  - SHA-256：`04fa505c91d7ce4c1a7c8619e68194eddaa0b76717a13c0836d3cd96340db1b0`
  - 基于 `main/32170.json` 的真实 Source-bound 片段；Q1/Q2 使用同一个问题。
- 问题：`圣园未花说自己一直在暗中支援哪个组织？`
- 总 Q1 deadline：120 秒；每个模型请求零重试。为给已授权回退和后续审计各保留一个窗口，v10 将单次主模型请求上限从 40 秒改为 25 秒；没有改变总预算、问题、Source 或评分规则。
- 实际使用模型：DeepSeek-V3.2（主推理）、GLM-4.5V（仅 transport fallback）、BGE-M3（embedding）。本 trial 没有使用 GLM-4.5-Air，也没有使用任何未授权模型或 per-edge 云端审核。
- 诊断配置：仅在初始候选槽缺失时才计划 follow-up；增长、cue fast path、promotion、restricted rewrite 和 rerank audit 均关闭。没有预置边、答案缓存、need 缓存、selected-ID 缓存或 Q2 gold。

## 原始失败 trial 的可重放复盘

原始失败 case 未被改写。离线导出的逐阶段复盘保留“没有保存就不补写”的边界：剩余预算、原始 prompt、required evidence 的首次入候选/选择/prompt 时间均为 `not_observed`，而不是推测值。

| Q1 HTTP 用途 | 状态 | 开始 → 结束（UTC） | Q1 开始时 ms | 剩余预算 / 重试或 fallback |
|---|---|---|---:|---|
| query intent | succeeded | 02:02:13.215 → 02:02:22.462 | 2.007 | not_observed / not_observed |
| initial vector retrieval | succeeded | 02:02:22.467 → 02:02:22.913 | 9,253.931 | not_observed / not_observed |
| follow-up planning | succeeded | 02:02:23.681 → 02:02:31.818 | 10,467.528 | not_observed / not_observed |
| follow-up vector retrieval | succeeded | 02:02:31.822 → 02:02:32.095 | 18,608.921 | not_observed / not_observed |
| evidence rerank selection | timeout | 02:02:32.744 → 02:04:02.764 | 19,530.897 | not_observed / not_observed |
| answer generation | late_discarded | 02:04:03.735 → 02:04:13.230 | 110,521.851 | not_observed / not_observed |

rerank 的具体类别是 transport `ReadTimeout`：底层 `_SerializedHTTPSConnectionPool(...api.siliconflow.cn:443)` 在 `read timeout=90.0` 时抛错；它不是 rerank JSON 校验失败、候选为空或 Source 校验失败。

本地故障注入测试确认原控制流：`_rerank_answer_episodes()` 捕获这类异常、写入 `rerank_trace.error`、返回空 rerank 结果；上层随后以 `reranked_episode_ids or episode_anchor_ids` 继续普通答案选择，因此会进入 `answer_generation`。原 case 此时没有保存成功 selection/prompt，且其最后请求在总 deadline 后才被丢弃。这解释了“rerank 失败后仍发出迟到 answer”这一行为；它不是一次隐式 rerank retry。相应的 local control-flow 回归为 `test_rerank_timeout_is_caught_and_current_flow_still_generates_answer`。

## 最小诊断修正

修正局限于已定位的编排与证据对齐问题，未扩大通用 deadline/journal/gold 门禁范围：

1. `missing_slots` 只在初始候选槽确实为空时调用 follow-up planner；v10 的记录为 `followup_planner_invoked=false`、`all_initial_candidate_slots_present`。
2. 单一 requirement 以原问题而非 planner 的首个改写作为绑定；只复用同一逻辑输入已有的向量绑定，不重新 embedding 或近邻猜测。
3. 合法的短 rerank 选择不再填充为宽候选尾部，避免把本应简短的答案证据 prompt 放大；Source fact locator 使用导入时声明的 compact reasoning view 坐标。
4. Q1 后的数据库 clone 只服务观察：`before_commit_probe` 在 Q1 终态后针对 pre-Q1 clone 运行，绝不并发暂停 Q1；随后才运行“实际发送后立即 Q2”。新边由公开 finalizer 生成，不调用内部 finalize 补边。
5. v10 仅降低单个请求等待上限至 25 秒，保留 120 秒总预算和零重试。它使主回答 25 秒 timeout 后的 GLM-4.5V 回退、以及随后的 answer audit 都有完成空间。

本地回归：123 项通过、0 failed/error，涵盖上述 Q1/Q2 runner、rerank timeout 控制流、Source locator、contribution coverage、公开 receipt/finalizer 和 saved-case postmortem。

## v10 真实闭环观测

Q1 用时 75,878.787 ms（engine 计时 75.832 s）。公开 finalizer 在 77,062 ms 进入、77,094 ms 返回；未注入 Q1 pause。其 provider ledger 为 5 logical batches、6 HTTP：5 成功、1 timeout、1 fallback。

| Q1 用途 | 实际模型 / 结果 | 网络 ms |
|---|---|---:|
| query intent | DeepSeek-V3.2 / succeeded | 16,125 |
| initial vector retrieval | BGE-M3 / succeeded | 578 |
| evidence rerank selection | DeepSeek-V3.2 / succeeded | 9,359 |
| answer generation | DeepSeek-V3.2 / `ReadTimeout` | 25,000 |
| answer generation fallback | GLM-4.5V / succeeded | 5,969 |
| answer evidence audit | DeepSeek-V3.2 / succeeded | 17,922 |

Q1 有 45 个候选、rerank 选择 2 个 evidence episode；最终所需槽由 episode `32` 覆盖。公共提交生成 receipt `1` 与 association `121`。这是一条真实执行记录，不等同于人工签核或正式质量分数。

| 同一 Q2 臂 | Candidate | Selected | Delivered | 实际交付 episode | HTTP（成功/失败或拒绝/fallback） | 耗时 ms | 边状态 |
|---|---:|---:|---|---|---|---:|---|
| before_commit_probe | 45 | 1 | True | 32 | 7 (5/2/2) | 91,535.397 | pre-Q1 clone，无边 |
| post_send_immediate_q2 | 45 | 1 | True | 32 | 6 (5/1/1) | 101,659.165 | Q1 公开调用返回后的 clone |
| learning_ready_q2 | 45 | 1 | True | 32 | 5 (5/0/0) | 56,323.400 | ready 边可见 |
| edge_masked_q2 | 45 | 1 | True | 32 | 6 (4/2/1) | 82,507.921 | 仅隐藏 edge 121 |
| ordinary_embedding_cache_q2 | 45 | 1 | True | 32 | 6 (5/1/1) | 85,726.005 | 隐藏 edge 121；仅复用预先生成的原问题向量 |
| edge_restart_q2 | 45 | 1 | True | 32 | 6 (5/1/1) | 71,151.477 | 新 app/index 后边可见 |

每个臂都记录 `contextual_attached_edges=[]`，而 `exact_revisit_executed_modules` 为 `null`；因此报告只能写“观察到没有 exact-revisit module”，不能把未记录模块补称为跳过。后续零模型 public-preflight 已确认该 case 在 runtime-manifest lookup 处 miss，而普通 selector 的已保存理由是 `no_unresolved_slots`。遮罩臂仍可由基础检索独立获得 episode 32，正是 mask 合约要求保留的路径。这个单一、未审核诊断例未发生可归因的边复用参与，因而没有任何正式分数，也不构成边参与后的收益或无收益结论。

## 可交付审核材料

以下均是已实际生成的文件，而非目录占位符：

| 材料 | 内容与状态 |
|---|---|
| `validation/v3-t04-real-source-review-20260906T233016Z/source_gold_review.full_local.json` | 19 atom 的真实 Source 原文、上下文、定位、span/hash、主张边界和争议说明；所有项仍为 `pending_source_span_review`。 |
| `validation/v3-t04-real-source-review-20260906T233016Z/source_gold_review.md` | 上述审核包的可读版。 |
| `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/q1_q2_case_postmortem_v1/q1_q2_case.postmortem.full_local.json` | 原始失败 case 的离线、未补写复盘。 |
| `validation/v3-q1-q2-real-source-pilot-20260907T001700Z/q1_q2_case_postmortem_v1/q1_q2_case.postmortem.report.md` | 原始六次 HTTP 的脱敏可读复盘。 |
| `validation/v3-q1-q2-real-source-diagnostic-v10-20260907T131038Z/q1_q2_case.full_local.json` | v10 的完整本地原始轨迹，含各臂 provider observation 与结果。仅限本地审核。 |
| `validation/v3-q1-q2-real-source-diagnostic-v10-20260907T131038Z/q1_q2_case.report.md` | v10 的可读脱敏 case report。 |

## 后续边界

T04 的 19 个项目不代签；在有人工 approved Source gold、source-disjoint holdout 和冻结的全量语料处置清单之前，不能把本例外推为旧库—全量库正式对比或 benchmark 成绩。全量静态快照准备可继续，但正式 promotion 与 canary 继续关闭。
