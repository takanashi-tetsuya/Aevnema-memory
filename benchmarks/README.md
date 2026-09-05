# Benchmarks

本目录保存离线实验、审计和语料准备工具。

- `manifests/`：可复用的基础评测问题。
- `validation/`：项目根目录下的清单与少量跨项目 catalog。
- 历史运行数据库、模型请求日志和中间结果不属于源码；整理后保存在 `_archive/`，可在确认不再需要复现实验后永久删除。

大部分历史阶段脚本依赖对应报告中记录的前置数据库或语料。执行前应先阅读 `docs/reports/` 中同阶段报告。

`run_rerank_stability_eval.py` 不重新检索候选。它读取一次已完成的基础检索报告，冻结其中每题的
Candidate@100，多次运行当前证据覆盖与 Top-20 压缩流程，统计每个必需事实槽的命中频率、两名覆盖
侦察器的分歧、最差 Recall@20 和延迟。该工具用于区分“基础候选没有召回”和“随机证据选择不稳定”。

历史报告若没有记录数据库哈希，应先运行 `find_frozen_candidate_database.py`。它用报告中持久化的
Episode ID + 文本作为语义指纹扫描 SQLite 快照；整数 ID 虽然存在但已被重建语料复用时不会被误判为
同一基线。

`audit_evidence_floor_policies.py` 对一次完整检索 trace 做离线反事实：保持 LLM coverage/compressor
输出不变，只切换 hard-floor 的来源和预算。它用于发现“召回候选”被误当成“必须进入最终答案的证据”
以及 floor 数量吞掉全部 Top-K 槽位的问题。

`run_atomic_cross_encoder_coverage_eval.py` 在冻结 Candidate@100 上分别测试整题 BGE、逐槽轮询 BGE 和
逐槽贪心覆盖。它同时保存每个查询对全部候选的排序，因此可离线分析 Top-20 最终选择以及 Top-30/40/50
注意力压缩池。该工具验证的是 Cross-Encoder 在多事实覆盖中的适用边界，不修改数据库和 Association。

`run_answer_claim_consistency_deadline_eval.py` 读取已经生成的冻结回答和 Answer Contract，不重新检索、
不写数据库。它分别运行本地结构守卫、Qwen/DeepSeek 原子语义审计、请求级总截止时间和长回答分块回退，
同时保留 completed-only 与 fail-safe 指标。`--prepare-only` 只生成本地人工标签和结构基线，不发送外部请求。

`run_atomic_answer_scope_eval.py` 将最终回答拆成带精确原文锚点的最小片段，使用 Qwen 快速抽取、失败子块
DeepSeek 恢复、DeepSeek 分批映射 Answer Contract，并只对首次 `unsupported` 的片段追加两次相同中立
判定。它不写数据库；embedding 不参与也不存在 fallback。

`run_atomic_answer_repair_eval.py` 读取原子范围结果，只把 `unsupported` 片段作为修复约束交给 DeepSeek，
随后运行可用性盲审、确定性合同质量检查和修复后原子复审。它支持带 inline `expected` 与人工标签的
自包含清单；没有确定性文本约束时会显式跳过旧关键词检查。`--ids` 可只复测指定案例。输出的 repaired
source 是实验资产，不是生产记忆。

`manifests/runtime_receipt_question_premise_v1.json` 验证 Answer Contract 中动作执行凭证与用户问题前提。
它区分 Supported Fact、Write Candidate、Runtime Receipt 和 Question Premise，不依赖特定剧情实体。

`run_request_scoped_generation_eval.py` 使用
`manifests/request_scoped_generation_v1.json` 对比旧式自由回答与请求级不可变合同渲染。它覆盖错目标/外来
回执、迟到回执快照、动作失败或取消、问题前提状态和混合动作，并生成可交给
`run_atomic_answer_scope_eval.py` 的控制组与请求级审计输入。模型盲审是离线质量指标，不属于线上调用链。

`manifests/request_scoped_generation_v2.json` 在同一组反事实案例上要求每个动作提供
`public_action_summary`，并把 limitation 分成 `action_state` 与 `evidence_gap`。它用于验证内部 action/target
不进入渲染提示、动作状态限制不与回执重复、事实缺口在检索成功后仍被保留。
