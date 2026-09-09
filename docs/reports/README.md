# 实验报告索引

本目录保存 Associative Memory 引擎各阶段的实验结论、测试计划和架构审计。

- `V3_IMPLEMENTATION_PROGRESS_AND_BENCHMARK_GATE_2026-09-07.md`：v3 当前实现、T18/T19 收尾硬化（含锁域隔离与旧记录 fail-closed）、T04 Source-gold/formal benchmark 门禁（原始 bytes、来源 span/全量冻结清单和本地路径检查）、695 项核心与 251 项聊天侧离线回归。

- `COMPREHENSIVE_SYSTEM_EXPERIMENT_REPORT_2026-08-26.md`：早期完整系统实验总结。
- `BASE_RETRIEVAL_RELIABILITY_REPORT_2026-08-26.md`：基础召回可靠性。
- `ARCHITECTURE_OPTIMIZATION_REPORT_2026-08-29.md`：架构和流程优化。
- `EXPERIMENT_REPORT_STAGE16_IMPORT_ORCHESTRATION_2026-08-29.md`：导入编排。
- `EXPERIMENT_REPORT_STAGE17_FULL_REBUILD_2026-08-29.md`：完整语料重建。
- `EXPERIMENT_REPORT_STAGE18_GROWTH_SAFETY_AND_RECALL_2026-08-29.md`：增长安全与召回。
- `EXPERIMENT_REPORT_STAGE19_FROZEN_PLAN_COUNTERFACTUAL_2026-08-29.md`：冻结计划与反事实门。
- `EXPERIMENT_REPORT_STAGE20_21_BASE_RECALL_AND_CROSS_QUERY_UTILITY_2026-08-29.md`：基础证据槽与跨问题增长收益。
- `PRODUCTION_KB_IMPORT_AND_FLEXIBILITY_AUDIT_2026-08-31.md`：production KB 全量导入、文件兼容性、证据修复、正式库启用与真实检索审计。
- `SERIHU_IMPORT_QUALITY_AUDIT_2026-08-31.md`：台词全量导入暂停后的 Episode/Concept 质量审计、模型对照、事实归因修复与恢复导入门禁。
- `DOMAIN_RULE_DEHARDENING_2026-09-01.md`：删除剧情派生的检索、增长和审计规则，保留领域无关的证据不变量。
- `IMPORT_QUALITY_POST_DEHARDCODE_CHECK_2026-09-01.md`：去剧情规则后的独立复导、严格审计和冻结证据槽检索复验。
- `BASE_RETRIEVAL_STABILITY_AND_RERANKING_REPORT_2026-09-01.md`：冻结 Candidate@100 的重复稳定性、hard-floor 反事实、LLM/BGE 重排对照及下一阶段分块覆盖方案。
- `EVIDENCE_DRIVEN_RETRIEVAL_ESCALATION_EXPERIMENT_2026-09-01.md`：120 条多语言跨域 Probe、证据角色判定、缺失事实槽两跳深搜、安全对照、模型延迟对照与证据驱动升级结论。
- `CROSS_DOMAIN_ANSWER_CONTRACT_AND_CANDIDATE_GATE_EXPERIMENT_2026-09-01.md`：17 类私人/公共/剧情/创作端到端 A/B、分帧 EvidenceContract、模型规划反证、Qwen/DeepSeek 同候选对照、BGE 分数语义校准及 Support/Context 双通道结论。
- `LAYERED_CANDIDATE_EVIDENCE_ESCALATION_EXPERIMENT_2026-09-01.md`：Support/Review/Context 三通道、required evidence domain、证据驱动 DeepSeek 升级、persona 创作约束、回答渲染反证、跨平台隔离与请求级总截止时间结论。
- `ANSWER_CLAIM_CONSISTENCY_AND_DEADLINE_EXPERIMENT_2026-09-01.md`：最终自然回答是否越过证据合同、原子语义审计、Qwen/DeepSeek 对照、请求级总截止时间、长回答分块回退与角色扮演即时评价边界。
- `ANSWER_PERSISTENCE_BOUNDARY_AND_REPAIR_EXPERIMENT_2026-09-01.md`：回答持久性新增与即时角色表达边界、原文锚定原子审计、局部 fallback、中立三判、合同受限修复、16 样本扩展结果及 runtime receipt/问题前提缺口。
- `RUNTIME_RECEIPT_AND_QUESTION_PREMISE_EXPERIMENT_2026-09-01.md`：请求级动作凭证与问题前提状态、12 个反事实样本三轮稳定性、无凭证写入边界、受限修复及生产接入门槛。
- `REQUEST_SCOPED_PREGENERATION_CONTRACT_EXPERIMENT_2026-09-01.md`：不可变请求快照、精确回执隔离、问题前提、必答语义单元、生成前约束 A/B、16 样本独立原子审计及生产接入缺口。
- `PUBLIC_ACTION_AND_TYPED_LIMITATION_EXPERIMENT_2026-09-01.md`：公开动作摘要、action_state/evidence_gap 类型化限制、渲染字段最小化、原文锚点审计、16 样本复验及 Persona/事实槽后续缺口。

其余文件是对应阶段的详细实验记录和预注册计划。
