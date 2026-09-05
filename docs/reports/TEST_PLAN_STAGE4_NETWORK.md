# 第四阶段：网络级长链联想测试方案

## 目标

本方案测试的不是“能否找到一个正确片段”，而是系统能否在不使用剧情外知识的前提下，把人物动机、制度、历史场所、现场事件、事后解释和认识论限制组织成可审计答案。

上一阶段的每题通常要求命中 3 组关键 Episode。本阶段每题要求 7–9 组独立证据组，作为“理论上约两倍 Association/推理连接”的出题复杂度目标。实际运行中的新增、强化或复用边数量不属于评测指标；增长模式只在确实发生变化时审查边是否正确并被使用。

唯一允许的剧情来源列在 [证据清单](validation/stage4-network-evidence-manifest.json) 的 `allowed_source_keys`。当前范围是 11 个已导入文件，不允许使用未导入主线、Wiki、角色常识或模型训练记忆补全答案。

## 统一回答约束

每道题的回答都必须明确写出：

1. 原文事实：哪一个 Episode、哪个 `source_key` 直接说了什么。
2. 综合推论：由哪些事实组合而来，且为什么仍只是推论。
3. 文本未知/仅为推测：不能把假说、人物发言或政治背景升级为已证实的具体机制。

缺少任一层，即使“故事结论”看似正确，也不应通过正式评分。

## 正式题集

问题原文在 [题集 JSON](validation/evaluation-questions-stage4-network.json)，评分锚点在 [证据清单](validation/stage4-network-evidence-manifest.json)。

| ID | 核心链条 | 证据组 | 必须守住的边界 |
|---|---|---:|---|
| `eden_obligation_to_new_institution` | 圣娅的义务 → 老师相信学生 → 原 ETO → 戒律历史 → 夏莱代理 → 新 ETO → 双 ETO 冲突 | 7 | 不得说守护者已明确服从新 ETO。 |
| `old_cathedral_symbol_infrastructure_and_limits` | 条约目的 → 古圣堂历史 → 整修/地下废墟 → 墓穴 → 阿里乌斯部署 → 爆炸假说 → 现场推测 → 新 ETO | 8 | 不得把墓穴、炸药、会场选择者或未花写成已证实的渗透机制。 |
| `arius_two_patrons_and_control_boundaries` | 未花利用阿里乌斯 vs 真琴利用阿里乌斯 → 各自的资源/反噬边界 | 8 | 不得把未花与真琴写成统一指挥链。 |
| `mika_motive_epistemic_four_layers` | 未花直述 → 花子推测 → 未花否认/修正 → 乐园悖论的可知性限制 | 8 | 不得宣告“真正动机”已被证明。 |
| `hifumi_azusa_identity_coalition_and_declaration` | 梓的档案身份 → 面具身份 → 日富美/浮士德 → 援军 → Happy End 反驳 → 新 ETO | 8 | 不得把日富美写成爆炸主使，或把新 ETO写成她个人单独建立。 |
| `hina_public_duty_personal_exhaustion_and_teacher_response` | ETO/真琴职责 → 爆炸战斗 → 星野比较 → 崩溃与求关注 → 老师的支持 → 重返指挥 | 9 | 星野经历是日奈引用的比较，不等于文本证明的直接行动因果。 |
| `kisaki_mika_governance_analogy_with_boundary` | 山海经排外焦虑与月影祭治理 vs 未花的排外/政变治理 | 8 | 只能做类比，不能虚构两条剧情的直接因果或相互影响。 |

## 执行方式

### 阶段 A：逐题诊断

每题分别在新的评测副本中运行 `vector_only`、`graph_static`、`graph_growing`。此阶段用于定位问题属于：召回不足、图遍历不足、关系增长不足，还是回答因果越界。

```powershell
$env:MEMORY_DB_PATH = "validation/ba-stage3-deep-v37.db"
python src/main.py evaluate `
  validation/evaluation-questions-stage4-network.json `
  validation/evaluation-stage4-network-isolated `
  --modes vector_only graph_static graph_growing

python benchmarks/score_stage4_network.py `
  validation/evaluation-stage4-network-isolated/evaluation-report.json `
  --output validation/evaluation-stage4-network-isolated/scorecard.json
```

正式运行时，建议先从单题 JSON 文件切分并保存单独日志；这样一次失败不会掩盖其他题的召回质量。

### 阶段 B：连续增长

七题按照题集顺序，在同一 `graph_growing` 副本连续执行。检查每个新 Association 是否在当题答案路径使用，并检查后题是否开始复用前题建立的关系。

### 阶段 C：顺序扰动

至少使用三种不同题目顺序重复阶段 B。关键指标不是每次生成完全相同的边，而是：

- 每题仍通过 Source 原文增强后的答案审计；
- 精确 Episode 组和预设来源覆盖作为诊断值单独报告；
- 末次答案审计仍有效；
- 无错误身份、无虚构路径、无推测升级；
- 若发生复用，记录哪些后题实际使用了前题关系；不设复用次数或比例目标。

命令行可直接指定顺序，并在网络或电脑异常后从断点继续：

```powershell
python src/main.py evaluate questions.json output `
  --modes graph_growing `
  --question-order reverse `
  --resume
```

### 阶段 D：人工审计

逐条审阅每条新增/强化 Association 的：

```text
claim_level
audit_status
evidence_json
audit_json
relation_text
最终 answer path 是否使用
```

特别审查 `political_precondition`、`historical_support_context`、`evidence_bridge` 和跨 Source 边。

可复现的增长审计命令：

```powershell
python benchmarks/audit_stage4_growth.py `
  output/evaluation-report.json `
  --output output/growth-audit.json
```

## 评分

[评分器](benchmarks/score_stage4_network.py) 将检查：

- 答案包含问题要求的核心实体/概念；
- 结构化 `evidence_episodes` 必须包含可回溯且位于允许语料内的来源；
- 最后一轮答案审计为 `valid`；
- 证据来源没有越出允许语料；
- `vector_only` 不使用 Association；
- `graph_static` 使用既有 Association；
- `graph_growing` 的新建/强化边必须在最终答案路径中出现。

精确 Episode 组、预设来源集合和“事实/推论/未知”标签词仍会完整输出，但仅作诊断：Episode 可以通过 `source_id` 回溯到相邻 Source record，同一事实也可能在允许语料的另一文件中被完整重述，不能让预设 ID 或文件名覆盖经原文审计后的正确答案。

新增或强化边数量不记录为成功指标。发生增长时，只检查它是否安全、是否有证据、是否进入答案路径；零增长本身既不加分也不扣分。

## 通过标准与失败分类

硬性通过：核心实体/概念、结构化证据、回答审计和模式行为均通过，且无越界来源。正文是否打印 `source_key`、精确 Episode 组、预设来源和标签覆盖用于解释检索与展示质量，不单独决定整题正确性。

失败应按原因记录，而不是只记录“回答错误”：

| 类型 | 例子 | 优先修复位置 |
|---|---|---|
| 召回失败 | 7 组中缺某一远距离 Episode | 原子查询、锚点预算、向量/关键词混合检索。 |
| 图路径失败 | Episode 都进池但关键路径没进入答案 | beam、路径重排、关系权重。 |
| 增长失败 | 候选正确但审计/门卫拒绝 | 检查关系类型和 claim 层级，而非放宽因果规则。 |
| 事实强度失败 | 把墓穴、炸药或政治合作写成已证实机制 | 回答审计与确定性因果门卫。 |
| 复用失败 | 新边写入但没有进入答案/后续题 | 增长价值排序、重复边抑制。 |
| 语料越界 | 使用未导入剧情或外部设定 | 提示词、来源审计、回答生成上下文。 |

## 当前不测的内容

本阶段不用于评估全量 Blue Archive 时间线、模型升级、联网补充、ANN 性能或长期记忆衰减。它首先验证的是：在当前 11 个文件的封闭语料内，Association 网络是否能安全地从“三跳问答”进入“七到九组证据的可审计联想”。
