# 全文总结 / Document Map 导入实验报告

## 结论

全文信息对理解局部片段有帮助，但不应把自由文本“全文总结”直接作为默认 Episode 提取前置步骤，也不应把全文级人物名单直接注入局部 Episode 提取器。

本轮最好的使用方式是：把全文结果限制为**可丢弃的导航上下文**，Episode 仍须绑定原始 Source 行号并接受原文蕴含审计。即便如此，在 5 文件 / 7 Source 的同条件 A/B 中，全文地图仍增加了 26.6% token，没有缩短总耗时，并把 53 条 Episode 压缩为 36 条。部分压缩合理，但也出现事实归因和别名错误，因此当前只保留为可回退实验档。

第二轮已经实现并实测“仅在物理分片后触发、只允许原文词面锚点”的 `adaptive_anchor_map`。它消除了上一轮观察到的具体幻觉措辞，但仍使 token 增加 38.9%、提取拒绝增多，并把 48 条 Episode 压缩成 36 条。问题不再主要来自自由文本总结，而是来自**全文级先验本身会改变局部抽取边界和身份判断**。

因此当前推荐方案是：默认采用局部 `single_pass_audited`，把人物身份严格约束到每条 Episode 自己的 `evidence_spans`；全文派生数据只保留为检索导航或证据不足时的请求级辅助，不参与事实生成。`adaptive_anchor_map` 保留为默认关闭的研究档，方便以后测试“仅位置/时间导航、不传人物和语义锚词”的更窄版本。

## 名词

- **全文总结**：模型阅读一个逻辑文件后生成的自然语言概览。它是有损压缩，可能遗漏事实，也可能加入推断。
- **Document Map（文档地图）**：用于说明片段在全文中的位置、事件阶段和时间模式的派生导航数据。
- **导航上下文**：只帮助模型定位，不允许作为事实证据写入 Episode。
- **摘要锚定**：后续模型受先前摘要措辞影响，倾向于复制摘要的归因、边界或错误。
- **蕴含审计**：逐项检查 Episode 的主体、谓词、宾语、否定、时间、模态和归因是否由其原始 Source 行号证据支持。
- **提取拒绝**：模型返回内容未通过 JSON、行号范围、证据范围或覆盖约束，程序拒绝该次结果并重试。

## 安全边界与回退

实验没有修改现有数据库，也没有切换默认档。所有运行使用独立数据库和日志目录。

可用档位：

- `legacy`：原多阶段流程，仍是默认值。
- `single_pass_evidence`：单次 Episode 提取与行号证据。
- `single_pass_audited`：在行号证据后增加紧凑蕴含审计，不使用全文地图。
- `document_map_assisted`：地图阶段强制控制 Episode 数量和边界；实验证明过强，不推荐。
- `document_map_contextual`：地图只作上下文，不控制 Episode 数量和边界；本报告的实验组。
- `adaptive_anchor_map`：仅当一个逻辑文件被切成多个 Source 时生成逐字锚点；已实测，仍不推荐用于默认导入。

实验前快照位于 `validation/rollback/v3.45_pre_single_pass_20260831/`。上述档位均为请求级选择，切回 `legacy` 或 `single_pass_audited` 不需要数据库迁移。

embedding 在硬盘与内存中仍统一为 float32。embedding 模型始终是固定的 BGE-M3，只允许同模型重试，从不 fallback 到其他 embedding 模型。推理 fallback 只适用于 Episode、Concept、地图和审计等语言任务。

## 方案演化

### 自由文本全文总结

最初先生成一段全文概览，再把概览传给 Episode 提取器。它能说明场景处于全文哪个阶段，但没有纠正事实强化：模型仍把“没有其他选择”写成“接受了提议”，并新增原文没有的“早晨”。

结论：自由文本概览不能成为事实层，也不能代替原文审计。

### 强制结构化地图

地图返回每个 segment 的 `event_stages`、行号范围、时间模式和未决问题；程序要求每个阶段恰好产生一个 Episode。

结果是地图把样本粗分为 4 个阶段，而主模型连续两次只返回 1 条 Episode，最终依赖推理 fallback 才完成 4 条。强制阶段同时压缩了细节。

结论：地图不应拥有 Episode 边界的最终决定权。

### 上下文式结构化地图

新增 `document_map_contextual`：仍先生成结构化全文地图，但提示词明确要求后续提取器：

- 地图可能不完整或有误；
- 地图不是证据；
- 不服从地图的阶段数量；
- 独立依据 Source 决定 Episode 边界和 `evidence_spans`；
- 完成后只使用重建的原文证据做蕴含审计。

旧的强制地图档保持不变，因此该修改可直接回退。

### 自适应逐字 Anchor Map

新增 `adaptive_anchor_map`（`v4.7_adaptive_literal_anchor_audited`）：

- 单 Source 文件完全不生成地图；
- 只有一个逻辑文件实际被切成多个 Source 时触发；
- 不允许 `overview`、因果总结或结论；
- 只允许返回原文中逐字存在的 `anchor_terms`、`participants`，以及枚举化的 `role_in_document`、`time_mode`；
- 程序再次验证所有词面是否确实存在于原文，不存在就拒绝整张地图；
- Episode 仍必须提供自己的 `evidence_spans`，地图永远不是证据。

这验证了“去掉自由总结是否足以消除负作用”。实测结果表明：还不够。即使每个锚词都是真的，把其他 Source 的人物和主题提前暴露给局部提取器，仍会改变 Episode 边界并造成身份先验串扰。

## 单文件实验

样本：`000__主线剧情_最終編_第1章__segment-0000.txt`。

| 方案 | Episode | LLM 请求 | Token | 耗时 | 观察 |
|---|---:|---:|---:|---:|---|
| 单次证据，无审计 | 7 | 4 | 12,928 | 3:47 | 有“没有选择→接受”、残句等问题 |
| 自由文本地图 | 6 | 7 | 25,800 | 6:01 | 未修复强化，新增“早晨” |
| 无地图 + 蕴含审计 v4.4 | 5 | 5 | 15,398 | 3:16 | 修复主要强化，但仍有局部关系措辞问题 |
| 强制结构地图 + 审计 | 4 | 8 | 32,147 | 4:05 | 两次阶段覆盖失败并依赖 fallback |
| 地图仅作上下文 + 审计 | 5 | 6 | 20,603 | 3:50 | 无提取重试，关系表述更好，但成本增加 |

上下文式地图在该次单文件运行中正确保留了“老师是会长指定的人”和“出于对会长的义理”之间的关系，也没有把“没有其他选择”强化为已经接受。

## 五文件正式 A/B

冻结样本：`validation/serihu-quality-canary-v6-structural-corpus-20260831/`，共 5 个逻辑文件、7 个 Source。

共同条件：DeepSeek-V3.2 主推理、GLM-4.5V 推理 fallback、4 个准备 worker、float32 BGE-M3、关闭推断关系构建、关闭旧 factual audit、独立数据库。

| 指标 | 无地图 `single_pass_audited` | 地图仅作上下文 `document_map_contextual` |
|---|---:|---:|
| 状态 | completed | completed |
| 文件 / Source | 5 / 7 | 5 / 7 |
| Episode | 53 | 36 |
| Concept | 111 | 91 |
| Association | 159 | 124 |
| LLM 请求 | 39 | 43 |
| 提取拒绝 | 2 | 0 |
| 审计修订 Episode | 26 | 16 |
| prompt token | 85,015 | 121,043 |
| completion token | 41,801 | 39,454 |
| 总 token | 126,816 | 160,497 |
| 总耗时 | 25:53 | 26:02 |

相对变化：

- 总 token 增加约 26.6%；
- 总耗时增加约 8.5 秒，基本持平；
- Episode 减少约 32.1%；
- 提取拒绝从 2 次降为 0 次。

确定性数据库审计在两组中均未发现 embedding、外键、空内容、participant 完整性、Concept 输出或 alias collision 问题。

## 语义覆盖检查

使用数据库内 float32 Episode embedding，在相同 `source_key` 内做双向 cosine 最近邻。以 0.80 为严格观察阈值：

- 对照组 53 条中，20 条在地图组有不低于 0.80 的最近邻，覆盖率 37.7%；
- 地图组 36 条中，19 条在对照组有不低于 0.80 的最近邻，覆盖率 52.8%。

该指标不能直接等同事实 recall：一个较大的 Episode 可以覆盖多个小 Episode，却因文本粒度不同低于阈值。因此随后进行了人工检查。

### 合理收益

- 把连续会议发言、共同分工和同一解决方案合并，减少逐句碎片。
- 全文位置与时间模式更清楚。
- 五文件运行没有发生 Episode 结构/证据拒绝。
- 第一份样本中，对“老师是会长指定的人”的关系方向表述优于部分无地图运行。

### 风险与实际错误

- 在一个长会议片段中把 `アユム（步梦）`写成`亚瑠`。两个名字都可能在整个 Source 出现，因此“名称存在于 Source”这种全局 grounding 无法发现证据范围内的错误身份。
- 把阿罗娜对噩梦的回应压缩为“掩饰了自己梦中被枪击的事实”，原文并未明确支持这个更强结论。
- 把多个独立小组的任务、行动者和对象合并为一个很长的 Episode。虽然可读，单个 embedding 会平均掉局部检索锚点。
- 仍出现近重复 Episode，说明全文地图没有自动解决重复抽取。
- 地图自身的自然语言 `overview` 和 `hint` 会给提取器提供有偏先验；后续的蕴含审计能修复一部分，但不能保证发现所有摘要锚定。

## 技术判断

“全文先总结、再理解片段”的思想成立，但应该拆成两个不同职责：

```text
全文结构导航
    只回答：片段在哪、邻近什么、属于何种时间模式
    不回答：最终事实是什么

局部事实提取
    只从 Source 行号证据创建 Episode
    地图不能决定边界、身份或因果
```

目前 `document_map_contextual` 已实现这种隔离的第一版，但地图仍包含自然语言概览，所以不足以成为默认流程。

## 第一轮后提出的下一版（第二轮已验证）

采用证据驱动升级，而不是所有文件固定总结：

```text
局部单次提取 + 原文行号证据
        ↓
检查局部上下文质量
        ↓
若代词未解析 / 时间模式冲突 / 人物别名歧义 / 边界不稳定
        ↓
生成抽取式 Document Anchor Map
        ↓
只重做有问题的 Source
```

Anchor Map 不生成全文叙事摘要，只返回：

- segment 顺序；
- Source 行号范围；
- 从原文逐字复制的锚词和人物标记；
- `current / past / memory / reported / mixed / unknown`；
- 前后片段的引用关系；
- 未解析项，不给出自行补全的答案。

同时把 participant grounding 从“名字出现在整个 Source”收紧为“名字必须出现在该 Episode 的 `evidence_spans`，或拥有显式、已验证的 alias Association”。这可以直接发现 `アユム → 亚瑠` 一类错误。

上述方案随后实现为 `adaptive_anchor_map`。第二轮结果见下文：严格 participant 证据约束有效，全文人物/语义锚点直接进入局部提取则仍然没有净收益。

## 第一轮资产

- 上下文式地图档：`v4.6_document_map_context_audited`
- 单元测试：地图有两个阶段时，提取器仍可按 Source 只生成一个 Episode
- 引擎回归：270 项通过
- 机器人回归：159 项通过，13 项按环境跳过
- 数据库审计：
  - `validation/serihu-ab-v18-single-pass-audited-fullcanary-audit-20260831.json`
  - `validation/serihu-ab-v19-document-map-contextual-fullcanary-audit-20260831.json`
- 覆盖差异：`validation/serihu-ab-v18-v19-episode-coverage-20260831.json`
- 通用对比工具：`benchmarks/compare_episode_imports.py`

两个实验数据库只用于验证，不是机器人活动知识库。

## 第二轮：证据身份约束与自适应 Anchor A/B

冻结样本、模型、并发和 embedding 条件与第一轮相同。两组均启用更严格的人物证据约束：每条 Episode 声明的 participant 必须出现在该 Episode 自己的 `evidence_spans` 中，仅仅出现在同一 Source 的其他位置不再算作有效依据。

| 指标 | 严格局部证据 `single_pass_audited` | 自适应逐字地图 `adaptive_anchor_map` |
|---|---:|---:|
| 状态 | completed | completed |
| 文件 / Source | 5 / 7 | 5 / 7 |
| Episode | 48 | 36 |
| Concept | 103 | 87 |
| Association | 153 | 125 |
| LLM 请求 | 40 | 51 |
| Episode 提取拒绝 | 4 | 10 |
| 地图拒绝 | 0 | 1 |
| 外层任务重试 | 0 | 1 |
| 审计修订 Episode | 24 | 22 |
| prompt token | 93,334 | 144,218 |
| completion token | 46,316 | 49,745 |
| 总 token | 139,650 | 193,963 |
| 总耗时 | 32:05.90 | 30:52.58 |

相对变化：

- 自适应地图总 token 增加约 38.9%；
- 墙钟时间减少约 3.8%，没有形成足以抵消 token 和稳定性成本的速度收益；
- Episode 数量减少 25%；
- Episode 提取拒绝从 4 次增加到 10 次，并新增一次地图拒绝和一次外层重试；
- 地图只在确实发生 Source 分片的两个文件触发，触发条件本身符合设计。

同 `source_key` 内用 float32 Episode embedding 做双向最近邻，在 cosine 不低于 0.80 时：

- 对照组 48 条中，29 条能在地图组找到对应项，覆盖率 60.4%；
- 地图组 36 条中，27 条能在对照组找到对应项，覆盖率 75.0%。

覆盖率提高主要来自大 Episode 合并多个小事实，不能解释为检索质量提高。人工差异检查发现，地图组在会议和任务分配片段中把多个行动者、目标和结果合并成巨型 Episode；这种文本看起来完整，但会削弱单一事实的 embedding 锚点，并使后续 Association 难以精确连接。

### 严格证据约束带来的收益

第一轮的 `アユム（步梦）→ 亚瑠` 错误之所以能漏过旧检查，是因为两个名字都可能出现在整个 Source。新检查把范围缩到 Episode 自己引用的证据行，能够直接拒绝这种跨证据身份替换。

审计请求现在同时携带 participant，并明确检查：

- Episode 文本中的人物是否由证据行支持；
- participant 是否逐字出现在证据行，或能通过显式、已验证 alias 解释；
- 同文件其他位置出现的姓名不能替代本 Episode 的证据；
- 主体、谓词、宾语、否定、时间、模态和信息来源仍需逐项成立。

### 导入后处理暴露的通用缺陷

第二轮对照组还暴露了另一个与当前剧情无关的通用问题：片假名人物名可能被更长的损坏词串包含，例如 `マリナ` 被误认为出现在 `ヒマリナ…` 中，进而把错误人物补进 participant。

已作两层修复：

1. 对带 `evidence_quotes` 的新式 Episode，别名清理、姓名修复、文本清洗、人物 grounding 和 participant 补全全部只读取该 Episode 的证据文本，不再读取整个 Source；旧式、没有证据引用的草稿继续兼容原行为。
2. 日文平假名、片假名、韩文和拉丁字符分别采用文字系统边界，短名字不能在同文字系统的更长 token 内命中。

这不是针对某个角色写死规则，而是把“人物名出现”从裸子串判断升级为证据范围内、文字系统感知的实体匹配。

## 最终架构判断

实验否定的是“把全文总结或全文锚点直接喂给局部事实提取”这一实现，不是否定全文信息的价值。更合适的职责分离是：

```text
原始全文
  ├─ 局部 Source → Episode + evidence_spans → 严格事实审计
  └─ 派生导航索引 → 片段定位 / 时间模式 / 检索扩展

派生导航索引可以帮助找到证据，
但不能成为 Episode 的事实来源，也不能提供全局人物候选名单。
```

如果以后继续研究全文层，优先顺序应是：

1. 把地图用于查询时的 Source 路由，而不是导入时的 Episode 生成；
2. 若导入时确实需要，只传 segment 序号、原文行区间和时间模式，不传人物、别名、主题或因果；
3. 由局部提取先运行，只对代词未解析、时间模式冲突或证据覆盖不足的 Episode 请求级升级；
4. 用完整问答评测验证收益，不能再用 Episode 数量减少或提示词看起来更完整作为成功标准。

## 更新后的资产与验证

- 严格证据 participant 校验：`MemoryExtractor._single_pass_participant_evidence_errors`
- 自适应研究档：`v4.7_adaptive_literal_anchor_audited`
- 通用 A/B 对比工具：`benchmarks/compare_episode_imports.py`
- 第二轮数据库审计：
  - `validation/serihu-ab-v20-evidence-grounded-audited-fullcanary-audit-20260831.json`
  - `validation/serihu-ab-v21-adaptive-anchor-fullcanary-audit-20260831.json`
- 第二轮 embedding 覆盖差异：`validation/serihu-ab-v20-v21-episode-coverage-20260831.json`
- 记忆引擎完整回归：277 项通过
- 机器人集成回归：159 项通过，13 项按环境跳过

所有 A/B 数据库和日志均为隔离实验资产，活动机器人数据库未被修改。
