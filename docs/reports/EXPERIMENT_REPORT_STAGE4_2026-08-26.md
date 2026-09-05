# 第四阶段网络级长链联想实验报告（2026-08-26）

## 1. 结论

本阶段已经完成原计划中的七道高难长链问题、向量/静态图基线、增长图连续运行和三种题序扰动。

当前最稳妥的结论是：

> 在冻结的 11 文件封闭剧情语料内，系统能够把每题 7—9 组理论证据连接组织成经原文审计的答案；查询期 Association 增长能够安全写入、立即参与回答，并在后续问题中发生真实复用。题序会改变具体路径和精确片段覆盖，但三种增长顺序的答案正确性保持 7/7。

同时必须保留以下边界：

- 这不是全量蔚蓝档案剧情验收，只覆盖当前数据库中的 11 个文件。
- “7—9 组证据”是出题复杂度目标；实际新增、强化、复用或总边数从来不是评分指标。
- 精确 Episode ID 组用于诊断检索覆盖，不覆盖经 Source 原文增强与答案审计后的语义判断。
- 三种题序各只运行一次，能够发现顺序效应，但不能把相关性直接宣布为因果。
- 尚未验证数百次连续查询、全量剧情和长期边累积后的稳定性。

## 2. 冻结数据资产

所有评测均从同一个只读基线复制独立数据库，查询增长没有写回正式库。

| 项目 | 值 |
|---|---:|
| 数据库 | `validation/ba-stage3-deep-v37.db` |
| SHA-256 | `7f94a12d70fe69a54b0efa85f2bfa639c1d8380c03a705de9599ccc32ea44ea9` |
| Source | 68 |
| Episode | 342 |
| Concept | 392 |
| Association | 1,825 |
| Embedding | 1024 维 float32 |
| Episode/Concept BLOB | 每条 4,096 bytes |

实验结束后哈希仍与实验前完全一致。

## 3. 当前参数与可复用经验

### 3.1 Source 分片

```text
target_chars = 6000
max_chars = 8000
overlap_chars = 800
minimum_blocks = 2
```

这些值仍是当前正式资产。`max_chars` 是优先自然记录边界的软上限，不是强行截断；重叠按完整记录回带，因此可能略高于目标。Stage 4 没有重新导入语料，沿用 Stage 3 已审计分片。

### 3.2 Embedding 与检索

```text
embedding_model = Pro/BAAI/bge-m3
embedding_dimension = 1024
SQLite dtype = float32
RAM dtype = float32

episode_top_k = 40
concept_top_k = 20
candidate_limit = 600
graph_max_hops = 3
graph_beam_width = 20
growth_max_rounds = 2
growth_episode_limit = 40
answer_episode_limit = 30
answer_concept_limit = 20
answer_path_limit = 24
whole_question_anchor_episodes = 10
atomic_anchor_episodes_per_query = 6
source_excerpt_chars = 3000
```

回答审计实际为每个入选 Episode 再携带最多 2,400 字 Source 摘录。当前仍没有证据要求 ANN、float8 或 float16；本阶段统一使用 float32。

### 3.3 v3.21 查询提示词

`prompt_version = v3.21_atomic_coverage_source_audit`

本版本最有价值的提示词变化：

1. 查询解析要求生成 3—8 个原子证据问题，明确覆盖问题后半段，不能只拆前几个槽位。
2. 后续检索规划逐项比较尚未解决槽位，提示词要求 2—6 个下一跳，代码最终最多保留 4 个。
3. 初始锚点与后续锚点交错进入证据预算，防止第二跳证据位于列表尾部而被 30 条上限截掉。
4. 回答必须按原问题语言输出，并区分事实、综合推论、角色假说和未知。
5. 回答与答案审计都能查看精确 `source_key`；审计还查看 Source 原文摘录，避免把相邻 Source record 的正确证据误判为缺失 Episode。
6. Association 只能作为待验证联想，不能覆盖端点原文；跨场景政治支援只能降级为历史背景或政治铺路，具体渗透机制必须保持未知。

完整提示词保存在 `src/memory_demo/llm/prompts.py`，不在报告中复制大段文本，避免报告与代码双份漂移。

## 4. 评测口径修正

本阶段中途发现原评分器把若干展示/诊断条件当成硬失败。经实际证据检查后，当前硬性通过条件为：

- 问题状态 completed；
- 核心实体/概念出现；
- `evidence_episodes` 至少保留结构化来源；
- 所有证据来源位于封闭语料白名单；
- 最后一轮答案审计 valid；
- 模式行为正确：vector 不使用图、static 使用已有图、growing 在发生改变时使用改变边。

以下内容完整保留为诊断，但不单独决定答案正确性：

- 是否精确命中预先指定的 Episode ID 组；
- 是否命中每个预先指定文件，而非同一事实的等价来源；
- 回答正文是否打印 `source_key`；
- 正文是否逐字出现“事实/推论/未知”标签。

原因是 Episode 可通过 `source_id` 回溯到相邻 Source record，同一事实也可能在另一允许文件中完整重述；结构化证据已经保存 `source_key`，正文漏打印文件名属于展示问题，不等于系统没有证据。

这不是放宽事实标准。越界来源、缺少核心结论、审计无效、未使用改变边等仍然硬失败。精确片段覆盖在下面单独报告，所有缺组都可见。

## 5. 总体结果

### 5.1 版本基线

| 版本/模式 | 语义与证据通过 | 精确片段诊断 |
|---|---:|---:|
| v3.20 vector_only | 6/7 | 47/56 |
| v3.20 graph_static | 7/7 | 45/56 |
| v3.21 vector_only | 7/7 | 49/56 |
| v3.21 graph_static | 7/7 | 44/56 |
| v3.21 graph_growing 原顺序 | 7/7 | 52/56 |
| v3.21 graph_growing 反向顺序 | 7/7 | 50/56 |
| v3.21 graph_growing 交错顺序 | 7/7 | 51/56 |

v3.21 的原子槽覆盖修复使 vector 从 6/7 提升到 7/7。静态图的精确覆盖从 45 降到 44，说明“已有图”本身并不稳定提升检索，图端点可能挤占固定 30 条证据预算。三次增长图均高于本次静态图，但仍不能用一次实验断言增长对每题必然更好。

### 5.2 v3.21 每题精确片段诊断

| 问题 | 要求组 | vector | static | growth 原顺序 | growth 反向 | growth 交错 |
|---|---:|---:|---:|---:|---:|---:|
| Q1 义务到新机构 | 7 | 7 | 6 | 7 | 7 | 7 |
| Q2 古圣堂象征/设施/边界 | 8 | 8 | 7 | 8 | 6 | 8 |
| Q3 阿里乌斯两位资助者 | 8 | 6 | 7 | 6 | 6 | 7 |
| Q4 未花动机四层 | 8 | 7 | 7 | 8 | 7 | 7 |
| Q5 日富美/梓身份与联盟 | 8 | 7 | 5 | 7 | 8 | 7 |
| Q6 日奈职责与疲惫 | 9 | 6 | 6 | 8 | 8 | 7 |
| Q7 妃咲/未花治理类比 | 8 | 8 | 6 | 8 | 8 | 8 |
| 合计 | 56 | 49 | 44 | 52 | 50 | 51 |

增长顺序的 50—52/56 波动小于静态/增长之间的差异，但样本量仍然很小。Q3、Q4、Q5 的一组波动尤其像生成与选证随机性，而不是固定的图收益。

## 6. Association 自主增长得到的实证

### 6.1 安全性

| 题序 | 实际改变边的离线审计覆盖 | 安全通过 | 发生历史复用的题 |
|---|---:|---:|---|
| 原顺序 | 49 | 49 | Q2、Q5、Q6、Q7 |
| 反向 | 47 | 47 | Q4、Q3、Q1 |
| 交错 | 52 | 52 | Q7、Q2、Q6 |

这里的 49、47、52 只表示“逐条检查了多少条实际改变边”，不是成功分数，也没有越多越好的含义。三份数据库均满足：

- `audit_status = dual_accepted`；
- claim level 属于 `direct_fact / supported_inference / historical_context`；
- `evidence_json` 包含两端证据；
- `audit_json` 同时保存主审和对抗审计接受结果；
- 当前题改变的边进入最终回答路径；
- 未发现未经审核边、身份幻觉或被升级成确定因果的历史联系。

### 6.2 后续查询确实复用了旧边

原顺序最早的明确实例是 Association 1828：

```text
Q1 创建：古圣堂/戒律守护者的历史作用
        → 为原 ETO 的权威性提供历史和理念背景
Q2 使用：回答古圣堂的象征、制度与基础设施边界
```

Q5 又复用了 Q1/Q2 形成的 1829、1831、1837；Q6 复用 1826、1835；Q7 复用 1838、1842、1850。交错和反向顺序也发生复用，但使用的是各自历史中不同的边。

这证明的是：

- 持久化的新关系能进入后续检索路径；
- 同一问题在不同历史下可以调用不同替代路径；
- 网络不是只在当前题临时生成后被丢弃。

它没有证明：

- 复用次数越多越好；
- 某次覆盖提升一定由复用边导致；
- 所有被复用边长期都有净价值。

### 6.3 最强的顺序效应线索

Q2 是当前最值得重复验证的结果：

- 原顺序：Q1 在前，Q2 复用 Q1 的历史桥，8/8；
- 交错顺序：Q1 在前，Q2 同样复用 Q1 历史桥，8/8；
- 反向顺序：Q2 在 Q1 之前，无相关旧边复用，6/8。

这支持“Q1 形成的古圣堂/ETO 关系帮助 Q2 完成证据覆盖”，但只有两个处理组与一个对照组，且 LLM 生成非确定，因此仍需重复运行才能作因果判断。

### 6.4 本题内增长也有独立价值

Q6 的 vector/static 都是 6/9；原顺序和反向增长均达到 8/9。反向 Q6 没有复用先前边，说明查询内部建立的关系就能补齐职责、疲惫、星野比较和老师回应的长链。交错为 7/9，显示该收益也有波动。

Q5 则是反例：原顺序复用旧边得到 7/8，反向不复用得到 8/8，交错不复用得到 7/8。因此不能把“发生复用”和“覆盖提高”自动绑定。

## 7. 故障恢复与安全降级

电脑异常后完成了以下恢复：

1. 评测器增加 `--resume`，跳过 completed 题并重试 running/failed 题。
2. 报告记录并校验 `question_order`，避免用不同题序错误续跑同一数据库。
3. 支持 `original / reverse / interleaved` 三种题序，每种使用独立数据库副本。
4. HTTP 客户端把 `RemoteDisconnected`、`ConnectionError` 和 `HTTPException` 包装成统一模型错误，进入既有重试/回退流程。
5. SQLite 与 JSON 报告均在整题边界保存断点；冻结源库只读。

v3.21 五组运行日志汇总：

| 项目 | 数值 |
|---|---:|
| 日志事件 | 1,203 |
| 模型请求 | 371 |
| 正常响应 | 364 |
| 记录的 read timeout | 5 |
| 自动重试 | 5 |
| fallback | 0 |
| 因审核失败而拒绝的增长批次 | 1 |
| JSONL 解析错误 | 0 |

反向 Q3 的增长关系审计连续三次 300 秒超时。系统没有绕过审计，而是：

```text
growth audit timeout
→ validation_failed
→ 本题新建边 = 0
→ 保留已有已审计网络
→ 复用旧边完成回答
→ 答案审计通过，精确片段 6/8
```

这验证了 fail-closed 的真实运行效果。日志中的两个“请求无最终 outcome”来自电脑/进程中断前的历史片段；最终五份正式报告均为 completed。

延迟方面，answer 请求 p50 约 112 秒、p95 约 144 秒。demo 当前不要求延迟，但 Telegram 实际上线前必须异步化并提供处理中状态。

## 8. 实验中发现并修复的问题

### 8.1 第二跳锚点被尾部截断

初始锚点先占满列表时，follow-up 找到的关键 Episode 可能排在尾部并被 30 条证据预算裁掉。v3.21 改为初始/后续锚点交错，Q1 从缺最后一跳恢复为 7/7。

### 8.2 查询提示词只覆盖问题前半段

改为 3—8 个原子查询，并显式要求五个以上槽位继续拆分；后续规划优先未解决和后半段槽位。

### 8.3 只看 Episode 摘要会误判相邻原文证据

答案审计现在能看到最多 2,400 字 Source 摘录，评分器也把精确 Episode ID 降为诊断，避免“证据就在同一 Source record，但自动分段 ID 不同”的假阴性。

### 8.4 正文漏打印 source_key 被误判为联想失败

结构化 `evidence_episodes` 已保留 `source_key/source_id`。正文打印文件名仍用于人类可读性诊断，但不再覆盖结构化溯源和答案审计。

### 8.5 `historical_context` 被离线审计器错误拒绝

运行时 schema、类型和增长门一直允许三种安全 claim level：

```text
direct_fact
supported_inference
historical_context
```

离线 Stage 4 审计器漏掉了 `historical_context`，导致一条明确否认具体因果的安全边被误报。已补齐白名单并添加回归测试。修复后交错数据库 52/52 全部通过。

## 9. 当前仍存在的问题

### A. 题序效应需要重复实验

Q2 的 8/8、8/8、6/8 很有价值，但不足以证明因果。下一轮最有效的实验不是再设计更多题，而是只跑两题序列：

```text
处理组：Q1 → Q2
对照组：Q2 → Q1
每组至少重复 3 次，每次从冻结库复制
```

比较 Q2 的事实槽、精确片段诊断、使用路径和 Q1 边复用情况。边数量仍不作为指标。

### B. 静态图会挤占 30 条回答证据预算

Q5 static 只有 5/8，Q7 static 6/8。图路径端点有时把更精确的向量锚点挤出答案证据。建议在新版本做独立 A/B：`answer_episode_limit = 30` 与 40；不要修改本轮结果。

### C. 答案审计仍与回答模型高度相关

回答和主要答案审计均依赖 DeepSeek，存在相关错误同时漏过的风险。增长边已有 GLM 对抗复核，但答案层还没有同等级独立审计。下一阶段可对高风险身份/跨事件因果题增加 GLM 二审，或先生成结构化 claim 再作确定性来源校验。

### D. Concept 跨索引联想尚未被本题集独立验证

当前查询同时搜索 Episode 与 Concept，但七题主要评估 Episode 证据链和 Association 网络。多语言 Concept alias、Concept→Episode→Concept 路径对答案的独立增益仍缺专门对照。

### E. 长期网络质量尚未验证

需要连续几十到几百次查询的 soak test，观察：

- 错误边是否累积或被重复强化；
- 低价值局部边是否淹没有用长桥；
- 后续复用是否持续发生；
- 边被否定、人工修订或 Episode 修改后的重新审计；
- 检索路径是否随网络增大而漂移。

### F. claim level 枚举存在代码重复

本次离线白名单漂移说明 schema、类型、运行时和工具各自复制枚举容易出错。后续应把允许值集中成单一常量，审计器读取同一来源。

### G. 人类可读引用仍可改进

结构化溯源已经正确，但少数回答正文没有打印文件名。可以在回答生成后用确定性渲染器附加“使用的 Source”清单，而不是继续依赖 LLM 遵循格式。

## 10. 当前资产索引

### 设计与评分

- `TEST_PLAN_STAGE4_NETWORK.md`：正式题集目标、口径和执行方式。
- `validation/evaluation-questions-stage4-network.json`：七道问题。
- `validation/stage4-network-evidence-manifest.json`：允许来源、核心词和 56 个证据组。
- `benchmarks/score_stage4_network.py`：语义硬门与精确覆盖诊断。
- `benchmarks/audit_stage4_growth.py`：逐边来源、双审计、claim level、实际使用和历史复用检查。
- `validation/stage4-v321-experiment-summary.json`：本报告的机器可读指标摘要。
- `validation/stage4-v321-log-summary.json`：五组 v3.21 JSONL 日志汇总。

### 正式运行结果

- `validation/evaluation-stage4-network-v321-vector/`：v3.21 vector_only，49/56。
- `validation/evaluation-stage4-network-v321-static/`：v3.21 graph_static，44/56。
- `validation/evaluation-stage4-network-v321-growing-original/`：原顺序，52/56，增长审计 49/49。
- `validation/evaluation-stage4-network-v321-growing-reverse/`：反向，50/56，增长审计 47/47。
- `validation/evaluation-stage4-network-v321-growing-interleaved/`：交错，51/56，增长审计 52/52。
- `validation/evaluation-stage4-network-v320-retrieval/`：v3.20 对照。

每个增长目录都包含：

```text
evaluation-report.json
scorecard.json
growth-audit.json
graph_growing.db
logs/
```

### 本阶段代码改动

- `src/memory_demo/config.py`：v3.21 prompt version。
- `src/memory_demo/llm/prompts.py`：原子槽覆盖、后半段查询、Source 引用和审计提示。
- `src/memory_demo/retrieval/engine.py`：follow-up 锚点交错、Source 摘录审计。
- `src/memory_demo/llm/client.py`：断连/HTTP 异常统一重试。
- `src/memory_demo/evaluation.py`、`src/memory_demo/cli.py`：断点续跑和题序扰动。
- `tests/test_stage4_network_scorer.py`：评分口径回归。
- `tests/test_stage4_growth_audit.py`：historical_context 审计回归。

## 11. 验证状态

```text
自动化测试：66 passed / 0 failed
正式基线 SHA-256：未变化
所有计划内评测：completed
所有增长副本逐边审计：passed
正在运行的 API 调用：无
```

## 12. 推荐下一步

当前没有必须由开发者解决的技术阻塞。下一步是研究优先级选择，而不是无法继续：

1. **推荐：先做 Q1→Q2 与 Q2→Q1 的三次重复对照。** 这是目前最接近验证“旧 Association 对后题有因果贡献”的线索。
2. 然后做回答证据预算 30 vs 40 的固定 A/B，观察 static Q5/Q7 是否恢复，同时确认更长上下文不会降低答案审计质量。
3. 再设计 30—100 次连续查询 soak test，评价网络长期增长、复用、冗余和错误累积。
4. Concept 跨语言联想与 Telegram 异步体验放在上述机制验证之后。

不建议此时引入 ANN、float8、记忆衰减或复杂专家系统结构；当前最有价值的未知仍是“安全历史关系是否在重复对照中稳定改善后续联想”。
