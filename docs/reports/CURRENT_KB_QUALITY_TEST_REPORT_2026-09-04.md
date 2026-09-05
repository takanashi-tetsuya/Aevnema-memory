# 当前剧情知识库质量测试报告

日期：2026-09-04  
测试对象：`chat_bot/data/knowledge/blue_archive.db`  
语料：`人类可读剧情知识库_20260901/可导入文本`  
状态：导入已按要求安全暂停，未完成文件保留在可恢复进度账本中。

## 1. 测试范围

本次测试分为四类：

1. 导入进度与数据库结构一致性；
2. embedding、SQLite、全文索引和来源追溯完整性；
3. 新版提取结果与旧版污染隔离；
4. 五个跨文件剧情问题的标准检索，以及 reranker 开关对照。

本报告不把“测试脚本超时”当成知识库失败。此前的 3 秒探针由于 deadline 过短，全部在召回完成前结束；本次实际对照使用 20 秒检索上限。

## 2. 导入状态

语料目录共发现 215 个支持文件：

| 状态 | 数量 |
|---|---:|
| 已完成 | 188 |
| 当前中断文件 | 1 |
| 尚未开始 | 26 |
| 失败 | 0 |
| 部分完成 | 0 |

因此当前数据库只代表约 87.4% 的文件，不能视为全量语料验收结果。进度文件为：

`chat_bot/data/knowledge/import-progress-human-readable-20260902.json`

恢复导入时，中断文件会先回滚该 `source_key` 的派生记录，再重新处理。

## 3. 数据库结构检查

当前数据库统计：

| 表 | 数量 |
|---|---:|
| Source | 4,017 |
| Paragraph | 7,019 |
| Episode | 13,764 |
| Concept | 10,675 |
| Association | 42,689 |

检查结果：

- `PRAGMA integrity_check` 返回 `ok`；
- Episode、Paragraph、Concept 的 embedding 均为 4,096 字节，即 float32 × 1024 维；
- 没有空 Episode 或空 Paragraph；
- Episode 和 Paragraph 没有孤立的 `source_id`；
- `source_fts`、`source_bigram_fts`、`episode_fts`、`episode_bigram_fts` 行数分别与 Source/Episode 完全一致；
- Source 长度为 122–6,469 字符，平均 1,224.89 字符；
- Episode 长度为 3–666 字符，平均 65.11 字符；
- Paragraph 长度为 122–1,578 字符，平均 802.67 字符。

结论：持久化、embedding 和索引层通过结构验收。

## 4. Association 图检查

当前 Association 的全部 42,689 条都是：

`Episode -> Concept / relation_key=involves / relation_type=semantic`

其中 generation=0 有 42,625 条，generation=1 有 64 条；全部标记为 `direct_fact`、`not_required`、正向关系。

这与本次导入命令中的 `--defer-inference-relations` 一致。也就是说，当前图能支持“Episode 涉及哪些 Concept”和按 Concept 扩展候选，但还没有导入阶段生成的：

- Episode → Episode；
- Concept → Concept；
- 带有完整因果/雇佣/身份关系的推理边。

因此，跨文件联想的当前上限主要由 Episode/Concept 的共同出现和查询阶段扩展决定，不能把本次 Association 数量理解为已经完成知识图谱构建。

## 5. 新旧提取质量

### 5.1 机器伪标记

使用通用模式 `[A-Z]{3,}[_-][0-9*]+` 扫描所有派生文本：

| 表 | 含伪标记的行 | 出现次数 | 新运行（run >= 140） |
|---|---:|---:|---:|
| Episode | 173 | 255 | 0 |
| Concept | 106 | 382 | 0 |
| Association 文本 | 71 | 71 | 0 |

所有命中均来自旧运行（run ≤ 57），典型形式包括 `SPERKER_003`、`SPREAKER_001` 等。当前 v5/source-scoped 提取没有产生新的伪标记，说明新版“模型只输出普通文本、程序负责结构化”的路径有效；旧污染仍需在导入完成后按来源键重导入清理。

### 5.2 新版参与者绑定

当前新版运行（run >= 140）产生 3,044 个 Episode：

- 137 个没有 participants；
- 其中 43 个以“她/他/其”等代词开头，约 1.4%；
- 这些记录大多来自单角色好感剧情，文本本身仍可读，但检索时自包含性较弱；
- 新版没有出现空文本或机器标记。

这是当前最明确的导入质量改进点：可以用通用的“单一 speaker 标签确定时补全参与者”程序逻辑处理，而不应加入具体剧情人物规则。

另有一个旧运行的纯标点 Episode：`！”。`（event_810，run 94）。它不是新版产生，应在旧来源重导入时删除。

## 6. 语义 embedding 探针

一次批量生成 12 个中、英、日、韩及省略上下文问题的 embedding，耗时约 1.34 秒。Episode/Concept 最高余弦相似度示例：

| 问题 | Episode top | Concept top |
|---|---:|---:|
| 白子人物特征（中文） | 0.723 | 0.571 |
| 补课部真正原因 | 0.653 | 0.641 |
| Hoshino 人物特征（英文） | 0.603 | 0.510 |
| ミカとナギサ初次相遇（日文） | 0.621 | 0.534 |
| 省略上下文的“为什么这么做” | 0.602 | 0.603 |
| 早上好（无知识问题） | 0.714 | 0.644 |

结论：多语言向量可正常工作，但纯向量分数不能判断“是否真的有答案”。闲聊/无知识问题也会得到较高相似度，必须依靠 sparse、证据覆盖、来源上下文和回答阶段审计，而不能仅用 top score 做有无证据判定。

## 7. 五题标准检索

测试配置：standard、candidate_limit=30、graph_hops=1、paragraph retrieval 开启、evidence slots 开启、检索上限 20 秒。

### 7.1 结果摘要

| 问题 | 关键证据情况 | 结论 |
|---|---|---|
| 补课部真正原因 | 命中“补课部”“内鬼”，未在前 8 条证据中命中“叛徒” | 基本可答，词形有差异 |
| 乐园悖论提出者与危机 | 命中“悖论”，但精确词“乐园悖论”和“圣亚”未进入前 8 条证据 | 证据缺槽 |
| 未花与阿里乌斯/政变 | 命中“阿里乌斯”“未花”，未命中“政变” | 因果动作缺槽 |
| 古圣堂袭击 | “巡航导弹”“阿里乌斯”“未花”均命中 | 当前五题中最完整 |
| 情侣专用菜单→社长→代号 | 命中“情侣专用菜单”，未命中“阿鲁”“浮士德” | 跨实体链不完整 |

### 7.2 数据层原因

这五题中至少两项不是单纯的 reranker 排序问题：

1. 当前语料使用 `爱露`、`アル`、`阿露` 等形式，没有 `阿鲁` 这个别名；对应 Concept 也没有 `阿鲁` alias。
2. “情侣专用菜单” Episode（`10788`）关联的是 `アル` 和菜单概念；“浮士德” Episode（`3814`、`3815`）明确描述的是日富美/ヒフミ作为覆面水着团领袖。当前数据库没有一条直接证据说明“阿鲁以浮士德代号被雇佣”。因此系统没有自动把这两个实体强行合并是正确的保守行为，测试问题本身需要先由人工确认语料是否真的包含该关系。
3. “乐园悖论”在当前 Episode 文本中没有精确字符串，只有相近的“关于‘乐园’的悖论”；提出者在当前片段中被提取为花子，而不是问题预设的圣亚。这个差异应归入语料事实/摘要版本核对，不应靠提示词硬编码修正。

## 8. Reranker 对照

同一五题、同一候选预算下：

| 配置 | 平均耗时 | 最大耗时 | 关键证据词覆盖 |
|---|---:|---:|---:|
| BGE reranker 开启 | 12.05 秒 | 12.35 秒 | 9/15（60%） |
| BGE reranker 关闭 | 10.40 秒 | 10.96 秒 | 9/15（60%） |

本样本中 reranker 增加约 1.65 秒，但没有提高关键证据词覆盖，也没有修复“阿鲁→浮士德”或“乐园悖论→圣亚”缺槽。它仍可能改善候选排序，但不能替代：

- 别名归一化；
- 事实槽覆盖选择；
- 跨 Episode/Concept 的关系边；
- 事实冲突检测。

## 9. 当前验收结论

### 已通过

- SQLite 数据库完整性；
- float32 1024 维 embedding 存储；
- Source→Paragraph/Episode 来源追溯；
- FTS 与 bigram FTS 同步；
- 新版提取不再产生旧式 speaker 伪标记；
- 补课部原因、古圣堂袭击等问题已有可用证据；
- 多语言 embedding 请求可以正常完成。

### 尚未通过或不能判定

- 全量语料尚未导入（26 个文件未开始，1 个中断）；
- 旧运行产生的伪标记尚未清理；
- Association 仍以 Episode→Concept 为主，无法证明复杂推理边已建立；
- 代词开头 Episode 仍有 43 条新版记录；
- “阿鲁→浮士德”和“乐园悖论→圣亚”测试无法仅靠当前证据闭环；
- 当前 reranker 没有在五题小样本中带来可见覆盖收益。

## 10. 建议的后续顺序

1. 恢复进度文件，完成剩余 27 个文件导入；
2. 对旧 run ≤57 的 18 个污染来源键执行回滚并按当前 v5 profile 重导入；
3. 增加通用的单 speaker 参与者补全，并只重导入受影响来源；
4. 重新运行本报告的结构检查和五题检索；
5. 再决定是否保留 reranker：若目标是最低延迟，当前五题结果支持 standard 默认关闭；若后续覆盖选择器能利用其排序，再重新评估。

详细机器结果保存在 `chat_bot/experiments/results/`：

- `structure-audit-current.txt`
- `fts-audit-current.txt`
- `token-audit-all-current.txt`
- `semantic-evidence-current.json`
- `custom-standard-current.json`
- `custom-standard-no-reranker-current.json`
- `anchor-coverage-current.json`

另外，私人/公共域隔离烟测通过 6/6 项检查（同一原生用户 ID 在 Telegram 与 Discord 下使用不同数据库；私人事实不会泄漏到另一用户；公共事实可共享）。临时烟测报告位于：

`chat_bot/experiments/results/domain-smoke-current/20260904T105440.228072Z/report.json`
