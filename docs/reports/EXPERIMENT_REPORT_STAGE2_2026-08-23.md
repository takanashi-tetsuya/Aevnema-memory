# Associative Memory Demo 第二阶段联想实验报告

日期：2026-08-23  
语料：Blue Archive `main / favor / event` 直接证据子集  
当前代码提示词版本：`v3.4_cross_scope_chronology_guard`  
正式无增长基准库：`validation/ba-stage2-direct-k4.db`  
最新真实 API 增长库：`validation/evaluation-stage2-growth-v33-q3/graph_growing.db`

## 1. 阶段结论

这轮已经验证了 Demo 最核心的两种能力，并找到了它们的边界：

1. **跨文件 Episode 联想可以工作。**“老师的义务→乐园悖论”能跨 `main/31010.json` 与 `main/33030.json` 被召回并综合；“山海经待客→伊甸条约会议”能跨 `event` 与 `main` 建立对比。
2. **缺失证据可以被正确识别。**阿露与便利屋有直接证据，但当前语料没有“便利屋受凯撒雇佣”的直接证据；系统明确中止证据链，没有用模型作品知识补洞。
3. **查询期 Association 自主增长已经从“完全不长”调到“有证据才长”。**v2.9 三题均为 0 边；v3.0 一次长出 18 边但有污染；v3.1 收敛为 `[4, 0, 6]`，缺证据题保持 0 边；v3.2 修正对比关系 polarity；v3.3 保证新路径的 Episode 端点进入回答证据。
4. **图已经能保存解释性桥，但“跨查询复用”尚未贯通到最终回答。**第三题把 `Episode 246 ↔ Episode 84` 固化为“生活化款待 vs 制度化谈判”的主题对比；复述查询的离线重放也确实遍历到这些边，但它们被高权重 `involves` 路径和候选预算淘汰，没有进入最终证据。
5. **当前端到端瓶颈是 LLM，不是 NumPy。**47 个 Source 的导入耗时 6505 秒；embedding p50 仅 2.49 秒，而 Concept/关系请求存在 200—300 秒长尾。回答提示词达到 56k—66k 字符，是查询延迟和成本的主要来源。
6. **`float32 + NumPy brute-force` 仍适合 Demo。**100 万 × 1024 单索引约 3.815 GiB，本机实测 Top-100 约 86 ms；此时引入 ANN 不会解决当前真正的 LLM 长尾问题。

阶段判断：系统已经证明“跨片召回、证据边界、解释性 Association 增长”三件事都能成立；新一轮还证明了旧边会被图遍历和部分强化，但尚未证明它能稳定改善后续回答。下一阶段应优先修复强化信号传递、查询相关路径排序、语义边去重和证据压缩，而不是继续增加节点/关系类型。

## 2. 正式语料与导入结果

### 2.1 直接证据范围

正式库只导入 7 个文件，避免把模型外部常识或无关剧情混入首轮判定：

```text
main/31010.json
main/33030.json
main/33050.json
main/33060.json
main/33190.json
favor/10000/100002.json
event/10038/1003805.json
```

配置：

```text
Source target_chars:       6000
Source max_chars:          8000（软上限）
Source overlap_chars:       800
minimum_blocks:               2
prepare_workers:               8
relation_batch_size:          24
Episode relation candidate k:  4
Embedding dtype:          float32（硬盘与内存一致）
Embedding dimension:         1024
```

预检得到 47 个 Source。正式导入结果：

| 文件 | Source | Episode | 二次修订 | 最终失败 | 用时 |
|---|---:|---:|---:|---:|---:|
| `main/31010.json` | 2 | 7 | 0 | 0 | 4m28s |
| `main/33030.json` | 12 | 63 | 12 | 0 | 17m30s |
| `main/33050.json` | 6 | 34 | 7 | 0 | 12m13s |
| `main/33060.json` | 8 | 43 | 9 | 0 | 13m54s |
| `main/33190.json` | 12 | 59 | 23 | 0 | 42m50s |
| `favor/10000/100002.json` | 2 | 9 | 0 | 0 | 4m28s |
| `event/10038/1003805.json` | 5 | 31 | 5 | 0 | 13m03s |
| **合计** | **47** | **246** | **56** | **0** | **108m25s** |

数据库最终基线统计：

```text
Source:       47
Episode:     246
Concept:     330
Association: 1336
```

Episode/Source 最小 2、平均 5.23、最大 12；56/246（22.8%）Episode 进入了第二遍理解。Embedding BLOB 统一为 4096 bytes，重复 Episode 对为 0。

Association 分布：

| 类型 | 数量 |
|---|---:|
| semantic | 1012 |
| temporal | 237 |
| identity | 65 |
| recall_trigger | 9 |
| causal | 7 |
| interpersonal | 3 |
| co_occurrence | 3 |

### 2.2 导入时发现的工程问题

- `main/33050` 的 Episode 91 被模型写成 `timeline_scope="past"`。现已规定 scope 完全由 `source_key` 决定，回忆只写入 `story_time_text`；基线库已备份后修复为 `main`。
- 最终三个 scope 的 temporal 图均无环。Association 511 的 `same_time_as` 是合法非排序时间边，审计器现已单独列为 `non_ordering_temporal_edge_ids`，不再误报为未知边。
- 导入日志有 1 次 `episode_relations_batch` 校验失败：模型遗漏 Episode 196、204 的 group；后续降级路径完成处理，最终任务失败为 0。
- 原 runner 把全语料 564 个候选文件写成 `selected_files`，即使本轮只 limit 到 7 个。代码现改为同时记录 `eligible_files` 与真正的 `selected_files`；历史 ledger 中的 564 应按“可选总数”解释。

## 3. API 与性能统计

正式导入日志为 33.57 MB、3191 个事件，JSON 解析错误为 0。

```text
LLM/API request:  402
success response: 369
llm_error:         33
retry:             33
fallback:           6
HTTP 429:          25
read timeout:       8
最终失败任务:       0
```

请求延迟（包含失败请求到 error 的耗时）：

| 类别 | 次数 | p50 | p95 | 最大 |
|---|---:|---:|---:|---:|
| embedding | 138 | 2.49s | 5.26s | 7.20s |
| Episode 提取 | 47 | 37.99s | 81.37s | 187.33s |
| Concept 提取 | 89 | 77.52s | 300.82s | 301.42s |
| 关系判断 | 59 | 42.86s | 202.36s | 301.28s |
| 第二遍理解 | 28 | 12.37s | 44.68s | 58.46s |

`workers=8` 确实让互不依赖的 Source 准备并发，但供应端仍会在高负载时拒绝请求。甚至正式评测的单线程查询也出现 2 次 429，所以“允许任意多线程”不等于服务端能无限承载。模型客户端现对 429 使用 8 秒起步、指数增长、0—4 秒抖动的退避，并读取 `Retry-After`；v2.9 导入日志产生在该改动前，因此历史 retry 事件没有等待秒数字段。

关系候选从 k=7 改为 k=4 的依据：

```text
k=7 scaling probe:
  38 Episode / 239 Association
  最大关系 prompt 约 24,436 字符
  多次 3—5 分钟尾延迟

k=4 formal import:
  246 Episode / 1336 Association
  关系 prompt p50 3,827；p95 14,589；max 14,904 字符
```

k=4 把最大关系提示缩短约 39%，同时三道测试所需联想仍然可建立，因此继续作为当前默认值。

查询回答仍过重：v2.9 九次 A/B/C 的 answer prompt 为 56k—66k 字符，回答请求 p50 75.75 秒。v3.3 单题 answer prompt 为 66,152 字符，回答耗时 84.11 秒。下一轮性能优化应先压缩 Source excerpt、按问题选择证据字段、对路径做摘要，而不是优化 1—2 秒的 query embedding。

## 4. 三道正式测试结果

### 4.1 老师义务与乐园悖论

结果：**通过。**

关键证据：

- Episode 7，`main/31010.json`：圣娅要求老师不要移开视线并见证到最后，这是做出选择后的老师义务。
- Episode 62—65，`main/33030.json`：再次引用第五条公案，明确“抵达乐园者无法在外部被观测”，并把不可证明性转向“只能选择相信”。
- Episode 66—67：花子指出老师在充满怀疑的故事中从未怀疑学生；老师说明不相信学生就无从开始。

纯向量模式已经同时召回两个远隔文件并给出正确结论：核心是《伊甸条约篇》，老师不是在形式逻辑上证明乐园，而是在不可证明他人真心时仍承担见证、相信和行动的责任。回答明确把这部分标成综合推论。

v2.9 图增长在本题返回 0 边；v3.1 最终保留 4 条核心 semantic 边：

```text
Episode 7 --thematic_response--> Episode 65
Episode 5 --evidence_bridge----> Episode 62
Episode 7 --thematic_contrast--> Episode 62
Episode 7 --thematic_contrast--> Episode 66
```

其中 7→65 与 7→66 是最可复用的“问题提出→后文实践回应”连接。7→62 的表述仍有轻微过度解释风险：Episode 62 主要是重新引用公案，“选择相信”的完整结论更明确地位于 Episode 65。说明即使有门卫，解释性边仍需要 confidence、后续使用反馈或人工抽查。

### 4.2 阿露、便利屋与凯撒

结果：**通过，且证据缺口处理正确。**

证据链：

```text
Episode 215：阿露发现情侣专用菜单后内心慌乱、表面逞强
Episode 214：阿露称老师为“我们便利屋的经营顾问”
Episode 165：浮士德曾炮击凯撒 PMC
```

结论：人物是阿露，所属社团是便利屋；当前数据没有任何一段直接证明便利屋与凯撒的雇佣或冲突关系。`main/33190` 的凯撒证据属于浮士德/阿拜多斯语境，不能偷换成便利屋。

三个 v2.9 模式都明确报告证据链在“社团→凯撒”处中断。v3.1 查询增长新增 0 边，证明系统在题目主动暗示一条关系时仍能拒绝制造该关系。这比“每次查询都必须增长”更符合长期记忆系统的目标。

### 4.3 山海经待客与圣三一条约会议

结果：**通过；该题最能体现 Association 增长价值。**

活动侧证据：

- Episode 246：水果、月饼、药材点心与“撑肠拄腹的接风”。
- Episode 244—245：在内部忧虑下仍照常举办月影祭，以传统、河灯、烟花和演奏维持共同体氛围。

主线侧证据：

- Episode 1：条约旨在结束圣三一与格黑娜的长期敌对并建立信任。
- Episode 84，`main/33050.json`：签署后创立伊甸条约机构，长期敌对双方被迫共同解决纷争，受到古圣堂神圣戒律约束。
- Episode 199：老师代理发起人、各政治势力聚集于古圣堂废墟，包含权限与历史正当性的博弈。

纯向量和早期静态图能给出“温暖生活化 vs 紧张政治化”的正确概括，但最终 20 个 Episode 没有保留 `main/33050` 的 Episode 84，主要依赖 `main/33190` 的事后分析。v3.3 做了两层修正：

1. 增长创建 6 条 `thematic_contrast`，全部 `polarity=1`；对比不是负关系，polarity 表示关系断言是否成立。
2. 新增长路径的 Episode 端点优先进入证据预算。最终 evidence 明确包含 Episode 84，24 条返回路径的所有 Episode 端点都可在 `evidence_episodes` 找到。

最有代表性的边：

```text
Episode 246 --thematic_contrast--> Episode 84
丰盛、感性、邀请性的饮食待客
vs
正式、制度化、带强制约束的政治机构
```

v3.3 回答仍曾把独立的 `event:10038` 与 `main` 内容放进一个“实际时间顺序”列表，虽然随后注明不能直接比较。v3.4 已把回答规则改为：只有问题确实要求先后且证据支持时才给时间顺序，不同 `timeline_scope` 禁止强排；4.4 节的三道真实 API 复述题已确认该规则生效。

### 4.4 Association 跨查询复用验证

为了验证“图会不会在后续问题中真正帮助系统”，新增三道不复用原始引文的渐进式复述题：

1. 直接比较“软性文化凝聚”和“硬性制度约束”；
2. 不说出山海经、圣三一和格黑娜名称，只描述开放庆典与神圣场所协定；
3. 进一步抽象为“日常传统建立关系”和“法律机构促成合作”。

对照数据库：

```text
baseline:  ba-stage2-direct-k4.db                         1336 Association
learned:   evaluation-stage2-growth-v33-q3/graph_growing.db 1342 Association
            其中 #1337—#1342 是上一问题增长的 6 条解释性边
```

#### 静态图结果

两组的三道答案都能正确识别山海经的月影祭/开放待客与圣三一、格黑娜的伊甸条约机构，并区分原文事实和综合推论。v3.4 没有把 `event:10038` 与 `main` 强排成一个时间线；第二题还明确写出“问题未要求时间顺序，且分属不同 timeline_scope，无法也不应排序”。

但是，baseline 与 learned 三题的最终 20 个证据 Episode 逐项相同；learned 组没有一题返回 #1337—#1342，六条边的 `use_count` 增量为 0，Episode 246 三次都没有进入最终证据。这说明答案正确来自原有向量/图证据，而不是已经保存的 246↔84 桥。

离线使用日志中原样保存的查询 embedding 重放遍历后，定位到真正的截断位置：

| 复述题 | 旧边在原始遍历中出现次数 | 旧边最高全局路径名次 | Episode 246 候选名次 | 结果 |
|---|---:|---:|---:|---|
| 机制对比 | 7 | 255 | 165 | 超过 `candidate_limit=120` 与 `answer_path_limit=24` |
| 身份识别 | 6 | 248 | 165 | 同上 |
| 抽象协调 | 3 | 254 | 228 | 同上 |

所以旧边不是“未被图遍历”，而是“被遍历后未被回答消费”。所有 288 条静态/生长最终路径都属于 `semantic/involves`，其 weight 为 0.895—0.960、confidence 为 1.0；查询增长边约为 weight 0.755—0.765、confidence 0.65—0.75。当前全局乘法排序必然让通用 Episode→Concept 边占满 24 条预算。即使只排除 `involves`，代表性旧边仍仅排在非 `involves` 路径第 36—55 名，说明固定类型配额只能缓解，不能代替查询相关重排。

#### 自主生长结果

相同三问分别在无学习库和已学习库上连续运行：

| 起点 | 新建边 | 强化事件 | 预先存在边被强化 | 总耗时 | API 请求 |
|---|---:|---:|---:|---:|---:|
| baseline 1336 边 | 17 | 1 | 0 | 447.10s | 15 |
| learned 1342 边 | 13 | 3 | #1339 | 440.88s | 15 |

learned 组确实少建 4 条边，并精确强化了旧边 #1339；同一轮中新建的 #1344 又在第二、第三题各被强化一次。这证明存储层已经具有有限的跨复述识别能力。不过，`AssociationGrowthEngine.grow()` 当前只把 created IDs 返回查询引擎：

- reinforced IDs 只写进日志，不写进最终结果；
- 只发生强化而没有新建时，查询循环会误判为“没有变化”并停止；
- 强化边不会进入 `preferred_association_ids`，因此即使刚被当前问题确认，也不会优先进入答案路径；
- #1339 的 `evidence_count +1`、weight `+0.04`，但 `use_count +0`，完整复现了这个断点。

此外，baseline 新建边中至少有 1 条与源库已有 semantic 边具有相同端点和关系类型，只因 `relation_key`/表述不同而另建。learned 组新增的 13 条多数是不同证据端点上的合理细化，不应一概视作重复；但目前只能按精确指纹去重，长期运行仍有近义边膨胀风险。

结论：自主增长已验证到“发现→保存→遍历→部分强化”，但还没有完成“强化/旧边→查询相关重排→证据→回答”的闭环。下一版应以这一断点作为 P0，而不是继续提高增长数量。

## 5. 提示词与确定性守卫的有效结论

### 5.1 Episode 粒度

当前有效的软约束：

```text
Episode 通常 80—300 中文字
连续 15—30 条短对白通常约 3—6 个 Episode
同一时间/地点/目标/冲突的微动作合并
实际故事时间、地点、核心目标或因果阶段改变才拆分
当前对话与对白中提到的过去事件必须拆开
```

这套规则在正式样本上得到平均 5.23 Episode/Source，没有发现 embedding 高相似重复对。不要把 3—6 当硬数量；倒叙和独立因果阶段优先于长度。

### 5.2 Source 分片

全 564 文件预检结果仍支持 `6000 / 8000 / 800`：

```text
文件: 564
解析失败: 0
Source: 1593
长度 p50: 5211
长度 p95: 7816
长度 max: 8071
轻微超过软上限: 26
```

超过 8000 的原因是保持完整 record、header 或人物别名表，当前应接受，不要为了硬截断破坏证据边界。重叠上下文 800 尚未发现明显重复 Episode 问题。

### 5.3 当前增长提示词的关键思想

v3.4 不再把“只能存原文事实关系”与“可以存解释性联想”混为一谈：

- 原文事实关系可以使用 temporal、causal、identity、interpersonal 等准确类型。
- 两个节点共同支持、可复用的主题回应或对比允许写成 `semantic`。
- 解释性边的 `relation_text` 必须以“查询综合推论：”开头，不能伪装成角色原话或强因果。
- 每条边必须直接帮助当前问题，同一节点对一次查询最多一条 semantic。
- 题目措辞不算证据；“阿露属于便利屋 + 浮士德攻击凯撒”不能生成“便利屋与凯撒有纠葛”。
- `polarity=-1` 只表示关系断言被证据否定，不表示内容相反或情绪负面。

仅靠提示词仍不够，因此代码还有确定性门卫：

- `timeline_scope` 由 Source 路径强制决定。
- temporal 统一 canonical 成 `earlier --before--> later`，并检查文件/分片来源顺序冲突。
- Episode identity 不能把共享人物误当成同一事件。
- `recall_trigger` 只有 Episode 原文含“想起/回忆起/思い出/떠올”等明确记忆语言时才允许。
- 同查询内 semantic 按节点对/类型去重。
- 新边路径必须携带可审计 Episode 端点。

### 5.4 回答提示词的有效约束

- 禁止模型使用数据库之外的作品知识。
- 明确区分事实、角色说法、综合推论和未知。
- Episode ID 只取对象顶层 `id`；`source_text` 中的 `[record:N]` 必须称作 `Source record N`。这个规则修复了早期把 Source record 388 写成 Episode 388 的引用错误。
- 不同 timeline scope 不强行排序。

## 6. 当前资产索引

### 6.1 代码与工具

| 资产 | 路径 | 本轮变化 |
|---|---|---|
| 查询引擎 | `src/memory_demo/retrieval/engine.py` | 增长路径端点进入证据预算；路径全部可审计 |
| 增长引擎 | `src/memory_demo/associations/growth.py` | created/reinforced 区分、结果统计、同查询去重 |
| 关系守卫 | `src/memory_demo/associations/builder.py` | 新增 recall_trigger 原文门卫 |
| 模型客户端 | `src/memory_demo/llm/client.py` | 429/5xx/Retry-After 带抖动退避及日志 |
| 导入管线 | `src/memory_demo/ingestion/pipeline.py` | timeline_scope 强制继承 Source，pass2 不可改写 |
| 提示词 | `src/memory_demo/llm/prompts.py` | 解释性 semantic、polarity、引用、跨 scope 时间规则 |
| 评测器 | `src/memory_demo/evaluation.py` | 原子增量报告、可筛选 `--modes` |
| 可恢复导入 | `benchmarks/import_corpus.py` | 修正 eligible/selected 文件统计 |
| 安全清理 | `benchmarks/cleanup_interrupted_run.py` | 仅清理无 Episode、无跨 run 引用的中断 run；默认 dry-run |
| 数据修复 | `benchmarks/repair_database.py` | 新增 source_key 推导 timeline_scope 修复 |
| 数据审计 | `benchmarks/audit_database.py` | 同时/非排序时间边与未知时间边分开 |
| 日志汇总 | `benchmarks/summarize_jsonl_logs.py` | 事件、模型、耗时、prompt 字符、429、fallback 汇总 |
| 静态复用对照 | `benchmarks/analyze_association_reuse.py` | 对比证据、回答路径、旧边 use_count 与邻接排名 |
| 遍历重放 | `benchmarks/trace_association_reuse.py` | 从 JSONL 恢复原始 query embedding，离线复现 seed、beam、候选和路径截断 |
| 生长复用审计 | `benchmarks/analyze_growth_reuse.py` | 区分 created/reinforced、旧边强化、回答消费和潜在 semantic 重复 |
| 回归测试 | `tests/` | 当前 32 项全部通过 |

### 6.2 数据库、评测与日志

| 资产 | 路径 | 说明 |
|---|---|---|
| 正式基线库 | `validation/ba-stage2-direct-k4.db` | 47 / 246 / 330 / 1336 |
| scope 修复前备份 | `validation/ba-stage2-direct-k4-before-scope-repair.db` | 可复现 Episode 91 的 `past` scope |
| 基线完整审计 | `validation/ba-stage2-direct-k4-audit.json` | 行级数据、BLOB、重复、时间图 |
| 7 文件证据清单 | `validation/stage2-direct-evidence-manifest.json` | 实际导入范围与 k=4 配置 |
| 18 文件宽范围清单 | `validation/stage2-evidence-manifest.json` | 最初 102 Source 方案与人工 ground truth |
| k=7 扩展探针 | `validation/ba-stage2-scaling-probe-k7.db` | 38 Episode / 239 Association 的成本样本 |
| 导入 ledger | `validation/stage2-direct-import-progress.json` | 每文件时间、Source、Episode、修订与失败 |
| 导入原始日志 | `validation/stage2-direct-import-logs/` | 33.57 MB 完整请求/响应 |
| 导入压缩统计 | `validation/stage2-direct-import-log-summary.json` | 3191 事件与延迟/错误统计 |
| v2.9 A/B/C | `validation/evaluation-stage2-direct-k4-v29/` | 9 次查询，增长 0 边 |
| v3.0 增长实验 | `validation/evaluation-stage2-growth-v30/` | 能增长但偏多，用于复现污染 |
| v3.1 质量实验 | `validation/evaluation-stage2-growth-v31/` | 三题 `[4,0,6]` |
| v3.2 polarity 单题 | `validation/evaluation-stage2-growth-v32-q3/` | 6 条对比边全为正断言 |
| v3.3 证据端点单题 | `validation/evaluation-stage2-growth-v33-q3/` | 最新真实 API 结果；新增 6 边 |
| v3.3 最终审计 | `validation/evaluation-stage2-growth-v33-q3/graph-growing-audit.json` | 1342 Association、无时间环、无重复 Episode |
| 复述问题集 | `validation/evaluation-questions-association-reuse.json` | 三个不复用原引文、逐步抽象的跨查询复用问题 |
| 无学习静态组 | `validation/evaluation-reuse-baseline-static-v34/` | 三题均答对，但未含 Episode 246 |
| 已学习静态组 | `validation/evaluation-reuse-learned-static-v34/` | 证据与 baseline 相同，旧边 use_count 增量 0 |
| 静态复用分析 | `validation/association-reuse-static-v34-analysis.json` | 0/3 使用 #1337—#1342，0/3 改变证据集 |
| 遍历重放明细 | `validation/association-reuse-static-v34-trace.json` | 旧边确实出现 7/6/3 次，但全局排名约 248—255 |
| 无学习生长组 | `validation/evaluation-reuse-baseline-growing-v34/` | 三题新增 17 边、1 次强化 |
| 已学习生长组 | `validation/evaluation-reuse-learned-growing-v34/` | 三题新增 13 边、3 次强化，命中旧边 #1339 |
| 生长复用分析 | `validation/association-reuse-growing-v34-analysis.json` | created/reinforced、消费情况和重复候选的逐边审计 |
| 人工评分卡 | `validation/stage2-evaluation-scorecard.json` | 三题结论与 v2.9→v3.3 生长演化 |
| 当前三题 | `evaluation_questions.json` | 取代早期六题标准 |
| 旧六题归档 | `evaluation_questions_v1.json` | 保留历史，不作为本阶段判定 |
| 上阶段总报告 | `EXPERIMENT_REPORT_2026-08-22.md` | 全语料预检、早期六题、float32 基准 |

实验数据库副本应保持原样用于复现，不应把 v3.0 或本轮复述实验的增长边合并回正式基线库。当前可作为下一阶段起点的是无查询污染的 `ba-stage2-direct-k4.db`；如果需要演示已经增长的图，则使用 v3.3 的 `graph_growing.db`。本轮两个 growing 数据库只用于复现增长与强化行为。

## 7. 规模判断与不过度设计结论

当前 47 Source 产生 246 Episode，若极粗略按 5.23 Episode/Source 外推 1593 Source，会得到约 8330 Episode，而不是 100 万。这个估算样本偏向长主线直接证据，只能用于数量级，不应作为最终容量设计依据。

按本轮 138.4 秒/Source 线性外推，全 1593 Source 约 61 小时；关系图随库增大、供应端限流和 fallback 会使真实时间可能更长。全量导入前应先改善任务调度和上下文长度，而不是把 workers 从 8 继续盲目提高。

当前不是过度设计的部分：

- Source/Episode/Concept/Association 四层结构；
- Episode/Concept 分离 float32 索引；
- SQLite source of truth 与可重建 RAM cache；
- 查询增长、时间守卫、完整日志、人工修正。

当前不应提前引入的部分：

- ANN/vector database：CPU brute-force 尚不是瓶颈；
- 专家系统式大量额外表：解释性关系可以继续落在 Association；
- float8/float16 两阶段量化：用户已明确统一 float32，且 RAM 足够当前阶段；
- 复杂线上并发架构：真实问题首先是供应端 429 与长提示词。

## 8. 下一阶段优先级

### P0：直接影响答案与增长质量

1. **把 reinforced 信号贯通查询引擎。**`grow()` 返回 created 与 reinforced 两组 ID；两者都触发重遍历并进入当前问题的 preferred paths；结果和日志分别暴露两组 ID。必须为“只强化、无新建”的轮次加回归测试。
2. **查询相关的 Association 路径重排。**不要再让高权重 `involves` 占满 24 条预算。先保留较大的 raw path pool，再综合 query↔`relation_text` 相关性、端点 seed 分数、weight/confidence、关系类型多样性排序；至少分别给 Episode↔Episode 解释边和结构边保留预算。关系文本 embedding 是否落库应由小规模 benchmark 决定。
3. **跨查询 semantic 去重。**精确指纹匹配前，先找相同/反向节点对的 semantic 边，让 LLM 判断“等价则强化、互补则并存、冲突则保留否定/冲突关系”。不能只比较 `relation_key` 字符串，也不宜粗暴规定一个节点对永远只有一条边。
4. **压缩回答证据。**把 66k 字符降到约 20k—30k：为 Source record 做查询相关片段压缩，去掉重复多语言内容，只为真正被路径采用的节点带 Source excerpt。
5. **Association 质量评估。**为解释性 semantic 边记录人工 `accepted/rejected` 或用户反馈，并比较它们是否在后续问题中真正提高召回；仅统计“长了几条边”没有意义。

### P1：全量导入可行性

1. 对 API 增加全局并发信号量、429 共享冷却窗口，而不是让每个线程各自重试。
2. 将 Concept/关系长尾任务写入可恢复队列；文件完成状态应区分“事实已落库”和“关系补全待处理”。
3. 用 50—100 个随机文件做分层样本，不再只测直接证据文件；统计 Episode 数、Concept 数、边密度与人工质量。
4. 修正 Source 软上限 oversize 的报告口径，但继续保留完整自然 record。

### P2：检索能力扩展

1. 在语料明显增大后加入 BM25/关键词精确召回，与向量召回 merge；这会改善专有名词和原句锚点。
2. 只有当 float32 实测延迟或 RAM 达到约束时再评估 ANN。
3. Concept 自动合并仍应保持“相似候选→LLM 判断→后续用户确认”的渐进路线，不要现在做不可逆大规模合并。

## 9. 可复现命令

```powershell
# 32 项本地回归
python -m unittest discover -s tests -v

# 基线库完整审计
python benchmarks/audit_database.py validation/ba-stage2-direct-k4.db `
  --output validation/ba-stage2-direct-k4-audit.json

# A/B/C 全模式
python src/main.py evaluate evaluation_questions.json evaluation-output

# 只测自主增长
python src/main.py evaluate evaluation_questions.json growth-output `
  --modes graph_growing

# 压缩日志统计
python benchmarks/summarize_jsonl_logs.py validation/stage2-direct-import-logs `
  --output validation/stage2-direct-import-log-summary.json

# 对比无学习/已学习静态结果
python benchmarks/analyze_association_reuse.py `
  validation/evaluation-reuse-baseline-static-v34/evaluation-report.json `
  validation/evaluation-reuse-learned-static-v34/evaluation-report.json `
  validation/evaluation-stage2-growth-v33-q3/graph_growing.db `
  validation/evaluation-reuse-learned-static-v34/graph_static.db `
  --learned-id-min 1337 --learned-id-max 1342 `
  --output validation/association-reuse-static-v34-analysis.json
```

## 10. 最终状态

```text
代码：v3.4，32 tests passed
基线数据库：47 Source / 246 Episode / 330 Concept / 1336 Association
最新增长数据库：+6 semantic thematic_contrast = 1342 Association
Embedding：硬盘与 RAM 均 float32，1024 维，4096-byte BLOB
时间图：三个 scope 均无环
最终失败任务：0
三道测试：全部通过，其中第二题正确报告证据缺口
复述静态组：3/3 答案可用，但 0/3 消费旧边、0/3 改变证据集
复述生长组：baseline 新建17/强化1；learned 新建13/强化3，命中旧边#1339
复用闭环：遍历与存储强化已成立，最终证据/回答消费尚未成立
```

本阶段最重要的新信息不是“模型能答出三道题”，而是把长期联想闭环拆出了可验证的边界：查询能把两个 Episode 拉进同一证据窗口并写入解释性 semantic 边；后续复述确实能在 raw traversal 中再次遇到旧边，增长模型也能部分强化它；但当前路径排序和返回协议会把强化信号丢在最终证据之前。系统已经是可增长的联想网络雏形，但还不能宣称旧 Association 已稳定改善后续回答。这个负结果比单次答对更能指明下一版应修的核心。
