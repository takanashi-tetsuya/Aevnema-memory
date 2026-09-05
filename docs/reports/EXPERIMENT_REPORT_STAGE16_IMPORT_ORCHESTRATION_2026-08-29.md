# Stage16：导入编排质量与 Token 实测报告

日期：2026-08-29（Asia/Tokyo）  
最终提示版本：`v3.25_balanced_orchestration_guarded`

## 1. 本阶段回答的问题

Stage15 已把导入默认流程从历史 `strict` 改为 `balanced`：

- 时间边界审计和粒度/证据审计由两次调用合并为一次；
- 二次理解保留完整目标 Episode，但 Source 使用紧凑多语言视图，相邻 Episode 作为共享上下文只发送一次。

Stage15 的导入收益只有历史日志重建估算和一个小型 smoke import。本阶段用完全相同的历史输入做在线
重放，回答两个更严格的问题：

1. 调用数和 token 是否真的下降；
2. 合并审计和上下文压缩是否造成事实遗漏、时间混合、粒度恶化或未知人物被擅自命名。

最终结论：**保留 `balanced` 为默认值**。48 个固定案例全部结构有效，语义
coverage@0.70 为 100%，可比生产 token 减少 52.0%。6 个启发式异常交给独立模型盲审后，结果为
balanced 胜 4、平局 1、strict 胜 1；差异全部为 `minor/none`，没有 `major` 回归。

## 2. 术语定义

- **strict**：历史高成本路径。一个高风险 Source 先做时间边界审计，再把结果交给粒度/证据审计；
  二次理解重复发送每个目标自己的 nearby 上下文和完整 Source。
- **balanced**：当前默认路径。时间、粒度和人物证据一次审计；二次理解去掉重复 nearby，并使用共享上下文
  与紧凑 Source。
- **冻结重放（frozen replay）**：从已完成的 Stage14 日志中保存模型当时看到的输入及其 strict 输出，
  新流程只重跑同一输入，避免语料、分片或首轮 Episode 提取变化干扰对比。
- **结构有效**：JSON 可解析、Episode 字段可校验；二次理解的每个 `episode_id` 恰好返回一次。
- **strict→balanced coverage**：对每条历史 strict Episode，找到 balanced 输出中最相似的 Episode；相似度达到
  阈值的比例。它衡量“历史事实阶段是否仍能在新输出中找到”，不是最终剧情正确率。
- **coverage@0.70**：上述最大 cosine similarity 至少为 0.70 的比例。
- **自适应盲审**：只有结构规则、语义阈值或身份规则报异常的案例才交给 GLM；A/B 位置按 case ID 交替，
  裁判不知道哪一组是 balanced。
- **可比生产 token**：只计算 Episode 审计、二次理解及必要 JSON 修复。评测专用 embedding 和盲审 token
  不计入产品收益。

## 3. 冻结集如何建立

来源是 Stage14 完整导入的 4 份 JSONL 日志。流式扫描后得到：

| 类型 | 历史请求 | 无效响应 | 可配对/有效批次 | 最终选择 |
|---|---:|---:|---:|---:|
| 时间边界审计 | 280 | 7 | 与粒度审计配对后共 92 个 Source | 24 |
| 粒度/证据审计 | 217 | 14 | 同上 | 24 |
| batch 二次理解 | 225 | 3 | 222 个有效唯一 batch | 24 |

审计样本按时间词、未知人物标记和 Episode 数量排序；二次理解样本按可节省的提示字符比例排序。两组都按
`main/event/favor` 分层，并限制同一 `source_key` 最多两个样本。最终分布：

| 集合 | main | event | favor | 合计 |
|---|---:|---:|---:|---:|
| 合并审计 | 11 | 3 | 10 | 24 |
| 二次理解 | 10 | 5 | 9 | 24 |

冻结文件为 `validation/evaluation-stage16-import-orchestration/frozen-cases.json`。它保存完整 Source、候选、
历史 strict 输出、请求 ID 和历史 usage，因此可以在以后模型或提示升级时重复使用。

## 4. 最终在线结果

48 案例全量数值来自最终提示与身份清洗版本。该轮暴露出的唯一 5→16 极端膨胀随后直接产生了确定性保护；
因此下表保留原始模型输出用于诚实比较，正式导入时该一个输出会被保护规则拒绝。保护不增加请求，不改变
下述 token 数。

### 4.1 质量

| 指标 | 合并审计 | 紧凑二次理解 |
|---|---:|---:|
| 案例数 | 24 | 24 |
| 结构有效 | 24/24 | 24/24 |
| 平均语义相似度 | 0.9510 | 0.9691 |
| 平均 coverage@0.70 | 100% | 100% |
| 最差单案例 coverage@0.70 | 100% | 100% |
| 触发独立盲审 | 3 | 3 |
| 平均单请求时间 | 40.56 s | 32.52 s |

4 路并发下，24 个审计请求从首个请求到最后响应约 256.5 秒；24 个二次理解请求约 210.7 秒。历史日志
没有严格可比的同机墙钟基线，因此本报告不声称延迟降低，只确认调用数、token 和当前并发稳定性。

6 个异常盲审：

| 裁判结论 | 数量 |
|---|---:|
| balanced 更好 | 4 |
| 平局 | 1 |
| historical strict 更好 | 1 |
| 裁判错误 | 0 |
| `major` 差异 | 0 |

唯一 strict 胜例为 `main/32070.json`：一次在线生成把 5 个候选拆成 16 条，历史 strict 为 4 条，事实并未
缺失但粒度过碎。最终代码加入零调用保护：审计输出同时满足“超过原候选 2.5 倍”和“至少多出 8 条”时，
拒绝该审计结果并保留审计前候选。该规则只处理明显爆炸，不试图用条目数判断所有粒度优劣。

### 4.2 Token

| 阶段 | historical strict | final balanced | 减少 |
|---|---:|---:|---:|
| Episode 审计 | 211,937 | 104,658 | 50.6% |
| 二次理解 | 264,768 | 124,180 | 53.1% |
| 合计 | 476,705 | 228,838 | **52.0%** |

请求数从 72 次（48 次审计 + 24 次二次理解）降到 48 次，减少 33.3%。拆分 usage：

| usage | historical strict | final balanced | 变化 |
|---|---:|---:|---:|
| prompt tokens | 414,266 | 186,463 | -55.0% |
| completion tokens | 62,439 | 42,375 | -32.1% |
| total tokens | 476,705 | 228,838 | -52.0% |

评测另外使用 28,799 embedding token 和 40,591 裁判 token。这些只为本次离线质量验证服务，不属于正式
导入路径，因此没有混进上表。

## 5. 实验中发现并修复的问题

### 5.1 合并审计曾经合并过度

早期提示把“同一时间、地点、目标”视为可合并条件，模型会把同场景中的不同话题、承诺、行动阶段和情绪转折
揉在一起。最终提示改为：只合并不能独立检索的寒暄、重复确认和微小反应；每项候选事实、参与者、回忆来源
和承诺都必须在输出中有落点。

### 5.2 收紧提示后出现粒度爆炸

少数重放会走向反面，把连续对白拆得过细。因为极端爆炸有明确的数量信号，最终加入上述 2.5 倍且 +8 条的
本地拒绝规则，不增加模型调用。中等粒度差异仍交给模型本身，避免用武断的数量阈值误伤正常细分。

### 5.3 `老师（???）` 绕过未知人物规则

紧凑二次理解偶尔不直接删除 `???`，而是生成“老师（???）”。这仍是在未知标记前塞入身份猜测。最终
`sanitize_episode_draft()` 会把这种明确矛盾恢复为 `???`；若参与者字段出现“姓名（未知标记）”，通过完整
参与者字符串做精确替换。

开发中第一版正则范围过宽，曾把“爱丽丝向老师（???）报告”错误清洗成“???报告”。定向重放发现后，规则
改为只识别明确身份标签，并新增句法回归测试。最终形式为“爱丽丝向???报告”，不再吞掉主语或谓词。

### 5.4 评测器把未知标记同义形式误报

`???`、`未标注发言者`、`未知发言者` 是同一不确定性类别，出现次数会因措辞和 Episode 合并而变化。最初按
字符串计数产生大量假阳性。最终只有“不确定性从整个输出完全消失”或“姓名（???）”才触发身份审查；独立
盲审仍检查 Source 是否真的提供了身份依据。

## 6. 最终架构决定

1. 默认保持 `MEMORY_OPTIMIZATION_PROFILE=balanced`。
2. `strict` 继续作为完整历史路径，可用于人工争议样本、高保真重跑和未来 A/B 基线。
3. 不添加普遍性的第三次“是否该回退”LLM 调用：本实验没有找到可靠的本地信号判断所有轻微粒度差异；
   强行回退会抵消 token 收益并把许多 balanced 胜例误判为失败。
4. 保留两个确定性门卫：未知身份矛盾清洗、极端 Episode 数量爆炸拒绝。
5. SQLite 与 RAM 中的 embedding 继续统一为 float32；本阶段没有改动向量格式、Source/Episode/Concept/
   Association 语义或 generation 规则。

## 7. 局限

- 历史 strict 输出是强基线，不是人工标注的绝对真值；strict 自身也会遗漏或过度合并。
- cosine coverage 只能发现语义缺口，不能单独证明每个细节完全正确，所以异常案例另加了 Source 约束盲审。
- 裁判模型仍有随机性；本阶段通过冻结输入、A/B 交替和只报告 `minor/major` 缓解，但不能代替未来人工抽检。
- 本阶段重放 48 个高风险样本，没有重新导入全部 208 个文件。它验证了最昂贵的两个编排环节，不等于一次
  全语料 v3.25 重建。
- 平均延迟来自当前供应商和本机的一次运行；Demo 尚无并发/延迟 SLA。

## 8. 复现与资产

核心脚本：

- `benchmarks/prepare_stage16_import_orchestration.py`：从大型 JSONL 日志流式建立冻结集；
- `benchmarks/run_stage16_import_orchestration_eval.py`：在线重放、embedding 对齐、异常筛查和盲审。

主要结果：

- `validation/evaluation-stage16-import-orchestration/final-balanced-eval.json`：最终 48 案例结果；
- `validation/evaluation-stage16-import-orchestration/final-balanced.jsonl`：完整 API 请求、响应与 usage；
- `validation/evaluation-stage16-import-orchestration/expansion-guard-replay.json`：把最终全量运行中 5→16 的
  计数信号确定性重放到保护规则；
- `validation/evaluation-stage16-import-orchestration/expansion-guard-eval.json`：同一案例的随机在线复跑；
  该次模型返回 8 条，没有跨过“极端爆炸”阈值，因而没有触发保护；
- `validation/evaluation-stage16-import-orchestration/guarded-targeted-eval.json` 与
  `guarded-regex-fix-eval.json`：提示和未知身份清洗的迭代证据；
- `validation/evaluation-stage16-import-orchestration/balanced-online-eval.json`：修复前首次全量基线。

最终确定性测试：**133/133 通过**。测试不调用外部 API。
