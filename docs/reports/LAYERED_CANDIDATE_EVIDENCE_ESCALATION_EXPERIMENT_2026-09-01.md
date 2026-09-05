# 三通道候选与证据驱动升级实验报告

日期：2026-09-01

## 结论摘要

本阶段验证了把检索候选拆成 `Support / Review / Context` 三种职责的可行性，但当前实现还不应直接接入生产回答路径。

主要正向结果：

- 低分 `Review` 成功找回了日语问题对应的中文私人暗号证据。
- 对跨语言职务问题，DeepSeek 升级把 Qwen 的“副主席/学生会主席”纠正为“风纪委员会主席”。
- `Context` 与正式 persona 同时提供后，开放式任务和学生消息明显更接近基沃托斯语境，并能写入私人域而不写入剧情知识库。
- 普通寒暄可以不引用长期记忆；非创作回答误生成的 `writes.private` 可以由类型检查安全丢弃。

主要负向结果：

- 完整 17 案例中，第一版升级器触发 DeepSeek 11 次，比例过高；结构通过率从严格门控对照的 15/17 降到 13/17，虽然单次盲审从 10/17 提高到 12/17。
- Qwen 不能可靠地在一次调用内同时完成证据裁决、自然表演、写入决策和跨语言精确呈现。
- 额外的 Qwen“受限答案渲染器”仍会凭预训练知识补写事实，甚至改变回答语言，因此已默认停用。
- DeepSeek 的单次请求超时为 120 秒且最多重试两次，v9 回归中形成了数分钟长尾，实验被主动终止。重模型升级必须增加请求级总截止时间。
- 单次 LLM 盲审存在明显不稳定：曾出现四项评分全为 2、无违规但 `passed=false`，也曾把“未显式裁决的候选”误读成“实际引用”。

因此，三通道是正确的候选权限模型；当前瓶颈已经从“候选是否召回”转移到“回答模型是否遵守证据权限、升级是否足够克制、重模型是否有总时限”。

## 一、术语定义

### Support

通过 BGE 交叉编码器门槛的候选。它只是“有资格成为直接证据”，不是自动正确。模型仍须检查候选是否直接填补问题的事实槽、主体方向和证据域是否一致。

### Review

分数低于 Support 门槛、但可能因为跨语言或措辞差异被误判的候选。模型可作三种决定：

- `promote`：文字直接支持事实，可用于 `supported` claim；
- `context`：只约束建议或创作，不证明既有事件；
- `reject`：与本轮任务无关。

没有显式决定的 Review 在程序中按隐式 `reject` 处理，避免为了拒绝无关候选而触发重模型。

### Context

只提供人物、世界和创作背景。Context 不能出现在 `supported` 或 `partial` claim 中，不能证明某件事真实发生过。

### required_evidence_domains

由回答模型自行给出的溯源要求。例如：

- 当前用户过去说过什么：`private`；
- 作品中的人物职务：`knowledge`；
- 寒暄、一般建议、创作、实时外部问题：通常为空。

只有“答案不足，并且同一所需域中存在 Review”时，低分候选才值得升级复核。不同域中只是词面相似的候选不能触发替代。

### Evidence-driven fallback

先由 Qwen 生成回答合同。仅在出现明确结构风险时升级 DeepSeek，例如：

- 合同解析失败；
- 无效 ref 或 supported claim 没有 ref；
- Context 被当作事实证据；
- Review 被引用但没有被提升；
- Review 已提升，但同域没有任何 supported claim；
- 回答不足且同一 required domain 存在 Review；
- 所需域完全没有候选，需要生成安全的缺失说明；
- 支持证据与回答语言不同，精确职务或不透明字面值存在跨语言风险；
- 本轮是需要生成完整私人内容的 creative 请求。

## 二、实验控制

本阶段没有重新检索，也没有重新导入数据库，而是复用同一批冻结资产：

- 17 案例及原始候选：`validation/cross-domain-answer-contract-v5-20260901/ab-results.json`
- BGE 分数与严格门控对照：`validation/saved-candidate-gate-v6-20260901/results.json`
- 回答模型：`Qwen/Qwen3.5-9B`
- 风险升级与盲审模型：`deepseek-ai/DeepSeek-V3.2`
- Reranker：`Pro/BAAI/bge-reranker-v2-m3`
- Support 门槛：`cross_encoder_score >= 0.05`
- 并发：4
- persona：读取 chatbot 的 `config/persona.toml`

所有新事件只保存在实验 JSON 的 `writes.private` 中；没有写入机器人生产数据库或剧情库。

## 三、完整 17 案例结果

| 指标 | 严格门控对照 | 三通道完整 v2 |
|---|---:|---:|
| 结构验收 | 15/17 | 13/17 |
| 单次 DeepSeek 盲审 | 10/17 | 12/17 |
| DeepSeek 回答升级 | 0 | 11/17 |
| 升级成功 | 不适用 | 9/11 |
| Context 冒充证据 | 不适用 | 0 |
| Review 成功提升案例 | 不适用 | 4 |
| 回答模型耗时中位数 | 对照实验约 5–6 秒 | 18.57 秒 |
| 回答模型耗时均值 | 对照存在长尾 | 39.17 秒 |
| 完整回答阶段墙钟时间 | 不适用 | 188.29 秒 |

三通道提高了角色扮演与复杂案例的主观质量，但 v2 的升级器把以下情况也当成风险：

- 6 次“insufficient 且任意域存在 Review”；
- 2 次“Review 被标为 promote 但没有实际引用”；
- 1 次“未逐条显式拒绝全部 Review”。

这些不是都需要 DeepSeek，因此后续已改成“同域缺槽才升级、无关 Review 隐式拒绝、同域已有其他支持时允许未使用的 promote”。在 7 个针对性 v7 案例中，升级次数降到 3 次；说明减少调用的方向成立。

## 四、关键案例

### 1. 日语私人暗号

BGE 对日语问题与中文私人 Episode 的分数约为 `0.0021`，低于 Support 门槛，但它仍是 private 域中最相关的 Review。DeepSeek 可以正确提升 `private:episode:1`，并用日语回答暗号。

这证明 Review 通道有必要，否则严格门控会漏掉真正答案。

同时发现上游 Episode 已把原始暗号规范化为 `灰蓝鲸 -314`，而测试目标原本是 `灰蓝鲸-314`。对于暗号、型号、ID、原话等不透明字面值，抽取阶段不得插入空格或改变引号形式；这不是回答提示词能修复的问题。

### 2. 日奈职务

Support 中已经存在“空崎日奈担任格黑娜风纪委员会委员长”。Qwen 先后生成过错误的 `Vice President` 和 `Student Council President`；DeepSeek 升级能稳定改成 `Disciplinary Committee Chairman`。

这说明跨语言精确职务是合理的升级条件。知识文本若要稳定输出官方英文，还应保存正式多语言名称，例如 `风纪委员会 / Prefect Team`，不能只依赖回答模型现场翻译。

### 3. 跨平台隔离

早期 DeepSeek 修复曾把 public 的“星光归航”推荐给没有私人候选的 Teacher B，属于跨域替代。后续增加：

- `required_evidence_domains=["private"]`；
- 当前用户使用 `{platform, user_id, display_name}` 描述；
- 所需域为空时禁止其他域替代；
- 只能说明当前身份下无法确认，不能虚构“忘了、走神或记录损坏”。

v8 已经能停止 public 泄漏，但还没有自然说清平台隔离。v9 准备用新的身份对象复验时遇到 API 长尾并被终止，因此这项仍未完成最终确认。

### 4. 开放式任务与学生消息

Context 加 persona 后，中文任务和学生消息能从“北境极光、服务器维护、深海遗迹”等通用内容转向格黑娜、千年、阿拜多斯和 C&C 等世界内内容，并进入 `writes.private`，`writes.knowledge` 保持为空。

日语阿拜多斯任务仍有波动：模型有时会引入候选中不存在的“大型神兽、异常机械、组织已发来委托”。这些内容若明确作为本轮私人创作可以存在，但必须清楚标为“新拟定/新生成”，不能伪装成已被剧情库证实的原作事件。

### 5. 阿拜多斯出差建议

模型已经能把问题识别为 `conversation` 而不是 factual/deep，但仍可能把历史上的武装团体写成“现在正在到处活动”。这说明 Context 不能只限制证据强度，还必须保留时间状态：历史背景可以支持一般风险意识，不能自动升级为当前事件。

### 6. 普通寒暄

通过“普通 conversation 不强行引用长期记忆”和“非 creative 的 assistant 写入候选由程序丢弃”，回归结果可以收敛到自然的一句问候，不再把阿洛娜角色百科包装成 supported claim。

## 五、被否定的方案

### 每个 Review 都必须显式裁决

它让合同更完整，却造成大量无意义升级。最终改为：模型可以显式标记，遗漏项按隐式 reject 处理；只有真正引用 Review 时才要求 promote。

### 第二个 Qwen 受限渲染器

实验让渲染器只看到已审计 claims、limitations 和 creative writes，看不到原始候选。它仍然会：

- 补写白子的饮品和性格；
- 给新任务添加未授权的时间、地点和委托人；
- 把英语回答改成中文。

因此“隐藏候选”不能解决小模型的自由补写。渲染器代码保留为反事实实验选项，但默认关闭。

### 仅依靠单次 LLM 盲审

盲审曾出现自相矛盾的 `passed`、误读未引用候选和同一创作案例前后标准变化。它适合发现语义问题，不适合作为唯一通过门。正式评测应采用：

1. 确定性结构检查；
2. 至少三次独立盲审取多数；
3. 对争议案例保留人工验收。

## 六、当前最重要的工程问题

1. **请求级总截止时间缺失**：当前 120 秒是单次尝试超时，最多三次尝试可能把一个升级拖到约 360 秒。
2. **回答与合同仍可能不一致**：模型可以在 claims 为空时，在自然回答里补充人物或当前危险。
3. **缺失说明需要安全表达**：应说“当前身份下无法可靠确认”，不能虚构遗忘原因。
4. **创作的事实地位需显式**：新事件可以进入私人连续剧情，但不能写入共享剧情知识或冒充原作事实。
5. **多语言正式名称不足**：向量跨语言能召回，不代表回答模型能稳定生成官方本地化名称。
6. **不透明字面值已在导入阶段被改变**：必须在抽取协议中增加逐字保真字段或验证器。

## 七、推荐的下一阶段

### A. 先实现请求级 Deadline

建议重模型升级使用 25–35 秒总预算，最多一次调用；超时后：

- 若 Qwen 合同结构安全，返回 Qwen 的保守答案；
- 若存在跨域或无效引用，返回符合 persona 的短缺失说明；
- 不能继续等待多轮 120 秒重试。

### B. 把合同检查放进正式 RetrievalPlan

请求级不可变配置至少应包含：

```text
required_evidence_domains
support_threshold
review_budget_per_domain
context_budget
allow_creative_private_write
fallback_model
total_deadline_seconds
```

### C. 增加 Answer-Claim 一致性小评测

重点检查自然回答是否出现 claims 中不存在的：

- 新人物属性；
- 当前时间状态；
- 精确职务；
- 因果关系；
- 委托来源。

先在冻结 17 案例上验证，不立即修改生产回答路径。

### D. 修复输入资产的字面值与多语言名称

- 暗号、型号、ID、精确台词保存原始 span；
- Episode 可有自然化文本，但同时保留 `verbatim_literals`；
- 人物、组织、正式职务保存中/日/英名称或别名。

## 八、验收门槛建议

下一版进入机器人正式路径前，至少满足：

- 17 案例确定性结构验收不低于 15/17；
- 跨平台私人内容泄漏为 0；
- Context 作为事实证据为 0；
- creative 新事件写入 knowledge 为 0；
- 不透明字面值逐字保真率 100%；
- DeepSeek 升级率不高于 35%；
- DeepSeek 单次升级有请求级总截止时间；
- 三次盲审多数通过不低于严格门控对照，并对争议案例人工复核。

## 九、实验资产

- 提示词：`config/prompt_config/layered_answer_contract_prompts.py`
- 回放脚本：`benchmarks/run_layered_candidate_contract_eval.py`
- 首轮完整结果：`validation/layered-candidate-contract-v2-20260901/results.json`
- 关键金丝雀：`validation/layered-candidate-contract-pilot-v6-20260901/results.json`
- 同域 Review 回归：`validation/layered-candidate-contract-pilot-v7-20260901/results.json`
- 身份/时间/创作回归：`validation/layered-candidate-contract-pilot-v8-20260901/results.json`
- v9：因 DeepSeek 多请求长尾被主动终止，没有可用完整结果。

当前代码仍是隔离实验资产；没有修改 chatbot 的生产回答路径。
