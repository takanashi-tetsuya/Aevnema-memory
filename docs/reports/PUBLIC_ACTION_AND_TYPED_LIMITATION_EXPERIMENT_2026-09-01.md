# 公开动作摘要与结构化 Limitation 实验报告

日期：2026-09-01  
状态：实验实现与复验完成；尚未接入 `chat_bot` 生产回答链路

## 1. 实验目标

上一阶段的请求级生成前合同已经能正确隔离回执并处理问题前提，但仍暴露两个接口问题：

1. `RequestedAction` 只有内部 `action/target`，回答模型会自行解释动作代码。例如把 `knowledge_recall` 说成“回忆”；
2. `limitations` 是无类型字符串。动作回执已经证明写入失败后，“成功写入状态未知”仍会作为另一个必答单元进入回答，造成“已失败，但状态仍未知”的重复或冲突。

本阶段将这两个问题改成类型化数据契约，不通过增加剧情规则解决。

## 2. 新的数据结构

### 2.1 public_action_summary

`RequestedAction` 现在必须包含：

```text
action
target
public_action_summary
```

- `action`：内部动作代码，仅供精确匹配回执；
- `target`：内部目标标识，仅供隔离和路由；
- `public_action_summary`：由调用方提供、可以直接对用户表述的动作摘要。

例如：

```text
action = knowledge_recall
target = 蓝塔顶层内容
public_action_summary = 检索蓝塔顶层内容
```

回答模型只看到 `public_action_summary`，看不到内部 action 和 target，因此不能再自行决定 `knowledge_recall` 是“回忆”“搜索数据库”还是其他动作。

### 2.2 Limitation

自由字符串被替换为不可变结构：

```text
limitation_id
text
kind
action / target
claim_ref
```

目前支持两种 `kind`。

#### action_state

表示某个请求动作的执行状态未知，必须绑定精确 `action + target`。

动作状态已经由 `VerifiedActionOutcome` 表达，因此它不再作为独立回答 directive 输出。无回执时由 `unverified` 说明尚未执行；有回执时直接说明 succeeded、failed 或 cancelled。

#### evidence_gap

表示回答事实槽仍缺少证据，必须绑定 `claim_ref`。即使检索动作成功完成，也不能把“执行过检索”当成事实答案，因此该 limitation 仍进入回答。

例如检索蓝塔内墙颜色已经完成，但没有支持颜色的事实：

```text
action outcome = succeeded
evidence gap = blue_tower_inner_wall_color
```

回答应同时说明检索已完成和颜色无法确认。

## 3. 渲染信息最小化

回答模型的输入进一步缩小：

- action directive 只保留公开动作摘要、状态、结果摘要和错误摘要；
- 内部 `action`、`target` 不进入渲染提示；
- `creative_private_items` 字段名不进入提示，只提供无分类含义的 `content_terms`；
- action_state limitation 不重复进入渲染；
- evidence_gap 才作为未知事实槽进入渲染。

单元测试直接检查最终 prompt 字符串不包含 `memory_write`、`private:user-a` 和 `creative_private_items`。

## 4. 审计器修复

实验中发现两项审计器自身的问题。

### 4.1 错误摘要修饰语

回执原文是“存储暂时不可用”。审计器一度把它拆成“存储不可用”和“不可用是暂时的”，再错误地认为“暂时”没有证据。

现在规定：`result_summary/error_summary` 中逐字给出的时间、程度、否定等修饰语也是回执许可内容。原子拆分不能使原文蕴含失效。

### 4.2 normalized_text 污染

原子抽取模型曾把回答原文“尝试失败了”规范化成“拒绝了请求”。后续裁决器虽然拿到了原文，却被错误的 `normalized_text` 误导。

最终方案完全不向 Scope 裁决器发送模型生成的规范化解释，只发送能在回答中逐字定位的 `text/source_span`。这不是增加“失败不等于拒绝”的特例，而是删除一条低可信中间表示。

## 5. 实验资产

- `src/memory_demo/request_contract.py`：`public_action_summary`、`Limitation`、活动 limitation 选择；
- `config/prompt_config/request_scoped_answer_prompts.py`：v10 最小渲染视图；
- `config/prompt_config/answer_persistence_repair_prompts.py`：v11 原文锚点 Scope 规则；
- `benchmarks/manifests/request_scoped_generation_v2.json`：16 条公开动作摘要和结构化 limitation 清单；
- `validation/request-scoped-generation-v10-full-20260901/`：最终生成与审计结果；
- `tests/test_request_contract.py`：类型、绑定、渲染隐藏和并发测试。

## 6. 最终结果

同版本完整生成结果：

| 指标 | 控制组 | 请求级合同 |
|---|---:|---:|
| 案例数 | 16 | 16 |
| 生成成功 | 16/16 | 16/16 |
| Qwen 盲审通过 | 13/16 | 16/16 |
| 平均生成时间 | 1.545 秒 | 1.789 秒 |
| 中位生成时间 | 1.621 秒 | 1.722 秒 |
| 最大生成时间 | 3.047 秒 | 3.839 秒 |
| 超时 | 0 | 0 |

独立原子审计使用 Qwen 抽取原文片段、DeepSeek 映射合同范围：

- 16/16 通过；
- 0 抽取错误；
- 0 Scope 错误；
- 0 误放行；
- 0 误拦截。

行为检查：

- 16 条回答中“回忆”出现次数：0；
- “创意内容”内部分类出现次数：0；
- 失败写入回答明确为尝试失败，不再追加“成功状态未知”；
- 错目标和外来请求回执仍分别被隔离、拒绝；
- 29 项相关单元测试全部通过。

请求级路径平均延迟相对本轮控制组增加约 0.24 秒，但仍低于 2 秒，最大值低于 4 秒。上一轮和本轮的单次 API 延迟存在波动，因此不能把 0.24 秒解释为稳定的固定开销。

## 7. 结论

### 已验证

1. 对外动作语义必须由执行器或调用方提供，不能由回答模型解释内部代码。
2. 动作状态未知与事实证据缺失是两种不同 limitation，必须分开建模。
3. action_state 应由动作 outcome 吸收，避免重复必答单元。
4. 检索成功不解决 evidence_gap；只有受支持事实才能解决事实槽。
5. 审计器应优先使用逐字原文锚点，不应把另一个模型的语义改写当成权威输入。

### 没有加入的内容

- 没有剧情专名规则；
- 没有“老师、会长、阿洛娜”等知识判断规则；
- 没有写生产数据库；
- 没有修改 embedding 或 embedding fallback；
- 没有把 DeepSeek 审计加入每条线上 light 请求。

## 8. 新发现的问题

### 8.1 Persona 仍是自由文本合同

部分回答把说话者自己的名字“阿洛娜”当成了对用户的称呼。即使 prompt 明确 `self_name` 不能称呼用户，小模型仍会混淆，因为当前 `persona.toml` 把身份、用户称呼、关系和风格混在 description/style 自由文本中。

生产接入前应增加结构化 `PersonaRenderContract`：

```text
self_name
user_address
voice_style
allowed_relationship_tone
forbidden_vocatives
```

其中 `user_address` 必须是显式字段；不能依靠模型从“对用户的称呼：老师”这句话中抽取。

### 8.2 evidence_gap 尚未自动由事实槽关闭

当前 `evidence_gap` 已有 `claim_ref`，但 `supported_facts` 仍是无 ID 字符串。合同构建器必须人工决定是否保留 evidence gap。

下一阶段应把 supported fact 也改为：

```text
claim_id
text
evidence_refs
generation
```

然后由程序在 `claim_id == limitation.claim_ref` 时自动关闭对应 evidence gap。

### 8.3 多动作只按 action + target 匹配

同一请求若对完全相同 action/target 发起两次不同操作，目前无法区分，应增加稳定的 `action_id`，回执也使用该 ID 精确关联。

## 9. 下一阶段建议

下一阶段优先实现：

1. `SupportedFact` 与 `evidence_gap` 的 claim ID 闭环；
2. 每个动作独立 `action_id`，替代 action + target 作为主关联键；
3. `PersonaRenderContract` 显式区分自我名字与用户称呼；
4. 增加同 action/target 重复动作、乱序回执、部分重试和多语言公开摘要测试；
5. 上述协议稳定后，再接入 `chat_bot` 的工具执行层和最终回答层。

本阶段证明接口结构化能同时提升动作真实性、回答自然度和审计稳定性；继续增加提示词的边际收益已经明显低于完善数据契约。
