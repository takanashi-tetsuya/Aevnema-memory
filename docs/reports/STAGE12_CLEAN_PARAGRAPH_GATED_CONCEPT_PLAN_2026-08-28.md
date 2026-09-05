# Stage 12 实验路线：Clean Paragraph + Gated Concept

本阶段针对 Stage 11 已定位的两个根因做最小可逆修复。

## Paragraph 路线

完整 Paragraph 继续存入 SQLite，保证 Source 追溯不丢信息；embedding 输入去掉 source_key、segment_index、speaker_alias_legend、record 编号和 script_raw，只保留发言者与可读多语言剧情文本。

检索采用 recall-only 语义：Paragraph 只能追加基线没有命中的 Episode seed，不给已有 Episode 加分，不参加原子锚点轮转。因此 Paragraph 在没有新增召回时不能改变基线顺序。

## Concept 路线

fine_grained 提取继续积极发现候选，所有模型响应和准入决定写入 JSONL 日志。持久化前增加独立 LLM 准入审计：

- promote：长期可复用的命名实体、制度、事件、理论、地点、物品、持续动机/信念/创伤；
- reuse：与相似候选中的已有 Concept 确为同一稳定语义；
- transient：一次性动作、泛词、局部形容、整句改写、上下文绑定复合事实。

准入返回缺失时采用保守回退：至少两个 Episode 重复出现才晋升，否则只留日志。

## 评价分层

机制挑战集只回答“修复后的通道在目标缺口上能否工作”；Stage 11 的七个网络题和五个范围控制题回答“加入通道后是否破坏成熟基线”。两类指标分别报告。

完整规则已在 `validation/stage12-mechanism-preregistration.json` 冻结，之后不得根据实验结果修改问题或门槛。
