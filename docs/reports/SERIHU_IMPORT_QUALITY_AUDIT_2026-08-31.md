# Serihu 剧情导入暂停质量审计

日期：2026-08-31  
状态：全量导入继续暂停；正式 chatbot 剧情库未被本轮实验修改。

## 1. 审计目标

本轮不是检查 SQLite 是否能写入，而是回答三个质量问题：

1. 新台词语料能否被切成可独立检索、不过度合并的 Episode；
2. 摘要是否会写反发言者、动作主体、动作对象或人物职务；
3. 是否存在一种在质量、调用次数和墙钟时间之间更合理的导入路径。

这里使用的术语：

- **边界错误**：把不同地点、人物群或行动阶段合成一个 Episode，或把连续事件拆成逐句碎片。
- **事实归因错误**：把“谁说了什么”“谁看见谁做什么”“谁具有某身份”等关系写反。
- **确认偏差**：复核模型看到一个已经写错的候选后，倾向于沿用候选的说法，而不是重新从 Source 判断。
- **独立抽取**：复核模型只读取 Source，不读取第一模型的 Episode，因此不会直接继承候选措辞。
- **事实角色审计**：只复核发言者、动作主客体、身份职务与专名，不负责重新设计 Episode 边界。
- **单调改进门禁**：第二模型的结果只有在可检测风险分数下降时才允许替换原条目；否则保留主抽取结果。

## 2. 当前资产状态

### 2.1 正式资产

- chatbot 正式剧情库：未修改。
- 本轮没有激活任何新候选库。
- 全量导入进程保持停止。

### 2.2 语料

- `validation/serihu-corpus-20260831/`
- 180 个 TXT 文件。
- 按当前 `SegmentConfig(target=6000, max=8000, overlap=800)` 可形成 787 个 Source 分片。

### 2.3 候选数据库

- `validation/serihu-full-safeparallel-candidate-20260831.db`
  - 旧的暂停全量候选；只完成一小部分，不可激活。
  - 数据库总量：80 Source、567 Episode、807 Concept、3366 Association。
  - 相对正式库的新数据：17 Source、93 Episode、231 Concept、496 Association。
- `validation/serihu-quality-pilot-v2-20260831.db`
  - CLI 参数未进入嵌入引擎时产生的无效对照；不可激活。
- `validation/serihu-quality-pilot-v3-20260831.db`
  - 只完成 `主线剧情_最終編_第1章.txt` 第 0 分片的诊断候选；不可激活。

## 3. 旧候选暴露的问题

### 3.1 Episode 过度合并

旧 Episode 529 把以下两个现场写入同一条：

1. 黑服、マエストロ、ゴルコンダ讨论色彩、箱舟和各自计划；
2. 凯撒特种部队报告位置，ミヤコ准备突入シャーレ。

同一 Source 中存在章节、地点和主要人物群切换，仅凭它们位于同一 record 不能判为同一事件。

### 3.2 通用时间标签没有信息量

旧增量 93/93 Episode 使用“当前”“当前时间”“当前故事时间”等标签。这些值不能帮助倒叙排序，反而制造
伪精度。现在完全相同的通用值会在落库前清空，只有“上次”“童年”“某事件之后”等实际时间证据保留。

### 3.3 Concept 名称与别名污染

旧候选出现：

- canonical_name 中直接串联多个别名；
- 同一别名指向多个 Concept；
- `凯撒 PMC` 与 `凯撒PMC` 因 CJK 空格产生重复；
- description 中残留“此处修正”“应提取”等模型自检文字；
- `黑野`、`葛叶` 等 Source 未提供的中文翻译进入 Episode 后继续污染 Concept。

### 3.4 身份边误判

发现过 `圣三一复制体 = Divi:Sion`、`响 = 歌原` 等无原文身份依据的等同边。相邻出现、描述相似和共同
参与事件都不能单独证明身份相同。

## 4. 两个高风险 Source 的对照结果

### 4.1 第 0 分片：预知梦、会长指定关系和多场景切换

Qwen 主抽取加候选审计产生过两项严重错误：

- 把“扭曲光环的少女向老师举枪”写成“セイア向老师举枪”；
- 把“老师是联邦学生会长指定的人”写成“リン是会长”或“会长（指老师）”。

DeepSeek 独立抽取的第一轮正确保留了持枪者为未命名少女，也正确写成“老师是会长指定的人”。第二次独立
采样仍正确处理持枪者，但把会长关系写成“联邦学生会长（指老师）”。结论是：DeepSeek 明显优于旧候选
路径，但单次生成仍有随机事实误读，不能只凭模型名宣布问题解决。

当前确定性后处理把第二种语法误读改为“联邦学生会长（指定老师的人）”，条件是 Source 本身明确包含
“会长指名/指定”的关系；不会在普通的“会长（指某人）”文本上无条件改写。

### 4.2 第 8 分片：远距离现场切换

旧结果只有 2 条 Episode，分别混入多个现场。DeepSeek 独立抽取形成 5 个事件阶段：

1. 黑服等人的秘密会议；
2. D.U. 通信恢复及相关撤退；
3. フランシス的身份与叙事宣告；
4. 老师让アロナ联系“大家”；
5. 凯撒特殊部队与ミヤコ准备突入シャーレ。

这五条不是逐句切分，且正确保留了场景和行动阶段边界。说明改进不只对第 0 分片有效。

## 5. 模型与调用路线实验

### 5.1 DeepSeek 独立抽取

观测到的三个请求：

| Source | 墙钟时间 | 总 token | 结果 |
|---|---:|---:|---|
| 第 8 分片 | 60.46 秒 | 4,376 | 边界显著改善 |
| 第 0 分片，第一次 | 167.63 秒 | 8,313 | 事实主体正确 |
| 第 0 分片，第二次 | 143.93 秒 | 8,880 | 持枪者正确，会长关系仍需修复 |

DeepSeek 适合作为这批台词的主抽取模型，但供应端长尾明显。

### 5.2 让第二模型审计已有候选

- DeepSeek 查看 Qwen 错误候选时保留了两项严重错误，并把 9 条扩成 14 条；这是确认偏差和过度切分。
- GLM 对 11 条完整候选做事实角色审计：132.01 秒、14,780 token。该轮纠正了核心关系，但调用昂贵。
- GLM 只审计 2–3 条风险候选的两次请求：62.80 秒/7,654 token 和 28.84 秒/7,590 token。
  两次都没有稳定纠正已知错误。

因此，“第二模型复核”目前没有稳定净收益。功能保留用于实验，但默认关闭，不能作为全量导入的最终权威。

### 5.3 自适应事实审计触发率

最初使用任一关键词触发时，787 个分片中有 384 个触发，占 48.8%。主要误触来自普通的“会长”“少女”
和“梦”。组合条件收紧后，只有以下情况触发：

- 明确身份、正体、化名、别名词；
- 梦境与少女/持枪主体歧义同时出现；
- 会长与指名/指定关系同时出现。

触发数降为 88/787，即 11.18%。虽然成本已经受控，但由于第二模型质量收益尚未成立，默认仍为 `off`。

## 6. 已实施且保留的修复

### 6.1 Episode

- 提示词明确章节、地点、主要人物群和行动现场切换属于边界。
- 强制核对发言者、动作主体和动作对象。
- 禁止补写 Source 未提供的职务、组织身份、全名和中文译名。
- 通用“当前时间”标签在落库前清空。
- Source 中明确的“会长指定 X”可确定性修复错误的“会长（指 X）”简写。
- 非中文专名的跨文字系统单字符抄写错误可按 Source 唯一名称修复，例如：
  - `フランシ斯 -> フランシス`
  - `カイザ尔 -> カイザー`
  - `ノノ미 -> ノノミ`
- 同文字系统近似词不做模糊替换，避免 `アロハ -> アロナ` 之类误修。
- participants 的别名只保留 Source 确实出现的部分；有原名时移除模型自创译名。

### 6.2 Concept

- canonical_name 中的别名链拆为一个显示名和多个 aliases。
- CJK 名称去重时忽略内部空格，Latin 名称仍保留有意义的空格。
- exact resolution 同时考虑 canonical_name 和 aliases。
- 多个语言别名一致指向同一 Concept 时，可越过单个宽泛中文名的歧义。
- description/embedding_text 中含模型自检文字的 Concept 被拒绝。
- 身份关系仍要求名字、别名或 Source 证据，不能由共现和描述相似直接建立。

### 6.3 时间与 Association

- 时间环审计与时间排序忽略 `polarity <= 0` 的否定边；旧报告中的一个时间环属于审计误报。
- SQLite 中的正向时间图未发现真实环。

### 6.4 Chatbot 导入入口

`import_knowledge.py` 新增仅对当前进程生效的选项：

- `--reasoning-model`
- `--fallback-model`
- `--episode-factual-audit off|adaptive|always`
- `--episode-factual-audit-model`

embedding 模型不受这些参数影响，仍只允许同一模型重试，不允许 fallback。

## 7. 当前推荐导入配置

下一轮小样本候选建议使用：

```text
主抽取模型       deepseek-ai/DeepSeek-V3.2
推理 fallback    zai-org/GLM-4.5V
边界审计         combined + adaptive
事实角色审计     off
Episode/Concept embedding  Pro/BAAI/bge-m3
embedding dtype  float32（RAM 与 SQLite 一致）
prepare workers  2
relation workers 2
relation batch   12
```

不再使用 `--episode-audit-always` 做全量导入。该选项只用于少数人工质量样本，因为曾出现 300 秒超时和
数分钟长尾。

## 8. 恢复全量导入前的门禁

先从 787 个 Source 中冻结 12–20 个分层样本：

- 4 个多场景/章节切换；
- 4 个梦境、转述、身份或职务关系；
- 4 个普通短对话；
- 可选 4–8 个回忆、推测、多语言别名和长 record。

必须满足：

1. 严重发言者/动作主客体错误为 0；
2. 严重身份或职务错误为 0；
3. 不同现场的强制合并错误为 0；
4. 不出现逐句对白式碎片化；
5. 新 participants 中没有 Source 未提供且无法映射回原名的专名；
6. 通用 story_time 标签为 0；
7. 新 Concept 无 meta commentary、alias chain 和新增错误 identity 边；
8. 数据库审计、向量维度、Source/Episode/Concept/Association 闭包全部通过。

通过后才创建新的全量候选库。旧暂停候选不续写，以免混合不同 prompt_version 和后处理规则。

## 9. 验证资产

- `validation/serihu-full-paused-quality-v2-audit-20260831.json`
- `validation/formal-baseline-quality-v2-audit-20260831.json`
- `validation/serihu-quality-pilot-v3-audit-20260831.json`
- `validation/serihu-quality-pilot-v3-segment0-deepseek-independent-20260831.json`
- `validation/serihu-old-segment8-deepseek-independent-20260831.json`
- `validation/serihu-quality-pilot-v4-segment0-deepseek-current-20260831.json`
- `validation/serihu-quality-pilot-v4-segment0-deepseek-glm-factual-20260831.json`
- `validation/serihu-quality-pilot-v4-segment0-subset-factual-20260831.json`
- `validation/serihu-quality-pilot-v4-segment0-qwen-subset-factual-20260831.json`
- `logs/memory/episode-segment-fidelity-audit.jsonl`

## 10. 结论

本轮最重要的结论不是“再加一个审计模型”，而是把三个问题分开处理：

1. **边界问题**：DeepSeek 直接抽取与更严格边界提示已显示稳定收益；
2. **可确定的名称/语法问题**：用 Source 约束的确定性后处理解决，避免再次调用模型；
3. **开放式事实判断**：第二模型复核仍不稳定，只能作为实验建议，不能自动覆盖主结果。

因此全量导入仍不应立即恢复。下一步应先完成 12–20 个冻结 Source 的新配置小样本门禁，而不是继续在旧候选库
中追加数据。
