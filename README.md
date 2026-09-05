# Associative Memory

一个以 SQLite、NumPy、embedding 和 Association 图为核心的长期记忆引擎。项目只负责材料导入、证据检索、图关系增长和可审计结果，不负责聊天平台、用户身份、角色语气或最终回复。

## 数据模型

- `Source`：长度受控的原始证据文本。Episode、Paragraph 通过 `source_id` 回到原文。
- `Episode`：自包含的事件、状态或可检索事实，保存参与者、时间语义、证据状态、generation 和 float32 embedding。
- `Paragraph`：可选的局部语义召回层，只提高候选召回，不替代 Source 或 Episode。
- `Concept`：实体、抽象概念、情绪、关系主题及其多语言别名。
- `Association`：Episode/Concept 之间的有向关系，保存关系文本、权重、证据状态、generation、前提和使用统计。

`generation=0` 表示直接材料或不依赖推论的结构；更大的 generation 表示推断链离直接证据更远。可信度与 generation 分开记录。

Embedding 在 SQLite 和 RAM 中统一使用 float32。SQLite 是持久化真实来源，RAM 索引可以从数据库重建。

## 代码边界

```text
src/memory_demo/
├── __init__.py          公共 Python API
├── app.py               组合根：数据库、仓库、索引和模型客户端
├── contracts/           与上层应用共享的请求/回答合同
├── adapters/            JSON/TXT 输入兼容层
├── ingestion/           分片、提取、审计、目录顺序和事务写入
│   └── ordering.py      通用目录时间线分组与自然文件排序
├── retrieval/           向量、稀疏、Paragraph 和图检索
│   ├── context.py       有界 Source 证据摘录
│   └── query_planning.py 模型 intent 到证据槽的确定性转换
├── associations/        Association 建立、增长和遍历
├── chronology/          时间顺序与人工审查
├── repositories/        SQLite 表级读写
├── embeddings/          float32 编码和 RAM 矩阵索引
├── llm/                 模型调用、prompt facade 和结构校验
├── config.py            引擎配置
├── database.py          连接工厂、事务和 schema 初始化
└── types.py             引擎内部领域类型

config/prompt_config/    所有模型提示词
benchmarks/              当前离线评测与运维工具
benchmarks/support/      评测共用代码，不属于运行时包
validation/              实验结果和固定验收资产
tests/                   单元与回归测试
_archive/                历史实验或重构回退资产
```

上层应用只应依赖：

```python
from memory_demo import AppConfig, Database, MemoryApplication
from memory_demo.contracts import RequestAnswerContract
```

不要从 `benchmarks` 引用生产逻辑，也不要让聊天机器人直接操作 repositories。

## 导入流程

```text
文件
→ InputAdapter 解析
→ NaturalSegmenter 形成 SourceSegment
→ Source 写入
→ 文档锚点/全文结构理解（按配置）
→ Episode 提取与证据审计
→ Concept 提取、别名解析与去重
→ float32 embedding
→ Episode/Concept/Paragraph 写入
→ 直接 Association 与推论候选
→ 事务提交
→ RAM 索引更新
→ 日志记录成功、部分失败和重试信息
```

Embedding 模型没有 fallback。推理、提取和审计模型可以按配置重试或切换备用模型。导入失败必须保留文件、Source、任务、prompt 版本、模型输出和异常信息，目录导入可以继续处理其他文件并汇总失败。

## 查询流程

```text
问题与请求级 RetrievalPlan
→ 意图/槽位解析
→ Episode、Concept、Paragraph、稀疏索引并行召回
→ Source cohort 与 Association 候选扩展
→ 可选 cross-encoder/LLM rerank
→ Coverage Selector 保证实体和事实槽覆盖
→ 时间线与证据状态检查
→ 选择可回答证据和 Association 路径
→ 可选候选边审计与增长
→ 返回证据、质量指标、路径和完整 trace
```

`light`、`standard`、`deep` 是请求级预设，不应修改共享全局配置。普通问题先用 standard；证据槽缺失、冲突或时间线不完整时再升级 deep。

引擎不认识具体作品的章节、人物或地点。调用方可以通过请求级 intent 提供实体、关系、时间、因果和答案槽；确定性规划层只保存这些结构，不补写领域结论。

## 安装与配置

```bash
python -m pip install -e .
cp .env.example .env
```

至少配置：

```dotenv
SILICONFLOW_API_KEY=
MEMORY_DB_PATH=database/memory_demo.db
MEMORY_LOG_DIR=logs
```

模型、分片、审计、检索和增长参数位于 `.env`、`config/model_config.toml` 与 `config/prompt_config/`。模型输出维度固定为 1024，持久化和内存 dtype 固定为 float32。

## CLI

```bash
memory-demo init
memory-demo prepare ./documents
memory-demo import ./documents
memory-demo query "问题"
memory-demo stats
memory-demo rebuild-index
memory-demo timeline --help
memory-demo association --help
memory-demo concept --help
memory-demo episode --help
```

在未安装 editable package 时可使用：

```bash
python -m memory_demo.cli --help
```

## 测试

```bash
python -m pytest -q
```

`pyproject.toml` 将测试限制在 `tests/`，不会递归执行 `_archive`、validation、日志或数据库目录中的历史脚本。

## 设计约束

- 不在 prompt 或代码中写死某一部剧情的结论。
- Source 原文不可被模型改写；摘要、事实与推论分层保存。
- 不把相关性分数当成完整证据覆盖。
- 不把 generation 当成可信度；两者分别参与最终判断。
- 新增长边必须保存依据、关系文本和审计结果，并能被检索 A/B 评估。
- 数据库是真实来源；内存索引失败时必须可以重建。
- 实验阶段编号只存在于 benchmarks 或 `_archive`，不能进入生产包 API。
