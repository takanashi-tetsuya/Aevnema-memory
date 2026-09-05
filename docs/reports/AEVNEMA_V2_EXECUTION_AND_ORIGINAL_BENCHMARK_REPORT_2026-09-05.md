# Aevnema v2 计划执行与原始 Benchmark 报告

报告日期：2026-09-05  
范围：自收到 `Aevnema_contextual_association_revised_plan_v2_2026-09-05.md` 起的实际代码、配置、导入与验证工作。  
测试对象：独立测试库 `validation/contextual-association-benchmark-20260905/testing-kb.db`，不包含正式知识库写入。

## 执行结论

计划中的 M2（EvidenceSlot/slot selector）和 M3（残差门控双键联想接入）已实现；M0、M1、M4、M5、M6 只完成部分，不能宣称 v2 全部完成或已达到上线条件。

测试库已重建至 560/564 文件 completed（99.29%），数据库完整性正常，但仍有 3 个 partial 与 1 个 failed，因此它是“近全量诊断库”，不是计划要求的全量 clean KB。

原始五题 Benchmark 已完成四组 120 秒兼容重跑。最佳单次关键词覆盖为 10/15（66.67%，关闭 BGE reranker）。不过当前测试库没有 contextual edge/cue prototype，且原 runner 没有提供 contextual 机制所需的 slot 向量；四组运行均产生 0 个 contextual candidate。因此此次结果是原始检索诊断，不能作为“条件联想有效/无效”的因果结论。

最紧急的未处理风险是导入 ledger 的运行时配置快照可能含敏感模型凭据。原始 ledger 不应分发；应立即轮换相应密钥，并在后续改动中对配置快照脱敏。

## 一、原始五题 Benchmark

### 口径与执行方式

该 Benchmark 固定 5 个问题，每题有 3 个预声明关键词，共 15 项。分数是最终前 8 条 Episode 证据中命中的关键词数量，不是回答的真实性、完整性或用户体验评分。

历史参考记录为 9/15（60%），但本次结果与其并非严格同条件可比：当前 `Paragraph` 为 0、模型规划存在波动、知识库仍有 4 个未 completed 文件，且本次按独立快照运行。

首次按原脚本固定的 20 秒总时限运行 `baseline + BGE reranker`，5/5 均在模型意图解析和后续本地检索之间触发 deadline，结果为 0/15。该失败回执已保留，没有被覆盖。

为使诊断能完成，新增了 `--deadline-seconds` 参数，默认值仍是原来的 20 秒；兼容重跑仅将该值明确设为 120 秒。四个 arm 各自使用从同一测试库复制出的独立 SQLite 快照，避免 `Association.use_count/last_used` 等使用元数据互相污染或写入主测试库。模型角色保持为 DeepSeek-V3.2（检索规划）、GLM-4.5V（仅失败回退）、BGE-M3（embedding），BGE reranker 只在启用 arm 中使用；模型请求上限为 16，正文日志关闭。

### 结果

| 模式 | BGE reranker | 完成题数 | 关键词覆盖 | 平均耗时 | 最大耗时 |
|---|---:|---:|---:|---:|---:|
| 严格历史预算 baseline | 开 | 0/5 | 0/15（0.00%） | — | — |
| baseline | 开 | 5/5 | 8/15（53.33%） | 29.760s | 33.581s |
| baseline | 关 | 5/5 | 10/15（66.67%） | 32.324s | 42.670s |
| contextual | 开 | 5/5 | 9/15（60.00%） | 31.270s | 41.276s |
| contextual | 关 | 5/5 | 10/15（66.67%） | 26.628s | 37.539s |

单次结果显示关闭 reranker 的两个 arm 都达到 10/15；这不是足以作出 reranker 优劣结论的统计实验。当前链路会调用模型规划，单次结果仍可能受模型输出和候选排序波动影响，必须在冻结 query intent/vector 的新基准中复现后才可归因。

更重要的是，两个 contextual arm 的每个问题均为 `candidate_count=0`、无 attached contextual edge、无 treatment selected edge；测试库中 `association_mode=contextual_recall` 和 `association_cue_prototype` 的行数也都是 0。因而 8→9 的差异不能归因给条件联想，最多是普通检索路径的运行差异。

20 秒失败的直接原因也已定位：请求级 deadline 只在阶段边界检查，已开始的 DeepSeek 请求没有获得“剩余时间”并不可中断。首轮意图解析约为 12.80–24.39 秒，后接 embedding 与本地检索后才抛出超时。这揭示了 M5 尚未完成，而不是数据库、429、Windows socket 或未授权模型错误。

### Benchmark 产物

- [严格 20 秒失败回执](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-after-retry-20260905/original-five-question-baseline-bge_enabled.json)
- [四 arm 120 秒结果目录](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-extended-deadline-20260905T215025)
- [baseline + BGE](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-extended-deadline-20260905T215025/original-five-question-baseline-bge_enabled.json)
- [baseline，无 BGE](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-extended-deadline-20260905T215025/original-five-question-baseline-bge_disabled.json)
- [contextual + BGE](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-extended-deadline-20260905T215025/original-five-question-contextual-bge_enabled.json)
- [contextual，无 BGE](/C:/Users/Admin/PycharmProjects/PythonProject/validation/contextual-association-benchmark-20260905/original-five-question-extended-deadline-20260905T215025/original-five-question-contextual-bge_disabled.json)

## 二、测试知识库重建结果

导入来源为 `event`、`favor`、`main` 三个剧情目录。账本状态为：560 completed、3 partial、1 failed。

| 项目 | 当前值 |
|---|---:|
| 文件完成率 | 560/564（99.29%） |
| Sources | 4,723 |
| Episodes | 13,127 |
| Concepts | 9,215 |
| Associations | 36,646 |
| Paragraphs | 0 |
| `PRAGMA quick_check` | ok |
| 外键错误 | 0 |
| running extraction run/task | 0/0 |
| contextual recall edge / cue prototype | 0 / 0 |

未完成项按当前真实状态保留，没有伪造为 completed：

1. `favor/10070/100703.json`：旧 segment 14 无有效 Episode；后续 GLM-4.5-Air 对抗审核确认其为控制卡，任务已标记 skipped，但 ledger 保留历史 partial。
2. `main/31090.json`：同类 next-episode 标题预告；审核确认 safe skip，任务已标记 skipped，ledger 仍保留 partial。
3. `favor/10108/101085.json`：已持久化 8 Sources、12 Episodes；仍有 1 个 second-pass 清理任务失败，因此是实际 partial。
4. `main/33235.json`：adapter 本地判定无可用文本块，没有调用模型，也没有创建 Source/Episode，故为 failed。

`Paragraph=0` 是当前配置状态，并非完整性检查失败；但它使得本次 Benchmark 与启用 paragraph 检索的历史条件不完全可比。

## 三、自计划书以来的实际改动

### 1. v2 残差门控双键联想与 selector

| 文件 | 已完成改动 |
|---|---|
| `src/memory_demo/types.py` | 扩展请求级 `QueryVectorBundle`、`EvidenceSlot`、`SlotCandidate`、`ContextualSlotHit` 和效用观测结构；slot 不写入长期知识库。 |
| `src/memory_demo/retrieval/coverage.py` | 新增确定性加权 set-cover：required slot 优先、直接相关性/来源质量加权、同 Source 冗余惩罚；contextual bonus 只能填补新槽。实现 Treatment/Masked、single-edge、leave-one-out 的本地归因 receipt。 |
| `src/memory_demo/retrieval/contextual_association.py` | 双键 matcher 改为“已激活基础 anchor + unresolved slot”残差门控；复用请求向量和本地索引，`external_calls=0`；要求目标对当前 slot 具有本地支持。 |
| `src/memory_demo/retrieval/engine.py` | contextual target 不再作为零分普通 seed，而作为 slot candidate；先做 masked selector，仅在有缺槽时运行 contextual；输出 treatment/masked、缺槽和严格归因 trace。shadow 时真实计算 treatment，但返回 masked。加入 request-level `deadline_seconds` 和阶段检查。 |
| `src/memory_demo/repositories/association.py` | 收紧 retrieval-only contextual edge 的读写和生命周期：generation-0 endpoint、probation/active/retired、query hash 去重、harm/no-op 衰减；默认禁止 probation→active。 |
| `src/memory_demo/app.py`、`src/memory_demo/retrieval/__init__.py` | 应用启动重建 context/need cue 索引，并将 v2 matcher、repository、配置接入查询引擎。 |
| `src/memory_demo/associations/builder.py` | 关联草稿保留 Episode 的来源、认识状态和 generation，避免推理关系掩盖源证据。 |

### 2. 导入并发、软暂停、崩溃恢复和传输保护

| 文件 | 已完成改动 |
|---|---|
| `benchmarks/import_corpus.py` | 增加文件级 ledger、`--file-workers`、`--workers`、`--retry-interrupted`、`--retry-failed`、`--pause-file`；失败重试在只读 preflight 通过后才进行，并保存尝试历史。暂停时停止派发并等待已启动文件收尾。 |
| `src/memory_demo/ingestion/interruption.py` | 增加 PID-backed `.active` lease；下次启动检测陈旧 running run，逐 source ownership 审核后恢复；不确定所有权时 fail-closed。 |
| `benchmarks/recover_interrupted_file_runs.py` | 新增人工恢复/清理工具，默认 dry-run；只有显式 `--include-failed --apply` 才能处理 failed，completed/partial 不会自动删除。 |
| `src/memory_demo/repositories/extraction.py` | 只清理已证实中断 run 的关联、Episode、任务和无引用 Source；终态 run 的替换需要额外 source ownership 审计。 |
| `src/memory_demo/database.py` | SQLite 短写事务由进程内锁串行化、加入有限 busy retry 和一次 WAL 初始化，使 16 个文件 worker 可以并行准备/请求而不争抢 writer。 |
| `src/memory_demo/llm/client.py` | 增加请求 semaphore、共享 HTTPS adapter/连接池；只串行新 TCP/TLS 连接，不串行已复用请求。Windows socket resource error 触发共享 transport circuit，阻止重试放大并让文件安全中断待恢复。 |
| `src/memory_demo/ingestion/pipeline.py` | 区分 transport unavailable、SQLite busy 与中断状态；关系批次可并发准备、集中写入；中断时关闭 running task。 |
| `docs/SAFE_IMPORT_PAUSE.md` | 新增中文安全暂停、旧进程恢复、模型并发和 SQLite 写入操作规范。 |

### 3. Blue Archive 输入卫生与空 Episode 对抗审核

| 文件 | 已完成改动 |
|---|---|
| `src/memory_demo/adapters/blue_archive.py` | 精确过滤纯 `#nextepisode` 预告卡，以及无韩文叙事、全为 `#st/#clearST` 本地化控制语的 UI 卡；不再宽泛删除一般 `#` 指令或剧情文本。 |
| `config/prompt_config/memory_prompts.py` | 新增 `empty_episode_adversarial_audit_v1` 结构化提示词。 |
| `src/memory_demo/llm/validation.py` | 审核必须覆盖每一条实质物理 Source 行；伪造 quote、漏行、跨行 safe skip、事件/歧义分类均 fail-closed。 |
| `src/memory_demo/ingestion/extractor.py` | 增加 `EmptyEpisodeExtraction`、`EmptyEpisodeAudit`；审核模型必须不同于本次所有实际提取尝试模型；只有 `safe_skip` 才跳过，`episode_required` 仅允许一次受证据约束的救援提取，`uncertain` 失败关闭。 |
| `src/memory_demo/ingestion/pipeline.py` | 将审核接入真实导入路径；safe skip 不持久化 Source/Paragraph/Episode，而写入可追溯 receipt。 |
| `src/memory_demo/config.py`、`.env.example` | 明确模型角色：DeepSeek-V3.2 主提取、GLM-4.5V 回退、GLM-4.5-Air 空 Episode 独立审核、BGE embedding/rerank；未加入多余的模型白名单机制。 |

早期有 3 次手动历史审核误用了未授权的 GLM-5.1。用户指出后已立即停止；它未被写入当前程序或配置，最终两条有效审核结论均由 GLM-4.5-Air 重新给出。保留旧日志只用于审计，不作为有效审核依据。

### 4. Benchmark、运行验证与依赖配置

| 文件 | 已完成改动 |
|---|---|
| `benchmarks/manifests/original_five_question_quality_benchmark.json` | 冻结原五题及每题 3 个关键词，保留 15 项关键词覆盖口径。 |
| `benchmarks/run_original_five_question_benchmark.py` | 提供 baseline/contextual × BGE 开/关四 arm runner，记录候选、选中 Episode、关键词覆盖、耗时及 contextual trace；本次新增 `--deadline-seconds`，默认保留 20 秒历史预算。 |
| `validation/contextual-association-benchmark-20260905/run_benchmark_suite.ps1` | 四 arm 编排脚本。其 564/564 completed gate 与当前 560/564 状态不兼容，本次没有依赖它启动。 |
| `pyproject.toml` | 增加 `requests>=2.32` 以支持连接池客户端，并排除生成目录避免 pytest 误收集。 |

## 四、计划完成度对照

| 计划项 | 状态 | 证据与边界 |
|---|---|---|
| M0 冻结基线与口径 | 部分完成 | 有原五题 manifest、独立测试库、历史快照和可重跑 runner；未见计划要求的三题九槽正式 manifest、逐 Source span 冻结报告。 |
| M1 知识库卫生 | 部分完成 | 已完成 564 文件的近全量重建、卡片过滤与空 Episode 审核；仍为 560 completed、3 partial、1 failed，不能称完整 clean KB。 |
| M2 EvidenceSlot 与 selector | 已实现 | `EvidenceSlot`/`SlotCandidate`、local support map、set-cover、缺槽 trace 与 Source 冗余惩罚均已落地。 |
| M3 残差门控与锚点门 | 已实现 | active anchor、unresolved slot、target local support、contextual SlotCandidate、local-only matcher、shadow selector 均已落地。 |
| M4 严格 LOO 与效用更新 | 部分完成 | Treatment/Masked、single-edge、LOO receipt 已实现，promotion 默认关闭；尚未接通将 `strict_attribution` 自动提交为 `ContextualUtilityObservation` 的运行时闭环。 |
| M5 缺槽升级、回答合同、DeadlineBudget | 部分完成 | 有 request-level deadline 和 phase guard；没有 `DeadlineBudget`、fallback reserve 或完整的“只处理 missing slot”升级闭环。20 秒失败正是该缺口的运行证据。 |
| M6 Shadow 与 Canary | 部分完成 | shadow Treatment/Masked 已实现，默认 contextual disabled + shadow；未见 endpoint-limit=1 canary 编排或 E0–E7 真语料验收结果。 |
| E0–E7 实验 | 未完成 | 本报告的原五题诊断不能替代冻结三题九槽、Treatment/Masked、hard-negative、LOO 与回答级 A/B 实验。 |

因此准确总体判断是：M2/M3 完成；M0/M1/M4/M5/M6 部分完成；导入健壮性和空 Episode 防误处理是必要的工程增强；v2 尚未满足 plan 的 Go 条件。

## 五、验证、限制与风险

本轮实际验证完成了以下事项：

- 4 个 Benchmark JSON 均产生且每个 5/5 完成（120 秒兼容重跑）。
- 测试库 `quick_check=ok`、外键错误为 0、无 running run/task。
- benchmark runner 的语法编译与 `--deadline-seconds` CLI 帮助验证通过。
- 当前可用解释器缺少 `pytest`，完整 pytest 未能在本轮重新执行（收集前即报 `No module named pytest`）。项目既有报告中的 414 passed 是历史记录，不是本报告重新验证的断言结果；新参数也尚无专门 pytest 覆盖。

需要优先处理的限制与风险：

1. 导入 ledger 的运行时配置快照可能保存敏感模型凭据。应立即轮换密钥，并让 ledger 仅写入脱敏配置；本报告不链接或复制该文件。
2. 当前 Benchmark 的 contextual arm 没有任何 contextual candidate，不能用来验收 M3/M4；必须建立冻结的三题九槽金标准、source span、query vector bundle 和至少若干 probation edge。
3. 20 秒 deadline 不是端到端可取消预算；需要实现 M5 的 `DeadlineBudget`、剩余预算传递和确定性证据型 fallback。
4. `favor/10108/101085.json` 的 second-pass 失败以及 `main/33235.json` 的无可用文本块需要按独立数据质量流程处理，不能以改 ledger 状态替代修复。
5. 现有 runner 虽然不修改证据内容，但查询路径可能更新关联使用元数据；以后所有 benchmark 应继续在每 arm 独立快照上执行。

## 六、建议的下一步（未在本报告中擅自执行）

1. 先轮换凭据并实现 ledger 配置脱敏，再开放任何原始导入日志或状态文件。
2. 补齐 M5：request-wide DeadlineBudget、网络请求剩余预算、fallback reserve 和检索完成后的确定性回答。
3. 生成并冻结计划书要求的三题九槽 manifest（含 Source span 与 Episode alternatives），再用已构造 cue/edge 的开发库运行 E0–E4。
4. 仅当 strict Treatment>Masked、single-edge/LOO、hard-negative 和 source closure 都通过后，才考虑 M6 的 endpoint-limit=1 canary；继续保持 active promotion 关闭。
5. 在修复或明确分类 4 个非 completed 文件后，生成真正的全量 clean KB snapshot 与 hash，再重跑可比基准。
