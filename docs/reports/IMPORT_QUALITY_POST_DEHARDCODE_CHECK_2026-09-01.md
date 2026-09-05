# 去剧情规则后导入质量复验（2026-09-01）

## 结论

没有发现当前版本的导入质量出现明显下降。

去硬编码改动没有触及导入提示和导入执行路径：改动前后，`QUERY_SYSTEM` 之前的全部导入提示
32,589 个字符逐字相同；导入管线、配置和 LLM 响应解析器也未在该改动中修改。

使用此前最敏感的阿比多斯单文件 canary，在全新数据库中按相同的 `single_pass_audited`、float32、
延后推断边配置重新导入。严格审计全部通过；按独立事实槽修正评测清单后，3 个问题证据全部解析，
Recall@5/10/20 均为 100%，最差证据槽 rank 为 2。

## 对照配置

```text
输入：validation/serihu-quality-v32-abydos-audit-corpus-20260901/
Episode profile：single_pass_audited
Embedding：Pro/BAAI/bge-m3，1024 维 float32
推断型关系：导入时延后
数据库：独立新建，不复用历史结果
增长边：不参与检索评测
Reranker：不参与纯向量检索评测
```

## 结构与严格审计

| 指标 | 历史 v32 | 当前复验 |
|---|---:|---:|
| Source | 2 | 2 |
| Episode | 13 | 12 |
| Concept | 14 | 16 |
| 直接 Association | 21 | 28 |
| 证据映射问题 | 0 | 0 |
| Source 覆盖问题 | 0 | 0 |
| 参与者归属问题 | 0 | 0 |
| 重复 Episode 对 | 0 | 0 |
| 外键/SQLite/embedding 问题 | 0 | 0 |
| 未完成任务 | 0 | 0 |

Episode 和 Concept 数量不是质量指标。当前运行把一个场景拆得略细，同时提取了更多可复用 Concept；
没有 Source 内容、答案事实槽或持久证据丢失。

## 检索验收

按独立事实槽评测：

| 问题 | 各证据槽最佳 rank | 结论 |
|---|---:|---|
| 土地交易主体与学生会决议权 | 1, 1 | 完整命中 |
| “校内第一的笨蛋”与“两位笨蛋” | 1, 1 | 完整命中 |
| 白子翻包的原因与隐瞒判断 | 1, 2 | 两条相邻 Episode 完整命中 |

汇总：

```text
unresolved evidence questions = 0
Recall@1  = 66.67%
Recall@5  = 100%
Recall@10 = 100%
Recall@20 = 100%
MRR       = 0.8333
worst rank = 2
acceptance = passed
```

历史 v32 的三题 Recall@1 为 100%。当前 R@1 下降不是语义证据缺失，而是“白子翻包”和“星野有所
隐瞒”被拆成 Episode #11 与 #12，分别排第 1、2；对多事实问题，要求所有证据都塞进单一 Top-1 本身
不适合作为导入门槛。当前仍远高于既定 Recall@20 ≥ 95% 标准。

## 发现的评测资产问题

历史 `serihu-v28-abydos-retrieval-manifest-20260901.json` 对白子问题使用一个 matcher：

```json
{"all_terms": ["バッグ", "隠し事"]}
```

这强制两个原文词必须进入同一 Episode。当模型做出更细且合理的分段时，旧评测会把正确结果误报为
`unresolved`，使表面 Recall@20 降为 66.67%。

历史 manifest 保持不变以保留基线；本次新增
`validation/current-dehardcode-import-smoke-manifest-20260901.json`，把两个事实改为独立 matcher。
这符合系统一直采用的“多事实问题按独立证据槽评估”原则。

## 调用和耗时

| 指标 | 历史 v32 | 当前复验 |
|---|---:|---:|
| LLM request/response | 11/11 | 11/11 |
| Prompt tokens | 25,500 | 26,515 |
| Completion tokens | 9,932 | 9,987 |
| 局部覆盖补抽 | 1 | 1 |
| 蕴含审计批次 | 2 | 2 |
| 请求错误 | 0 | 0 |
| 总时间 | 239.11 秒 | 297.03 秒 |

当前耗时增加约 24%，但调用次数相同、token 只增加约 3%，且没有错误或重试；更符合供应商响应时间
波动，不是去剧情规则造成的导入流程增长。

## 资产

```text
validation/current-dehardcode-import-smoke-20260901.db
validation/current-dehardcode-import-smoke-logs/
validation/current-dehardcode-import-smoke-strict-audit-20260901.json
validation/current-dehardcode-import-smoke-dense-eval-20260901.json
validation/current-dehardcode-import-smoke-manifest-20260901.json
validation/current-dehardcode-import-smoke-dense-eval-v2-20260901.json
```

## 判断边界

本次可以否定“去剧情规则后导入质量明显下降”这一假设，但只覆盖一个高敏感 canary。它不能证明所有
文档类型都无回归。扩大导入时仍应按批次运行：严格证据审计、冻结证据槽 Recall@20、最差排名人工
抽查。尤其要避免把“必须同属一个 Episode”误当作多事实覆盖标准。
