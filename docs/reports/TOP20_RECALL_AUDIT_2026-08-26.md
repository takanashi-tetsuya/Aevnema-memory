# 无增长边 Top-20 Recall 审计

**日期：** 2026-08-26  
**目标：** 判断基础 Top-k 检索在完全关闭查询期 Association growth 时，能否可靠达到 `Recall@20 > 95%`。

## 1. 口径

本审计中的 recall 不是“答案看起来正确”，而是人工预注册证据组覆盖率：

```text
Recall@20 = Top-20 中至少命中一个 Episode 的证据组数 / 所有必需证据组数
```

对含 8 个证据组的问题，命中 7/8 只有 87.5%；要超过 95%，必须 8/8 全部命中。

区分两个 Top-20：

- `candidate_top20`：图遍历/证据分配之前，按候选得分取前 20 个 Episode；
- `selected_top20`：经过现有原子查询锚点和证据分配器后，最终交给回答器的 20 个 Episode。

测试模式：

- `vector_only`：纯 Episode/Concept embedding，`graph_max_hops=0`；
- `graph_static`：允许读取导入期已有 Association，但 `growth_max_rounds=0`；
- Paragraph 关闭；
- 使用上一阶段冻结的 10 份查询规划和同一“古圣堂”8 证据组标准，因此没有答案生成随机性，也没有新边写入。

## 2. 精确 Top-20 结果

| 模式 | 层次 | 平均 Recall@20 | 最低 | 最高 | 超过 95% 的次数 |
|---|---|---:|---:|---:|---:|
| vector_only | candidate_top20 | 66.25% | 50.0% | 75.0% | 0/10 |
| vector_only | selected_top20 | 78.75% | 62.5% | 100% | 1/10 |
| graph_static | candidate_top20 | 70.0% | 50.0% | 87.5% | 0/10 |
| graph_static | selected_top20 | 77.5% | 62.5% | 87.5% | 0/10 |

纯向量 selected Top-20 的 10 次结果为：

```text
75%, 75%, 75%, 87.5%, 75%, 87.5%, 75%, 100%, 75%, 62.5%
```

一次 100% 是偶然成功，不构成可靠性。若把“超过 95%”理解为每次都覆盖全部 8 组，当前成功率只有 1/10；静态图是 0/10。

## 3. 七题宽预算交叉检查

此前七道高难问题包含 56 个预注册证据组。纯向量模式即使把最终预算放宽到 Top-30，也只命中 49/56：

```text
micro Recall@30 = 87.5%
```

因此，当前系统不可能从现有结果合理外推“Recall@20 > 95%”。即使 Top-20 分配器经过调优，这个目标也需要新的候选召回与精排机制，而不是只改变 k 的数值。

## 4. 为什么基础 Top-k 不够

1. 每题要求同时覆盖 7—9 种语义不同的证据，不是找到一个最相似答案段落。
2. Dense embedding 会让多个描述同一主主题的 Episode 占据前排，挤掉地下结构、人物猜测、历史制度等较弱槽位。
3. Episode 摘要可能省略原文专有词或细节；纯 Episode embedding 无法检索已经被摘要丢失的词。
4. 原子查询规划存在随机性。某个证据槽没有被拆成独立查询时，增大单一查询 Top-k 也未必补回。
5. 静态 Association 只能略微改变候选可达性，当前没有稳定改善最终 Top-20 的证据覆盖。

## 5. “不用增长边”仍可采用的改进路线

完全关闭增长边，不等于只能使用单阶段 dense Top-20。更合理的目标架构是：

```text
多原子查询
  + Episode dense retrieval
  + Source/原文关键词或 BM25 sparse retrieval
  + Concept/alias 精确命中
        ↓
候选池 Top-100/200，目标 candidate recall > 99%
        ↓
按问题槽位分组的 evidence-aware reranker
        ↓
最终 20 个 Episode，目标 Recall@20 > 95%
```

关键是“先宽召回，再压缩到 20”，而不是要求原始 dense Top-20 直接达到 95%。Paragraph 当前可以继续关闭；Sparse 通道可直接索引 Episode 文本和 Source 原文，不必把 Paragraph 变成新图节点。

## 6. 下一步验收门槛

建议把无增长检索拆成两项硬指标：

1. `candidate Recall@100 ≥ 99%`：证明证据没有在第一阶段丢失；
2. `selected Recall@20 > 95%`：证明证据分配器能把宽候选压缩成有限上下文。

至少覆盖当前 7 道题、56 个证据组，并对查询规划做多次冻结重放。不能用单次 8/8 宣称通过。

## 7. 资产

```text
benchmarks/audit_top20_replay.py
validation/top20-no-growth-replay-audit.json
validation/evaluation-stage4-network-v321-vector/evaluation-report.json
validation/stage4-network-evidence-manifest.json
```

