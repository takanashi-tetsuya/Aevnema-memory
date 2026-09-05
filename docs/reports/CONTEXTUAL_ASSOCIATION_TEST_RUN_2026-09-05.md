# Contextual Association 测试记录

测试日期：2026-09-05  
测试对象：Aevnema 核心记忆系统、chatbot 接入层

## 结果摘要

| 范围 | 结果 |
|---|---|
| 核心 unittest | 383 项通过 |
| 双键联想专项 | 5 项通过 |
| 核心 + chatbot AST 解析 | 111 个 Python 文件通过 |
| chatbot 配置加载 | 通过；默认 `enabled=false`、`shadow=true` |
| chatbot 完整 unittest | 186 项通过 |

## 核心测试

核心套件使用项目可用的 Python 运行时，并设置核心 `src` 为导入路径：

```text
Ran 383 tests in 8.926s
OK
```

双键专项单独运行：

```text
Ran 5 tests in 0.184s
OK
```

专项覆盖：

1. blockwise `search_many` 与普通搜索结果一致性；
2. 请求级 embedding bundle 去重；
3. context + need 双门槛匹配；
4. future-only utility 记录与 probation→active 晋升；
5. overlay 对新边和 cue 的隐藏/回滚。

数据库启动和 contextual storage audit 也通过，当前审计结果为 `ok=true`，孤立 cue、非法向量、跨域边、非法目标和非法 claim level 均为 0。

## chatbot 测试

使用 chatbot 自带虚拟环境运行完整 unittest，共 186 项，全部通过。此前出现的 3 项 Windows 临时文件清理错误已修复。

修复位于 `src/cli/import_data.py`：回滚函数现在使用 `closing(sqlite3.connect(...))` 与事务上下文的组合，保证提交/回滚后立即关闭连接；三个测试夹具也显式关闭验证连接。因此 Windows 可以在测试退出时删除临时数据库。

另外，测试日志文件因当前运行账户的文件权限产生警告；它不影响测试断言。

## 配置 smoke

chatbot 配置成功加载并确认：

```text
contextual_association_enabled = False
contextual_association_shadow = True
endpoint_limit_light = 1
endpoint_limit_standard = 2
endpoint_limit_deep = 4
```

因此当前部署默认不会改变已有回答；启用后仍可先保持 shadow 进行观测。

## 结论

实现层的核心功能、迁移、索引、双键匹配和最小塑性闭环均通过。当前唯一重复出现的失败是 Windows 测试清理阶段的 SQLite 句柄未释放，建议作为单独的测试基础设施修复项处理。尚未据此宣称真实语料 E0–E8 实验或大规模召回率目标达标。
