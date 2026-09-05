# 可恢复文件导入的安全暂停流程

`benchmarks/import_corpus.py` 支持以文件为边界的安全暂停。暂停不会打断正在处理的文件，因此不会产生需要清理的半个提取 run。

## 启动时启用暂停控制

为导入命令加入 `--pause-file <控制文件路径>`。控制文件不存在时正常导入；创建该文件后，导入器停止派发新的文件任务，等待已开始的文件完成，然后在 ledger 中写入 `last_process.status = "paused"` 并退出。

```powershell
New-Item -ItemType File `
  C:\Users\Admin\PycharmProjects\PythonProject\validation\contextual-association-benchmark-20260905\import.pause
```

恢复前删除控制文件，再以相同的数据库、账本和并发参数重新执行导入命令。已完成文件会跳过，未开始文件继续导入。

```powershell
Remove-Item -LiteralPath `
  C:\Users\Admin\PycharmProjects\PythonProject\validation\contextual-association-benchmark-20260905\import.pause
```

## 旧进程没有暂停控制时

不要直接以 `--retry-interrupted` 续跑。先停止旧进程，再使用下面的工具进行 dry run，确认只识别到 ledger 中 `running` 的文件 run：

```powershell
python benchmarks/recover_interrupted_file_runs.py <测试库路径> --ledger <账本路径>
```

确认报告后加 `--apply`。工具会删除这些未完成 run 所创建的 Episode、关联、任务和无引用 Source，将对应 ledger 条目标记为 `interrupted`，并执行外键检查。它不会删除 completed 或 partial 文件。

恢复时使用 `--retry-interrupted`。partial/failed 文件仍需单独审计和重导入，避免把已保留的有效数据重复写入。

对于旧版本在短暂网络故障时留下的 `failed` 文件，先进行 dry run，再明确使用 `--include-failed --apply` 清理这些失败 run 的专属产物，最后用 `--retry-interrupted` 恢复。该选项永远不会清理 `completed` 或 `partial` 文件。

```powershell
python benchmarks/recover_interrupted_file_runs.py <测试库路径> `
  --ledger <账本路径> --include-failed
```

## 突然中断与自动恢复

导入器为账本创建一个原子活动租约（`<ledger>.active`）。同一账本不能同时启动第二个导入器；这能避免两个进程同时写入同一批文件。正常的 Ctrl+C、SIGTERM 或 Ctrl+Break 会进入可审计的收尾路径。

如果进程被任务管理器结束、断电或崩溃，租约和 ledger 中的 `running` 条目会保留。下一次启动时，导入器会先确认旧 PID 已不在运行，再以数据库为准恢复：

- 数据库已完成或部分完成、但 ledger 尚未来得及更新的文件，会保留数据库成果并回写正确状态；
- 仍在运行或已中断、且能与 ledger 一对一对应的 run，会删除其专属 Episode、关联、任务和无引用 Source，再标为可用 `--retry-interrupted` 重试的中断；
- 无法确认归属的记录会拒绝自动清理并停止启动，避免误删其他导入的数据。

每次恢复都会记录 run、文件键和删除汇总到 ledger 的 `last_recovery`。因此不要手工把 `running` 改为 `interrupted`；直接重新启动导入器即可先完成一致性恢复。
# 模型请求并发上限

文件并发和模型请求并发是两层不同的控制项。`--file-workers 16` 可以让 16 个文件准备、提交和等待，但同一时刻实际发出的模型 HTTP 请求默认最多为 8 个。模型客户端使用共享连接池复用已建立的 HTTP 连接；重试也会回到该连接池，而不是每次都重新建立 Windows 套接字。

当池子为空、连接失效或需要扩容时，新的 TCP/TLS 连接会通过进程级建连闸门逐个建立。这个闸门不会串行化已经建好的连接上的请求；它只阻止多个线程在同一时间创建新的 Windows 套接字。模型请求上限仍作为第二道保护，避免连接恢复时耗尽本地套接字资源并触发 `WinError 10013`。

如需调整该安全上限，在启动导入前设置 `MEMORY_MODEL_MAX_CONCURRENT_REQUESTS`（最小值为 1）。通常保持默认值；只有在确认本机网络和模型服务都能稳定承受时才提高。

## SQLite 写入并发

`--file-workers 16` 仍会让文本准备和模型调用并行，但 SQLite 对同一个数据库只有一个写入者。导入器现在会在进程内把短暂的写事务排队：不会降低文件与模型处理的并发，只会让实际写入测试库的那一小段依次提交。WAL 模式也只在进程首次打开数据库时配置一次，避免每个连接重复争用该设置。

如果数据库被本进程外的工具长期占用，导入器会在有限等待后把文件标记为可恢复的 `interrupted`，原因是 `sqlite_writer_busy`，清理该文件未完成 run 的专属数据，并停止派发新文件。关闭占用数据库的外部进程后，用 `--retry-interrupted` 恢复；不要把这类暂态竞争当成永久 `failed` 重跑。
