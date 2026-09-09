# Aevnema v3.2 综合执行计划包

主文：`Aevnema_v3_2_integrated_development_experiment_plan_2026-09-08.md`。

交给代理时保留本目录，并以 `AGENT_START_HERE.md` 为入口。主文自包含；JSON为便于连续推进的结构化索引与模板，不是应用代码。

- `execution_backlog.json`：20个工作包，包含文件定位、步骤、测试、验收、失败后的继续分支。
- `experiment_program.template.json`：11项实验与授权/预算默认边界；尚未执行。
- `campaign_state.template.json`：可续接状态模板，不是运行回执。
- `source_evidence_index.json`：本次实际依据的附件哈希与定位，不随包复制敏感原始轨迹。
- `validate_plan_package.py`：计划包一致性校验，标准库即可运行。
- `PACKAGE_VALIDATION.json`：创建本包时实际执行的计划校验结果。
- `SHA256SUMS`：包内文件校验和（自身不纳入自身哈希）。

所有具体程序变更与未来实验仍待代理执行。本包没有更改任何仓库、运行真实provider、重跑N12c或签署gold。
