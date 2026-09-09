# V3 Q2 公開復用入口診斷與單次完整對照（2026-09-07）

## 結論與邊界

v10 的原始軌跡完整保留，SHA-256 為
`b25bf6b810145f65429b88abdc954b39aed54915dbfac82c13769795ddc61556`。
它證明 Q1 經公開 finalizer 建立了 receipt `1` 與 edge `121`，但**不能**稱為「edge 已正常參與後的無收益驗證」。

對同一問題、同一 scope、同一 receipt/edge 執行的公開、零模型 preflight 觀察到：automatic exact-revisit 在 runtime-manifest lookup 直接 miss。普通 Q2 的已保存 trace 則顯示 contextual selector 因 `no_unresolved_slots` 在 matcher 前停止。因此本例屬於：

| 路徑 | 實際分類 | 依據 |
|---|---|---|
| V17 automatic exact-revisit | `not_entered` | receipt `1` 沒有 ready runtime manifest；公開入口回傳 `None`。 |
| 普通 V3 contextual contribution selector | 未進 matcher，非「匹配後拒絕」 | 已保存 `slot_selector_v3.reason=no_unresolved_slots`、`attached_edges=[]`、`matcher_backend=""`。 |
| edge 已使用但無收益 | `not_observed` | 沒有 edge 附著、source closure 或 contextual contribution 交付事件。 |

這不是正式 benchmark；gold 尚未人工簽核，promotion/canary 仍關閉。

## 凍結請求與有效設定

- Q2：`圣园未花说自己一直在暗中支援哪个组织？`
- request hash：`sha256:7d8e46e2b6e9f2828b38028ad4243ecc54464db88694b6b673a9f8ba844f92dd`
- domain：`knowledge`
- scope：`v3-real-source-diagnostic-pilot-32170`
- 有效 v10 設定：`contextual_association_enabled=true`、`shadow=false`、`growth_max_rounds=0`、`association_cue_enabled=false`、`association_cue_fast_path_enabled=false`、`contextual_promotion_enabled=false`、`contextual_restricted_rewrite_enabled=false`、`followup_planning_mode=missing_slots`。

cue fast path 關閉不是本次 V17 automatic exact-revisit miss 的近因。automatic route 的入口條件沒有讀取 cue/fast-path 設定；它獨立查詢 runtime manifest。cue fast path 是另一個 `association_cue` lane，而它在這個受控實驗中也被關閉，沒有替代或阻止 exact-revisit。

## request → receipt → edge → 來源 → 證據交付

| 順序 | 實際事件 | 結果 |
|---|---|---|
| 1 | Q1 的原公開 finalizer | 產生 receipt `1`，`ready` 於 `2026-09-07T13:11:55.858893+09:00`。 |
| 2 | Receipt durable state | `source_bound`；BGE-M3、1024 維、2 個 source facts、1 個 verification ref；Q1 source request hash 為 `sha256:df2776e03ee38a7880f704132417ee3e69f34d65f2a9c316bae761f66623d344`。 |
| 3 | Edge durable state | edge `121`：episode `21` → episode `32`，`contextual_recall`、`probation`、`retrieval_only`、`costly_success_reuse`；期限 `2026-10-07T13:11:55.831977+09:00`。 |
| 4 | endpoint 的已存來源驗證 | episode `21`：`main/32170.json`、source `9`、segment `8`；episode `32`：同檔、source `12`、segment `11`。兩者均有 1 個 literal quote 與 1 個 literal span，`evidence_origin=source`。這是資料庫既有 provenance，不是本次 exact path 執行的 source-closure。 |
| 5 | 公開 Q2 preflight | 呼叫 `QueryEngine.try_contextual_revisit`，只讀 clone；回傳 miss。receipt 的 V16 contract=0、V17 runtime manifest=0。第一個 ordinary-fallback 原因是 `runtime_manifest_lookup_miss`。 |
| 6 | 單次完整 Q2 control | automatic exact 仍為 `null`；普通 selector 的已保存理由為 `no_unresolved_slots`，未呼叫 matcher、未附著 edge。普通 retrieval 交付 episode `32`。 |

Q1 所存 requirement 帶有 `subject_terms`、`relation_hint`、`temporal_hint` 和 `negation_hint`。依現行 exact-template 投影規則，這些欄位使它不會投影成此種 whole-question exact contract/runtime manifest。這是由保存的 requirement shape 與程式規則做出的可重現推論；Q1 沒有保存「未投影」的逐事件理由，因此沒有把它偽稱為已觀測事件。

## 零模型前置復用診斷

輸出：
`validation/v3-q2-reuse-preflight-v2-20260907T133300Z/q2_reuse_preflight.full_local.json`
（SHA-256 `28cfba3cc7d09c553df7b22b452400690b58bc0abc82568e07e02c67c5c1434a`）

- 公開入口：`QueryEngine.try_contextual_revisit`；未直接呼叫 matcher、未手動補 edge、未繞過 scope/source-version 檢查。
- 結果：`miss`，耗時 `0.949 ms`。
- provider HTTP attempts：`0`；logical batches：`0`。
- 因 manifest lookup 失敗，未執行 contract 驗證、source closure、matcher、selector、planner、embedding、普通 retrieval、rerank 或 answer generation。
- 診斷使用獨立 SQLite clone；沒有修改 v10 的 receipt、edge、cue、contract、source 或原始 case。

## 一次完整 Q2 control（非重跑）

輸出：
`validation/v3-q2-reuse-full-answer-control-v1-20260907T133700Z/q2_reuse_preflight.full_local.json`
（SHA-256 `f8fb5bc001d8138b30b756b63b152c3f1f21c6bae8594dc4c7aa174215e9e80f`）

這是第一個 preflight 之後唯一一次完整回答請求，採同一 `.env` 的既有授權設定，寫入新 clone。它不是 v10 的替代、不是全套 benchmark 重跑，也未因結果好壞重試。

| 項目 | 觀測值 |
|---|---:|
| 結果 | `completed` |
| wall elapsed | 74,223.347 ms |
| logical batches / HTTP | 5 / 6 |
| 成功 / failed / fallback | 5 / 1 / 1 |
| Candidate / Selected / Delivered | 45 / 1 / 1（episode `32`） |
| exact revisit | `null` |
| attached contextual edges | `[]` |
| selector 事件 | `no_unresolved_slots`；`all_required_clauses_covered` |
| follow-up | 未呼叫 |

實際昂貴階段沒有被 exact reuse 跳過：intent（19.124 s）、BGE initial embedding（0.624 s）、initial retrieval（0.536 s）、rerank（6.935 s）、answer generation（46.832 s）及 audit 都執行。answer 的 DeepSeek-V3.2 HTTP 在 25.016 s ReadTimeout，依既有 transport fallback 以 GLM-4.5V 成功；沒有改用未授權模型。這是普通回退成本，不可歸因為 edge reuse 的收益或損失。

## 原始資料與可讀材料

- v10 原始 local trace：`validation/v3-q1-q2-real-source-diagnostic-v10-20260907T131038Z/q1_q2_case.full_local.json`
- v10 脫敏 case report：`validation/v3-q1-q2-real-source-diagnostic-v10-20260907T131038Z/q1_q2_case.report.md`
- 零模型 preflight raw/local 與 report：`validation/v3-q2-reuse-preflight-v2-20260907T133300Z/`
- 單次完整 control raw/local 與 report：`validation/v3-q2-reuse-full-answer-control-v1-20260907T133700Z/`

正式評分必須等待 approved Source gold 與獨立 holdout；本診斷不能作為 promotion、canary 或舊庫—全量庫的成績聲稱。
