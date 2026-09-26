# 可續接的逐步回想

這是 `MemoryApplication.recall()` 與 `memory-demo query --mode` 的現行使用說明。它與一般 `QueryEngine.query()` 是兩個入口；本文件描述程式能力及邊界，不把歷史實驗分數當作目前的召回保證。

## 模式與輸出

| 模式 | 每次呼叫的搜尋上限 | 每條線索種子數 | 每波圖擴展量 | 每波來源審閱數 |
| --- | ---: | ---: | ---: | ---: |
| `light` | 20 秒 | 2 | 32 | 2 |
| `standard` | 60 秒 | 4 | 128 | 3 |
| `deep` | 360 秒 | 8 | 512 | 4 |
| `max_effort` | 1800 秒 | 16 | 2048 | 6 |

這些值是獨立記憶引擎的搜尋政策，並非 chatbot 整輪對話的超時設定或模型服務保證。未指定模式時 `recall()` 預設 `deep`；明確要求盡最大努力回想的問句可選 `max_effort`。`--timeout` 只能縮短當次搜尋政策，不會延長模式上限。來源核對與檢查點保存可能使牆鐘返回時間略晚於協作式截止點。

回傳結果包含 `status`、`complete`、`answer`、`needs`、`missing_requirements`、`evidence`、`need_assessments`、`metrics`、`session_id` 與 `resumable`。`need_assessments` 分辨 `supported`、`partial`、`unknown` 和 `refuted`；未知不是反證，相關來源命中也不是完整支持。圖探索已盡但仍缺證據時，可得到 `search_exhausted` 和具體未滿足需求。

```bash
memory-demo --database ./working.sqlite query "問題" --mode deep --json
memory-demo --database ./working.sqlite query "問題" --mode max_effort --resume SESSION_ID --json
memory-demo --database ./working.sqlite query "問題" --mode deep --no-learn --json
memory-demo --database ./working.sqlite recall-feedback SESSION_ID positive --feedback-id event-001
```

`resume` 須使用同一問題和可見上下文，並由一個呼叫者串行續接。資料庫、來源、向量、圖或問題版本變更後不能盲目恢復舊進度。狀態 `time_budget`、`cancelled`、`wave_budget`、`technical_error` 可在符合檢查點條件時續接。檢查點的雜湊用於檢測損壞，不是對任意不可信檔案的身份認證。

## 搜尋與核驗

1. 模型根據問題和呼叫方提供的可見上下文提出事實需求及線索；已有 Episode 的 embedding 提供初始種子。
2. Episode／Concept 關聯圖傳播線索。不同來源的弱訊號可以匯合，但路徑權重、連通與放電本身只提供檢索順序，不自動證明事件、因果或人物立場。
3. 引擎讀取有限的 Source 原文窗口，將候選主張和逐字引文綁定到來源及版本，並重新審查需求覆蓋。仍缺證據時可取回更大上下文或從向量候選繼續查找，包括圖中沒有可用鄰居的 Episode。
4. 只交付通過當輪檢查的主張；未完成的窗口不標為已讀。技術錯誤或中斷保留可用檢查點，不把模型未輸出的判決當成通過。

檢索階段對 Source、Episode、Concept 和 Association 保持不同身份。`generation` 表示推論距離，不等於證據可信度；舊 Episode 摘要也不能代替其原文。Source 中出現的一句話可證明某角色說過什麼，未必直接證明世界事實或其動機。

## 學習與回饋

只有兩端均有原文依據、且被審閱接受的關係，才能形成或更新可歸因的回想捷徑。`--no-learn` 禁止當輪聯想寫入，仍允許本地檢查點。`recall-feedback` 要求穩定的 `feedback-id`，同一事件重試不重複調權；只讀到關聯或只命中主題不能成為正向回饋的理由。

再次回想仍須核對目前問題及 Source 版本，不把先前的答案當成永久真值。要證明捷徑真正有效，需在同一資料、問題及模型預算下比較有邊、遮邊與一般快取，分別報告事實支持度、來源閱讀量、圖工作量、模型呼叫和牆鐘時間。

本功能可以向配置的模型服務發送問題與來源片段，並產生費用。使用者材料、實驗資料庫與回想檢查點應存放在本機忽略目錄；公開提交只保留程式及可審閱的非私人測試。
