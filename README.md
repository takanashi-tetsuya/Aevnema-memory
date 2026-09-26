# Aevnema Memory

Aevnema Memory 是獨立的關聯記憶引擎，負責匯入材料、保存可回溯的來源、建立 Episode／Concept／Association 索引，以及按問題取回證據。它不處理聊天平台、使用者身份、人格提示詞或最終對話文字；這些由 [Aevnema chatbot](https://github.com/takanashi-tetsuya/Aevnema) 負責。

本倉庫同時是記憶演算法的開發與評測場所。研究中的召回率、來源可達率、片段充分性和最終答案正確率是不同指標；實驗報告不能視為目前產品的準確率保證。

## 安裝與首次使用

需要 Python 3.12 或更新版本。從倉庫根目錄執行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

在本機 `.env` 填入所使用模型服務的 `SILICONFLOW_API_KEY`，並按需設定 `MEMORY_DB_PATH`、模型及檢索參數。`.env` 不應提交；`.env.example` 是不含真實憑證的模板。此獨立引擎目前由 `AppConfig.from_env()` 讀取這些設定。chatbot 的模型路由配置是另一份文件，位於其自身倉庫的 `config/model_config.toml`。

```bash
memory-demo init
memory-demo prepare ./documents
memory-demo import ./documents
memory-demo query "要查詢的問題" --json
```

`init` 建立資料庫 schema；`prepare` 只解析並分段，不呼叫模型；`import` 和一般 `query` 可能向配置的模型服務發送材料、消耗配額。處理重要資料或執行實驗時，使用 `--database` 指向獨立的 SQLite 副本：

```bash
memory-demo --database ./working.sqlite import ./documents
memory-demo --database ./working.sqlite query "要查詢的問題" --mode deep --json
```

已安裝的命令入口為 `memory-demo`；也可使用 `python -m memory_demo.cli`。`memory-demo --help` 和各子命令的 `--help` 列出當前可用參數。

## 資料與執行邊界

| 物件 | 職責 |
| --- | --- |
| Source | 保存導入的原始材料及可回溯位置。 |
| Episode | 從材料提取的事件、狀態或候選事實；摘要本身不等於原文已證實整個主張。 |
| Paragraph | 可選的可逆局部片段，協助 Source 內定位。 |
| Concept | 實體、概念及其別名。 |
| Association | Episode／Concept 間的關係、權重與來源狀態；關聯不自動構成因果證明。 |

SQLite 保存來源與索引資料，RAM 向量索引可由資料庫重建。向量持久化及記憶體運算使用 float32；已建立的資料庫必須繼續使用相容的 embedding 空間。推理模型可以配置備援，embedding 模型不能任意切換到另一個向量空間。

匯入路徑由 `MemoryApplication` 組合資料庫、來源解析、分段、模型提取、審核、向量化及關聯寫入。查詢路徑在向量、詞面、Paragraph 和圖線索之間組合候選，再以來源及需求槽檢查交付品質。具體演算法可按請求配置；取得相關 Episode 或 Source 並不表示讀出的片段已完整支持答案。

`memory-demo query` 的一般入口沿用 `QueryEngine.query()`。指定 `--mode light|standard|deep|max_effort`、續接 ID，或在問題中明確要求最大程度回想時，會使用 `MemoryApplication.recall()`；可用 `--resume SESSION_ID` 續接、`--no-learn` 避免更新關聯權重，並透過 `recall-feedback` 對已交付且可歸因的關聯提供回饋。這些模式有不同的搜尋預設和本地工作預算，並非正確率等級。詳細的歷史設計與驗證界線見 [逐步回想說明](docs/PROGRESSIVE_RECALL.md)。

上層程式只應使用 `memory_demo` 的公開 API，例如 `AppConfig`、`Database`、`MemoryApplication` 與 `memory_demo.contracts`；不要直接修改 repositories 或引用 `benchmarks` 中的實驗實作。Aevnema chatbot 把已驗證的引擎程式複製為自身的 `memory-engine/` 執行快照，本倉庫仍保留實驗與測試。

## 常用命令

```bash
memory-demo prepare ./documents
memory-demo import ./documents --allow-partial
memory-demo query "問題" --json
memory-demo query "問題" --mode deep --no-learn --json
memory-demo query "問題" --mode max_effort --resume SESSION_ID --json
memory-demo recall-feedback SESSION_ID positive --feedback-id event-001
memory-demo stats
memory-demo rebuild-index
memory-demo timeline --help
memory-demo association --help
memory-demo concept --help
memory-demo episode --help
```

目錄匯入支援 `--source-root` 指定穩定的來源鍵；`--allow-partial` 只改變部分失敗時的退出碼，不代表失敗已修復。管理或刪除資料前應使用資料庫副本，並檢查子命令的確認選項。

## 專案結構與驗證

```text
src/memory_demo/       公開 API、匯入、檢索、關聯、儲存與模型客戶端
config/prompt_config/  引擎使用的抽取、檢索及審核提示詞
tests/                 單元與回歸測試
benchmarks/            可執行的離線評測工具，不屬於執行時 API
docs/                  目前架構說明與歷史研究報告
validation/            本地實驗結果與封存；大部分不進版本控制
```

```bash
python -m pip install pytest
python -m pytest -q
```

`pyproject.toml` 將預設測試收集限制在 `tests/`。真實語料、資料庫、日誌、模型回覆、實驗封包、憑證及私人對話均應留在本機；公開原始碼提交只包含可審閱的程式、測試、配置模板和文檔。歷史報告記錄當時的資料及協議，不能用其數字替代新版本驗收。

授權條款：AGPL-3.0-or-later，詳見 [LICENSE](LICENSE)。
