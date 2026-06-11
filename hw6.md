# Kaggle 50 Startups 利潤預測專案：今日工作成果與統計分析彙整 (hw6.md)

本文件詳細記錄了今日在 `c:\Users\a0970\OneDrive\Desktop\AI class\0611\HW6` 工作區中完成的所有開發工作、演算法實現、模型評估指標及應用程式部署細節。

---

## 📅 開發日期：2026-06-11

---

## 📌 一、工作完成摘要

今日工作嚴格遵循 **CRISP-DM** 數據挖掘生命週期，完成了從「數據理解」到「應用程式部署」的完整闭环：

1.  **專案設計與統計學論證 (`design.md`)**：
    *   補全了 CRISP-DM 六個階段的完整架構。
    *   完成了目標變數 `Profit` 的正態分佈檢驗（偏態係數為 `0.0233`）、註冊州別的平衡性分析（Calif 17, NY 17, FL 16），以及將 Row 51 定性為 **合法極端值 (Valid Outlier)** 予以保留的決策。
    *   計算設計矩陣的 **VIF (方差膨脹因子)**，證實無嚴重的多重共線性。
2.  **特徵漸進效能分析對齊 (`solve_50_startups.py` & `feature_selection_plot.png`)**：
    *   將特徵遞增評估的隨機分割調整為與樣板完全對齊的 `random_state=0`。
    *   **糾正樣板錯誤標籤**：樣板中 $k=5$ 的標籤誤標為 `State_California`，但實際參與計算並得出 RMSE=9137.99 與 R2=0.9347 的特徵是 **`Administration` (行政管理費用)**。為了不引入錯誤資訊，我們在代碼與表格中修正了此特徵名稱，實現了數據 100% 精準對齊。
3.  **五大特徵篩選算法全景對比 (`feature_selection_comparison.py` & `allinone.png`)**：
    *   同時實現了 **RFE (遞歸消除)**、**Lasso (L1正則)**、**SelectKBest (統計檢定)**、**Tree-Based (樹模型重要性)** 與 **SFS (序列前向)** 五大算法。
    *   在相同的 `random_state=0` 下評估，證實 Lasso 的篩選軌跡與樣板完美契合，所有算法一致表明 `R&D Spend + Marketing Spend` 是最優特徵組合。
4.  **獲利預估互動式應用程式 (`server.py` & `index.html`)**：
    *   **後端 (`server.py`)**：基於 Python 標準庫 `http.server` 構建輕量級伺服器，載入序列化隨機森林模型，開啟預測端點 `/predict`。
    *   **前端 (`index.html`)**：打造極具視覺質感的 **毛玻璃暗色主題 (Glassmorphism Dark Theme)**。提供研發、行政、行銷費用拖拉滑桿、註冊州切換按鈕、預估利潤動態計數動畫，以及動態 AI 財務顧問建議。同時內建 JavaScript 線性代數公式計算備援，確保高可用性。

---

## 📌 二、核心統計指標彙整

### 1. 模型評估對比（80/20 分割 與 5-Fold 交叉驗證）
*訓練使用的 Random State = 42*

| 評估模型 | 測試集 $R^2$ | 測試集 MAE | 測試集 RMSE | 5-Fold CV 平均 $R^2$ | 5-Fold CV 標準差 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multiple Linear Regression** | 0.8987 | \$6,961.48 | \$9,055.96 | 0.9279 | 0.0438 |
| **Ridge Regression** | 0.8954 | \$7,408.02 | \$9,202.87 | **0.9280** | 0.0447 |
| **Random Forest Regressor** | **0.9147** | **\$6,131.91** | **\$8,310.36** | 0.9277 | **0.0419** |

### 2. 特徵篩選效能評估表 (OLS, random_state=0)
*此數據與樣板圖片完全一致*

| 特徵數量 $k$ | 納入之特徵子集 | 測試集 RMSE | 測試集 $R^2$ |
| :---: | :--- | :---: | :---: |
| **1** | `[R&D Spend]` | 8,274.868018 | 0.946459 |
| **2** | `[R&D Spend, Marketing Spend]` | **8,198.797191** | **0.947439** |
| **3** | `[R&D Spend, Marketing Spend, State_New York]` | 8,309.059683 | 0.946015 |
| **4** | `[R&D Spend, Marketing Spend, State_New York, State_Florida]` | 8,409.916714 | 0.944697 |
| **5** | `[R&D Spend, Marketing Spend, State_New York, State_Florida, Administration]` | 9,137.990153 | 0.934707 |

---

## 📌 三、圖表註記說明

我們已更新了繪圖程式碼，在所有生成的圖片檔案中**直接嵌入了結果分析註記與 Footnotes**：
*   **[actual_vs_predicted.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/actual_vs_predicted.png)**：標註了模型隨機森林指標與測試集擬合良好的結論。
*   **[shap_summary_plot.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/shap_summary_plot.png)**：標註了研發支出的主導正向邊際貢獻，行政與地區變數的無效分佈。
*   **[feature_selection_plot.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/feature_selection_plot.png)**：標註了最優特徵個數 $k=2$ 拐點，以及後續特徵引入噪聲導致表現下滑的統計學解讀。
*   **[allinone.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/allinone.png)**：將 5 種算法在 `random_state=0` 分割下的各特徵數 Test $R^2$ 繪製在一張圖上，標註了不同算法特徵選擇路徑的異同點。

---

## 📌 四、啟動模擬器 App

後端 API 伺服器已於背景順利運行：
*   **伺服器地址**：`http://localhost:8000`
*   **如何操作**：
    1. 開啟瀏覽器訪問 `http://localhost:8000`。
    2. 調整 R&D、行政、行銷費用的數值滑桿，利潤會動態流暢變化。
    3. 點擊「自動優化預算分配」按鈕，體驗滑桿自動歸位至獲利最大化配置的動畫效果。

---

## 📌 五、專案成果與工作流程彙整圖 (Project Summary Dashboard)

本專案之開發工作成果與 CRISP-DM 流程彙整如下圖所示：

![Project Summary Dashboard](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/workflow.png)

### 📂 Draw.io 流程圖檔案說明
我們已生成結構化的 XML 流程圖，並存放在專案根目錄下：
*   **原始 Draw.io 檔案連結**：[workflow.drawio](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/workflow.drawio)

### 🛠️ 如何開啟與編輯該流程圖？
你可以使用以下兩種方式之一來載入並編輯 `workflow.drawio`：
1.  **線上網頁端開啟 (推薦)**：
    *   瀏覽 [Draw.io 官方網站 (https://draw.io)](https://draw.io/)。
    *   選擇 **「Open Existing Diagram」**，並選取本地的 [workflow.drawio](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/workflow.drawio) 檔案；或者直接將該檔案拖放至瀏覽器網頁視窗中，即可開始檢視與編輯。
2.  **VS Code 整合式編輯器**：
    *   在 VS Code 中安裝 **`Draw.io Integration`** 擴充功能 (Extension)。
    *   直接雙擊開啟 [workflow.drawio](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/workflow.drawio)，即可在編輯器內直觀地拖拉節點修改工作流。

### 📝 各階段工作細節與實作對照
*   **1. 商業理解 (Business Understanding)**：定義利用 R&D、行政、行銷預算來精準預估並極大化新創公司利潤的商業問題，並以 $R^2$ 評分、MAE、RMSE 作為專案成功指標。
*   **2. 資料理解 (Data Understanding)**：探索 `50_Startups.csv`，確認利潤呈現常態分佈（偏態係數為 `0.0233`），發現 `R&D Spend` 與利潤相關性高達 `0.973`。辨識出 Row 51 作為重要極端值，經商業解讀後認定其為合法合理特例並保留。
*   **3. 資料準備 (Data Preparation)**：採用 `StandardScaler` 標準化連續型特徵，One-Hot 編碼地區類別，排除 `California` 以防止虛擬變數陷阱 (Dummy Variable Trap)。進行 VIF 多重共線性檢驗，確認特徵 VIF 值小於 5，無共線性威脅。
*   **4. 模型建立 (Modeling)**：構建包括前處理與迴歸演算法的 ML Pipeline，建立多元線性迴歸 (OLS)、Ridge 迴歸及隨機森林迴歸模型。
*   **5. 模型評估 (Evaluation)**：對比 80/20 分割與 5-Fold 交叉驗證下的模型效能，隨機森林測試集表現最佳（$R^2 = 0.9147$）。實作 **RFE、Lasso、SelectKBest、Tree-Based、SFS Forward** 五大特徵篩選演算法，對齊 `random_state=0` 的測試集基準，一致選定 $k=2$ (`R&D Spend` + `Marketing Spend`) 為最佳特徵組合（$R^2 = 0.9474$, RMSE = `8198.80`）。使用 SHAP 值解釋模型預測，實行特徵效能分析。
*   **6. 模型部署 (Deployment)**：將最佳擬合 Pipeline 導出為 `best_startup_model.joblib`，編寫 `server.py` 微服務（API 端點 `/predict`），並實作毛玻璃暗黑科技感的前端 UI 模擬器 (`index.html`)，支援實時預測與自動滑桿分配動畫。
