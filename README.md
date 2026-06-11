# Kaggle 50 Startups Profit Prediction 🚀
## 基於 CRISP-DM 流程與 Scikit-learn 的機器學習專案

本專案利用 **Kaggle 50 Startups** 資料集，建立迴歸分析模型以預測新創公司的年度利潤（Profit）。專案內建了特徵篩選比較系統，並實作了一個採用毛玻璃暗色設計主題的互動式 Web 獲利模擬器。

### 📊 專案成果與工作流程彙整圖 (Project Summary Dashboard)
![Project Summary Dashboard](workflow.png)

---

## 📂 專案目錄結構

*   `50_Startups.csv`：原始資料集。
*   `solve_50_startups.py`：主建模管道程式碼。包含 EDA、VIF 多重共線性分析、模型訓練、5-Fold 交叉驗證、SHAP 可解釋性分析以及特徵篩選效能評估。
*   `feature_selection_comparison.py`：五大特徵篩選演算法（RFE, Lasso, SelectKBest, RandomForest, SFS Forward）全景對比評估腳本。
*   `server.py`：基於 Python 標準庫的輕量級 HTTP API 伺服器，載入最佳模型並開啟 `/predict` 端點。
*   `index.html`：互動式預估網頁前端（Glassmorphism 暗色風格，支援即時滑桿調節與動態優化）。
*   `best_startup_model.joblib`：儲存的隨機森林預測模型 Pipeline。
*   `design.md`：詳盡的 CRISP-DM 專案分析與統計設計文件。
*   `hw6.md`：今日工作成果與核心統計表格彙整報告。
*   `*.png`：生成的視覺化統計圖表（直接附帶解讀註釋）。

---

## ⚡ 快速啟動應用程式 App

要本地運行互動式模擬器，請在終端機中執行以下命令：

```bash
# 啟動 API 伺服器
python server.py
```

伺服器啟動後，請在瀏覽器中開啟以下網址：
👉 **[http://localhost:8000](http://localhost:8000)**

調整 R&D、行政、行銷費用的滑桿，利潤會以流暢的數字動畫顯示最新的預估結果，並動態產生 AI 預算專家諮詢建議。

---

## 📈 核心效能評估指標 (Test R2 Scores)

*評估模型基於 `random_state=0` 分割，與樣板對照圖完全一致：*

| 特徵數量 $k$ | RFE (遞歸消除) | Lasso (L1 正則) | SelectKBest (統計檢定) | Tree-Based (隨機森林) | SFS Forward (序列前向) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$k=1$** | **0.9465** | **0.9465** | **0.9465** | **0.9465** | **0.9465** |
| **$k=2$** | **0.9474** | **0.9474** | **0.9474** | **0.9474** | **0.9474** |
| **$k=3$** | 0.9451 | **0.9460** | 0.9394 | 0.9394 | 0.9394 |
| **$k=4$** | 0.9357 | **0.9447** | 0.9367 | 0.9357 | 0.9367 |
| **$k=5$** | 0.9347 | 0.9347 | 0.9347 | 0.9347 | 0.9347 |

*   **關鍵結論**：在特徵數 $k=2$（`R&D Spend` + `Marketing Spend`）時，模型預測效果達到全線最高峰（$R^2 = 0.9474$, RMSE = `8198.797191`）。額外拉入其他變數（如行政成本或註冊州別）會引入共線性噪聲，導致決定係數下跌。

---

## 🖼️ 視覺化成果圖表一覽

所有圖片已自動寫入註解文字，以提升閱讀效能：
1.  **[feature_selection_plot.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/feature_selection_plot.png)**：展示特徵數量遞增與 RMSE/R2 變化關係（對應 random_state=0 樣板）。
2.  **[allinone.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/allinone.png)**：五大篩選演算法全對比效能折線圖。
3.  **[actual_vs_predicted.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/actual_vs_predicted.png)**：預測值與真實值散點圖（含模型指標文字框）。
4.  **[shap_summary_plot.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/shap_summary_plot.png)**：SHAP 邊際貢獻可解釋性散點分布圖。
5.  **[correlation_heatmap.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/correlation_heatmap.png)**：變數線性相關係數熱力圖。
6.  **[residual_plot.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/residual_plot.png)**：殘差分析散點圖，驗證同方差性假設。
7.  **[workflow.png](file:///c:/Users/a0970/OneDrive/Desktop/AI%20class/0611/HW6/workflow.png)**：本專案遵循 CRISP-DM 流程之機器學習開發工作流程圖。
