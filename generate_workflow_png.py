import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set font for Traditional Chinese support in Windows
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Segoe UI', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(10, 13))
ax.axis('off')
fig.patch.set_facecolor('#0f172a') # Deep slate dark theme background to match premium app
ax.set_facecolor('#0f172a')

# Define stages
stages = [
    {
        "title": "1. Business Understanding (商業理解)",
        "color": "#1e293b",
        "border": "#38bdf8",
        "text_color": "#38bdf8",
        "body": "• 定義目標：透過 R&D、行政、行銷預算預估並極大化新創公司利潤。\n• 成功指標：測試集 R² 評分、MAE、RMSE。"
    },
    {
        "title": "2. Data Understanding (資料理解)",
        "color": "#1e293b",
        "border": "#34d399",
        "text_color": "#34d399",
        "body": "• 載入 50 Startups 資料集，確認利潤符合常態分佈（偏態 0.0233）。\n• 相關性分析：R&D 支出與利潤相關係數達 0.973。\n• 商業分析判定 Row 51 為合理極端值（合法離群值）並予以保留。"
    },
    {
        "title": "3. Data Preparation (資料準備)",
        "color": "#1e293b",
        "border": "#fbbf24",
        "text_color": "#fbbf24",
        "body": "• 數值型特徵進行標準化 (StandardScaler)。\n• 地區欄位進行 One-Hot 編碼，排除 California 避開虛擬變數陷阱。\n• 進行 VIF 多重共線性檢查，所有特徵 VIF < 5，排除共線性風險。"
    },
    {
        "title": "4. Modeling (模型建立)",
        "color": "#1e293b",
        "border": "#a78bfa",
        "text_color": "#a78bfa",
        "body": "• 構建資料處理與演算法之機器學習 Pipeline 管道。\n• 建立多元線性迴歸 (OLS)、Ridge 迴歸與隨機森林迴歸模型。"
    },
    {
        "title": "5. Evaluation (模型評估)",
        "color": "#1e293b",
        "border": "#f87171",
        "text_color": "#f87171",
        "body": "• 透過 5-Fold 交叉驗證小樣本泛化效能，隨機森林表現最優。\n• 實作 RFE、Lasso、SFS 等 5 大特徵篩選算法（對齊 random_state=0）。\n• 判定最佳特徵數 k=2 (R&D + Marketing)，達最高 R² = 0.9474。\n• 結合 SHAP 進一步解讀特徵邊際影響力。"
    },
    {
        "title": "6. Deployment (模型部署)",
        "color": "#1e293b",
        "border": "#e879f9",
        "text_color": "#e879f9",
        "body": "• 將最優的隨機森林 Pipeline 儲存至 best_startup_model.joblib。\n• 撰寫 server.py 基於標準庫之 API 服務，提供 /predict 預測端點。\n• 打造毛玻璃暗黑科技風格前端 Web 模擬器，支援即時滑桿與自動優化。"
    }
]

# Coordinates
box_width = 8.5
box_height = 1.3
gap = 0.6
start_y = 11.5

# Draw title
ax.text(5, 12.3, "CRISP-DM 機器學習專案工作流程", color='#ffffff', fontsize=20, fontweight='bold', ha='center', va='center')
ax.text(5, 12.0, "Kaggle 50 Startups Profit Prediction Project Workflow", color='#64748b', fontsize=11, ha='center', va='center')

for i, stage in enumerate(stages):
    y = start_y - i * (box_height + gap)
    
    # Draw box (FancyBboxPatch for rounded corners)
    rect = patches.FancyBboxPatch(
        (0.75, y), box_width, box_height,
        boxstyle="round,pad=0.1",
        linewidth=2, edgecolor=stage["border"], facecolor=stage["color"],
        mutation_scale=0.2, zorder=2
    )
    ax.add_patch(rect)
    
    # Title text
    ax.text(1.0, y + box_height - 0.25, stage["title"], color=stage["text_color"], fontsize=13, fontweight='bold', va='center', zorder=3)
    
    # Body text
    ax.text(1.0, y + box_height - 0.75, stage["body"], color='#e2e8f0', fontsize=10.5, va='top', linespacing=1.4, zorder=3)
    
    # Draw arrow to next stage
    if i < len(stages) - 1:
        arrow_y_start = y - 0.05
        arrow_y_end = y - gap + 0.05
        ax.annotate(
            '', xy=(5, arrow_y_end), xytext=(5, arrow_y_start),
            arrowprops=dict(arrowstyle="-|>", color='#64748b', lw=2.5, mutation_scale=15),
            zorder=1
        )

# Adjust axes limits
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 12.7)

plt.tight_layout()
os.makedirs("images", exist_ok=True)
output_path = os.path.join("images", "workflow.png")
plt.savefig(output_path, dpi=180, facecolor='#0f172a', bbox_inches='tight')
print(f"Successfully generated {output_path}!")
