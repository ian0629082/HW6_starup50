import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Enable matplotlib XKCD (hand-drawn sketch) mode
plt.xkcd(scale=1.5, length=100, randomness=2)

# Re-configure fonts for Chinese support after entering xkcd mode
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'Segoe UI', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(10, 13))
ax.axis('off')

# Set background to Excalidraw white canvas style
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Define stages with Excalidraw-like pastel colors
stages = [
    {
        "title": "1. Business Understanding (商業理解)",
        "fill_color": "#dbeafe",  # Light blue
        "border_color": "#2563eb",  # Blue
        "text_color": "#1e3a8a",
        "body": "• 定義目標：透過 R&D、行政、行銷預算預估並極大化新創公司利潤。\n• 成功指標：測試集 R² 評分、MAE、RMSE。"
    },
    {
        "title": "2. Data Understanding (資料理解)",
        "fill_color": "#d1fae5",  # Light green
        "border_color": "#059669",  # Green
        "text_color": "#064e3b",
        "body": "• 載入 50 Startups 資料集，確認利潤符合常態分佈（偏態 0.0233）。\n• 相關性分析：R&D 支出與利潤相關係數達 0.973。\n• 商業分析判定 Row 51 為合理極端值（合法離群值）並予以保留。"
    },
    {
        "title": "3. Data Preparation (資料準備)",
        "fill_color": "#fef3c7",  # Light amber
        "border_color": "#d97706",  # Amber
        "text_color": "#78350f",
        "body": "• 數值型特徵進行標準化 (StandardScaler)。\n• 地區欄位進行 One-Hot 編碼，排除 California 避開虛擬變數陷阱。\n• 進行 VIF 多重共線性檢查，所有特徵 VIF < 5，排除共線性風險。"
    },
    {
        "title": "4. Modeling (模型建立)",
        "fill_color": "#ede9fe",  # Light purple
        "border_color": "#7c3aed",  # Purple
        "text_color": "#4c1d95",
        "body": "• 構建資料處理與演算法之機器學習 Pipeline 管道。\n• 建立多元線性迴歸 (OLS)、Ridge 迴歸與隨機森林迴歸模型。"
    },
    {
        "title": "5. Evaluation (模型評估)",
        "fill_color": "#fee2e2",  # Light red
        "border_color": "#dc2626",  # Red
        "text_color": "#7f1d1d",
        "body": "• 透過 5-Fold 交叉驗證小樣本泛化效能，隨機森林表現最優。\n• 實作 RFE、Lasso、SFS 等 5 大特徵篩選算法（對齊 random_state=0）。\n• 判定最佳特徵數 k=2 (R&D + Marketing)，達最高 R² = 0.9474。\n• 結合 SHAP 進一步解讀特徵邊際影響力。"
    },
    {
        "title": "6. Deployment (模型部署)",
        "fill_color": "#fce7f3",  # Light pink
        "border_color": "#db2777",  # Pink
        "text_color": "#9d174d",
        "body": "• 將最優的隨機森林 Pipeline 儲存至 best_startup_model.joblib。\n• 撰寫 server.py 基於標準庫之 API 服務，提供 /predict 預測端點。\n• 打造毛玻璃暗黑科技風格前端 Web 模擬器，支援即時滑桿與自動優化。"
    }
]

# Coordinates
box_width = 8.5
box_height = 1.3
gap = 0.6
start_y = 11.5

# Draw title in Excalidraw handwritten style
ax.text(5, 12.3, "✏️ CRISP-DM 機器學習專案工作流程 (Excalidraw Style)", color='#1e293b', fontsize=18, fontweight='bold', ha='center', va='center')
ax.text(5, 11.9, "Kaggle 50 Startups Profit Prediction Project Workflow", color='#64748b', fontsize=11, ha='center', va='center')

for i, stage in enumerate(stages):
    y = start_y - i * (box_height + gap)
    
    # Draw box (Using Rectangle with sketch effect from xkcd)
    rect = patches.Rectangle(
        (0.75, y), box_width, box_height,
        linewidth=2, edgecolor=stage["border_color"], facecolor=stage["fill_color"],
        zorder=2
    )
    ax.add_patch(rect)
    
    # Title text
    ax.text(1.0, y + box_height - 0.25, stage["title"], color=stage["text_color"], fontsize=12.5, fontweight='bold', va='center', zorder=3)
    
    # Body text
    ax.text(1.0, y + box_height - 0.65, stage["body"], color='#334155', fontsize=10.5, va='top', linespacing=1.4, zorder=3)
    
    # Draw arrow to next stage
    if i < len(stages) - 1:
        arrow_y_start = y - 0.05
        arrow_y_end = y - gap + 0.05
        ax.annotate(
            '', xy=(5, arrow_y_end), xytext=(5, arrow_y_start),
            arrowprops=dict(arrowstyle="-|>", color='#64748b', lw=2, mutation_scale=15),
            zorder=1
        )

# Adjust axes limits
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 12.7)

plt.tight_layout()
plt.savefig("workflow.png", dpi=180, facecolor='#ffffff', bbox_inches='tight')
print("Successfully generated Excalidraw-style workflow.png!")
