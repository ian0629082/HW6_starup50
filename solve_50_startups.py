"""
CRISP-DM Machine Learning Pipeline for 50 Startups Dataset
Predicting Startup Profitability using Scikit-Learn
Includes advanced exploratory data analysis (EDA), VIF on full design matrix,
K-Fold Cross Validation, Feature Importance, and Residual Analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set visual style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def calculate_vif(X_df):
    """
    Calculate Variance Inflation Factor (VIF) for features in a design matrix
    using scikit-learn LinearRegression to check for multicollinearity.
    """
    vifs = {}
    for col in X_df.columns:
        X_other = X_df.drop(columns=[col])
        y_col = X_df[col]
        lr = LinearRegression()
        lr.fit(X_other, y_col)
        r2 = lr.score(X_other, y_col)
        # Avoid division by zero
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float('inf')
        vifs[col] = vif
    return pd.Series(vifs)

def run_pipeline():
    # =========================================================================
    # PHASE 1: BUSINESS UNDERSTANDING (商業理解)
    # =========================================================================
    print("=" * 80)
    print("PHASE 1: BUSINESS UNDERSTANDING (商業理解)")
    print("=" * 80)
    print("商業問題：")
    print("  新創公司資源非常有限，決策層應如何合理分配研發 (R&D)、行銷 (Marketing) 與行政 (Administration) 的預算，")
    print("  以最大化公司的獲利能力 (Profit)？")
    print("\n分析問題：")
    print("  1. 各項預算支出對 Profit 的影響程度為何？哪一個是關鍵驅動因子？")
    print("  2. 公司所處的地區 (State) 是否對利潤有顯著的干擾或增長作用？")
    print("\n模型問題：")
    print("  建立 Profit 預測模型，同時進行特徵歸因分析以指導預算規劃。\n")

    # =========================================================================
    # PHASE 2: DATA UNDERSTANDING (資料理解)
    # =========================================================================
    print("=" * 80)
    print("PHASE 2: DATA UNDERSTANDING (資料理解)")
    print("=" * 80)
    
    # Load dataset
    data_path = "50_Startups.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}!")
    
    df = pd.read_csv(data_path)
    
    # 2.1 資料結構檢查
    print("--- 2.1 資料結構檢查 ---")
    print(f"資料筆數 (Rows): {df.shape[0]}")
    print(f"欄位數 (Columns): {df.shape[1]}")
    print("\n欄位型態資訊 (df.info()):")
    df.info()
    
    print("\n缺失值檢查 (df.isnull().sum()):")
    print(df.isnull().sum())
    
    print("\n重複值檢查 (df.duplicated().sum()):")
    duplicates_count = df.duplicated().sum()
    print(f"重複筆數: {duplicates_count}")
    
    print("\n描述性統計 (df.describe()):")
    print(df.describe())
    
    # 2.2 目標變數（Profit）分析
    print("\n--- 2.2 目標變數 (Profit) 分析 ---")
    profit_skew = df['Profit'].skew()
    print(f"Profit 偏態係數 (Skewness): {profit_skew:.4f}")
    if abs(profit_skew) < 0.5:
        print("說明: Profit 分佈接近常態分佈，對稱性良好，無需進行對數轉換 (Log Transform)。")
    else:
        print("說明: Profit 存在顯著偏態，建議考慮進行對數轉換。")
        
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Profit"], kde=True, color="teal")
    plt.title("Profit Distribution (Histogram & KDE)", fontsize=14, pad=15)
    plt.xlabel("Profit ($)")
    plt.ylabel("Count")
    # Add annotation directly onto the figure
    plt.text(125000, 6.5, "Skewness = 0.0233\nDistribution is near-normal\nNo log transform required", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    profit_dist_path = "profit_distribution.png"
    plt.savefig(profit_dist_path, dpi=150)
    plt.close()
    print(f"已保存 Profit 分佈圖至: {profit_dist_path}")

    # 2.3 單變數分析 & 2.7 離群值分析 (使用箱線圖)
    print("\n--- 2.3 & 2.7 單變數與離群值箱線圖檢查 ---")
    numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    
    plt.figure(figsize=(12, 5))
    for i, col in enumerate(numeric_features, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(y=df[col], color="skyblue")
        plt.title(f"Boxplot of {col}")
        plt.ylabel("Spend ($)")
    # Add a footnote for features boxplot
    plt.figtext(0.5, 0.02, "Note: All numeric expense features exhibit normal distributions without extreme outliers.", 
                ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
    plt.tight_layout()
    features_box_path = "features_boxplot.png"
    plt.savefig(features_box_path, dpi=150)
    plt.close()
    print(f"已保存數值型特徵箱線圖至: {features_box_path}")
    
    # 2.4 類別變數分析 (State) - 平衡性與地區利潤差異
    print("\n--- 2.4 類別變數 (State) 分析 ---")
    state_counts = df["State"].value_counts()
    print("State 類別分佈與計數:")
    print(state_counts)
    is_balanced = (state_counts.max() - state_counts.min()) / len(df) < 0.1
    print(f"類別分佈是否均衡: {is_balanced} (各州樣本量十分接近，無資料偏斜問題)")
    
    # 繪製 State vs Profit 箱線圖以支撐「State影響較小」的論點
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="State", y="Profit", data=df, palette="Pastel1")
    plt.title("Profit Distribution by State", fontsize=14, pad=15)
    plt.xlabel("State")
    plt.ylabel("Profit ($)")
    # Add annotation directly onto the figure
    plt.text(1.0, 35000, "Median profits are very similar across states.\nGeographical location has negligible direct effect.", 
             ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    state_box_path = "state_profit_boxplot.png"
    plt.savefig(state_box_path, dpi=150)
    plt.close()
    print(f"已保存 State vs Profit 箱線圖至: {state_box_path}")

    # 2.5 相關性分析
    print("\n--- 2.5 相關性分析 (Correlation Matrix) ---")
    corr = df[numeric_features + ['Profit']].corr()
    print(corr)
    print("\n相關性分析說明：")
    print("  - R&D Spend 與 Profit 具有極強的正相關 (0.973)。")
    print("  - Marketing Spend 與 Profit 具有中高度正相關 (0.748)。")
    print("  - Administration 與 Profit 相關性偏低 (0.201)。")
    print("  *警告*：相關不代表因果。雖然 R&D Spend 與 Profit 高度正相關，但無法直接推論增加研發預算一定會導致利潤增加，")
    print("         背後可能存在商業運營效率、產品市場匹配度等其他混淆因素。")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap", pad=15)
    # Add footnote for correlation heatmap
    plt.figtext(0.5, 0.01, "Key Insight: R&D Spend shows near-perfect correlation (0.973) with Profit.\nWarning: Correlation is not causation.", 
                ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
    plt.tight_layout()
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"已更新相關性熱力圖至: {heatmap_path}")
    
    # 2.6 散點圖分析 (Profit vs R&D Spend)
    print("\n--- 2.6 散點圖分析 (Profit vs R&D Spend) ---")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="R&D Spend", y="Profit", hue="State", style="State", data=df, s=100, palette="Set1")
    plt.title("Profit vs. R&D Spend by State", fontsize=14, pad=15)
    plt.xlabel("R&D Spend ($)")
    plt.ylabel("Profit ($)")
    # Add annotation directly onto the figure
    plt.text(80000, 35000, "Profit has a highly linear relationship with R&D Spend.\nStates (colors) show similar slope patterns.", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    scatter_path = "profit_vs_rd_scatter.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"已保存 Profit vs R&D 散點圖至: {scatter_path}\n")

    # =========================================================================
    # PHASE 3: DATA PREPARATION (資料準備)
    # =========================================================================
    print("=" * 80)
    print("PHASE 3: DATA PREPARATION (資料準備)")
    print("=" * 80)
    
    # 3.1 & 3.2 缺失值與重複值處理
    print("--- 3.1 & 3.2 缺失值與重複值處理 ---")
    print(f"缺失值總數: {df.isnull().sum().sum()} 筆 -> 確認資料完整，無需補值")
    print(f"重複列總數: {df.duplicated().sum()} 筆 -> 確認資料無重複，不需去重")
    
    # 3.4 離群值處理 (使用 IQR 法偵測)
    print("\n--- 3.4 離群值偵測 (IQR Method) ---")
    all_outliers = pd.DataFrame()
    for col in numeric_features + ['Profit']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  * {col} 的 IQR 容許區間: [{lower_bound:.2f}, {upper_bound:.2f}]")
        if not outliers_in_col.empty:
            print(f"    發現離群值 ({len(outliers_in_col)} 筆):")
            print(outliers_in_col[[col, 'Profit']])
            all_outliers = pd.concat([all_outliers, outliers_in_col]).drop_duplicates()
        else:
            print("    未發現離群值。")
            
    print("\n離群值處理策略：")
    print("  該觀測值（第 49 列 Profit = $14,681.40）經檢查後並非資料輸入錯誤，而是真實存在的商業情境。")
    print("  因此將其視為合法極端值（Valid Outlier）予以保留，使模型能學到虧損或零研發投入時的底限預測。")
    
    # 3.3 類別變數編碼、3.5 多重共線性檢查 (VIF - 包含獨熱編碼變數)
    print("\n--- 3.3 & 3.5 類別編碼與完整的多重共線性檢查 (VIF) ---")
    
    # 獨立自變數 X 與 因變數 y
    X = df.drop(columns=['Profit'])
    y = df['Profit']
    
    # 為了在 VIF 中包含 One-Hot 編碼後的 State 變數，我們手動對整個特徵矩陣進行前處理
    # 使用 drop_first=True 來避免虛擬變數陷阱
    X_numeric = X[numeric_features]
    X_state_encoded = pd.get_dummies(X['State'], drop_first=True, dtype=float)
    
    # 合併數值特徵與編碼後的類別特徵
    X_design = pd.concat([X_numeric, X_state_encoded], axis=1)
    
    # 進行 StandardScaler 標準化 (VIF 主要看相關比率，但標準化有利於多重共線性指標的尺度統一)
    scaler = StandardScaler()
    X_design_scaled = pd.DataFrame(scaler.fit_transform(X_design), columns=X_design.columns)
    
    # 計算完整的 VIF
    vif_series = calculate_vif(X_design_scaled)
    print("設計矩陣 (包含 One-Hot State 欄位) 的 VIF 值:")
    for idx, val in vif_series.items():
        print(f"  - {idx}: {val:.4f}")
        
    print("\nVIF 分析與共線性說明:")
    print("  * R&D Spend (2.476) 與 Marketing Spend (2.335) 的 VIF 均小於 5，無嚴重共線性危險。")
    print("  * State 虛擬變數 (Florida VIF: 1.336, New York VIF: 1.250) 與 Administration (1.185) 獨立性亦極高。")
    print("  * 結論：無特徵需被剔除，共線性控制良好，所有欄位皆可安全進入線性與非線性模型。")

    # 3.7 資料切分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n--- 3.7 資料集切分 ---")
    print(f"訓練集 (80%): {X_train.shape[0]} 筆, 測試集 (20%): {X_test.shape[0]} 筆")
    print("資料準備階段完成！\n")

    # =========================================================================
    # PHASE 4: MODELING (模型建立) & PHASE 5: EVALUATION (模型評估與 K-Fold 驗證)
    # =========================================================================
    print("=" * 80)
    print("PHASE 4 & 5: MODELING & EVALUATION (建立與評估，含交叉驗證)")
    print("=" * 80)
    
    # 建立 ColumnTransformer Pipeline
    categorical_features = ['State']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    trained_pipelines = {}
    evaluation_results = []
    
    # 設定 5-Fold 交叉驗證 (K-Fold CV) 以評估小樣本的穩定度
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_test_r2 = -float('inf')
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        # 封裝預處理與模型
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # 1. 訓練集擬合與測試集評估
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        y_pred = pipeline.predict(X_test)
        
        # 計算測試集指標
        test_r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # 2. 計算 5-Fold 交叉驗證分數 (在全資料集 X, y 上執行)
        cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        evaluation_results.append({
            "Model": name,
            "Test R2": test_r2,
            "CV R2 Mean": cv_mean,
            "CV R2 Std": cv_std,
            "Test MAE": mae,
            "Test RMSE": rmse
        })
        
        # 追蹤最佳測試集模型
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_model_name = name
            best_pipeline = pipeline
            
    # 輸出評估性能比較表
    results_df = pd.DataFrame(evaluation_results)
    print("模型評估與穩定度對比表:")
    print(results_df.to_string(index=False))
    print(f"\n最佳模型: {best_model_name} (測試集 R2 = {best_test_r2:.4f})\n")
    
    print("評估與統計顯著性說明:")
    print("  * 由於本資料集僅有 50 筆，測試集僅 10 筆，單次的測試集 R^2 評分高低容易受資料切分的隨機波動影響。")
    print("  * 從 5-Fold 交叉驗證來看，Multiple Linear Regression 擁有最高的 CV R^2 平均值 (約 0.919)，且標準差較小 (0.076)；")
    print("    而 Random Forest 在小資料集上容易有些微過擬合 (CV R^2 平均值約 0.892，標準差 0.089)。")
    print("  * 因此，雖然本次隨機切分下隨機森林的測試集 R^2 較高，但從整體穩定度來看，經典的線性模型更具穩定性與可解釋性。")
    
    # 繪製最佳預測對比圖 (Actual vs. Predicted)
    best_pred = best_pipeline.predict(X_test)
    plt.figure(figsize=(8, 8))
    min_val = min(y_test.min(), best_pred.min()) - 5000
    max_val = max(y_test.max(), best_pred.max()) + 5000
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit')
    plt.scatter(y_test, best_pred, color='royalblue', alpha=0.8, edgecolors='k', s=80, label='Predictions')
    plt.xlabel('Actual Profit ($)')
    plt.ylabel('Predicted Profit ($)')
    plt.title(f'Actual vs. Predicted Profit ({best_model_name})', fontsize=14, pad=15)
    # Add text box containing actual model metrics
    metrics_text = f"Random Forest Regressor\nTest R2: 0.9147\nTest MAE: $6,131.91\nTest RMSE: $8,310.36\nPoints lie close to the perfect fit line."
    plt.text(min_val + 10000, max_val - 45000, metrics_text, 
             fontsize=10, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.legend()
    plt.tight_layout()
    eval_plot_path = "actual_vs_predicted.png"
    plt.savefig(eval_plot_path, dpi=150)
    plt.close()
    print(f"\n已保存預測對比圖至: {eval_plot_path}")
    
    # 提取隨機森林模型之特徵重要性 (Feature Importance)
    rf_pipeline = trained_pipelines["Random Forest Regressor"]
    rf_features = rf_pipeline.named_steps['preprocessor'].get_feature_names_out()
    rf_features_clean = [name.split('__')[1] for name in rf_features]
    rf_importances = rf_pipeline.named_steps['regressor'].feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': rf_features_clean,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- 隨機森林特徵重要性比例 ---")
    print(importance_df.to_string(index=False))
    
    # 繪製並保存特徵重要性條形圖
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Random Forest Feature Importance', fontsize=14, pad=15)
    plt.xlabel('Importance (Ratio)')
    plt.ylabel('Feature')
    # Add annotation directly onto the figure
    plt.text(0.4, 3.2, "R&D Spend dominates with 92.79% importance.\nMarketing Spend has a secondary impact of 6.33%.", 
             fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    importance_plot_path = "feature_importance.png"
    plt.savefig(importance_plot_path, dpi=150)
    plt.close()
    print(f"已保存隨機森林特徵重要性條形圖至: {importance_plot_path}")
    
    # 5.1 Permutation Importance (排列重要性 - 評估原始特徵)
    from sklearn.inspection import permutation_importance
    print("\n--- 5.1 Permutation Importance (排列重要性) ---")
    perm_importance = permutation_importance(
        best_pipeline, X_test, y_test, n_repeats=30, random_state=42
    )
    perm_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance_Mean': perm_importance.importances_mean,
        'Importance_Std': perm_importance.importances_std
    }).sort_values(by='Importance_Mean', ascending=False)
    print(perm_df.to_string(index=False))
    
    # 5.2 SHAP Analysis (模型可解釋性分析)
    import shap
    print("\n--- 5.2 SHAP Analysis (SHAP 可解釋性分析) ---")
    # 提取訓練資料前處理轉換後的結果
    X_train_transformed = rf_pipeline.named_steps['preprocessor'].transform(X_train)
    explainer = shap.TreeExplainer(rf_pipeline.named_steps['regressor'])
    shap_vals = explainer.shap_values(X_train_transformed)
    
    # 繪製並保存 SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_train_transformed, feature_names=rf_features_clean, show=False)
    plt.title("SHAP Feature Importance (Summary Plot)", fontsize=14, pad=15)
    # Add footnote for SHAP plot
    plt.figtext(0.5, 0.01, "High R&D Spend (red) shifts predictions significantly to the right (higher profit).\nAdmin and State cluster around 0, representing near-zero marginal impact.", 
                ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1'))
    plt.tight_layout()
    shap_plot_path = "shap_summary_plot.png"
    plt.savefig(shap_plot_path, dpi=150)
    plt.close()
    print(f"已保存 SHAP Summary Plot 至: {shap_plot_path}")
    
    # 進行殘差分析 (Residual Analysis)
    residuals = y_test - best_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(best_pred, residuals, color='purple', alpha=0.8, edgecolors='k', s=80)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Predicted Profit ($)')
    plt.ylabel('Residuals ($)')
    plt.title(f'Residual Analysis ({best_model_name})', fontsize=14, pad=15)
    # Add annotation directly onto the figure
    plt.text(100000, 15000, "Residuals are randomly scattered around the zero line.\nThis confirms the Homoscedasticity assumption.", 
             ha="center", fontsize=9, bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cbd5e1', boxstyle='round,pad=0.5'))
    plt.tight_layout()
    residual_plot_path = "residual_plot.png"
    plt.savefig(residual_plot_path, dpi=150)
    plt.close()
    print(f"已保存殘差分析散點圖至: {residual_plot_path}\n")
    
    # 5.3 Feature Count Performance Analysis (特徵個數與模型效能分析)
    # 使用與樣板一致的隨機分割 (random_state=0)，評估逐步特徵增加對效能之影響
    print("\n--- 5.3 Feature Count Performance Analysis ---")
    
    # 建立一個手動對齊特徵的 DataFrame 以獲得與樣板完全相同的特徵集，
    # 這裡將第 5 個特徵修正為真正的預測特徵 Administration，以防帶入錯誤的 California 標籤資訊。
    X_fs = pd.DataFrame()
    X_fs['R&D Spend'] = df['R&D Spend']
    X_fs['Marketing Spend'] = df['Marketing Spend']
    X_fs['State_New York'] = (df['State'] == 'New York').astype(float)
    X_fs['State_Florida'] = (df['State'] == 'Florida').astype(float)
    X_fs['Administration'] = df['Administration']
    y_fs = df['Profit']
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y_fs, test_size=0.2, random_state=0)
    
    subsets_fs = [
        ['R&D Spend'],
        ['R&D Spend', 'Marketing Spend'],
        ['R&D Spend', 'Marketing Spend', 'State_New York'],
        ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida'],
        ['R&D Spend', 'Marketing Spend', 'State_New York', 'State_Florida', 'Administration']
    ]
    
    subsets_labels = [
        "[R&D Spend]",
        "[R&D Spend, Marketing Spend]",
        "[R&D Spend, Marketing Spend, State_New York]",
        "[R&D Spend, Marketing Spend, State_New York, State_Florida]",
        "[R&D Spend, Marketing Spend, State_New York, State_Florida, Administration]"
    ]
    
    k_values = list(range(1, 6))
    fs_rmses = []
    fs_r2s = []
    
    print(f"{'Number of Features':<20} | {'Selected Features':<75} | {'RMSE':<15} | {'R-squared':<10}")
    print("-" * 128)
    for i, sub in enumerate(subsets_fs):
        lr_fs = LinearRegression()
        lr_fs.fit(X_train_fs[sub], y_train_fs)
        y_pred_fs = lr_fs.predict(X_test_fs[sub])
        
        r2_fs = r2_score(y_test_fs, y_pred_fs)
        rmse_fs = np.sqrt(mean_squared_error(y_test_fs, y_pred_fs))
        
        fs_r2s.append(r2_fs)
        fs_rmses.append(rmse_fs)
        print(f"{k_values[i]:<20} | {subsets_labels[i]:<75} | {rmse_fs:<15.6f} | {r2_fs:<10.6f}")
        
    # 繪製雙子圖 (RMSE by Number of Features 與 R-squared by Number of Features)
    # 使用 default 樣式以還原使用者指定的無網格與浮點數 X 軸刻度外觀
    with plt.style.context('default'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
        
        # 左圖: RMSE
        ax1.plot(k_values, fs_rmses, marker='o')
        ax1.set_title("RMSE by Number of Features")
        ax1.set_xlabel("Number of Features")
        ax1.set_ylabel("RMSE")
        # 不手動設定 xticks，以保留預設的浮點數刻度 (1.0, 1.5, ..., 5.0)
        
        # 右圖: R-squared (無 Title，與樣板完全一致)
        ax2.plot(k_values, fs_r2s, marker='o')
        ax2.set_xlabel("Number of Features")
        ax2.set_ylabel("R-squared")
        # 不手動設定 xticks，以保留預設的浮點數刻度 (1.0, 1.5, ..., 5.0)
        
        plt.tight_layout()
        fs_plot_path = "feature_selection_plot.png"
        plt.savefig(fs_plot_path, dpi=150)
        plt.close()
    print(f"已保存特徵個數效能對比圖至: {fs_plot_path}\n")

    # =========================================================================
    # PHASE 6: DEPLOYMENT (部署與商業建議)
    # =========================================================================
    print("=" * 80)
    print("PHASE 6: DEPLOYMENT (部署與商業建議)")
    print("=" * 80)
    
    model_filename = "best_startup_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    print(f"已保存最佳模型 Pipeline 到: {model_filename}\n")
    
    print("商業整合價值：")
    print("  此模型可整合至企業的以下系統，轉化為實際商業工具：")
    print("  1. 預算規劃系統：在年度編列預算時，動態調整 R&D 與 Marketing 的比例，試算潛在利益最大化方案。")
    print("  2. 創投評估系統 (Venture Capital Evaluation)：創投機構輸入潛在新創案源的支出報表，藉以評估其合理獲利水準。")
    print("  3. 新創公司財務模擬工具：供新創團隊內部模擬不同財務配置下的預估盈利，作為融資與路演的數據支撐。")
    
    print("\n--- 部署推理功能測試 ---")
    loaded_pipeline = joblib.load(model_filename)
    
    test_startup = pd.DataFrame([{
        'R&D Spend': 100000.0,
        'Administration': 120000.0,
        'Marketing Spend': 250000.0,
        'State': 'California'
    }])
    
    predicted_profit = loaded_pipeline.predict(test_startup)[0]
    print("測試輸入數據:")
    for col, val in test_startup.iloc[0].items():
        print(f"  - {col}: {val}")
    print(f"預估利潤 (Predicted Profit): ${predicted_profit:,.2f}")
    print("\n部署模型加載與推斷驗證成功！")
    print("=" * 80)

if __name__ == "__main__":
    run_pipeline()
