import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE, f_regression, f_classif, chi2, 
    mutual_info_regression, SequentialFeatureSelector
)
from sklearn.metrics import r2_score, mean_squared_error

# Set visual style
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv("50_Startups.csv")

# Train-test split (using random_state=0 for feature selection matching the template)
X_raw = df.drop(columns=['Profit'])
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=0)

# Preprocess numeric features (Standard Scaling)
num_cols = ['R&D Spend', 'Administration', 'Marketing Spend']
scaler = StandardScaler()
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

# Preprocess categorical feature (One-Hot Encoding, drop California)
X_train_state = pd.get_dummies(X_train['State'], drop_first=True, dtype=float)
X_test_state = pd.get_dummies(X_test['State'], drop_first=True, dtype=float)

# Align columns (ensure Florida and New York exist)
for col in ['Florida', 'New York']:
    if col not in X_train_state.columns:
        X_train_state[col] = 0.0
    if col not in X_test_state.columns:
        X_test_state[col] = 0.0

X_train_state = X_train_state[['Florida', 'New York']].rename(columns={'Florida': 'State_Florida', 'New York': 'State_New York'})
X_test_state = X_test_state[['Florida', 'New York']].rename(columns={'Florida': 'State_Florida', 'New York': 'State_New York'})

# Combine processed features
X_train_scaled = pd.concat([X_train_num, X_train_state], axis=1)
X_test_scaled = pd.concat([X_test_num, X_test_state], axis=1)
features = list(X_train_scaled.columns)

print("Preprocessed features:", features)

# =========================================================================
# IMPLEMENTING THE 9 FEATURE SELECTION ALGORITHMS
# =========================================================================
rankings = {}

# 1. Correlation Analysis (Filter)
corr_scores = [abs(np.corrcoef(X_train_scaled[col], y_train)[0, 1]) for col in features]
rankings['Correlation Analysis'] = [features[i] for i in np.argsort(corr_scores)[::-1]]

# Build min-max scaled features for Chi-Square scoring.
scaler_minmax = MinMaxScaler()
X_train_minmax = pd.DataFrame(scaler_minmax.fit_transform(X_train_scaled), columns=features)

# Pre-bin target for classification-based filter methods (Chi-Square & ANOVA F-Test Classification)
y_train_binned = pd.qcut(y_train, q=3, labels=False, duplicates='drop')

# 2. Chi-Square Test (Filter)
chi2_scores, _ = chi2(X_train_minmax, y_train_binned)
rankings['Chi-Square Test'] = [features[i] for i in np.argsort(chi2_scores)[::-1]]

# 3. ANOVA F-Test (Filter)
f_class_scores, _ = f_classif(X_train_scaled, y_train_binned)
rankings['ANOVA F-Test'] = [features[i] for i in np.argsort(f_class_scores)[::-1]]

# 4. Mutual Information (Filter)
mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
rankings['Mutual Information'] = [features[i] for i in np.argsort(mi_scores)[::-1]]

# 5. SelectKBest (Filter)
f_reg_scores, _ = f_regression(X_train_scaled, y_train)
rankings['SelectKBest'] = [features[i] for i in np.argsort(f_reg_scores)[::-1]]

# 6. Recursive Feature Elimination (RFE) (Wrapper)
rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_scaled, y_train)
rankings['RFE'] = [features[i] for i in np.argsort(rfe.ranking_)]

# 7. Forward Selection (Wrapper)
sfs_rank = []
remaining_features = list(features)
for k in range(1, 5):
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=5)
    sfs.fit(X_train_scaled, y_train)
    selected = [features[i] for i in range(len(features)) if sfs.support_[i]]
    new_feat = [f for f in selected if f not in sfs_rank]
    if new_feat:
        sfs_rank.append(new_feat[0])
        remaining_features.remove(new_feat[0])
sfs_rank.extend(remaining_features)
rankings['Forward Selection'] = sfs_rank

# 8. Lasso (L1 Regularization) (Embedded)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_coefs = np.abs(lasso.coef_)
rankings['Lasso Regression'] = [features[i] for i in np.argsort(lasso_coefs)[::-1]]

# 9. Tree-Based Feature Importance (Embedded)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rankings['Tree-Based Importance'] = [features[i] for i in np.argsort(rf.feature_importances_)[::-1]]

# Print the ranked features for each algorithm
print("\n--- Ranked Features by Method ---")
for method, ranked_f in rankings.items():
    print(f"{method:<25}: {ranked_f}")

# =========================================================================
# EVALUATING PERFORMANCE AT EACH k (1 to 5)
# =========================================================================
k_values = [1, 2, 3, 4, 5]
r2_results = {method: [] for method in rankings.keys()}
rmse_results = {method: [] for method in rankings.keys()}

for k in k_values:
    for method, ranked_f in rankings.items():
        subset = ranked_f[:k]
        lr = LinearRegression()
        lr.fit(X_train_scaled[subset], y_train)
        y_pred = lr.predict(X_test_scaled[subset])
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        r2_results[method].append(r2)
        rmse_results[method].append(rmse)

# Print R2 results table
print("\n--- Test R2 Scores at each k ---")
r2_df = pd.DataFrame(r2_results, index=[f"k={k}" for k in k_values])
print(r2_df.round(6))

# Print RMSE results table
print("\n--- Test RMSE Scores at each k ---")
rmse_df = pd.DataFrame(rmse_results, index=[f"k={k}" for k in k_values])
print(rmse_df.round(2))

# =========================================================================
# PLOTTING THE COMPARISON (LINE CHARTS + SUMMARY TABLE)
# =========================================================================
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[3.1, 1.9], hspace=0.35, wspace=0.18)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

colors = ['#06b6d4', '#10b981', '#f59e0b', '#ec4899', '#f43f5e', '#3b82f6', '#14b8a6', '#6366f1', '#a855f7']
markers = ['o', '^', 'D', 'v', 'p', '*', 'h', 'H', 'X']

# Left subplot: RMSE
for i, (method, rmse_scores) in enumerate(rmse_results.items()):
    ax1.plot(k_values, rmse_scores, label=method, color=colors[i], marker=markers[i], linewidth=2, markersize=7)
ax1.set_title("RMSE by Number of Features", fontsize=12, fontweight='bold')
ax1.set_xlabel("Number of Features", fontsize=11)
ax1.set_ylabel("RMSE", fontsize=11)
ax1.set_xticks(k_values)
ax1.grid(True, linestyle='--', alpha=0.5)

# Right subplot: R-squared
for i, (method, r2_scores) in enumerate(r2_results.items()):
    ax2.plot(k_values, r2_scores, label=method, color=colors[i], marker=markers[i], linewidth=2, markersize=7)
ax2.set_title("R-squared by Number of Features", fontsize=12, fontweight='bold')
ax2.set_xlabel("Number of Features", fontsize=11)
ax2.set_ylabel("R-squared", fontsize=11)
ax2.set_xticks(k_values)
ax2.set_ylim(0.80, 0.96)
ax2.grid(True, linestyle='--', alpha=0.5)

# Summary table under the line charts
method_types = {
    'Correlation Analysis': 'Filter',
    'Chi-Square Test': 'Filter',
    'ANOVA F-Test': 'Filter',
    'Mutual Information': 'Filter',
    'SelectKBest': 'Filter',
    'RFE': 'Wrapper',
    'Forward Selection': 'Wrapper',
    'Lasso Regression': 'Embedded',
    'Tree-Based Importance': 'Embedded',
}
table_rows = []
for i, method in enumerate(rmse_results.keys()):
    best_idx = int(np.argmax(r2_results[method]))
    best_k = k_values[best_idx]
    selected_features = ', '.join(rankings[method][:best_k])
    table_rows.append([
        method,
        method_types[method],
        f"k={best_k}",
        f"{r2_results[method][best_idx]:.4f}",
        f"{rmse_results[method][best_idx]:,.2f}",
        selected_features,
    ])

table = ax_table.table(
    cellText=table_rows,
    colLabels=['Method', 'Type', 'Best k', 'Best R2', 'Best RMSE', 'Selected features at best k'],
    loc='center',
    cellLoc='left',
    colLoc='left',
    colWidths=[0.20, 0.10, 0.08, 0.09, 0.11, 0.42],
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.45)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#cbd5e1')
    if row == 0:
        cell.set_facecolor('#e2e8f0')
        cell.set_text_props(weight='bold')
    else:
        cell.set_facecolor('#f8fafc' if row % 2 else 'white')

plt.suptitle("Top 9 Feature Selection Methods Performance Comparison (50 Startups)", fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

os.makedirs("feature_selection_plots", exist_ok=True)
# Save both to feature_selection_plot.png and allinone.png to keep all links working
plot_path = os.path.join("feature_selection_plots", "feature_selection_plot.png")
allinone_path = os.path.join("feature_selection_plots", "allinone.png")

plt.savefig(plot_path, dpi=180, bbox_inches='tight')
plt.savefig(allinone_path, dpi=180, bbox_inches='tight')
plt.close()

print(f"\nSuccessfully generated and saved comparison plots to:\n  - {plot_path}\n  - {allinone_path}")
