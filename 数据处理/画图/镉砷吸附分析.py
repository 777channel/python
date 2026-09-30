import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score
import shap
import warnings

# 屏蔽警告
warnings.filterwarnings('ignore')

# ================= 1. 全局样式配置 (新罗马 24号 加粗) =================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.unicode_minus'] = False 

# ================= 2. 数据处理与模型训练 =================
file_path = r"C:\Users\AMDYE\OneDrive\桌面\分析.csv"
df = pd.read_csv(file_path)

X_raw = df[['DW-D', 'W', 'Cd']]
y_raw = df['As']

# 标准化
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_raw)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y_raw.values.reshape(-1, 1)).ravel()

# SVR 训练
param_grid = {'C': [1, 10, 50, 100], 'gamma': ['scale', 'auto', 0.01]}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=cv, scoring='r2')
grid_search.fit(X_scaled, y_scaled)
best_svr = grid_search.best_estimator_

# 预测还原
y_pred = scaler_y.inverse_transform(best_svr.predict(X_scaled).reshape(-1, 1)).ravel()
final_r2 = r2_score(y_raw, y_pred)

# ================= 3. 图 1: 修改后的标准拟合图 =================
plt.figure(figsize=(10, 10))
# 改为黑色点，保持加粗边缘 (根据用户要求，散点颜色设为黑色)
plt.scatter(y_raw, y_pred, s=150, color='#b4d4e1', edgecolor='#b4d4e1', alpha=0.7, label='Experimental Data')

# 1:1 线 (y=x)
low = min(y_raw.min(), y_pred.min())
high = max(y_raw.max(), y_pred.max())
plt.plot([low, high], [low, high], 'k--', lw=3, alpha=0.8, label='y=x (Ideal)')

# 文本框
bbox = dict(boxstyle="round,pad=0.4", edgecolor="black", facecolor="white", lw=1.5)
plt.gca().text(0.95, 0.05, f'Total $R^2$ = {final_r2:.3f}', transform=plt.gca().transAxes, ha='right', va='bottom', bbox=bbox)
plt.gca().text(0.05, 0.95, 'Model: SVR', transform=plt.gca().transAxes, ha='left', va='top', bbox=bbox)

plt.xlabel('Measured As ($\mu$g/kg)')
plt.ylabel('Predicted As ($\mu$g/kg)')
plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.88), frameon=False, fontsize=20)
plt.grid(True, ls='--', alpha=0.3)
plt.savefig('SVR_Standard_Fitting_Black.pdf', format='pdf', bbox_inches='tight', dpi=600)
plt.close()

# ================= 4. 图 2: 特征重要性图 (从左往右伸展) =================
explainer = shap.KernelExplainer(best_svr.predict, shap.kmeans(X_scaled, 10))
shap_v = explainer.shap_values(X_scaled)
imp_vals = np.abs(shap_v).mean(axis=0)
relative_imp = imp_vals / np.sum(imp_vals)

imp_df = pd.DataFrame({'Feature': X_raw.columns, 'Importance': relative_imp})
imp_df = imp_df.sort_values('Importance', ascending=True)

cat_map = {'Cd': 'Pollutant', 'W': 'Environment', 'DW-D': 'Time'}
colors = {'Pollutant': '#07AEE3', 'Environment': '#35B257', 'Time': '#B3D49D'}
imp_df['Color'] = imp_df['Feature'].map(cat_map).map(colors)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(range(len(imp_df)), imp_df['Importance'], color=imp_df['Color'], height=0.6)

ax.set_xlabel('Contribution', fontweight='bold')
ax.set_xlim(0, imp_df['Importance'].max() * 1.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.set_yticks(range(len(imp_df)))
ax.set_yticklabels([]) 

for i, (idx, row) in enumerate(imp_df.iterrows()):
    ax.text(row['Importance'] + 0.01, i, f"{row['Importance']:.1%}", va='center', ha='left', fontweight='bold')
    ax.text(0.005, i, row['Feature'], color='black', va='center', ha='left', fontweight='bold', zorder=14)

ax_pie = fig.add_axes([0.55, 0.15, 0.35, 0.35])
cat_sums = imp_df.groupby(imp_df['Feature'].map(cat_map))['Importance'].sum()
pie_data = [cat_sums[c] for c in colors if c in cat_sums]
pie_clrs = [colors[c] for c in colors if c in cat_sums]
wedges, _ = ax_pie.pie(pie_data, colors=pie_clrs, startangle=140, wedgeprops=dict(width=0.5, edgecolor='white', lw=2))

for i, p in enumerate(wedges):
    ang = np.deg2rad((p.theta2 - p.theta1) / 2. + p.theta1)
    ax_pie.text(1.4 * np.cos(ang), 1.4 * np.sin(ang), f"{pie_data[i]:.1%}", ha='center', va='center', fontweight='bold', fontsize=20)

plt.savefig('Feature_Importance_Reverse.pdf', format='pdf', bbox_inches='tight')
plt.close()

# ================= 5. 新增：全组合量化影响分析脚本 =================
print("\n" + "="*50)
print("Cd Addition Impact Analysis (SVR Model Inference)")
print("="*50)

results_list = []
days_levels = [6, 12, 18, 24, 30]
water_levels = [0.4, 0.6, 0.8, 1.0]

for d in days_levels:
    for w in water_levels:
        # 构造对比样本并标准化
        raw_no_cd = np.array([[d, w, 0]])
        raw_with_cd = np.array([[d, w, 1]])
        
        # 预测与还原
        val_no_cd = scaler_y.inverse_transform(best_svr.predict(scaler_X.transform(raw_no_cd)).reshape(-1, 1)).ravel()[0]
        val_with_cd = scaler_y.inverse_transform(best_svr.predict(scaler_X.transform(raw_with_cd)).reshape(-1, 1)).ravel()[0]
        
        reduction = val_no_cd - val_with_cd
        percent_drop = (reduction / val_no_cd) * 100
        
        results_list.append({
            'Days': d, 'Water': w, 'As(No_Cd)': val_no_cd, 
            'As(With_Cd)': val_with_cd, 'Reduction': reduction, 'Drop(%)': percent_drop
        })

impact_df = pd.DataFrame(results_list)
# 打印结果表，方便直接复制数据
print(impact_df.to_string(index=False, formatters={
    'As(No_Cd)': '{:.2f}'.format, 'As(With_Cd)': '{:.2f}'.format,
    'Reduction': '{:.2f}'.format, 'Drop(%)': '{:.1f}'.format
}))

print("\n[完成] 黑色散点拟合图、特征重要性图已保存，Cd 抑制效应量化表已输出。")