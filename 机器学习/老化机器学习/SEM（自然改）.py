import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- 1. 数据读取与预处理 ---
file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"
df = pd.read_csv(file_path)

# 剔除 T-Aging 列
if 'T-Aging' in df.columns:
    df = df.drop(columns=['T-Aging'])

# 强制转换所有列为数值类型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 定义目标变量
M_col = 'pH-S-CH'  # 中介变量 (ΔpH)
Y_col = 'Cd-CH'    # 因变量 (ΔCd)
output_columns = [M_col, Y_col]

# 筛选输入特征 (建议根据自然老化的特点筛选核心特征以提高 N/K Ratio)
X_cols = [col for col in df.columns if col not in output_columns]

# 清理缺失值
df = df.dropna(subset=output_columns)
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
df[X_cols] = knn_imputer.fit_transform(df[X_cols])

# 数据标准化 (获取标准化系数 Beta)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- 2. 构建核心回归函数 ---
def run_regression_analysis(predictors, target, data, model_name):
    """执行线性回归并提取详细指标"""
    X = sm.add_constant(data[predictors])
    y = data[target]
    model = sm.OLS(y, X).fit()
    
    # 路径系数表
    path_table = pd.DataFrame({
        'Beta (Std)': model.params,
        'Std.Error': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues,
        'Sig': model.pvalues.apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns')))
    }).drop('const', errors='ignore')
    
    # 模型性能指标
    metrics = {
        'Model': model_name,
        'N (Samples)': int(model.nobs),
        'K (Paths)': len(predictors),
        'N/K Ratio': round(model.nobs / len(predictors), 2),
        'R2': round(model.rsquared, 4),
        'Adj_R2': round(model.rsquared_adj, 4),
        'AIC': round(model.aic, 2),
        'BIC': round(model.bic, 2)
    }
    return model, path_table, metrics

# --- 3. 执行多模型分析 ---

# 路径 1: 所有自变量 -> 中介变量 (X -> M)
model_XM, path_XM, metrics_XM = run_regression_analysis(X_cols, M_col, df_scaled, "Path_X_to_M")

# 路径 2: 直接效应对比模型 (X -> Y)
model_XY_dir, path_XY_dir, metrics_XY_dir = run_regression_analysis(X_cols, Y_col, df_scaled, "Direct_Model")

# 路径 3: 中介全模型 (X + M -> Y)
model_SEM, path_SEM, metrics_SEM = run_regression_analysis(X_cols + [M_col], Y_col, df_scaled, "Full_SEM_Model")

# --- 4. 中介效应定量拆解 ---
mediation_results = []
b_path = model_SEM.params[M_col] # M -> Y 的系数

for x in X_cols:
    a_path = model_XM.params[x]       # X -> M
    c_prime = model_SEM.params[x]     # X -> Y (直接)
    
    indirect_eff = a_path * b_path
    total_eff = c_prime + indirect_eff
    
    # 判断中介类型
    p_direct = model_SEM.pvalues[x]
    if p_direct > 0.05 and abs(indirect_eff) > 0.01:
        concl = "Full Mediation"
    elif p_direct <= 0.05 and abs(indirect_eff) > 0.01:
        concl = "Partial Mediation"
    else:
        concl = "No Mediation"

    mediation_results.append({
        'Factor': x,
        'Direct (c\')': round(c_prime, 4),
        'Indirect (a*b)': round(indirect_eff, 4),
        'Total (c)': round(total_eff, 4),
        'Mediation Type': concl
    })

df_mediation = pd.DataFrame(mediation_results)

# --- 5. 打印结果与对比分析 ---
print("\n" + "="*80)
print("表 1: 自然老化模型评估与 AIC 对比 (响应编辑意见 1 & 4)")
print("-" * 80)
comparison_df = pd.DataFrame([metrics_XY_dir, metrics_SEM])
print(comparison_df.set_index('Model').T)

delta_aic = metrics_XY_dir['AIC'] - metrics_SEM['AIC']
print(f"\n关键判别: ΔAIC = {delta_aic:.2f}")
if delta_aic > 2:
    print("✅ 结论: 引入 ΔpH 中介路径后 AIC 显著降低，模型结构更优。")
else:
    print("⚠️ 结论: 引入中介路径后 AIC 提升不明显，请检查中介变量的显著性。")

print("\n" + "="*80)
print("表 2: 全路径模型系数 (用于绘图)")
print("-" * 80)
print(path_SEM)

print("\n" + "="*80)
print("表 3: 自然老化中介效应路径解析")
print("-" * 80)
print(df_mediation.to_string(index=False))

# --- 6. 保存至 Excel ---
with pd.ExcelWriter('Natural_Aging_SEM_Results.xlsx') as writer:
    comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    path_SEM.to_excel(writer, sheet_name='Full_Model_Coefficients')
    df_mediation.to_excel(writer, sheet_name='Mediation_Analysis', index=False)

print(f"\n[完成] 自然老化 SEM 结果已保存至 'Natural_Aging_SEM_Results.xlsx'")