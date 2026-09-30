import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# --- 1. 数据读取与预处理 ---
file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
df = pd.read_csv(file_path)

# 剔除无关列
if 'T_aging' in df.columns:
    df = df.drop(columns=['T_aging'])

# 定义关键变量
M_col = 'pH-S-CH'  # 中介变量 (ΔpH)
Y_col = 'Cd-CH'    # 因变量 (ΔCd)
# 排除目标变量，剩余全部作为自变量
X_cols = [col for col in df.columns if col not in [M_col, Y_col]]

# 处理缺失值
df = df.dropna(subset=[M_col, Y_col])
imputer = KNNImputer(n_neighbors=5)
df[X_cols] = imputer.fit_transform(df[X_cols])

# 数据标准化 (用于获得标准化路径系数 Beta)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# --- 2. 构建核心回归函数 ---
def run_regression(predictors, target, data):
    X = sm.add_constant(data[predictors])
    y = data[target]
    model = sm.OLS(y, X).fit()
    
    # 提取详细统计表
    results = pd.DataFrame({
        'Beta (Std)': model.params,
        'Std.Error': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues,
        'Sig': model.pvalues.apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns')))
    }).drop('const')
    
    # 提取模型整体指标
    metrics = {
        'N (Samples)': int(model.nobs),
        'K (Paths)': len(predictors),
        'N/K Ratio': round(model.nobs / len(predictors), 2),
        'R-squared': round(model.rsquared, 4),
        'Adj. R-squared': round(model.rsquared_adj, 4),
        'AIC': round(model.aic, 2),
        'BIC': round(model.bic, 2),
        'F-statistic': round(model.fvalue, 2)
    }
    return model, results, metrics

# --- 3. 执行多模型分析 ---

# 模型 A: 直接效应模型 (仅 X -> Y)
model_A, res_A, met_A = run_regression(X_cols, Y_col, df_scaled)

# 模型 B: 中介过程模型 (X -> M)
model_B, res_B, met_B = run_regression(X_cols, M_col, df_scaled)

# 模型 C: 全路径模型 (X + M -> Y)
model_C, res_C, met_C = run_regression(X_cols + [M_col], Y_col, df_scaled)

# --- 4. 中介效应定量计算 ---
mediation_data = []
b_path = model_C.params[M_col] # M -> Y 的系数

for x in X_cols:
    a_path = model_B.params[x]       # X -> M 的系数
    c_prime_path = model_C.params[x] # X -> Y (控制M后) 的系数 (直接效应)
    
    indirect_eff = a_path * b_path
    total_eff = c_prime_path + indirect_eff
    
    # 计算占比 (Proportion Mediated)
    prop = (indirect_eff / total_eff) if abs(total_eff) > 1e-5 else 0
    
    mediation_data.append({
        'Factor': x,
        'Direct (c\')': round(c_prime_path, 4),
        'Indirect (a*b)': round(indirect_eff, 4),
        'Total (c)': round(total_eff, 4),
        'Mediation Ratio': f"{prop*100:.2f}%" if np.sign(indirect_eff) == np.sign(total_eff) else "Competitive"
    })

df_mediation = pd.DataFrame(mediation_data)

# --- 5. 格式化输出详细结果 ---

print("\n" + "="*80)
print("表 1: 模型整体性能与对比 (响应编辑意见 1 & 4)")
print("-" * 80)
comparison_df = pd.DataFrame([met_A, met_C], index=['Direct Model (No ΔpH)', 'Full SEM Model (With ΔpH)'])
print(comparison_df.T)
print(f"\n关键判别: ΔAIC = {round(met_A['AIC'] - met_C['AIC'], 2)} (正值且>2表示中介模型显著更优)")

print("\n" + "="*80)
print("表 2: 全路径模型系数 (X + ΔpH -> ΔCd)")
print("-" * 80)
print(res_C)

print("\n" + "="*80)
print("表 3: 中介效应路径解析 (Mediation Effect Breakdown)")
print("-" * 80)
print(df_mediation.to_string(index=False))

# 保存到 Excel
with pd.ExcelWriter('Detailed_SEM_Results.xlsx') as writer:
    comparison_df.to_excel(writer, sheet_name='Model_Comparison')
    res_C.to_excel(writer, sheet_name='Path_Coefficients')
    df_mediation.to_excel(writer, sheet_name='Mediation_Analysis', index=False)

print("\n[系统通知] 详细结果已导出至 'Detailed_SEM_Results.xlsx'")