import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

# 1. 读取数据
file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging(TEXT).csv"
df = pd.read_csv(file_path)

# 2. 预处理
# 剔除 T_Aging 列
if 'T-Aging' in df.columns:
    df = df.drop(columns=['T-Aging'])

# 强制转换所有列为数值类型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 定义输出列
output_columns = ['pH-S-CH', 'Cd-CH']

# 分离输入列和输出列
input_columns = [col for col in df.columns if col not in output_columns]

print(f"处理前缺失值统计：")
print(df.isnull().sum())
print(f"\n原始数据形状: {df.shape}")

# 删除输出列中有缺失值的行
df = df.dropna(subset=output_columns)
print(f"\n删除输出列缺失后数据形状: {df.shape}")

# 对输入列使用 KNN 插补
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
df[input_columns] = knn_imputer.fit_transform(df[input_columns])

# 确认无缺失值
print(f"\nKNN填充后缺失值统计：")
print(df.isnull().sum())
print(f"\n最终数据形状: {df.shape}\n")

# 3. 定义变量
# 自变量 X (所有输入列)
X_columns = input_columns
# 中介变量 M
M_column = 'pH-S-CH'
# 因变量 Y
Y_column = 'Cd-CH'

# 4. 数据标准化 (为了得到标准系数 Beta)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def get_regression_table(predictors, target, df_data):
    """构建回归模型并返回格式化表格"""
    X = df_data[predictors]
    X = sm.add_constant(X)  # 添加截距项
    y = df_data[target]
    
    model = sm.OLS(y, X).fit()
    
    # 提取统计指标
    results_df = pd.DataFrame({
        'Standardized Beta': model.params,
        'Std.Error': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues
    })
    
    # 添加显著性标记
    results_df['Significance'] = results_df['p-value'].apply(
        lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else 'ns'))
    )
    
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    f_pvalue = model.f_pvalue
    
    return results_df, r2, adj_r2, f_pvalue, model

# --- 执行三个模型分析 ---

# 模型 1: 所有变量 -> pH-S-CH
table1, r2_1, adj_r2_1, f_p1, model1 = get_regression_table(X_columns, M_column, df_scaled)

# 模型 2: 所有变量 -> Cd-CH
table2, r2_2, adj_r2_2, f_p2, model2 = get_regression_table(X_columns, Y_column, df_scaled)

# 模型 3: 所有变量 + pH-S-CH -> Cd-CH
table3, r2_3, adj_r2_3, f_p3, model3 = get_regression_table(X_columns + [M_column], Y_column, df_scaled)

# 5. 打印结果
print("="*60)
print("模型 1: 所有变量 -> pH-S-CH (中介变量预测)")
print(f"R-squared: {r2_1:.4f}, Adjusted R-squared: {adj_r2_1:.4f}")
print(f"F-pvalue: {f_p1:.4e}")
print("-"*60)
print(table1.drop('const'))

print("\n" + "="*60)
print("模型 2: 所有变量 -> Cd-CH (总效应分析)")
print(f"R-squared: {r2_2:.4f}, Adjusted R-squared: {adj_r2_2:.4f}")
print(f"F-pvalue: {f_p2:.4e}")
print("-"*60)
print(table2.drop('const'))

print("\n" + "="*60)
print("模型 3: 所有变量 + pH-S-CH -> Cd-CH (直接效应与中介路径分析)")
print(f"R-squared: {r2_3:.4f}, Adjusted R-squared: {adj_r2_3:.4f}")
print(f"F-pvalue: {f_p3:.4e}")
print("-"*60)
print(table3.drop('const'))

# 6. 中介效应分析
print("\n" + "="*60)
print("中介效应分析 (Mediation Analysis)")
print("="*60)

mediation_results = []

for x_var in X_columns:
    # 提取系数
    # a路径: X -> M (从模型1)
    a = model1.params[x_var]
    
    # b路径: M -> Y (控制X后，从模型3)
    b = model3.params[M_column]
    
    # c'路径: X -> Y (控制M后，从模型3)
    c_prime = model3.params[x_var]
    
    # c路径: X -> Y (总效应，从模型2)
    c = model2.params[x_var]
    
    # 间接效应 (Indirect effect): a * b
    indirect_effect = a * b
    
    # 直接效应 (Direct effect): c'
    direct_effect = c_prime
    
    # 总效应 (Total effect): c (或者 c' + a*b)
    total_effect = c
    
    # 中介占比 (Agency Share / Proportion Mediated)
    if abs(total_effect) > 0.0001:  # 避免除以0
        agency_share = indirect_effect / total_effect
    else:
        agency_share = np.nan
    
    # 判断中介类型
    if abs(agency_share) >= 0.8 and np.sign(indirect_effect) == np.sign(total_effect):
        conclusion = "Fully Mediated"
    elif abs(agency_share) < 0.8 and abs(agency_share) > 0.2 and np.sign(indirect_effect) == np.sign(total_effect):
        conclusion = "Partially Mediated"
    elif np.sign(indirect_effect) != np.sign(direct_effect) and abs(agency_share) > 1:
        conclusion = "Competitive Mediation"
    elif abs(direct_effect) < 0.0001 or model3.pvalues[x_var] > 0.05:
        conclusion = "Fully Mediated"
    else:
        conclusion = "No Mediation"
    
    mediation_results.append({
        'Pathway': f"{x_var} → {M_column} → {Y_column}",
        'Indirect effects': round(indirect_effect, 3),
        'Direct effect': round(direct_effect, 3),
        'Total effect': round(total_effect, 3),
        'Agency Share': round(agency_share, 3) if not np.isnan(agency_share) else 'NA',
        'Conclusion': conclusion
    })

mediation_df = pd.DataFrame(mediation_results)
print(mediation_df.to_string(index=False))

# 7. 保存结果到Excel
with pd.ExcelWriter('regression_results.xlsx') as writer:
    table1.drop('const').to_excel(writer, sheet_name='Model1_X_to_M')
    table2.drop('const').to_excel(writer, sheet_name='Model2_X_to_Y')
    table3.drop('const').to_excel(writer, sheet_name='Model3_X+M_to_Y')
    mediation_df.to_excel(writer, sheet_name='Mediation_Analysis', index=False)
    
print("\n结果已保存到 'regression_results.xlsx'")
