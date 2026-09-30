import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
import os

# 忽略警告和设置中文字体
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False


def create_data():
    """创建实验数据"""
    base_data = {
        'pH': [2, 3, 4, 5, 6, 7],
        'As200': [16.044, 32.658, 39.386, 46.706, 55.969, 41.626],
        'Cd100As200': [38.914, 66.38, 68.999, 69.2215, 76.086, 78.774],
        'Cd200As200': [42.46, 67.611, 74.338, 78.0945, 81.851, 77.047],
        'Cd400As200': [42.147, 67.179, 70.651, 69.46867, 73.632, 72.218]
    }

    df = pd.DataFrame(base_data)

    # 添加pH=5特殊实验数据
    ph5_index = df.index[df['pH'] == 5].tolist()[0]
    as_concentrations = {
        'As5': 3.82218, 'As10': 6.99845, 'As20': 12.2402, 'As40': 24.101,
        'As60': 30.2617, 'As80': 35.4105, 'As100': 43.308,
        'As150': 38.971, 'As400': 114.704, 'As600': 162.345,
        'As800': 158.848
    }

    for col, value in as_concentrations.items():
        df[col] = np.nan
        df.at[ph5_index, col] = value

    return df


def reshape_data(df):
    """重组数据为长格式，包含所有数据"""
    long_form = []

    # 处理原始实验数据
    for index, row in df.iterrows():
        # 添加As200数据
        long_form.append({
            'pH': row['pH'], 'Cd_conc': 0, 'As_conc': 200,
            'As_adsorbed': row['As200']
        })

        # 添加不同镉浓度数据
        for cd_conc, col in [(100, 'Cd100As200'), (200, 'Cd200As200'), (400, 'Cd400As200')]:
            long_form.append({
                'pH': row['pH'], 'Cd_conc': cd_conc, 'As_conc': 200,
                'As_adsorbed': row[col]
            })

    # 处理pH=5时不同砷浓度数据
    ph5_index = df.index[df['pH'] == 5].tolist()[0]
    as_columns = {
        'As5': 5, 'As10': 10, 'As20': 20, 'As40': 40, 'As60': 60,
        'As80': 80, 'As100': 100, 'As150': 150, 'As200': 200,
        'As400': 400, 'As600': 600, 'As800': 800
    }

    for col, conc in as_columns.items():
        if col in df.columns and not pd.isna(df.at[ph5_index, col]):
            long_form.append({
                'pH': 5, 'Cd_conc': 0, 'As_conc': float(conc),
                'As_adsorbed': df.at[ph5_index, col]
            })

    return pd.DataFrame(long_form).dropna().reset_index(drop=True)


def reshape_data_fixed_as(df):
    """重组数据为长格式，仅保留As浓度固定为200mg/L的数据"""
    long_form = []

    # 处理原始实验数据
    for index, row in df.iterrows():
        # 添加As200数据
        long_form.append({
            'pH': row['pH'], 'Cd_conc': 0, 'As_conc': 200,
            'As_adsorbed': row['As200']
        })

        # 添加不同镉浓度数据
        for cd_conc, col in [(100, 'Cd100As200'), (200, 'Cd200As200'), (400, 'Cd400As200')]:
            long_form.append({
                'pH': row['pH'], 'Cd_conc': cd_conc, 'As_conc': 200,
                'As_adsorbed': row[col]
            })

    return pd.DataFrame(long_form).dropna().reset_index(drop=True)


def plot_ph5_relationship(df):
    """绘制pH=5时砷浓度与吸附量关系"""
    ph5_data = df[(df['pH'] == 5) & (df['Cd_conc'] == 0)].sort_values('As_conc')

    if len(ph5_data) > 0:
        plt.figure(figsize=(12, 8))
        plt.plot(ph5_data['As_conc'], ph5_data['As_adsorbed'],
                 marker='o', linestyle='-', color='green', linewidth=2)
        plt.xlabel('砷浓度 (mg/L)', fontsize=14)
        plt.ylabel('砷吸附量 (mg/g)', fontsize=14)
        plt.title('pH=5条件下砷浓度对吸附量的影响', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.savefig('pH5砷浓度吸附关系.png', dpi=300)
        plt.show()
        plt.close()


def analyze_cd_effect(df):
    """专门分析镉浓度对砷吸附的影响"""
    # 筛选固定砷浓度(200)数据
    fixed_as_df = df[df['As_conc'] == 200]

    # 按pH分组绘制镉浓度与吸附量的关系
    plt.figure(figsize=(14, 8))
    for ph in sorted(fixed_as_df['pH'].unique()):
        ph_data = fixed_as_df[fixed_as_df['pH'] == ph]
        plt.plot(ph_data['Cd_conc'], ph_data['As_adsorbed'],
                 marker='o', linewidth=2, label=f'pH={ph}')

    plt.xlabel('镉浓度 (mg/L)', fontsize=14)
    plt.ylabel('砷吸附量 (mg/g)', fontsize=14)
    plt.title('不同pH下镉浓度对砷吸附的影响', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('镉浓度影响分析.png', dpi=300)
    plt.show()
    plt.close()

    # 计算镉的促进效应
    promotion_df = []
    for ph in sorted(fixed_as_df['pH'].unique()):
        ph_data = fixed_as_df[fixed_as_df['pH'] == ph]
        if len(ph_data) > 1:  # 确保有足够数据点
            # 无镉时的吸附量
            no_cd_data = ph_data[ph_data['Cd_conc'] == 0]
            if len(no_cd_data) > 0:
                no_cd_adsorption = no_cd_data['As_adsorbed'].values[0]

                # 有镉时的平均吸附量
                with_cd_data = ph_data[ph_data['Cd_conc'] > 0]
                if len(with_cd_data) > 0:
                    with_cd_adsorption = with_cd_data['As_adsorbed'].mean()

                    # 计算促进百分比
                    promotion_pct = (with_cd_adsorption - no_cd_adsorption) / no_cd_adsorption * 100
                    promotion_df.append({
                        'pH': ph,
                        'promotion_pct': promotion_pct
                    })

    promotion_df = pd.DataFrame(promotion_df)

    # 绘制促进效应图
    if len(promotion_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(promotion_df['pH'], promotion_df['promotion_pct'])
        plt.xlabel('pH', fontsize=14)
        plt.ylabel('镉促进砷吸附效应 (%)', fontsize=14)
        plt.title('不同pH下镉对砷吸附的平均促进效应', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig('镉促进效应.png', dpi=300)

        plt.close()

    return promotion_df


def plot_correlation(df):
    # 1. 定义要进行相关性分析的原始列名列表
    corr_columns_original = ['pH', 'Cd_conc', 'As_conc', 'As_adsorbed']
    
    # 2. 定义固定的列名映射
    fixed_column_map = {
        'pH': 'pH',
        'Cd_conc': 'Cd',
        'As_conc': 'As',
        'As_adsorbed': 'As Adsorption'
    }

    # 3. 检查原始DataFrame中实际存在的列，并创建副本
    existing_cols = [col for col in corr_columns_original if col in df.columns]
    
    if not existing_cols:
        print("错误：输入 DataFrame 中不包含任何指定的原始列，无法生成相关性矩阵。")
        print(f"（需要以下列中的至少一个：{', '.join(corr_columns_original)}）")
        return
    
    if len(existing_cols) < len(corr_columns_original):
        missing_cols = [col for col in corr_columns_original if col not in existing_cols]
        print(f"警告：原始 DataFrame 中缺少以下指定列：{', '.join(missing_cols)}。这些列将不会出现在图中。")
        
    temp_df = df[existing_cols].copy()

    # 4. 应用固定的列名映射
    # rename只会作用于temp_df中实际存在的列
    temp_df.rename(columns=fixed_column_map, inplace=True)
    # temp_df现在的列名已经是映射后的名称了，例如 'Cd', 'As' 等

    # 5. 计算相关性矩阵
    # correlation_matrix 将使用映射后的列名
    correlation_matrix = temp_df.corr()

    if correlation_matrix.empty or correlation_matrix.shape[0] < 2: # 需要至少两列/行才能有意义的相关性
        print("错误：无法计算有效的相关性矩阵。处理后的数据不足（少于2个有效列/行）。")
        return

    # 6. 绘制热图
    plt.figure(figsize=(10, 8))

    heatmap_ax = sns.heatmap(correlation_matrix,
                             annot=True,
                             annot_kws={'color': 'black','size':18}, 
                             cmap='coolwarm',
                             vmin=-1, vmax=1,
                             square=True,
                             xticklabels=correlation_matrix.columns, 
                             yticklabels=correlation_matrix.columns, 
                            )

    plt.title('Correlation Matrix', fontsize=22, fontweight='bold', color='black') 

    plt.xticks(color='black', rotation=0, ha='center',fontsize=20) 
    plt.yticks(color='black',fontsize=20) 
    if heatmap_ax and hasattr(heatmap_ax, 'figure') and heatmap_ax.figure.axes and len(heatmap_ax.figure.axes) > 1:
        cbar = heatmap_ax.figure.axes[-1]
        if hasattr(cbar, 'yaxis') and hasattr(cbar.yaxis, 'label'):
            cbar.yaxis.label.set_color('black')
        cbar.tick_params(colors='black')
    cbar.tick_params(colors='black', labelsize=18)
    plt.tight_layout()
    plt.savefig('相关性矩阵.png', dpi=600)
    plt.close()
    print(f"相关性矩阵图片 '相关性矩阵.png' 已生成，并使用了预设的列名（例如 'Cd', 'As' 等）。")
    return df

def build_optimized_model(df):
    """构建优化的回归模型，包含线性项、二次项和二阶交互项"""
    # 准备特征
    analysis_df = df.copy()

    # 添加二次项
    analysis_df['pH_sq'] = analysis_df['pH'] ** 2
    analysis_df['Cd_conc_sq'] = analysis_df['Cd_conc'] ** 2
    analysis_df['As_conc_sq'] = analysis_df['As_conc'] ** 2

    # 标准化变量用于重要性比较
    scaler = StandardScaler()
    std_vars = ['pH', 'Cd_conc', 'As_conc']
    std_df = analysis_df.copy()
    std_df[std_vars] = scaler.fit_transform(std_df[std_vars])

    # 为标准化数据添加二次项
    std_df['pH_sq'] = std_df['pH'] ** 2
    std_df['Cd_conc_sq'] = std_df['Cd_conc'] ** 2
    std_df['As_conc_sq'] = std_df['As_conc'] ** 2

    # 构建完整模型公式（仅包含二阶项）
    full_formula = ("As_adsorbed ~ pH + Cd_conc + As_conc + " +  # 线性项
                    "pH_sq + Cd_conc_sq + As_conc_sq + " +  # 二次项
                    "pH:Cd_conc + pH:As_conc + Cd_conc:As_conc")  # 二阶交互项

    # 使用原始数据构建模型
    full_model = sm.formula.ols(formula=full_formula, data=analysis_df).fit()
    print("\n完整回归模型结果:")
    print(full_model.summary())

    # 提取显著项构建优化模型
    significant_terms = []
    for i, p_value in enumerate(full_model.pvalues):
        if p_value < 0.05:  # 使用0.05作为显著性水平
            term = full_model.pvalues.index[i]
            significant_terms.append(term)

    # 构建优化后的模型公式
    if len(significant_terms) > 1:  # 除了截距外还有其他显著项
        optimized_formula = "As_adsorbed ~ " + " + ".join(significant_terms[1:])  # 排除截距项
        optimized_model = sm.formula.ols(formula=optimized_formula, data=analysis_df).fit()
    else:
        optimized_model = full_model

    print("\n优化后的回归模型结果:")
    print(optimized_model.summary())

    # 构建优化模型方程
    opt_coefs = optimized_model.params
    opt_intercept = opt_coefs[0]

    opt_eq_terms = [f"{opt_intercept:.4f}"]
    for i, name in enumerate(optimized_model.params.index[1:]):
        coef = opt_coefs[i + 1]
        if abs(coef) > 0.0001:
            # 美化变量名称
            pretty_name = name
            pretty_name = pretty_name.replace(':pH', '·pH')
            pretty_name = pretty_name.replace(':Cd_conc', '·[Cd]')
            pretty_name = pretty_name.replace(':As_conc', '·[As]')
            pretty_name = pretty_name.replace('pH_sq', 'pH²')
            pretty_name = pretty_name.replace('Cd_conc_sq', '[Cd]²')
            pretty_name = pretty_name.replace('As_conc_sq', '[As]²')
            pretty_name = pretty_name.replace('pH', 'pH')
            pretty_name = pretty_name.replace('Cd_conc', '[Cd]')
            pretty_name = pretty_name.replace('As_conc', '[As]')

            if coef > 0:
                opt_eq_terms.append(f"+ {coef:.6f}·{pretty_name}")
            else:
                opt_eq_terms.append(f"- {abs(coef):.6f}·{pretty_name}")

    optimized_equation = "As_adsorbed = " + " ".join(opt_eq_terms)
    print("\n优化后的回归方程:")
    print(optimized_equation)

    # 保存优化方程到TXT文件
    with open('优化回归方程.txt', 'w', encoding='utf-8') as f:
        f.write(optimized_equation)

    # 生成优化方程图片
    plt.figure(figsize=(14, 4))
    plt.text(0.5, 0.5, optimized_equation, fontsize=14, ha='center', va='center', family='serif')
    plt.axis('off')
    plt.savefig('优化回归方程.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 模型评估
    y = analysis_df['As_adsorbed']
    y_opt_pred = optimized_model.predict(analysis_df)
    opt_r2 = r2_score(y, y_opt_pred)
    opt_rmse = np.sqrt(mean_squared_error(y, y_opt_pred))
    print(f"\n优化模型性能:")
    print(f"R² = {opt_r2:.4f}")
    print(f"RMSE = {opt_rmse:.4f}")

    # 计算变量重要性
    # 使用标准化数据构建模型用于计算重要性
    std_opt_model = sm.formula.ols(
        formula=optimized_formula if len(significant_terms) > 1 else full_formula,
        data=std_df
    ).fit()

    importance = {}
    for var in ['pH', 'Cd_conc', 'As_conc']:
        # 获取与该变量相关的所有项（主效应、二次项和交互项）
        related_terms = [term for term in std_opt_model.params.index
                         if var in term]
        # 计算绝对系数之和
        importance[var] = sum(abs(std_opt_model.params[term])
                              for term in related_terms)

    # 计算相对重要性百分比
    total = sum(importance.values())
    relative_importance = {k: v / total * 100 for k, v in importance.items()}

    # 绘制重要性饼图
    plt.figure(figsize=(10, 8))
    labels = {'pH': 'pH', 'Cd_conc': '[Cd]', 'As_conc': '[As]'}
    plt.pie([relative_importance[k] for k in labels.keys()],
            labels=[labels[k] for k in labels.keys()],
            autopct='%1.1f%%',
            colors=['skyblue', 'lightgreen', 'salmon'],
            textprops={'fontsize': 14})
    plt.title('优化模型：各因素相对重要性', fontsize=16)
    plt.savefig('优化模型重要性饼图.png', dpi=300)
    plt.show()
    plt.close()

    # 保存重要性到TXT
    with open('优化模型重要性.txt', 'w', encoding='utf-8') as f:
        f.write("优化模型各因素相对重要性(%):\n")
        for var in labels.keys():
            f.write(f"{labels[var]}: {relative_importance[var]:.2f}%\n")

    # 实际值vs预测值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_opt_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual As Adsorption (mg/g)', fontsize=14,fontweight='bold')
    plt.ylabel('Predicted As adsorption (mg/g)', fontsize=14,fontweight='bold')
    plt.title('Actual vs Predicted\nR² = {:.4f}'.format(opt_r2), fontsize=16,fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('实际值vs预测值.pdf', dpi=600, bbox_inches='tight')
    plt.close()

    return {
        'model': optimized_model,
        'equation': optimized_equation,
        'r2': opt_r2,
        'rmse': opt_rmse,
        'importance': relative_importance
    }


def build_optimized_model_simplified(df):
    """构建优化的回归模型（简化版，固定砷浓度），包含线性项、二次项和二阶交互项"""
    # 准备特征
    analysis_df = df.copy()

    # 添加二次项
    analysis_df['pH_sq'] = analysis_df['pH'] ** 2
    analysis_df['Cd_conc_sq'] = analysis_df['Cd_conc'] ** 2

    # 标准化变量用于重要性比较
    scaler = StandardScaler()
    std_vars = ['pH', 'Cd_conc']
    std_df = analysis_df.copy()
    std_df[std_vars] = scaler.fit_transform(std_df[std_vars])

    # 为标准化数据添加二次项
    std_df['pH_sq'] = std_df['pH'] ** 2
    std_df['Cd_conc_sq'] = std_df['Cd_conc'] ** 2

    # 构建完整模型公式（仅包含二阶项）
    full_formula = ("As_adsorbed ~ pH + Cd_conc + " +  # 线性项
                    "pH_sq + Cd_conc_sq + " +  # 二次项
                    "pH:Cd_conc")  # 二阶交互项

    # 使用原始数据构建模型
    full_model = sm.formula.ols(formula=full_formula, data=analysis_df).fit()
    print("\n完整回归模型结果（简化版）:")
    print(full_model.summary())

    # 提取显著项构建优化模型
    significant_terms = []
    for i, p_value in enumerate(full_model.pvalues):
        if p_value < 0.05:  # 使用0.05作为显著性水平
            term = full_model.pvalues.index[i]
            significant_terms.append(term)

    # 构建优化后的模型公式
    if len(significant_terms) > 1:  # 除了截距外还有其他显著项
        optimized_formula = "As_adsorbed ~ " + " + ".join(significant_terms[1:])  # 排除截距项
        optimized_model = sm.formula.ols(formula=optimized_formula, data=analysis_df).fit()
    else:
        optimized_model = full_model

    print("\n优化后的回归模型结果（简化版）:")
    print(optimized_model.summary())

    # 构建优化模型方程
    opt_coefs = optimized_model.params
    opt_intercept = opt_coefs[0]

    opt_eq_terms = [f"{opt_intercept:.4f}"]
    for i, name in enumerate(optimized_model.params.index[1:]):
        coef = opt_coefs[i + 1]
        if abs(coef) > 0.0001:
            # 美化变量名称
            pretty_name = name
            pretty_name = pretty_name.replace(':pH', '·pH')
            pretty_name = pretty_name.replace(':Cd_conc', '·[Cd]')
            pretty_name = pretty_name.replace('pH_sq', 'pH²')
            pretty_name = pretty_name.replace('Cd_conc_sq', '[Cd]²')
            pretty_name = pretty_name.replace('pH', 'pH')
            pretty_name = pretty_name.replace('Cd_conc', '[Cd]')

            if coef > 0:
                opt_eq_terms.append(f"+ {coef:.6f}·{pretty_name}")
            else:
                opt_eq_terms.append(f"- {abs(coef):.6f}·{pretty_name}")

    optimized_equation = "As_adsorbed = " + " ".join(opt_eq_terms)
    print("\n优化后的回归方程（简化版）:")
    print(optimized_equation)

    # 保存优化方程到TXT文件
    with open('优化回归方程_固定砷.txt', 'w', encoding='utf-8') as f:
        f.write(optimized_equation)

    # 生成优化方程图片
    plt.figure(figsize=(14, 4))
    plt.text(0.5, 0.5, optimized_equation, fontsize=14, ha='center', va='center', family='serif')
    plt.axis('off')
    plt.savefig('优化回归方程_固定砷.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # 模型评估
    y = analysis_df['As_adsorbed']
    y_opt_pred = optimized_model.predict(analysis_df)
    opt_r2 = r2_score(y, y_opt_pred)
    opt_rmse = np.sqrt(mean_squared_error(y, y_opt_pred))
    print(f"\n优化模型性能（简化版）:")
    print(f"R² = {opt_r2:.4f}")
    print(f"RMSE = {opt_rmse:.4f}")

    # 计算变量重要性
    # 使用标准化数据构建模型用于计算重要性
    std_opt_model = sm.formula.ols(
        formula=optimized_formula if len(significant_terms) > 1 else full_formula,
        data=std_df
    ).fit()

    importance = {}
    for var in ['pH', 'Cd_conc']:  # 简化版只有这两个变量
        # 获取与该变量相关的所有项（主效应、二次项和交互项）
        related_terms = [term for term in std_opt_model.params.index
                         if var in term]
        # 计算绝对系数之和
        importance[var] = sum(abs(std_opt_model.params[term])
                              for term in related_terms)

    # 计算相对重要性百分比
    total = sum(importance.values())
    relative_importance = {k: v / total * 100 for k, v in importance.items()}

    # 绘制重要性饼图
    plt.figure(figsize=(10, 8))
    labels = {'pH': 'pH', 'Cd_conc': '[Cd]'}
    plt.pie([relative_importance[k] for k in labels.keys()],
            labels=[labels[k] for k in labels.keys()],
            autopct='%1.1f%%',
            colors=['skyblue', 'lightgreen'],
            textprops={'fontsize': 14})
    plt.title('优化模型：各因素相对重要性（固定砷浓度）', fontsize=16)
    plt.savefig('优化模型重要性饼图_固定砷.png', dpi=300)
    plt.show()
    plt.close()

    # 保存重要性到TXT
    with open('优化模型重要性_固定砷.txt', 'w', encoding='utf-8') as f:
        f.write("优化模型各因素相对重要性(%) - 固定砷浓度:\n")
        for var in labels.keys():
            f.write(f"{labels[var]}: {relative_importance[var]:.2f}%\n")

    # 实际值vs预测值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(y, y_opt_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('实际值', fontsize=14)
    plt.ylabel('预测值', fontsize=14)
    plt.title('实际值 vs 预测值（固定砷浓度）\nR² = {:.4f}'.format(opt_r2), fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('实际值vs预测值_固定砷.png', dpi=300)
    plt.show()
    plt.close()

    return {
        'model': optimized_model,
        'equation': optimized_equation,
        'r2': opt_r2,
        'rmse': opt_rmse,
        'importance': relative_importance
    }


def main():
    """主函数，运行完整分析"""
    output_dir = r"C:\Users\AMDYE\OneDrive\桌面"
    # 第一部分：包含所有数据的完整分析
    all_data_dir = os.path.join(output_dir, "砷吸附分析结果_全部数据")
    os.makedirs(all_data_dir, exist_ok=True)
    os.chdir(all_data_dir)

    # 数据处理
    original_df = create_data()
    df = reshape_data(original_df)
    print(f"数据处理完成，共{len(df)}行。")

    # 数据分析与可视化
    plot_ph5_relationship(df)
    promotion_df = analyze_cd_effect(df)
    plot_correlation(df)

    # 使用新的优化模型进行建模分析
    model_results = build_optimized_model(df)

    print("\n全部数据分析完成！所有结果已保存至当前目录。")

    # 恢复工作目录
    os.chdir("..")

    print("\n所有分析已完成！")


if __name__ == "__main__":
    main()