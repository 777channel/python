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


def format_equation(model, params):
    """格式化方程输出"""
    # 获取截距
    intercept = params[0]
    equation_terms = [f"{intercept:.4f}"]

    # 处理变量名称转换
    name_map = {
        'pH': 'pH',
        'pH_sq': 'pH²',
        'pH:Cd_conc': 'pH·[Cd]',
        'pH:As_conc': 'pH·[As]',
        'Cd_conc': '[Cd]',
        'Cd_conc_sq': '[Cd]²',
        'As_conc': '[As]',
        'As_conc_sq': '[As]²',
        'Cd_conc:As_conc': '[Cd]·[As]'
    }

    # 处理其他参数
    for name, coef in params[1:].items():
        if abs(coef) > 0.0001:
            # 获取美化后的变量名
            pretty_name = name_map.get(name, name)

            # 格式化系数
            if coef > 0:
                equation_terms.append(f"+ {coef:.6f}·{pretty_name}")
            else:
                equation_terms.append(f"- {abs(coef):.6f}·{pretty_name}")

    # 组合完整方程
    full_equation = "Cd_adsorbed = " + " ".join(equation_terms)
    return full_equation


def create_data():
    """创建实验数据"""
    base_data = {
        'pH': [2, 3, 4, 5, 6, 7],
        'Cd400': [34.6, 239.2, 250.6, 254.8, 255.4, 282.4],  # 无砷时的镉吸附数据
        'Cd100As200': [8.95, 56.2, 74.1, 73.45, 76.1, 74.3],
        'Cd200As200': [28.7, 75.2, 98.6, 112.9, 125.4, 138.6],
        'Cd400As200': [28.2, 134.2, 167.6, 175.6, 177.2, 194.2]
    }

    df = pd.DataFrame(base_data)

    # 添加pH=5特殊实验数据
    ph5_index = df.index[df['pH'] == 5].tolist()[0]
    cd_concentrations = {
        'Cd5': 5, 'Cd10': 10, 'Cd20': 20, 'Cd40': 40,
        'Cd80': 80, 'Cd100': 100, 'Cd150': 149,
        'Cd200': 189.7, 'Cd300': 238.75, 'Cd400': 256.8,
        'Cd600': 360.8, 'Cd800': 342
    }

    for col, value in cd_concentrations.items():
        df[col] = np.nan
        df.at[ph5_index, col] = value

    return df


def reshape_data(df):
    """重组数据为长格式"""
    long_form = []

    # 处理原始实验数据
    for index, row in df.iterrows():
        # 添加无砷时的镉吸附数据
        long_form.append({
            'pH': row['pH'], 'As_conc': 0, 'Cd_conc': 400,
            'Cd_adsorbed': row['Cd400']
        })

        # 添加不同镉浓度数据
        for cd_conc, col in [(100, 'Cd100As200'), (200, 'Cd200As200'), (400, 'Cd400As200')]:
            long_form.append({
                'pH': row['pH'], 'Cd_conc': cd_conc, 'As_conc': 200,
                'Cd_adsorbed': row[col]
            })

    # 处理pH=5时不同镉浓度数据
    ph5_index = df.index[df['pH'] == 5].tolist()[0]
    additional_columns = ['Cd5', 'Cd10', 'Cd20', 'Cd40', 'Cd80', 'Cd100',
                          'Cd150', 'Cd200', 'Cd300', 'Cd400', 'Cd600', 'Cd800']

    for col in additional_columns:
        # 提取浓度值（去掉'Cd'前缀并转换为数字）
        conc = float(col.replace('Cd', ''))

        # 只有当值不是NaN时才添加
        if not pd.isna(df.at[ph5_index, col]):
            long_form.append({
                'pH': 5, 'Cd_conc': conc, 'As_conc': 0,  # As_conc为0
                'Cd_adsorbed': df.at[ph5_index, col]
            })

    return pd.DataFrame(long_form).dropna().reset_index(drop=True)
def plot_ph5_relationship(df):
    """绘制pH=5时镉浓度与吸附量关系"""
    ph5_data = df[(df['pH'] == 5) & (df['As_conc'] == 0)].sort_values('Cd_conc')

    plt.figure(figsize=(12, 8))
    plt.plot(ph5_data['Cd_conc'], ph5_data['Cd_adsorbed'],
             marker='o', linestyle='-', color='green', linewidth=2)
    plt.xlabel('镉浓度 (mg/L)', fontsize=14)
    plt.ylabel('镉吸附量 (mg/g)', fontsize=14)
    plt.title('pH=5条件下镉浓度对吸附量的影响', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.savefig('pH5镉浓度吸附关系.png', dpi=300)
    plt.close()


def analyze_as_effect(df):
    """专门分析砷浓度对镉吸附的影响"""
    # 筛选固定镉浓度数据
    fixed_cd_df = df[(df['Cd_conc'] == 400) | (df['Cd_conc'] == 200)]

    # 按pH分组绘制砷浓度与吸附量的关系
    plt.figure(figsize=(14, 8))
    for ph in sorted(fixed_cd_df['pH'].unique()):
        ph_data = fixed_cd_df[fixed_cd_df['pH'] == ph]
        plt.plot(ph_data['As_conc'], ph_data['Cd_adsorbed'],
                 marker='o', linewidth=2, label=f'pH={ph}')

    plt.xlabel('砷浓度 (mg/L)', fontsize=14)
    plt.ylabel('镉吸附量 (mg/g)', fontsize=14)
    plt.title('不同pH下砷浓度对镉吸附的影响', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('砷浓度影响分析.png', dpi=300)
    plt.close()

    # 计算砷的影响效应
    effect_df = []
    for ph in sorted(fixed_cd_df['pH'].unique()):
        ph_data = fixed_cd_df[fixed_cd_df['pH'] == ph]
        if len(ph_data) > 1:  # 确保有足够数据点
            # 无砷时的吸附量
            no_as_data = ph_data[ph_data['As_conc'] == 0]
            if len(no_as_data) > 0:
                no_as_adsorption = no_as_data['Cd_adsorbed'].values[0]

                # 有砷时的平均吸附量
                with_as_data = ph_data[ph_data['As_conc'] > 0]
                if len(with_as_data) > 0:
                    with_as_adsorption = with_as_data['Cd_adsorbed'].mean()

                    # 计算影响百分比
                    effect_pct = (with_as_adsorption - no_as_adsorption) / no_as_adsorption * 100
                    effect_df.append({
                        'pH': ph,
                        'effect_pct': effect_pct
                    })

    effect_df = pd.DataFrame(effect_df)

    # 绘制影响效应图
    if len(effect_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.bar(effect_df['pH'], effect_df['effect_pct'])
        plt.xlabel('pH', fontsize=14)
        plt.ylabel('砷对镉吸附的影响效应 (%)', fontsize=14)
        plt.title('不同pH下砷对镉吸附的影响程度', fontsize=16)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig('砷影响效应.png', dpi=300)
        plt.close()

    return effect_df


def plot_correlation(df):
    # 1. 定义要进行相关性分析的原始列名列表
    corr_columns_original = ['pH', 'Cd_conc', 'As_conc', 'Cd_adsorbed']
    
    # 2. 定义固定的列名映射
    fixed_column_map = {
        'pH': 'pH',
        'Cd_conc': 'Cd',
        'As_conc': 'As',
        'Cd_adsorbed': 'Cd Adsorption'
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
    full_formula = ("Cd_adsorbed ~ pH + Cd_conc + As_conc + " +  # 线性项
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
        optimized_formula = "Cd_adsorbed ~ " + " + ".join(significant_terms[1:])  # 排除截距项
        optimized_model = sm.formula.ols(formula=optimized_formula, data=analysis_df).fit()
    else:
        optimized_model = full_model

    print("\n优化后的回归模型结果:")
    print(optimized_model.summary())

    # 构建优化模型方程
    optimized_equation = format_equation(optimized_model, optimized_model.params)
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
    plt.close()

    # 模型评估
    y = analysis_df['Cd_adsorbed']
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
    plt.xlabel('Actual Cd Adsorption (mg/g)', fontsize=14,fontweight='bold')
    plt.ylabel('Predicted Cd adsorption (mg/g)', fontsize=14,fontweight='bold')
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


def main():
    """主函数，运行完整分析"""
    output_dir = r"C:\Users\AMDYE\OneDrive\桌面"
    # 第一部分：包含所有数据的完整分析
    all_data_dir = os.path.join(output_dir, "镉吸附分析结果_全部数据")
    os.makedirs(all_data_dir, exist_ok=True)
    os.chdir(all_data_dir)

    # 数据处理
    original_df = create_data()
    df = reshape_data(original_df)
    print(f"数据处理完成，共{len(df)}行。")

    # 数据分析与可视化
    plot_ph5_relationship(df)
    effect_df = analyze_as_effect(df)
    plot_correlation(df)

    # 使用新的优化模型进行建模分析
    model_results = build_optimized_model(df)

    print("\n全部数据分析完成！所有结果已保存至当前目录。")

    # 恢复工作目录
    os.chdir("..")

    print("\n所有分析已完成！")


if __name__ == "__main__":
    main()