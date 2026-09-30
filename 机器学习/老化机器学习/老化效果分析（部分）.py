import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# 添加图例相关的导入
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter  # 添加格式化导入

# --- 导入TabPFN ---
try:
    from tabpfn import TabPFNRegressor
    print("成功导入 TabPFNRegressor。")
    tabpfen_available = True
except ImportError:
    print("严重错误: 'tabpfen' 库未找到。模型将不可用。")
    tabpfen_available = False
    TabPFNRegressor = None

# --- 1. 全局参数与字体设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 3      # 设置坐标轴框线线宽
plt.rcParams['xtick.major.width'] = 2.5 # 设置X轴主刻度线宽度
plt.rcParams['ytick.major.width'] = 2.5 # 设置Y轴主刻度线宽度
plt.rcParams['xtick.major.size'] = 9   # 设置X轴主刻度线长度
plt.rcParams['ytick.major.size'] = 9  # 设置Y轴主刻度线长度
plt.rcParams['xtick.labelsize'] = 28 # 单独设置X轴刻度标签的字号
plt.rcParams['ytick.labelsize'] = 28  # 单独设置Y轴刻度标签的字号

# =============================================================================
# [!!请在这里配置您的文件信息!!]
DATA_FILE_PATH = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
VERSION_SUFFIX = "v4_tabpfen_dynamic_device"
# =============================================================================

# --- 2. 数据加载与情景定义 ---
def load_and_process_data(file_path):
    """加载数据并对所有列进行初步数值转换。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")
    return df

FIXED_PARAMS = {
    'CEC': 15, 'RM': 2, 'T': 400, 'pH-B': 9.7, 'BET': 3.6, 'BR': 0.03
}

# 只保留 pH=5, Cd=3 的情况
TARGET_CASE = {'name': 'pH-S=5, Cd-C=3', 'params': {'pH-S': 5, 'Cd-C': 3}}

AGING_METHODS = {
    1: 'DW',
    2: 'FT',
    3: 'C(25°C,60%W)',
    4: 'DW + FT'
}


# --- 3. 核心功能函数 ---

def prepare_data(df, target_column_name, all_potential_targets):
    """准备数据：清洗、定义特征、分层抽样并返回数据集。"""
    df_processed = df.copy()
    df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.empty: return None, None, None, None
    
    strata_col_name = 'T_aging'
    cols_to_exclude = list(all_potential_targets) + [strata_col_name]
    
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]; y = df_processed[target_column_name]
    
    strata = None
    if strata_col_name in df_processed.columns and all(df_processed[strata_col_name].value_counts() >= 2):
        strata = df_processed[strata_col_name]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strata)
    return X_train, X_test, y_train, y_test

def train_tabpfn_model(X_train, y_train, X_test, y_test, target_column_name):
    """使用 TabPFN 模型进行训练。"""
    version_suffix = f"{target_column_name}_{VERSION_SUFFIX}"
    print(f"\n--- 为目标 '{target_column_name}' 训练 TabPFN (人工老化) ---")
    
    n_samples = len(X_train)
    n_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
    preprocessor = KNNImputer(n_neighbors=n_neighbors)
    if not tabpfen_available: print("错误: TabPFNRegressor 不可用。"); return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = TabPFNRegressor(device=device)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    
    try:
        print(f"开始在 {device} 上拟合...")
        full_pipeline.fit(X_train, y_train)
        print("模型拟合完成！")
        r2_test = full_pipeline.score(X_test, y_test)
        print(f"TabPFN - 测试集 R²: {r2_test:.4f}")
        
        model_save_dir = './saved_models'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'pipeline_{version_suffix}.joblib')
        joblib.dump(full_pipeline, model_save_path)
        print(f"已保存模型至: {model_save_path}")
        return True
    except Exception as e:
        print(f"模型训练流程失败: {e}"); traceback.print_exc()
        return False

def run_single_case_analysis_with_ci(target_column_name, X_train, y_train, n_bootstrap=500):
    version_suffix = f"{target_column_name}_{VERSION_SUFFIX}"
    print(f"\n--- 为 '{target_column_name}' 进行 pH=5, Cd=3 的四种老化方式分析 ---")
    
    # 用于标题显示的映射
    column_name_map = {
        'Cd-CH': 'ΔCd',
        'pH-S-CH': 'ΔpH'
    }
    
    # 专门为y轴标签设置的映射
    ylabel_map = {
        'Cd-CH': 'ΔCd(%)',
        'pH-S-CH': 'ΔpH'
    }
    
    display_name = column_name_map.get(target_column_name, target_column_name)
    ylabel_display = ylabel_map.get(target_column_name, target_column_name)

    model_filename = f'./saved_models/pipeline_{version_suffix}.joblib'
    if not os.path.exists(model_filename):
        print(f"ERROR: Model file not found at '{model_filename}'.")
        return
    full_pipeline = joblib.load(model_filename)
    
    print("正在计算模型残差...")
    train_predictions = full_pipeline.predict(X_train)
    residuals = y_train.values - train_predictions
        
    output_parent_dir = f'Analysis_Results_Single_Case_for_{target_column_name}'
    os.makedirs(output_parent_dir, exist_ok=True)

    aging_day_limits = {1: 60, 2: 60, 3: 365, 4: 120}
    
    # 生成预测数据
    all_results_list = []
    print(f"正在生成 {TARGET_CASE['name']} 的预测数据...")
    
    for t_aging_code, aging_name in AGING_METHODS.items():
        if t_aging_code == 3: 
            step = 10
        else: 
            step = 5
        day_limit = aging_day_limits[t_aging_code]
        days_range = np.arange(0, day_limit + 1, step)
        virtual_samples_features = []
        virtual_samples_info = []
        
        for day in days_range:
            sample = {feat: 0 for feat in X_train.columns}
            sample.update(FIXED_PARAMS); 
            sample.update(TARGET_CASE['params'])
            sample['T_aging'] = t_aging_code
            if t_aging_code == 1: 
                sample['DW-D'] = day
            elif t_aging_code == 2: 
                sample['FT-D'] = day
            elif t_aging_code == 3: 
                sample['C-D'], sample['C-T'], sample['C-W'] = day, 25, 0.6
            elif t_aging_code == 4: 
                sample['DW-D'], sample['FT-D'] = day, day
            virtual_samples_features.append(sample)
            virtual_samples_info.append({'case': TARGET_CASE['name'], 'aging_method': aging_name, 'T_aging': t_aging_code, 'days': day})
        
        df_virtual = pd.DataFrame(virtual_samples_features)[X_train.columns]
        df_info = pd.DataFrame(virtual_samples_info)
        point_predictions = full_pipeline.predict(df_virtual)
        bootstrap_residuals = np.random.choice(residuals, size=(len(df_virtual), n_bootstrap), replace=True)
        bootstrap_predictions = point_predictions[:, np.newaxis] + bootstrap_residuals
        
        df_preds_wide = pd.DataFrame(bootstrap_predictions, index=df_info.index)
        df_combined_wide = pd.concat([df_info, df_preds_wide], axis=1)
        results_df_long = df_combined_wide.melt(id_vars=['case', 'aging_method', 'T_aging', 'days'], var_name='bootstrap_run', value_name='prediction')
        all_results_list.append(results_df_long)

    # 合并所有结果
    results_df = pd.concat(all_results_list, ignore_index=True)
    results_df = results_df.sort_values(by=['aging_method', 'bootstrap_run', 'days'])
    
    csv_filename = os.path.join(output_parent_dir, f'predicted_data_single_case_with_ci.csv')
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"已保存详细数值到: {csv_filename}")

    # --- 绘制单个图表，包含四个子图 ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 18), sharex=False, sharey=True)
    
    # 设置总标题
    fig.suptitle(f'{display_name} pH=5, Cd=3\nFour Aging Methods with 95% CI', 
                 fontweight='bold', fontsize=30, y=0.90)  # y参数控制垂直位置，0.95表示距离顶部5%
    
    prediction_line_color = 'darkorange'
    prediction_ci_color = 'salmon'
    
    # 定义显示顺序：第一行 DW, FT；第二行 C, DW+FT
    display_order = ['DW', 'FT', 'C(25°C,60%W)', 'DW + FT']
    
    for idx, aging_name in enumerate(display_order):
        aging_group = results_df[results_df['aging_method'] == aging_name]
        ax = axes[idx // 2, idx % 2]  # 根据索引计算行列位置
        
        # 计算该老化方式对应的天数范围
        t_aging_code_current = aging_group['T_aging'].iloc[0]
        day_limit_current = aging_day_limits[t_aging_code_current]
        integer_ticks = np.linspace(0, day_limit_current, 12, dtype=int)
        
        # 绘制置信区间带
        sns.lineplot(data=aging_group, x='days', y='prediction', ax=ax, ci=95, 
                    err_style="band", n_boot=1000, color=prediction_ci_color, 
                    linewidth=0, legend=False)
        
        # 绘制预测线
        sns.lineplot(data=aging_group, x='days', y='prediction', ax=ax, 
                    marker='o', markersize=8, color=prediction_line_color, 
                    linewidth=3, ci=None)
        
        # 根据目标变量类型设置y轴格式化
        if target_column_name == 'Cd-CH':
            # 对于Cd数据，乘以100显示为百分数
            def percentage_formatter(x, pos):
                return f'{x*100:.1f}'
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        # 对于pH-S-CH数据，不进行格式化，保持原始数值
        
        ax.set_title(f'{aging_name}', fontsize=28, fontweight='bold')
        ax.grid(True, linestyle='--', linewidth=0.5)
        
        # 添加子图标签
        label_char = chr(97 + idx)  # a, b, c, d
        ax.text(0.99, 0.98, f'({label_char})', transform=ax.transAxes, 
                fontsize=30, fontweight='bold', ha='right', va='top', color='black')
        
        # 设置x轴刻度
        ax.set_xticks(integer_ticks)
        ax.set_xlabel('Aging Days', fontsize=30, fontweight='bold')
        
        # 只有左侧两个子图显示y轴标签，使用对应的标签
        if idx % 2 == 0:
            ax.set_ylabel(ylabel_display, fontsize=30, fontweight='bold')
        else:
            ax.tick_params(labelleft=False)
        
        # 添加图例
        legend_elements = [
            Line2D([0], [0], color=prediction_line_color, lw=3, marker='o', 
                   markersize=8, label='Trendline'),
            Patch(facecolor=prediction_ci_color, alpha=0.3, label='95% CI')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=26,
                 bbox_to_anchor=(1.01, 0.25), frameon=False)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    # 保存图片
    safe_filename = f'{display_name}_Four_Aging_Methods_pH5_Cd3_with_95pct_CI.pdf'
    safe_filename = safe_filename.replace('Δ', 'Delta')  # 处理希腊字母
    output_filename = os.path.join(output_parent_dir, safe_filename)
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存预测图: {output_filename}")
    
    # --- 计算复合老化相对于其他老化方式的提升百分比 ---
    print(f"\n--- {display_name} 复合老化效果分析（60天节点）---")
    
    # 获取每种老化方式在60天节点的平均预测值
    target_day = 60
    final_predictions = {}
    for aging_name in display_order:
        aging_data = results_df[results_df['aging_method'] == aging_name]
        # 获取60天节点的数据
        target_day_data = aging_data[aging_data['days'] == target_day]
        if not target_day_data.empty:
            # 计算该老化方式在60天节点的平均预测值
            final_predictions[aging_name] = target_day_data['prediction'].mean()
        else:
            print(f"警告：{aging_name} 在第{target_day}天没有数据")
            final_predictions[aging_name] = None
    
    # 获取复合老化(DW + FT)的效果值
    composite_effect = final_predictions.get('DW + FT')
    
    if composite_effect is None:
        print("错误：复合老化在60天节点没有数据，无法计算提升百分比")
        return
    
    # 计算相对于其他老化方式的提升百分比
    improvements = {}
    for aging_name in ['FT', 'DW', 'C(25°C,60%W)']:
        other_effect = final_predictions.get(aging_name)
        if other_effect is not None:
            # 计算提升百分比：(复合老化效果 - 其他老化效果) / |其他老化效果| * 100
            if abs(other_effect) > 1e-10:  # 避免除零错误
                improvement_pct = ((composite_effect - other_effect) / abs(other_effect)) * 100
            else:
                improvement_pct = 0
            improvements[aging_name] = improvement_pct
        else:
            improvements[aging_name] = None
    
    # 根据目标变量类型调整描述
    if target_column_name == 'Cd-CH':
        effect_description = "Cd固定率"
        unit = "mg/kg"
    else:  # pH-S-CH
        effect_description = "pH缓冲效果"
        unit = "pH单位"
    
    # 输出结论
    print(f"\n=== 结论分析（60天节点）===")
    print(f"在土壤pH为5，初始Cd浓度为3.0 mg/kg时，复合老化条件下生物炭的{effect_description}（60天）分别较：")
    if improvements['FT'] is not None:
        print(f"- 冻融循环提升 {improvements['FT']:.1f}%")
    if improvements['DW'] is not None:
        print(f"- 干湿循环提升 {improvements['DW']:.1f}%")
    if improvements['C(25°C,60%W)'] is not None:
        print(f"- 恒温恒湿提升 {improvements['C(25°C,60%W)']:.1f}%")
    
    # 保存结论到文件
    conclusion_filename = os.path.join(output_parent_dir, f'conclusion_analysis_{target_column_name}_60days.txt')
    with open(conclusion_filename, 'w', encoding='utf-8') as f:
        f.write(f"=== {display_name} 复合老化效果分析（60天节点）===\n")
        f.write(f"分析条件：土壤pH=5，初始Cd浓度=3.0 mg/kg\n")
        f.write(f"分析时间点：第60天\n\n")
        f.write(f"各老化方式在60天的效果值：\n")
        for aging_name, value in final_predictions.items():
            if value is not None:
                f.write(f"- {aging_name}: {value:.4f} {unit}\n")
            else:
                f.write(f"- {aging_name}: 数据缺失\n")
        f.write(f"\n复合老化相对提升（60天节点）：\n")
        f.write(f"在土壤pH为5，初始Cd浓度为3.0 mg/kg时，复合老化条件下生物炭的{effect_description}（60天）分别较：\n")
        if improvements['FT'] is not None:
            f.write(f"- 冻融循环提升 {improvements['FT']:.1f}%\n")
        if improvements['DW'] is not None:
            f.write(f"- 干湿循环提升 {improvements['DW']:.1f}%\n")
        if improvements['C(25°C,60%W)'] is not None:
            f.write(f"- 恒温恒湿提升 {improvements['C(25°C,60%W)']:.1f}%\n")
    
    print(f"已保存结论分析到: {conclusion_filename}")


# --- 4. 主程序执行 ---
if __name__ == "__main__":
    df_original = load_and_process_data(DATA_FILE_PATH)
    if df_original is not None:
        # 只分析两个目标变量
        targets_to_analyze = ['Cd-CH', 'pH-S-CH']
        for target in targets_to_analyze:
            print(f"\n{'='*40}\n   开始处理目标: {target} (pH=5, Cd=3)\n{'='*40}")
            
            X_train, X_test, y_train, y_test = prepare_data(df_original.copy(), target, targets_to_analyze)
            if X_train is None:
                print(f"数据准备失败，跳过目标 '{target}'。")
                continue
            
            version_suffix = f"{target}_{VERSION_SUFFIX}"
            model_path = f'./saved_models/pipeline_{version_suffix}.joblib'
            
            if not os.path.exists(model_path):
                print(f"\n模型 '{model_path}' 未找到，开始训练...")
                training_success = train_tabpfn_model(X_train, y_train, X_test, y_test, target)
                if not training_success:
                    print(f"模型训练失败，跳过对目标 '{target}' 的分析。")
                    continue
            else:
                print(f"\n模型 '{model_path}' 已存在，将直接用于分析。")

            run_single_case_analysis_with_ci(target, X_train, y_train) 
        
        print(f"\n{'='*10} 所有分析任务已全部完成 {'='*10}")
    else:
        print("数据加载失败，无法执行任何操作。")