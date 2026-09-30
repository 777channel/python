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
plt.rcParams['font.size'] = 32
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 3      # 设置坐标轴框线线宽
plt.rcParams['xtick.major.width'] = 2.5 # 设置X轴主刻度线宽度
plt.rcParams['ytick.major.width'] = 2.5 # 设置Y轴主刻度线宽度
plt.rcParams['xtick.major.size'] = 10   # 设置X轴主刻度线长度
plt.rcParams['ytick.major.size'] = 10   # 设置Y轴主刻度线长度
plt.rcParams['xtick.labelsize'] = 30 # 单独设置X轴刻度标签的字号
plt.rcParams['ytick.labelsize'] = 30 # 单独设置Y轴刻度标签的字号

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
PH_VALUES = [5, 6, 7]
CD_VALUES = [0.4, 3]
BASE_CASES = []
for soil_ph in PH_VALUES:
    for initial_cd in CD_VALUES:
        case_name = f'pH-S={soil_ph}, Cd-C={initial_cd}'
        BASE_CASES.append({'name': case_name, 'params': {'pH-S': soil_ph, 'Cd-C': initial_cd}})

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

def run_artificial_aging_analysis_with_ci(target_column_name, X_train, y_train, n_bootstrap=500):
    version_suffix = f"{target_column_name}_{VERSION_SUFFIX}"
    print(f"\n--- 为 '{target_column_name}' 进行带置信区间的区域分析 (人工老化) ---")
    
    # 用于标题显示的映射
    column_name_map = {
        'Cd-CH': 'ΔCd',
        'pH-S-CH': 'ΔpH'
    }
    
    # 专门为预测图y轴标签设置的映射
    prediction_ylabel_map = {
        'Cd-CH': 'ΔCd(%)',
        'pH-S-CH': 'ΔpH'
    }
    
    display_name = column_name_map.get(target_column_name, target_column_name)
    prediction_ylabel_display = prediction_ylabel_map.get(target_column_name, target_column_name)

    model_filename = f'./saved_models/pipeline_{version_suffix}.joblib'
    if not os.path.exists(model_filename):
        print(f"ERROR: Model file not found at '{model_filename}'.")
        return
    full_pipeline = joblib.load(model_filename)
    
    print("正在计算模型残差...")
    train_predictions = full_pipeline.predict(X_train)
    residuals = y_train.values - train_predictions
        
    output_parent_dir = f'Analysis_Results_Artificial_CI_for_{target_column_name}'
    os.makedirs(output_parent_dir, exist_ok=True)

    aging_day_limits = {1: 60, 2: 60, 3: 365, 4: 120}
    
    # 先生成所有区域的完整数据
    all_results_list = []
    print("正在生成所有情景的预测数据...")
    for case in BASE_CASES:
        for t_aging_code, aging_name in AGING_METHODS.items():
            if t_aging_code == 3: step = 10
            else: step = 5
            day_limit = aging_day_limits[t_aging_code]
            days_range = np.arange(0, day_limit + 1, step)
            virtual_samples_features = []
            virtual_samples_info = []
            for day in days_range:
                sample = {feat: 0 for feat in X_train.columns}
                sample.update(FIXED_PARAMS); sample.update(case['params'])
                sample['T_aging'] = t_aging_code
                if t_aging_code == 1: sample['DW-D'] = day
                elif t_aging_code == 2: sample['FT-D'] = day
                elif t_aging_code == 3: sample['C-D'], sample['C-T'], sample['C-W'] = day, 25, 0.6
                elif t_aging_code == 4: sample['DW-D'], sample['FT-D'] = day, day
                virtual_samples_features.append(sample)
                virtual_samples_info.append({'case': case['name'], 'aging_method': aging_name, 'T_aging': t_aging_code, 'days': day})
            
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
    results_df = results_df.sort_values(by=['case', 'aging_method', 'bootstrap_run', 'days'])
    
    # 修改衰减率计算为百分比变化率
    results_df['change_per_step'] = results_df.groupby(['case', 'aging_method', 'bootstrap_run'])['prediction'].pct_change() * 100
    
    csv_filename = os.path.join(output_parent_dir, f'predicted_data_artificial_with_ci.csv')
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"已保存所有人工老化详细数值到: {csv_filename}")

    # --- 循环为每种老化方式绘图 ---
    for aging_name, aging_group in results_df.groupby('aging_method'):
        print(f"--- 正在为老化方式 '{aging_name}' 绘图 ---")
        
        # 计算固定的x轴刻度（12个等间距刻度）
        t_aging_code_current = aging_group['T_aging'].iloc[0]
        day_limit_current = aging_day_limits[t_aging_code_current]
        integer_ticks = np.linspace(0, day_limit_current, 12, dtype=int)
        
        # --- 绘图 1: 预测值 ---
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 18), sharex=True, sharey=True)
        
        # 先定义标题
        prediction_title = f'{aging_name} with 95% CI Prediction ({display_name})'
        # 设置图表标题，为了美观将部分空格替换为换行符
        fig.suptitle(prediction_title.replace(" Prediction ", "\nPrediction "), fontweight='bold',fontsize=30,y=0.95)
        
        prediction_line_color = 'darkorange'
        prediction_ci_color = 'salmon'
        
        for row_idx, cd_val in enumerate(CD_VALUES):
            for col_idx, ph_val in enumerate(PH_VALUES):
                ax = axes[row_idx, col_idx]
                case_name = f'pH-S={ph_val}, Cd-C={cd_val}'
                subplot_data = aging_group[aging_group['case'] == case_name]
                sns.lineplot(data=subplot_data, x='days', y='prediction', ax=ax, ci=95, err_style="band", n_boot=1000, color=prediction_ci_color, linewidth=0, legend=False)
                sns.lineplot(data=subplot_data, x='days', y='prediction', ax=ax, marker='o', markersize=8, color=prediction_line_color, linewidth=3, ci=None)
                
                # 根据目标变量类型设置y轴格式化（仅对预测图）
                if target_column_name == 'Cd-CH':
                    # 对于Cd数据，乘以100显示为百分数
                    def percentage_formatter(x, pos):
                        return f'{x*100:.1f}'
                    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
                # 对于pH-S-CH数据，不进行格式化，保持原始数值
                
                if row_idx == 0: ax.set_title(f'pH = {ph_val}')
                if col_idx == 0: ax.set_ylabel(f'Cd-C = {cd_val}')
                ax.grid(False, linestyle='--', linewidth=0.5)
                label_char = chr(97 + row_idx * 3 + col_idx)
                ax.text(0.99, 0.98, f'({label_char})', transform=ax.transAxes, fontsize=28, fontweight='bold', ha='right', va='top', color='black')
                
                # 控制坐标轴刻度显示：只有第一列显示y轴刻度线和标签，只有第二行显示x轴刻度线和标签
                if col_idx != 0:  # 不是第一列，隐藏y轴刻度线和标签
                    ax.tick_params(labelleft=False, left=False)
                if row_idx != 1:  # 不是第二行，隐藏x轴刻度线和标签
                    ax.tick_params(labelbottom=False, bottom=False)
                
                # 添加图例（每个子图都添加）
                legend_elements = [
                    Line2D([0], [0], color=prediction_line_color, lw=3, marker='o', markersize=8, label='Trendline'),
                    Patch(facecolor=prediction_ci_color, alpha=0.3, label='95% CI')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=26,bbox_to_anchor=(1.01, 0.25),frameon=False)
        
        # 设置固定的x轴刻度（12个刻度）
        axes[0,0].set_xticks(integer_ticks)
        
        # 只为第二行设置x轴标签
        for col_idx in range(3):
            axes[1, col_idx].set_xlabel('Aging Days')
        
        # 为第一列设置对应的y轴标签
        for row_idx in range(2):
            axes[row_idx, 0].set_ylabel(prediction_ylabel_display)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # 基于标题创建一个对文件名安全的版本
        safe_filename_str = prediction_title.replace('"', '').replace(':', '').replace('%', 'pct').replace('°', 'deg')
        safe_filename_str = safe_filename_str.replace('(', '').replace(')', '').replace('+', 'and').replace('/', '-').replace('\\', '-')
        
        # 使用安全的文件名保存图片
        output_filename = os.path.join(output_parent_dir, f'{safe_filename_str}.pdf')
        plt.savefig(output_filename, dpi=600)
        plt.close(fig)
        print(f"   已保存预测图: {output_filename}")

        # --- 绘图 2: 变化量/衰减图（百分比变化率）---
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 18), sharex=True, sharey=True)
        
        # 先定义标题
        attenuation_title = f'{aging_name} with 95% CI Percentage Change Rate ({display_name})'
        # 设置图表标题
        fig.suptitle(attenuation_title.replace(" Percentage Change Rate ", "\nPercentage Change Rate "), fontweight='bold',fontsize=30,y=0.95)
        
        attenuation_line_color = '#88CFEC'
        attenuation_ci_color = '#B8E6FF'
        
        for row_idx, cd_val in enumerate(CD_VALUES):
            for col_idx, ph_val in enumerate(PH_VALUES):
                ax = axes[row_idx, col_idx]
                case_name = f'pH-S={ph_val}, Cd-C={cd_val}'
                plot_data = aging_group.dropna(subset=['change_per_step'])
                subplot_data = plot_data[plot_data['case'] == case_name]
                sns.lineplot(data=subplot_data, x='days', y='change_per_step', ax=ax, ci=95, marker='o', markersize=8, linewidth=3, color=attenuation_line_color)
                
                # 衰减率图：无论CD还是pH都不需要额外格式化，因为已经是百分比了
                # 但如果需要不显示小数，可以添加格式化
                # def percentage_formatter_no_decimal(x, pos):
                #     return f'{x:.0f}'
                # ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter_no_decimal))
                
                if row_idx == 0: ax.set_title(f'Soil pH = {ph_val}')
                if col_idx == 0: ax.set_ylabel(f'Change Rate (%) for Initial Cd = {cd_val}')
                ax.grid(False, linestyle='--', linewidth=0.5)
                ax.axhline(0, color='black', linestyle='--', linewidth=3)
                label_char = chr(97 + row_idx * 3 + col_idx)
                ax.text(0.99, 0.98, f'({label_char})', transform=ax.transAxes, fontsize=30, fontweight='bold', ha='right', va='top', color='black')
                
                # 控制坐标轴刻度显示：只有第一列显示y轴刻度线和标签，只有第二行显示x轴刻度线和标签
                if col_idx != 0:  # 不是第一列，隐藏y轴刻度线和标签
                    ax.tick_params(labelleft=False, left=False)
                if row_idx != 1:  # 不是第二行，隐藏x轴刻度线和标签
                    ax.tick_params(labelbottom=False, bottom=False)
                
                # 添加图例（每个子图都添加）
                legend_elements = [
                    Line2D([0], [0], color=attenuation_line_color, lw=3, marker='o', markersize=8, label='Trendline'),
                    Patch(facecolor=attenuation_ci_color, alpha=0.3, label='95% CI')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=26, bbox_to_anchor=(0.99, 0.25),frameon=False)
        
        # 设置固定的x轴刻度（12个刻度）
        axes[0,0].set_xticks(integer_ticks)
        
        # 只为第二行设置x轴标签
        for col_idx in range(3):
            axes[1, col_idx].set_xlabel('Aging Days')
        
        # 为第一列设置y轴标签为百分比变化率
        for row_idx in range(2):
            axes[row_idx, 0].set_ylabel('Change Rate (%)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # 基于标题创建一个对文件名安全的版本
        safe_filename_str = attenuation_title.replace('"', '').replace(':', '').replace('%', 'pct').replace('°', 'deg')
        safe_filename_str = safe_filename_str.replace('(', '').replace(')', '').replace('+', 'and').replace('/', '-').replace('\\', '-')

        # 使用安全的文件名保存图片
        output_filename = os.path.join(output_parent_dir, f'{safe_filename_str}.png')
        plt.savefig(output_filename, dpi=600)
        plt.close(fig)
        print(f"   已保存百分比变化率图: {output_filename}")


# --- 4. 主程序执行 ---
if __name__ == "__main__":
    df_original = load_and_process_data(DATA_FILE_PATH)
    if df_original is not None:
        # 在这里仍然使用原始列名来从DataFrame中选择数据
        targets_to_analyze = ['Cd-CH', 'pH-S-CH']
        for target in targets_to_analyze:
            print(f"\n{'='*40}\n   开始处理目标: {target} (人工老化)\n{'='*40}")
            
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
                    print(f"模型训练失败，跳过对目标 '{target}' 的区域分析。")
                    continue
            else:
                print(f"\n模型 '{model_path}' 已存在，将直接用于分析。")

            run_artificial_aging_analysis_with_ci(target, X_train, y_train) 
        
        print(f"\n{'='*10} 所有分析任务已全部完成 {'='*10}")
    else:
        print("数据加载失败，无法执行任何操作。")