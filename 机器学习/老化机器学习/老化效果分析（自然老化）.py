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
import re
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore') # 忽略警告

# --- 1. 导入 TabPFN 相关模块 (核心修改部分) ---
# 尝试导入标准 TabPFNRegressor 作为保底
try:
    from tabpfn import TabPFNRegressor
    print("成功导入标准 TabPFNRegressor。")
except ImportError:
    print("严重错误: 未找到 'tabpfn' 库。请 pip install tabpfn")
    TabPFNRegressor = None

# 尝试导入 AutoTabPFN (扩展版)
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    print("成功导入 AutoTabPFNRegressor (扩展版)。")
    autotabpfn_available = True
except ImportError:
    print("提示: 'tabpfn_extensions' 未找到，将默认使用标准 TabPFNRegressor。")
    autotabpfn_available = False
    AutoTabPFNRegressor = None

# --- 2. 全局参数与字体设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 32
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

# =============================================================================
# [!!请在这里配置您的文件信息!!]
DATA_FILE_PATH = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv" 
VERSION_SUFFIX_PREFIX = "Natural_Aging_AutoTabPFN" 

# [!!核心!!] 定义输出列的名称映射
TARGET_NAME_MAP = {
    'Cd-CH': 'ΔCd',
    'pH-S-CH': 'ΔpH'
}

# 专门为y轴标签设置的映射
YLABEL_MAP = {
    'Cd-CH': 'ΔCd(%)',
    'pH-S-CH': 'ΔpH'
}

# 全局变量存储区域分析结果
GLOBAL_REGIONAL_RESULTS = {}
# =============================================================================

# --- 3. 辅助函数、数据加载与情景定义 ---
def sanitize_filename(text):
    text = text.replace('\n', ' ').replace(':', '')
    text = re.sub(r'[^a-zA-Z0-9_()\-. ]', '', text)
    text = text.replace(' ', '_').replace('"', '')
    return text

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

REGIONAL_PARAMS = [
    {'name': 'Northeast',    'DW-D': 2,   'FT-D': 160, 'C-30%-40%-D': 60,  'C-100%-D': 60},
    {'name': 'North China',  'DW-D': 3,   'FT-D': 109, 'C-30%-40%-D': 60,  'C-100%-D': 60},
    {'name': 'Central China','DW-D': 4,   'FT-D': 48,  'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'East China',   'DW-D': 5,   'FT-D': 2,   'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'South China',  'DW-D': 10,  'FT-D': 0,   'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'Southwest',    'DW-D': 10,  'FT-D': 10,  'C-30%-40%-D': 120, 'C-100%-D': 120}
]
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

def prepare_data(df, target_column_name, all_potential_targets):
    df_processed = df.copy()
    df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.empty: return None, None, None, None
    strata_col_name = 'T-Aging'
    cols_to_exclude = list(all_potential_targets) + [strata_col_name]
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]; y = df_processed[target_column_name]
    strata = None
    if strata_col_name in df_processed.columns and all(df_processed[strata_col_name].value_counts() >= 2):
        strata = df_processed[strata_col_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=strata)
    return X_train, X_test, y_train, y_test

# --- 4. 核心训练函数 (已修改逻辑) ---

def train_autotabpfn_model(X_train, y_train, X_test, y_test, target_column_name):
    """
    尝试使用 AutoTabPFN 进行训练。
    如果 AutoTabPFN 缺失或训练报错，则自动降级使用标准的 TabPFNRegressor。
    """
    version_suffix = f"{VERSION_SUFFIX_PREFIX}_{target_column_name}"
    print(f"\n--- 为目标 '{target_column_name}' 准备训练 ---")
    
    n_samples = len(X_train)
    n_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
    preprocessor = KNNImputer(n_neighbors=n_neighbors)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 标志位：是否需要使用标准模型
    use_standard_model = False
    model_name_log = ""

    # ================= 尝试 1: AutoTabPFN (优先) =================
    if autotabpfn_available and AutoTabPFNRegressor is not None:
        try:
            print(f"尝试使用 AutoTabPFNRegressor (设备: {device})...")
            max_training_time = 300
            # 初始化 AutoTabPFN
            model_instance = AutoTabPFNRegressor(max_time=max_training_time, device=device)
            full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
            
            # 尝试训练
            full_pipeline.fit(X_train, y_train)
            print("AutoTabPFN 训练成功！")
            model_name_log = "AutoTabPFN"
            
        except Exception as e:
            print(f"警告: AutoTabPFN 训练过程中发生错误: {e}")
            print(">>> 正在切换到标准 TabPFNRegressor 作为备选方案 <<<")
            use_standard_model = True
    else:
        print("AutoTabPFN 未安装，直接使用标准 TabPFNRegressor。")
        use_standard_model = True

    # ================= 尝试 2: Standard TabPFN (备选) =================
    if use_standard_model:
        if TabPFNRegressor is None:
            print("错误: 无法使用标准模型，因为 tabpfn 库未正确导入。")
            return False
            
        try:
            print(f"开始使用 Standard TabPFNRegressor (设备: {device})...")
            # 初始化标准 TabPFN (注意：标准版通常不需要 max_time 参数)
            # 这里的参数根据你的 tabpfn 版本可能略有不同，v0.1.x 和 v2.x 都能接受 device
            model_instance = TabPFNRegressor(device=device) 
            full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
            
            # 尝试训练
            full_pipeline.fit(X_train, y_train)
            print("Standard TabPFN 训练成功！")
            model_name_log = "StandardTabPFN"
            
        except Exception as e:
            print(f"错误: 标准 TabPFN 训练也失败了: {e}")
            traceback.print_exc()
            return False

    # ================= 评估与保存 =================
    try:
        r2_test = full_pipeline.score(X_test, y_test)
        print(f"[{model_name_log}] - 测试集 R²: {r2_test:.4f}")
        
        model_save_dir = './saved_models'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'pipeline_{version_suffix}.joblib')
        joblib.dump(full_pipeline, model_save_path)
        print(f"已保存模型至: {model_save_path}")
        return True
    except Exception as e:
        print(f"保存或评估模型时出错: {e}")
        return False

# --- 5. 区域分析函数 (保持之前修复后的状态) ---
def run_regional_analysis_with_ci(target_column_name, X_train, y_train, n_bootstrap=500):
    global GLOBAL_REGIONAL_RESULTS
    version_suffix = f"{VERSION_SUFFIX_PREFIX}_{target_column_name}"
    print(f"\n--- 为 '{target_column_name}' 进行带置信区间的区域分析 ---")
    
    model_filename = f'./saved_models/pipeline_{version_suffix}.joblib'
    if not os.path.exists(model_filename):
        print(f"ERROR: Model file not found at '{model_filename}'.")
        return
        
    full_pipeline = joblib.load(model_filename)
    
    print("正在计算模型残差...")
    train_predictions = full_pipeline.predict(X_train)
    residuals = y_train.values - train_predictions
    
    max_year = 10; step = 1
    t_a_range = np.arange(1, max_year + step, step)
    
    pretty_target_name = TARGET_NAME_MAP.get(target_column_name, target_column_name)
    ylabel_display = YLABEL_MAP.get(target_column_name, target_column_name)
    output_parent_dir = f'Analysis_Results_Regional_CI_for_{pretty_target_name}'
    os.makedirs(output_parent_dir, exist_ok=True)

    all_regional_results = {}

    for region_data in REGIONAL_PARAMS:
        region_name = region_data['name']
        print(f"--- 正在处理区域: {region_name} ---")
        
        virtual_samples_info = []
        virtual_samples_features = []
        for case in BASE_CASES:
            for t_a_value in t_a_range:
                sample = {feat: 0 for feat in X_train.columns}
                sample.update(FIXED_PARAMS); sample.update(case['params'])
                for param_key, yearly_value in region_data.items():
                    if param_key != 'name': sample[param_key] = yearly_value * t_a_value
                sample['T-A'] = t_a_value
                virtual_samples_features.append(sample)
                virtual_samples_info.append({'case': case['name'], 'T-A': t_a_value})
        
        df_virtual = pd.DataFrame(virtual_samples_features)[X_train.columns]
        df_info = pd.DataFrame(virtual_samples_info)
        
        point_predictions = full_pipeline.predict(df_virtual)
        bootstrap_residuals = np.random.choice(residuals, size=(len(df_virtual), n_bootstrap), replace=True)
        bootstrap_predictions = point_predictions[:, np.newaxis] + bootstrap_residuals

        df_preds_wide = pd.DataFrame(bootstrap_predictions, index=df_info.index)
        df_combined_wide = pd.concat([df_info, df_preds_wide], axis=1)
        results_df = df_combined_wide.melt(id_vars=['case', 'T-A'], var_name='bootstrap_run', value_name='prediction')
        results_df = results_df.sort_values(by=['case', 'bootstrap_run', 'T-A'])
        results_df['change_per_step'] = results_df.groupby(['case', 'bootstrap_run'])['prediction'].pct_change() * 100
        all_regional_results[region_name] = results_df.copy()
        
        csv_filename = os.path.join(output_parent_dir, f'predicted_data_region_{region_name}_with_ci.csv')
        results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

        # --- 绘图 1: 预测值 ---
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 18), sharex=True, sharey=True)
        title_text = f'{region_name} with 95% CI\nPrediction ({pretty_target_name})'
        fig.suptitle(title_text, fontweight='bold', fontsize=30,y=0.95)
        
        prediction_line_color = 'darkorange'; prediction_ci_color = 'salmon'
        
        for row_idx, cd_val in enumerate(CD_VALUES):
            for col_idx, ph_val in enumerate(PH_VALUES):
                ax = axes[row_idx, col_idx]
                case_name = f'pH-S={ph_val}, Cd-C={cd_val}'
                subplot_data = results_df[results_df['case'] == case_name]
                
                sns.lineplot(data=subplot_data, x='T-A', y='prediction', ax=ax, ci=95, err_style="band", n_boot=1000, color=prediction_ci_color, linewidth=0, legend=False)
                sns.lineplot(data=subplot_data, x='T-A', y='prediction', ax=ax, marker='o', markersize=8, color=prediction_line_color, linewidth=3, ci=None)
                
                # Y轴格式化逻辑
                if target_column_name == 'Cd-CH':
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.1f}'))
                else:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.2f}'))
                
                if row_idx == 0: ax.set_title(f'pH = {ph_val}')
                label_char = chr(97 + row_idx * 3 + col_idx)
                ax.text(0.99, 0.06, f'({label_char})', transform=ax.transAxes, fontsize=30, fontweight='bold', ha='right', va='top', color='black')
                
                legend_elements = [Line2D([0], [0], color=prediction_line_color, lw=3, marker='o', markersize=8, label='Trendline'), Patch(facecolor=prediction_ci_color, alpha=0.3, label='95% CI')]
                ax.legend(handles=legend_elements, fontsize=26, bbox_to_anchor=(1.01, 0.99), loc='upper right', frameon=False)
                
                if col_idx == 0: ax.set_ylabel(f'Cd-I = {cd_val}\n{ylabel_display}')
                else: ax.set_ylabel('')
                if row_idx == 1: ax.set_xlabel('T-A')
                else: ax.set_xlabel('')
                ax.tick_params(axis='both', which='major', labelsize=24)
                if col_idx == 0: ax.tick_params(axis='y', which='major', left=True, labelleft=True)
                else: ax.tick_params(axis='y', which='major', left=False, labelleft=False)
                if row_idx == 1: ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)
                else: ax.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
        
        axes[0,0].set_xticks(t_a_range)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        safe_filename = sanitize_filename(title_text) + ".pdf"
        plt.savefig(os.path.join(output_parent_dir, safe_filename), dpi=600, bbox_inches='tight'); plt.close(fig)

        # --- 绘图 2: 变化率 ---
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 18), sharex=True, sharey=True)
        title_text_change = f'{region_name} with 95% CI\nPercentage Change Rate ({pretty_target_name})'
        fig.suptitle(title_text_change, fontweight='bold', fontsize=28,y=0.95)
        
        attenuation_line_color = '#88CFEC'; attenuation_ci_color = '#88CFEC'
        for row_idx, cd_val in enumerate(CD_VALUES):
            for col_idx, ph_val in enumerate(PH_VALUES):
                ax = axes[row_idx, col_idx]
                case_name = f'pH-S={ph_val}, Cd-C={cd_val}'
                plot_data = results_df.dropna(subset=['change_per_step'])
                subplot_data = plot_data[plot_data['case'] == case_name]
                
                sns.lineplot(data=subplot_data, x='T-A', y='change_per_step', ax=ax, ci=95, marker='o', markersize=8, linewidth=3, color=attenuation_line_color)
                if row_idx == 0: ax.set_title(f'pH = {ph_val}')
                ax.axhline(0, color='black', linestyle='--', linewidth=3)
                label_char = chr(97 + row_idx * 3 + col_idx)
                ax.text(0.99, 0.06, f'({label_char})', transform=ax.transAxes, fontsize=30, fontweight='bold', ha='right', va='top', color='black')
                legend_elements = [Line2D([0], [0], color=attenuation_line_color, lw=3, marker='o', markersize=8, label='Trendline'), Patch(facecolor=attenuation_ci_color, alpha=0.3, label='95% CI')]
                ax.legend(handles=legend_elements, fontsize=26, bbox_to_anchor=(1.01, 0.99), loc='upper right', frameon=False)
                if col_idx == 0: ax.set_ylabel(f'Cd-I = {cd_val}\nChange Rate (%)')
                else: ax.set_ylabel('')
                if row_idx == 1: ax.set_xlabel('T-A')
                else: ax.set_xlabel('')
                ax.tick_params(axis='both', which='major', labelsize=24)
                if col_idx == 0: ax.tick_params(axis='y', which='major', left=True, labelleft=True)
                else: ax.tick_params(axis='y', which='major', left=False, labelleft=False)
                if row_idx == 1: ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)
                else: ax.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
        if len(t_a_range) > 1: axes[0,0].set_xticks(t_a_range[1:])
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        safe_filename_change = sanitize_filename(title_text_change) + ".pdf"
        plt.savefig(os.path.join(output_parent_dir, safe_filename_change), dpi=600, bbox_inches='tight'); plt.close(fig)

    GLOBAL_REGIONAL_RESULTS[target_column_name] = all_regional_results

def perform_comprehensive_regional_analysis():
    global GLOBAL_REGIONAL_RESULTS
    if not GLOBAL_REGIONAL_RESULTS: return
    cd_results = GLOBAL_REGIONAL_RESULTS.get('Cd-CH')
    ph_results = GLOBAL_REGIONAL_RESULTS.get('pH-S-CH')
    if cd_results is None or ph_results is None: return
    
    print(f"\n--- 进行综合区域比较分析 ---")
    cd_south = cd_results.get('South China'); cd_northeast = cd_results.get('Northeast')
    ph_south = ph_results.get('South China'); ph_northeast = ph_results.get('Northeast')
    
    if all(data is not None for data in [cd_south, cd_northeast, ph_south, ph_northeast]):
        # 分析逻辑保持不变...
        case_ph5_cd3 = 'pH-S=5, Cd-C=3'; case_ph7_cd3 = 'pH-S=7, Cd-C=3'
        
        cd_south_ph5_cd3 = cd_south[(cd_south['case'] == case_ph5_cd3) & (cd_south['T-A'] == 10)]['prediction'].mean() * 100
        cd_northeast_ph5_cd3 = cd_northeast[(cd_northeast['case'] == case_ph5_cd3) & (cd_northeast['T-A'] == 10)]['prediction'].mean() * 100
        cd_south_ph7_cd3 = cd_south[(cd_south['case'] == case_ph7_cd3) & (cd_south['T-A'] == 10)]['prediction'].mean() * 100
        cd_northeast_ph7_cd3 = cd_northeast[(cd_northeast['case'] == case_ph7_cd3) & (cd_northeast['T-A'] == 10)]['prediction'].mean() * 100
        
        ph_south_ph5_cd3_u = ph_south[(ph_south['case'] == case_ph5_cd3) & (ph_south['T-A'] == 10)]['prediction'].mean()
        ph_northeast_ph5_cd3_u = ph_northeast[(ph_northeast['case'] == case_ph5_cd3) & (ph_northeast['T-A'] == 10)]['prediction'].mean()
        ph_south_ph7_cd3_u = ph_south[(ph_south['case'] == case_ph7_cd3) & (ph_south['T-A'] == 10)]['prediction'].mean()
        ph_northeast_ph7_cd3_u = ph_northeast[(ph_northeast['case'] == case_ph7_cd3) & (ph_northeast['T-A'] == 10)]['prediction'].mean()
        
        analysis_filename = 'Analysis_Results_Comprehensive_Regional_Comparison.txt'
        with open(analysis_filename, 'w', encoding='utf-8') as f:
            f.write(f"=== 综合区域比较分析结果 (第10年) ===\n\n")
            f.write(f"【Cd固定率】\n")
            f.write(f"pH=5, Cd=3 -> 华南: {cd_south_ph5_cd3:.1f}% | 东北: {cd_northeast_ph5_cd3:.1f}%\n")
            f.write(f"pH=7, Cd=3 -> 华南: {cd_south_ph7_cd3:.1f}% | 东北: {cd_northeast_ph7_cd3:.1f}%\n\n")
            f.write(f"【pH提升量】\n")
            f.write(f"pH=5, Cd=3 -> 华南: {ph_south_ph5_cd3_u:.2f} | 东北: {ph_northeast_ph5_cd3_u:.2f}\n")
            f.write(f"pH=7, Cd=3 -> 华南: {ph_south_ph7_cd3_u:.2f} | 东北: {ph_northeast_ph7_cd3_u:.2f}\n")
        print(f"综合分析已保存: {analysis_filename}")

# --- 6. 主程序 ---
if __name__ == "__main__":
    df_original = load_and_process_data(DATA_FILE_PATH)
    if df_original is not None:
        targets_to_analyze = ['Cd-CH', 'pH-S-CH']
        for target in targets_to_analyze:
            print(f"\n{'='*40}\n   开始处理目标: {target}\n{'='*40}")
            
            X_train, X_test, y_train, y_test = prepare_data(df_original.copy(), target, targets_to_analyze)
            if X_train is None:
                print(f"数据准备失败，跳过目标 '{target}'。")
                continue
            
            version_suffix = f"{VERSION_SUFFIX_PREFIX}_{target}"
            model_path = f'./saved_models/pipeline_{version_suffix}.joblib'
            
            if not os.path.exists(model_path):
                print(f"\n模型 '{model_path}' 未找到，开始训练...")
                # 这里会优先尝试 AutoTabPFN，失败则自动切回 Standard TabPFN
                training_success = train_autotabpfn_model(X_train, y_train, X_test, y_test, target)
                if not training_success:
                    print(f"所有模型训练尝试均失败，跳过对目标 '{target}' 的区域分析。")
                    continue
            else:
                 print(f"\n模型 '{model_path}' 已存在，将直接用于分析。")

            run_regional_analysis_with_ci(target, X_train, y_train)
        
        perform_comprehensive_regional_analysis()
        print(f"\n{'='*10} 所有分析任务已全部完成 {'='*10}")
    else:
        print("数据加载失败。")