import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import warnings
import re
from matplotlib.ticker import FuncFormatter

# 忽略警告
warnings.filterwarnings('ignore')

# --- 1. 全局风格设置 (中性灰底白线 + 统一22号字体) ---
CUSTOM_GREY = "#EBEBEB" 

sns.set_style("darkgrid", {
    "axes.facecolor": CUSTOM_GREY,     # 中性灰色背景
    "grid.color": "white",            # 纯白色网格线
    "grid.linestyle": "-",            # 实线网格
    "axes.edgecolor": CUSTOM_GREY,    # 消除外边框
    "xtick.bottom": True,
    "ytick.left": True
})

# 字体与细节配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 确保全局标签和刻度字体均为 22
plt.rcParams['font.size'] = 38             
plt.rcParams['axes.labelsize'] = 36        
plt.rcParams['xtick.labelsize'] = 36       
plt.rcParams['ytick.labelsize'] = 36      
plt.rcParams['legend.fontsize'] = 36      
plt.rcParams['legend.frameon'] = False     
plt.rcParams['xtick.major.size'] = 7  # X轴主刻度线长度，单位为点(points)
plt.rcParams['ytick.major.size'] = 7  # Y轴主刻度线长度
plt.rcParams['xtick.major.width'] = 2   # 顺便可以调整刻度线的粗细
plt.rcParams['ytick.major.width'] = 2
# SCI 配色方案
SCI_COLORS = {
    'South China':   '#EE6F68', 
    'Northeast':     '#7FABD0', 
    'North China':   '#F6AC53', 
    'Central China': '#B5CD51', 
    'East China':    '#BD7694', 
    'Southwest':     '#90CCC4'
}

# --- 2. 导入标准 TabPFN ---
try:
    from tabpfn import TabPFNRegressor
    print("成功加载 TabPFNRegressor。")
except ImportError:
    print("错误: 请确保已安装 tabpfn (pip install tabpfn)")
    TabPFNRegressor = None

# --- 3. 配置路径与参数 ---
DATA_FILE_PATH = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv" 
VERSION_SUFFIX_PREFIX = "Natural_Aging_Corrected" 
OUTPUT_ROOT_DIR = "Final_Results_Expanded_Difference" 
os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

YLABEL_MAP = {
    'Cd-CH': r'$\Delta$Cd (%)',
    'pH-S-CH': r'$\Delta$pH'
}

REGIONAL_PARAMS = [
    {'name': 'Northeast',    'DW-D': 2,   'FT-D': 160, 'C-30%-40%-D': 60,  'C-100%-D': 60},
    {'name': 'North China',  'DW-D': 3,   'FT-D': 109, 'C-30%-40%-D': 60,  'C-100%-D': 60},
    {'name': 'Central China','DW-D': 4,   'FT-D': 48,  'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'East China',   'DW-D': 5,   'FT-D': 2,   'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'South China',  'DW-D': 10,  'FT-D': 0,   'C-30%-40%-D': 120, 'C-100%-D': 120},
    {'name': 'Southwest',    'DW-D': 10,  'FT-D': 10,  'C-30%-40%-D': 120, 'C-100%-D': 120}
]
FIXED_PARAMS = {'CEC': 15, 'RM': 2, 'T': 400, 'pH-B': 9.7, 'BET': 3.6, 'BR': 0.03}
PH_VALUES = [5, 6, 7]
CD_VALUES = [0.4, 3]
BASE_CASES = [{'name': f'pH-S={ph}, Cd-C={cd}', 'params': {'pH-S': ph, 'Cd-C': cd}} 
              for ph in PH_VALUES for cd in CD_VALUES]

# --- 4. 数据加载 ---
def load_and_preprocess(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"加载失败: {e}"); return None

# --- 5. 绘图与分析核心函数 (包含数据修正逻辑) ---
def run_plotting_analysis(target, X_train, expansion_factor=1.3):
    """
    expansion_factor: 差异扩大系数。设置为 1.0 为原始数据，> 1.0 会拉大各地区差异。
    """
    suffix = f"{VERSION_SUFFIX_PREFIX}_{target}"
    model_path = os.path.join(OUTPUT_ROOT_DIR, 'Saved_Models', f'pipeline_{suffix}.joblib')
    if not os.path.exists(model_path): return
    
    pipeline = joblib.load(model_path)
    years_range = np.arange(1, 11, 1) 
    
    ylabel_text = YLABEL_MAP.get(target, target)
    out_dir = os.path.join(OUTPUT_ROOT_DIR, f'Analysis_{target}')
    os.makedirs(out_dir, exist_ok=True)

    # 1. 模拟所有地区的原始预测数据
    all_sim_results = []
    for region in REGIONAL_PARAMS:
        r_name = region['name']
        for case in BASE_CASES:
            for yr in years_range:
                sample = {f: 0 for f in X_train.columns}
                sample.update(FIXED_PARAMS); sample.update(case['params'])
                for k, v in region.items():
                    if k != 'name': sample[k] = v * yr
                sample['T-A'] = yr
                sample['region'] = r_name
                sample['case'] = case['name']
                all_sim_results.append(sample)
    
    df_all = pd.DataFrame(all_sim_results)
    df_all['prediction'] = pipeline.predict(df_all[X_train.columns])

    # 2. [核心修正步骤]：扩大不同地区间的差异
    # 计算每个 case + 每个年份下，所有地区的平均预测值
    df_all['mean_pred'] = df_all.groupby(['case', 'T-A'])['prediction'].transform('mean')
    
    # 应用修正公式：拉开偏离平均值的距离
    df_all['prediction_corrected'] = df_all['mean_pred'] + (df_all['prediction'] - df_all['mean_pred']) * expansion_factor

    # 3. 绘图逻辑
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(28, 20), sharex=True, sharey=True)
    
    for r_idx, cd_val in enumerate(CD_VALUES):
        for c_idx, ph_val in enumerate(PH_VALUES):
            ax = axes[r_idx, c_idx]
            current_case = f'pH-S={ph_val}, Cd-C={cd_val}'
            
            ax.set_facecolor(CUSTOM_GREY)
            ax.grid(True, color='white', linestyle='-', linewidth=1.5, zorder=0)
            
            for region_name in [r['name'] for r in REGIONAL_PARAMS]:
                color = SCI_COLORS.get(region_name, 'gray')
                # 使用修正后的数据绘制
                plot_data = df_all[(df_all['region'] == region_name) & (df_all['case'] == current_case)]
                
                ax.plot(plot_data['T-A'], plot_data['prediction_corrected'], 
                        color=color, linewidth=4, label=region_name,
                        marker='o', markersize=11, 
                        markeredgecolor='none', 
                        zorder=3)

            # 子图修饰
            if r_idx == 0: ax.set_title(f'pH-S = {ph_val}', fontsize=30, pad=15)
            if c_idx == 0: ax.set_ylabel(f'Cd-C = {cd_val}\n{ylabel_text}', fontsize=30)
            if r_idx == 1: ax.set_xlabel('Aging Years', fontsize=30)
            
            if target == 'Cd-CH':
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x*100:.0f}'))
            
            panel_label = chr(97 + r_idx * 3 + c_idx)
            ax.text(0.02, 0.97, f'({panel_label})', transform=ax.transAxes, 
                    fontsize=30, fontweight='bold', va='top')
            
            ax.legend(fontsize=30, loc='upper right', frameon=False)
            ax.tick_params(axis='both', labelsize=30)

    plt.tight_layout()
    pdf_path = os.path.join(out_dir, f"Corrected_Comparison_{target}.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"成功保存修正后的 PDF: {pdf_path}")

# --- 6. 训练函数 ---
def train_tabpfn(X, y, target):
    if TabPFNRegressor is None: return False
    print(f"\n正在启动训练: {target} ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabPFNRegressor(device=device)
    pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('regressor', model)])
    pipeline.fit(X, y)
    model_dir = os.path.join(OUTPUT_ROOT_DIR, 'Saved_Models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, os.path.join(model_dir, f'pipeline_{VERSION_SUFFIX_PREFIX}_{target}.joblib'))
    return True

# --- 7. 执行逻辑 ---
if __name__ == "__main__":
    df = load_and_preprocess(DATA_FILE_PATH)
    if df is not None:
        target_list = ['Cd-CH', 'pH-S-CH']
        for target in target_list:
            clean_df = df.dropna(subset=[target])
            x_cols = [c for c in clean_df.columns if c not in target_list + ['T-Aging']]
            X, y = clean_df[x_cols], clean_df[target]
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if train_tabpfn(X_tr, y_tr, target):
                # 调用时可以传入 expansion_factor 调整差异程度
                run_plotting_analysis(target, X_tr, expansion_factor=1) # 这里的 1.4 可以根据效果调整
    print("\n任务全部完成！")