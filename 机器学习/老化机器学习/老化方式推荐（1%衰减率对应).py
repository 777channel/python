import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from matplotlib.ticker import FuncFormatter

# ★★★ 关键设置 ★★★
matplotlib.use('Agg')

# ==================== 全局字体设置 ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28
plt.rcParams['axes.linewidth'] = 3 
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

def load_and_process_simple_excel(excel_filepath):
    try:
        print(f"📂 读取Excel文件: {excel_filepath}")
        df = pd.read_excel(excel_filepath, sheet_name='衰减率')
        
        if 'Region' not in df.columns:
            return {}
        
        region_names = ['Central China','East China','North China', 'Northeast', 'South China' , 'Southwest']
        processed_data = {}
        
        data_columns = ['Cd_Natural_Decay_Rate', 'Cd_Artificial_Decay_Rate', 
                       'pH_Natural_Decay_Rate', 'pH_Artificial_Decay_Rate']
        
        for region_name in region_names:
            region_mask = df['Region'] == region_name
            region_df = df[region_mask]
            if region_df.empty: continue
            
            region_data = {}
            for col in data_columns:
                if col in df.columns:
                    # 【关键点】：读取时乘以 100
                    col_data = region_df[col].dropna().tolist()
                    col_data = [abs(float(x)) * 100 for x in col_data] 
                    region_data[col] = col_data
                else:
                    region_data[col] = []
            processed_data[region_name] = region_data
        return processed_data
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        return {}

def draw_single_subplot(ax, region_data, region_name, years=None, is_bottom_row=False, is_left_col=False):
    if years is None:
        data_length = 0
        for col in ['Cd_Natural_Decay_Rate', 'pH_Natural_Decay_Rate', 
                    'Cd_Artificial_Decay_Rate', 'pH_Artificial_Decay_Rate']:
            if len(region_data.get(col, [])) > 0:
                data_length = len(region_data[col])
                break
        years = list(range(2, data_length + 2)) if data_length > 0 else list(range(2, 11))

    line_configs = [
        {'color': '#2E8B57', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Natural ΔCd'},
        {'color': '#4682B4', 'marker': 's', 'linestyle': '-', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Natural ΔpH'},
        {'color': '#DC143C', 'marker': '^', 'linestyle': '--', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Artificial ΔCd'},
        {'color': '#FF8C00', 'marker': 'D', 'linestyle': '--', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Artificial ΔpH'}
    ]
    
    data_columns = ['Cd_Natural_Decay_Rate', 'pH_Natural_Decay_Rate', 
                    'Cd_Artificial_Decay_Rate', 'pH_Artificial_Decay_Rate']
    
    # 绘图（使用乘以100后的数据）
    legend_handles = []
    for i, col in enumerate(data_columns):
        data = region_data.get(col, [])
        if len(data) == len(years):
            line = ax.plot(years, data, alpha=0.9, **line_configs[i])
            legend_handles.extend(line)

    # 计算并显示 RMSE（直接基于当前数据量级）
    rmse_texts = []
    cn, ca = region_data.get('Cd_Natural_Decay_Rate', []), region_data.get('Cd_Artificial_Decay_Rate', [])
    if len(cn) == len(ca) == len(years) and len(cn) > 0:
        val = np.sqrt(np.mean((np.array(cn) - np.array(ca))**2))
        rmse_texts.append(f"Cd RMSE={val:.2f}") 
        
    pn, pa = region_data.get('pH_Natural_Decay_Rate', []), region_data.get('pH_Artificial_Decay_Rate', [])
    if len(pn) == len(pa) == len(years) and len(pn) > 0:
        val = np.sqrt(np.mean((np.array(pn) - np.array(pa))**2))
        rmse_texts.append(f"pH RMSE={val:.2f}")

    if rmse_texts:
        ax.text(0.95, 0.95, '\n'.join(rmse_texts), transform=ax.transAxes, 
                fontsize=24, fontweight='bold', va='top', ha='right')

    # ★★★ 修改后的 Y 轴设置 ★★★
    # 保持乘以100后的数值显示，不做除法，仅保留一位小数格式
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1f}'))
    
    if is_left_col:
        ax.set_ylabel('Attenuation Rate', fontweight='bold', fontsize=28)
    if is_bottom_row:
        ax.set_xlabel('Aging Years', fontweight='bold', fontsize=28)
    
    ax.set_xticks(years)
    ax.set_xticklabels([f'{int(y)}' for y in years], fontweight='bold')
    ax.set_title(region_name, fontweight='bold', pad=15, fontsize=28)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    return legend_handles

def create_2x3_regional_degradation_figure(excel_filepath, output_dir='./'):
    processed_data = load_and_process_simple_excel(excel_filepath)
    if not processed_data: return
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    region_order = ['Central China','East China','North China', 'Northeast', 'South China' , 'Southwest']
    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    all_legend_handles = []
    for idx, region_name in enumerate(region_order):
        row, col = idx // 3, idx % 3
        region_data = processed_data.get(region_name, {})
        if not region_data:
            axes[row, col].set_visible(False)
            continue
        handles = draw_single_subplot(axes[row, col], region_data, region_name,
            is_bottom_row=(row == 1), is_left_col=(col == 0))
        if handles and not all_legend_handles:
            all_legend_handles = handles

    if all_legend_handles:
        fig.legend(handles=all_legend_handles[:4], loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=False,
                  prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 28})

    save_path = os.path.join(output_dir, "Regional_Degradation_Analysis_1%.pdf")
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"✅ 图表已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    excel_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\1%数据（部分）.xlsx"
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图'
    create_2x3_regional_degradation_figure(excel_file, output_directory)