import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import os
import warnings

warnings.filterwarnings('ignore')

# --- 常量 ---
CD_MIN_PLOT = 0.0
CD_MAX_PLOT = 3.0

# --- 最终版绘图函数 ---
def plot_final_style_with_uncertainty(thresholds_data_dict, cd_min_plot, cd_max_plot):
    # ***** 更改点 2：设置字体为 Times New Roman *****
    try:
        # 将字体设置为 Times New Roman，并加粗
        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['axes.labelweight'] = 'bold'
    except Exception as e:
        print(f"设置字体'Times New Roman'失败，请确保您的系统中已安装此字体。错误: {e}")
        # 如果失败，回退到通用无衬线字体
        plt.rcParams['font.family'] = ['sans-serif']

    ph_levels_plot = sorted(thresholds_data_dict.keys())
    n_ph = len(ph_levels_plot)
    
    # 颜色和标签映射
    colors = {
        'S1': '#BEDEAA',
        'S2': '#8BB8FE',
        'S3': '#FE8B7C'
    }
    legend_labels = { 
        'S1': 'S1: DW',
        'S2': 'S2: WF',
        'S3': 'S3: NR'
    }

    # 创建子图
    fig_height = max(n_ph * 0.8 + 1.2, 4.0)
    fig, axes = plt.subplots(n_ph, 1, figsize=(10, fig_height), sharex=True, constrained_layout=False)
    if n_ph == 1: axes = [axes]

    # ***** 更改点 1：调整遍历顺序，让pH=5.0在最上方 *****
    # 移除之前的倒序 [::-1]，现在按从小到大的顺序绘制
    for i, ph_val_plot in enumerate(ph_levels_plot):
        ax = axes[i]
        thresholds = thresholds_data_dict[ph_val_plot]
        
        # --- 数据提取 ---
        t2_data = thresholds.get('DW', {})
        t3_data = thresholds.get('WF', {})
        
        t2_center = t2_data.get('center')
        t3_center = t3_data.get('center')

        # --- 绘制分段颜色 (基于中心值) ---
        current_pos = cd_min_plot
        
        if t2_center is not None:
            end = min(t2_center, cd_max_plot)
            if end > current_pos: ax.axvspan(current_pos, end, facecolor=colors['S1'], alpha=0.9)
            current_pos = end
        
        start_s2 = t2_center if t2_center is not None else cd_min_plot
        if t3_center is not None:
            end = min(t3_center, cd_max_plot)
            if end > start_s2: ax.axvspan(start_s2, end, facecolor=colors['S2'], alpha=0.9)
            current_pos = end

        if current_pos < cd_max_plot:
            ax.axvspan(current_pos, cd_max_plot, facecolor=colors['S3'], alpha=0.9)
        
        # --- 绘制不确定性区间 ---
        if t2_data:
            ax.axvspan(t2_data['lower'], t2_data['upper'], facecolor='dimgray', alpha=0.35, edgecolor=None, zorder=2)
        if t3_data:
            ax.axvspan(t3_data['lower'], t3_data['upper'], facecolor='dimgray', alpha=0.35, edgecolor=None, zorder=2)

        # --- 绘制阈值虚线和数值标签 ---
        text_y_axes_coord = 1.05
        
        if t2_center is not None and cd_min_plot < t2_center < cd_max_plot:
            ax.axvline(x=t2_center, color='black', linestyle='--', linewidth=1.2, ymax=0.95, zorder=3)
            ax.text(t2_center, text_y_axes_coord, f'{t2_center:.2f}',
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=4)

        if t3_center is not None and cd_min_plot < t3_center < cd_max_plot:
            label_too_close = t2_center and abs(t3_center - t2_center) < 0.2 
            ax.axvline(x=t3_center, color='black', linestyle='--', linewidth=1.2, ymax=0.95, zorder=3)
            ax.text(t3_center, text_y_axes_coord, f'{t3_center:.2f}',
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=11, fontweight='bold', zorder=4,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0) if label_too_close else None)
        
        # --- 坐标轴和样式美化 ---
        ax.set_xlim(cd_min_plot, cd_max_plot)
        ax.set_yticks([])
        ax.set_ylabel(f"pH={ph_val_plot:.1f}", rotation=0, labelpad=30, 
                      va='center', ha='right', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', labelsize=12)

        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(1)
        ax.xaxis.set_tick_params(width=1, color='black')

        if i != n_ph - 1:
            ax.tick_params(axis='x', which='both', labelbottom=False)

    axes[-1].set_xlabel("Initial Soil Cd (mg·kg⁻¹)", fontsize=14, fontweight='bold')

    # --- 全局图例 ---
    s_patches = [mpatches.Patch(color=colors[key], label=legend_labels[key]) for key in ['S1', 'S2', 'S3']]
    uncertainty_label = 'Threshold Uncertainty (±10%)'
    uncertainty_patch = mpatches.Patch(facecolor='dimgray', alpha=0.35, label=uncertainty_label)
    all_patches = s_patches + [uncertainty_patch]
    
    fig.legend(handles=all_patches, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=12, frameon=False)
    
    plt.subplots_adjust(hspace=0.6, top=0.85, bottom=0.15, left=0.15, right=0.98)
    
    # 保存文件
    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'strategy_plot.pdf')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n图表已保存到: {plot_filename}")


# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 数据定义 ---
    data_from_image = {
        5.0: {'DW': {'center': '>3'}, 
              'WF': {'lower': 0.846, 'center': 0.940, 'upper': 1.034}},
        5.5: {'DW': {'center': '>3'}, 
              'WF': {'lower': 1.125, 'center': 1.250, 'upper': 1.375}},
        6.0: {'DW': {'center': '>3'},
              'WF': {'lower': 1.485, 'center': 1.680, 'upper': 1.815}},
        6.5: {'DW': {'lower': 0.280, 'center': 0.311, 'upper': 0.342},
              'WF': {'lower': 1.998, 'center': 2.120, 'upper': 2.222}},
        7.0: {'DW': {'lower': 0.851, 'center': 0.946, 'upper': 1.040},
              'WF': {'lower': 2.352, 'center': 2.420, 'upper': 2.561}}
    }

    # 处理数据：将 ">3" 的条目转换为空字典，使其不被绘制
    processed_data = {}
    for ph, values in data_from_image.items():
        processed_data[ph] = {}
        
        dw_bc_data = values.get('DW', {})
        if dw_bc_data.get('center') != '>3':
            processed_data[ph]['DW'] = dw_bc_data
            
        wf_bc_data = values.get('WF', {})
        if wf_bc_data:
            processed_data[ph]['WF'] = wf_bc_data

    # 调用绘图函数
    plot_final_style_with_uncertainty(
        thresholds_data_dict=processed_data, 
        cd_min_plot=CD_MIN_PLOT, 
        cd_max_plot=CD_MAX_PLOT
    )