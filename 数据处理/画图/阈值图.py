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
CD_MAX_PLOT = 3.0  # <-- 已按您的要求，将X轴范围设置为3.0

# --- 最终版绘图函数 (在您的代码风格上进行修改) ---
def plot_final_style_with_uncertainty(thresholds_data_dict, cd_min_plot, cd_max_plot):
    ph_levels_plot = sorted(thresholds_data_dict.keys())
    n_ph = len(ph_levels_plot)
    
    # 颜色和标签映射 (与您代码一致)
    colors = {
        'S1': '#BEDEAA',
        'S2': '#8BB8FE',
        'S3': '#FE8B7C'
    }
    legend_labels = { 
        'S1': 'S1: DW+BC',
        'S2': 'S2: WF+BC',
        'S3': 'S3: NR'
    }

    fig_height = max(n_ph * 0.6 + 1.2, 4.0)
    fig, axes = plt.subplots(n_ph, 1, figsize=(8, fig_height), sharex=True, constrained_layout=False)
    if n_ph == 1: axes = [axes]

    for i, ph_val_plot in enumerate(ph_levels_plot):
        ax = axes[i]
        thresholds = thresholds_data_dict[ph_val_plot]
        
        # --- 数据提取 (适配新的数据结构) ---
        t2_data = thresholds.get('T2', {})
        t3_data = thresholds.get('T3', {})
        t2_center = t2_data.get('center')
        t3_center = t3_data.get('center')

        # --- 绘制分段颜色 (基于中心值) ---
        current_pos = cd_min_plot
        if t2_center is not None:
            end = min(t2_center, cd_max_plot)
            if end > current_pos: ax.axvspan(current_pos, end, facecolor=colors['S1'], alpha=0.85)
            current_pos = end
        if t3_center is not None:
            end = min(t3_center, cd_max_plot)
            if end > current_pos: ax.axvspan(current_pos, end, facecolor=colors['S2'], alpha=0.85)
            current_pos = end
        if current_pos < cd_max_plot:
            ax.axvspan(current_pos, cd_max_plot, facecolor=colors['S3'], alpha=0.85)
        
        # --- 新增: 绘制不确定性区间 ---
        if t2_data:
            ax.axvspan(t2_data['lower'], t2_data['upper'], facecolor='grey', alpha=0.35, edgecolor=None, zorder=2)
        if t3_data:
            ax.axvspan(t3_data['lower'], t3_data['upper'], facecolor='grey', alpha=0.35, edgecolor=None, zorder=2)

        # --- 绘制阈值虚线和数值标签 (保持您代码的原有风格) ---
        text_y_axes_coord = 1.05
        
        if t2_center is not None and cd_min_plot < t2_center < cd_max_plot:
            ax.axvline(x=t2_center, color='grey', linestyle='--', linewidth=0.8, ymax=0.95, zorder=3)
            ax.text(t2_center, text_y_axes_coord, f'{t2_center:.2f}',
                    transform=ax.get_xaxis_transform(), 
                    ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=4)

        if t3_center is not None and cd_min_plot < t3_center < cd_max_plot:
             if not (t2_center and np.isclose(t3_center, t2_center)):
                ax.axvline(x=t3_center, color='grey', linestyle='--', linewidth=0.8, ymax=0.95, zorder=3)
                ax.text(t3_center, text_y_axes_coord, f'{t3_center:.2f}',
                        transform=ax.get_xaxis_transform(),
                        ha='center', va='bottom', fontsize=10, fontweight='bold', zorder=4)
        
        # --- 坐标轴和样式美化 (保持您代码的原有风格) ---
        ax.set_xlim(cd_min_plot, cd_max_plot)
        ax.set_yticks([])
        ax.set_ylabel(f"pH={ph_val_plot:.1f}", rotation=0, labelpad=20, 
                      va='center', ha='right', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', labelsize=11)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(0.6)
        ax.xaxis.set_tick_params(width=0.6, color='black')

        if i == n_ph - 1:
            ax.set_xlabel("Initial Soil Cd (mg·kg$^{-1}$)", fontsize=12, fontweight='bold')
        else:
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    # --- 全局图例 (新增灰色区域的图例项) ---
    s_patches = [mpatches.Patch(color=colors[key], label=legend_labels[key]) for key in ['S1', 'S2', 'S3']]
    uncertainty_label = 'Threshold Uncertainty (±10%)'
    uncertainty_patch = mpatches.Patch(facecolor='grey', alpha=0.35, label=uncertainty_label)
    all_patches = s_patches + [uncertainty_patch]
    
    fig.legend(handles=all_patches, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=4, fontsize=9, frameon=False)
    
    plt.subplots_adjust(hspace=0.5, top=0.88, bottom=0.15, left=0.12, right=0.98)
    
    output_dir = 'output_final_style'
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'strategy_final_style.pdf')
    plt.savefig(plot_filename, dpi=600, bbox_inches='tight')
    print(f"\n已按您的最终风格要求完成的图表已保存到: {plot_filename}")
    plt.close(fig)

# --- 主程序 ---
if __name__ == "__main__":
    
    # --- 数据更新: 采用新的数据结构, center值来自您的图片 ---
    threshold_data_with_uncertainty = {
        5.0: {'T2': {'center': 1.00, 'lower': 0.94, 'upper': 1.07}, 
              'T3': {'center': 1.25, 'lower': 1.18, 'upper': 1.33}},
        5.5: {'T2': {'center': 1.30, 'lower': 1.22, 'upper': 1.38}, 
              'T3': {'center': 1.60, 'lower': 1.51, 'upper': 1.70}},
        6.0: {'T2': {'center': 1.65, 'lower': 1.56, 'upper': 1.75}, 
              'T3': {'center': 1.95, 'lower': 1.84, 'upper': 2.06}},
        6.5: {'T2': {'center': 2.02, 'lower': 1.91, 'upper': 2.13}, 
              'T3': {'center': 2.30, 'lower': 2.18, 'upper': 2.42}},
        7.0: {'T2': {'center': 2.51, 'lower': 2.38, 'upper': 2.64}, 
              'T3': {'center': 2.85, 'lower': 2.72, 'upper': 2.99}}
    }
    
    try:
        plt.rcParams['font.family'] = ['Times New Roman']
        plt.rcParams['font.weight'] = 'bold'
        # ... (其他字体设置保持不变)
    except Exception as e:
        print(f"设置中文字体失败: {e}")

    # 调用更新后的函数
    plot_final_style_with_uncertainty(
        thresholds_data_dict=threshold_data_with_uncertainty, 
        cd_min_plot=CD_MIN_PLOT, 
        cd_max_plot=CD_MAX_PLOT
    )