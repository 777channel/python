# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager
import os
import numpy as np
import statsmodels.api as sm

# --- 配置区域 ---

# 1. 文件路径
csv_file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\生物炭结果\3% result.csv"
output_image_filename = 'ph_ci.pdf' # 保存的图片文件名 (更新)
output_directory = 'plots_grouped_unit' # 保存图片的子文件夹名 (更新)

# 2. 分组变量选择
grouping_variable_choice = 'Cd'

# 3. Lowess 平滑度参数
lowess_fraction = 0.9

# 4. 图例名称映射
legend_label_map = {
    'rice_straw': 'RS', 'animal_derived': 'A', 'waste': 'WA', 'other_crop_straw': 'PS',
    'nutshell': 'NS', 'corn_straw': 'CS', 'wood': 'WO', 'wheat_straw': 'WS'
}

# 5. 全局字体和线条设置
try:
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['mathtext.default'] = 'regular'
    line_thickness = 1.5
    plt.rcParams['axes.linewidth'] = line_thickness
    plt.rcParams['xtick.major.width'] = line_thickness
    plt.rcParams['ytick.major.width'] = line_thickness
    prop = matplotlib.font_manager.FontProperties(family='Times New Roman')
    matplotlib.font_manager.findfont(prop, fallback_to_default=False)
    print(f"字体 '{plt.rcParams['font.family']}' 设置成功。")
except Exception as e:
    print(f"\n警告：无法将字体设置为 Times New Roman ({e})。")
    plt.rcParams['font.family'] = 'sans-serif'
    line_thickness = 1.5
    plt.rcParams['axes.linewidth'] = line_thickness
    plt.rcParams['xtick.major.width'] = line_thickness
    plt.rcParams['ytick.major.width'] = line_thickness
    plt.rcParams['axes.unicode_minus'] = False

# 6. 字体大小
title_fontsize = 24
label_fontsize = 22
tick_label_fontsize = 20
legend_title_fontsize = 22
legend_text_fontsize = 20

# 7. 刻度线配置
tick_length = 4
tick_direction = 'in'

# 8. 定义绘图变量
y_var_en = 'Application Rate (%)'
y_var_col = '施炭比(%)'
x_var_en = 'Temperature (°C)'
x_var_col = '生物炭热解温度(°C)'
hue_var_en = 'Biochar Feedstock Type'
hue_var_col = '生物炭原料类型'

# 根据选择确定分组列名和单位
unit_latex = ''
if grouping_variable_choice.lower() == 'ph':
    grouping_column = '土壤pH'
    grouping_column_en = 'pH'
elif grouping_variable_choice.lower() == 'cd':
    grouping_column = '初始镉浓度(mg/kg)'
    grouping_column_en = 'Cd'
    unit_latex = ' mg·kg$^{-1}$'
else:
    raise ValueError(f"无效的分组变量选择: '{grouping_variable_choice}'。")


# --- 数据加载与预处理 (与之前相同) ---
try:
    df = pd.read_csv(csv_file_path, skiprows=1, header=0, encoding='utf-8')
    # ... (此处省略与之前完全相同的数据加载和预处理代码) ...
    required_cols = [grouping_column, x_var_col, y_var_col, hue_var_col, '灌溉方式']
    if any(col not in df.columns for col in required_cols):
        raise KeyError(f"CSV 文件缺少必需的列")
    df[grouping_column] = pd.to_numeric(df[grouping_column])
    value_counts = df[grouping_column].value_counts()
    if len(value_counts) < 5:
        top_grouping_values = sorted(value_counts.index.tolist())
    else:
        top_grouping_values = sorted(value_counts.nlargest(5).index.tolist())
    num_cols = len(top_grouping_values)
    value_to_exclude = 4
    if grouping_variable_choice.lower() == 'cd' and value_to_exclude in top_grouping_values:
        top_grouping_values.remove(value_to_exclude)
        num_cols = len(top_grouping_values)
    unique_feedstocks = sorted(df[hue_var_col].unique())
    num_feedstocks = len(unique_feedstocks)
    if num_feedstocks == 8:
        image_colors = ['#B7C6F2', '#33C37C', '#D20101', '#66B63C', '#F5660D', '#C979B8', '#8AC96A', '#3D4F87']
        custom_colors = {feedstock: image_colors[i] for i, feedstock in enumerate(unique_feedstocks)}
    else:
        palette = sns.color_palette('tab10', n_colors=num_feedstocks)
        custom_colors = {feedstock: palette[i] for i, feedstock in enumerate(unique_feedstocks)}
    df_cf = df[df['灌溉方式'] == '持续淹水'].copy()
    df_awd = df[df['灌溉方式'] == '干湿交替'].copy()
    
    # --- 创建图形和子图区域 ---
    fig, axes = plt.subplots(nrows=2, ncols=num_cols,
                             figsize=(6 * num_cols, 10),
                             sharex=True, sharey=True)

    # --- 绘图循环 ---
    irrigation_types_en = {
        'DW': {'data': df_awd, 'row': 0},
        'WF': {'data': df_cf, 'row': 1}
    }
    plot_successful = False

    for irr_name_en, irr_info in irrigation_types_en.items():
        df_irr = irr_info['data']
        row_idx = irr_info['row']

        for col_idx, group_value in enumerate(top_grouping_values):
            ax = axes[row_idx, col_idx] if num_cols > 1 else axes[row_idx]
            df_plot = df_irr[df_irr[grouping_column] == group_value]
            title_text = f'{irr_name_en}\n{grouping_column_en} = {group_value}{unit_latex}'

            if not df_plot.empty:
                # 1. 绘制散点图 (保持不变)
                sns.scatterplot(data=df_plot, x=x_var_col, y=y_var_col, hue=hue_var_col,
                                  ax=ax, palette=custom_colors, s=60, legend=False,
                                  hue_order=unique_feedstocks)

                # ==================== 修改开始 ====================
                # 2. 使用 sns.regplot 绘制实线拟合和95%置信区间
                #    我们不再手动计算lowess，而是让regplot自动完成
                if len(df_plot) > 2:
                    try:
                        sns.regplot(
                            data=df_plot,
                            x=x_var_col,
                            y=y_var_col,
                            ax=ax,
                            lowess=True,              # 启用Lowess平滑
                            scatter=False,            # 不重复绘制散点
                            ci=95,                    # 计算95%的置信区间
                            color='orange',           # 将置信区间的颜色设为橙色
                            line_kws={
                                'color': 'black',     # 单独设置线条颜色为黑色
                                'linestyle': '-',     # 设置线条为实线
                                'linewidth': line_thickness
                            }
                        )
                    except Exception as reg_err:
                        print(f"无法为 {irr_name_en}, {grouping_column_en}={group_value} 绘制拟合曲线。错误: {reg_err}")

                # (原有的手动计算Lowess的代码已被上面的regplot替代)
                # ==================== 修改结束 ====================

                # --- 其他设置 (与之前相同) ---
                ax.set_title(title_text, fontsize=title_fontsize)
                if col_idx == 0: ax.set_ylabel(y_var_en, fontsize=label_fontsize)
                if row_idx == 1: ax.set_xlabel(x_var_en, fontsize=label_fontsize)
                else: ax.tick_params(axis='x', labelbottom=False)
                ax.tick_params(axis='both', direction=tick_direction, length=tick_length, labelsize=tick_label_fontsize, width=line_thickness)
                plot_successful = True
            else:
                # ... (无数据时的处理与之前相同) ...
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, color='grey')
                ax.set_title(title_text, fontsize=title_fontsize)
                if col_idx == 0: ax.set_ylabel(y_var_en, fontsize=label_fontsize)
                if row_idx == 1: ax.set_xlabel(x_var_en, fontsize=label_fontsize)
                else: ax.tick_params(axis='x', labelbottom=False)
                ax.tick_params(axis='both', direction=tick_direction, length=tick_length, labelsize=24, width=1.5)

    # --- 创建统一图例 (与之前相同) ---
    if plot_successful:
        legend_handles = []
        for feedstock_name in unique_feedstocks:
            color = custom_colors.get(feedstock_name, 'grey')
            display_label = legend_label_map.get(feedstock_name, feedstock_name)
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                                   label=display_label,
                                                   markerfacecolor=color, markersize=12))
        if legend_handles:
            fig.legend(handles=legend_handles, title=hue_var_en,
                       loc='center left', bbox_to_anchor=(1.01, 0.5),
                       fontsize=legend_text_fontsize, title_fontsize=legend_title_fontsize,
                       frameon=False)

    # --- 调整整体布局和保存 (与之前相同) ---
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    save_path = os.path.join(output_directory, output_image_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已成功保存至: '{save_path}'")
    plt.close(fig)

# --- 异常处理 (与之前相同) ---
except Exception as e:
    print(f"\n执行脚本时发生未知错误: {e}")