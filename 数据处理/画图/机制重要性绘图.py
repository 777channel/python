import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- 1. 全局样式设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
# ==================== 字体大小修改处 ====================
# 将全局默认字体大小提升到 18
plt.rcParams['font.size'] = 22
# =======================================================
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'


# --- 2. 数据准备 ---
# (数据部分与之前相同)
data_cd = {
    'Solution Types': ['Cd$_{100}$As$_{200}$', 'Cd$_{200}$As$_{200}$', 'Cd$_{400}$As$_{200}$', 'Cd$_{400}$'],
    'Non_Mineral_Adsorption': [4.95, 28.3, 64.4, 35.3],
    'Non_Mineral_Error': [0.95, 1.7, 10, 2.7],
    'Mineral_Adsorption': [83.35, 145.1, 182.933, 219.5],
    'Mineral_Error': [1, 2.6, 3.867, 0.8],
    'Non_Mineral_Percentage': [5.61, 16.32, 26.04, 13.85],
    'Mineral_Percentage': [94.39, 83.68, 73.96, 86.15]
}
df_cd = pd.DataFrame(data_cd)

data_as = {
    'Solution Types': ['As$_{200}$', 'Cd$_{100}$As$_{200}$', 'Cd$_{200}$As$_{200}$', 'Cd$_{400}$As$_{200}$'],
    'Non_Mineral_Adsorption': [39.596, 43.091, 41.372, 47.3],
    'Non_Mineral_Error': [3.495, 1.699, 0.016, 1.996],
    'Mineral_Adsorption': [15.404, 22.156, 25.647, 18.717],
    'Mineral_Error': [3.495, 1.699, 0.016, 1.996],
    'Hydroxyl_Adsorption': [7.244, 15.813, 13.973, 12.024],
    'Hydroxyl_Error': [2.858, 0.104, 1.016, 2.701],
    'Carboxyl_Adsorption': [2.14, 10.662, 11.252, 7.206],
    'Carboxyl_Error': [2.122, 1.783, 0.249, 1.105],
    'Non_Mineral_Percentage': [71.99, 66.04, 61.73, 71.65],
    'Mineral_Percentage': [28.01, 33.96, 38.27, 28.35],
    'Hydroxyl_Percentage': [13.17, 24.24, 20.85, 18.21],
    'Carboxyl_Percentage': [3.89, 16.34, 16.79, 10.92]
}
df_as = pd.DataFrame(data_as)


# --- 3. 创建图表和子图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=600) # 稍微增加画布高度

# 子图 a: 镉 (Cd)
ax1_right = ax1.twinx()
color_non_mineral_cd = 'skyblue'
color_mineral_cd = '#0077b6'
bar_width = 0.35
x_cd = np.arange(len(df_cd['Solution Types']))
bar1_pos = x_cd - bar_width / 2
bar2_pos = x_cd + bar_width / 2
b1 = ax1.bar(bar1_pos, df_cd['Non_Mineral_Adsorption'], bar_width, yerr=df_cd['Non_Mineral_Error'], capsize=4, label='Non-mineral', color=color_non_mineral_cd)
b2 = ax1.bar(bar2_pos, df_cd['Mineral_Adsorption'], bar_width, yerr=df_cd['Mineral_Error'], capsize=4, label='Mineral', color=color_mineral_cd)
l1, = ax1_right.plot(x_cd, df_cd['Non_Mineral_Percentage'], marker='o', color=color_non_mineral_cd, linestyle='-')
l2, = ax1_right.plot(x_cd, df_cd['Mineral_Percentage'], marker='o', color=color_mineral_cd, linestyle='-')

# 统一所有局部字体大小为 18
ax1.set_ylabel('Cd adsorption capacity (mg·g⁻¹)', fontsize=22,fontweight='bold')
ax1_right.set_ylabel('Relative Contribution (%)', fontsize=22,fontweight='bold')
ax1.set_xlabel('Solution Types', fontsize=22, labelpad=15,fontweight='bold') # 增加标签与轴的距离
ax1.set_xticks(x_cd)
ax1.set_xticklabels(df_cd['Solution Types'], rotation=0, fontsize=22,fontweight='bold')
ax1.tick_params(axis='y', labelsize=22) # 设置Y轴刻度字体大小
ax1_right.tick_params(axis='y', labelsize=22) # 设置次Y轴刻度字体大小
ax1.set_ylim(0, 250)
ax1_right.set_ylim(0, 100)
ax1.set_title('a', loc='left', fontsize=22,fontweight='bold') # 子图标题可以更大一些

# 子图 b: 砷 (As)
ax2_right = ax2.twinx()
color_as_non_mineral = '#A6CEE3'
color_as_mineral = '#1F78B4'
color_as_hydroxyl = '#B2DF8A'
color_as_carboxyl = '#33A02C'
bar_width_as = 0.2
x_as = np.arange(len(df_as['Solution Types']))
bar1_as_pos = x_as - 1.5 * bar_width_as
bar2_as_pos = x_as - 0.5 * bar_width_as
bar3_as_pos = x_as + 0.5 * bar_width_as
bar4_as_pos = x_as + 1.5 * bar_width_as
b3 = ax2.bar(bar1_as_pos, df_as['Non_Mineral_Adsorption'], bar_width_as, yerr=df_as['Non_Mineral_Error'], capsize=3, label='Non-mineral', color=color_as_non_mineral)
b4 = ax2.bar(bar2_as_pos, df_as['Mineral_Adsorption'], bar_width_as, yerr=df_as['Mineral_Error'], capsize=3, label='Mineral', color=color_as_mineral)
b5 = ax2.bar(bar3_as_pos, df_as['Hydroxyl_Adsorption'], bar_width_as, yerr=df_as['Hydroxyl_Error'], capsize=3, label='-OH', color=color_as_hydroxyl)
b6 = ax2.bar(bar4_as_pos, df_as['Carboxyl_Adsorption'], bar_width_as, yerr=df_as['Carboxyl_Error'], capsize=3, label='-COOH', color=color_as_carboxyl)
l3, = ax2_right.plot(x_as, df_as['Non_Mineral_Percentage'], marker='o', color=color_as_non_mineral, linestyle='-')
l4, = ax2_right.plot(x_as, df_as['Mineral_Percentage'], marker='o', color=color_as_mineral, linestyle='-')
l5, = ax2_right.plot(x_as, df_as['Hydroxyl_Percentage'], marker='o', color=color_as_hydroxyl, linestyle='-')
l6, = ax2_right.plot(x_as, df_as['Carboxyl_Percentage'], marker='o', color=color_as_carboxyl, linestyle='-')

# 统一所有局部字体大小为 18
ax2.set_ylabel('As adsorption capacity (mg·g⁻¹)', fontsize=22,fontweight='bold')
ax2_right.set_ylabel('Relative Contribution (%)', fontsize=22,fontweight='bold')
ax2.set_xlabel('Solution Types', fontsize=22, labelpad=15,fontweight='bold') # 增加标签与轴的距离
ax2.set_xticks(x_as)
ax2.set_xticklabels(df_as['Solution Types'], rotation=0, fontsize=22,fontweight='bold')
ax2.tick_params(axis='y', labelsize=22) # 设置Y轴刻度字体大小
ax2_right.tick_params(axis='y', labelsize=22) # 设置次Y轴刻度字体大小
ax2.set_ylim(0, 50)
ax2_right.set_ylim(0, 80)
ax2.set_title('b', loc='left', fontsize=22,fontweight='bold') # 子图标题可以更大一些

# ==================== 图例修改处 ====================
# 独立控制图例
ax1.legend(handles=[b1, b2], labels=['Non-minerals', 'Minerals'], 
           loc='upper center', bbox_to_anchor=(0.5, 1.15),  # 稍微提高位置
           ncol=2, frameon=False, 
           fontsize=22,                          # 增大图例字体
           handletextpad=0.8,                    # 加大图例中色块和文字的距离
           columnspacing=1.2                     # 加大列与列之间的距离
          )

legend_elements_b = [Patch(facecolor=color_as_non_mineral,label='Non-minerals'), 
                     Patch(facecolor=color_as_mineral,  label='Minerals'), 
                     Patch(facecolor=color_as_hydroxyl,  label='-OH'), 
                     Patch(facecolor=color_as_carboxyl,  label='-COOH')]
ax2.legend(handles=legend_elements_b, 
           loc='upper center', bbox_to_anchor=(0.5, 1.15), # 稍微提高位置
           ncol=4, frameon=False, 
           fontsize=22,                          # 增大图例字体
           handletextpad=0.8,                    # 加大图例中色块和文字的距离
           columnspacing=1.1                    # 加大列与列之间的距离
          )
# ====================================================

# --- 4. 最终调整和显示 ---
plt.subplots_adjust(
    left=0.08,    # 调整左边距
    right=0.92,   # 调整右边距
    top=0.87,
    wspace=0.35
)

plt.savefig('adsorption_capacity_large_fonts.pdf', dpi=600)
plt.show() # 如果不需要在屏幕上显示，可以注释掉这行
# plt.ioff() # 这行在脚本末尾通常不是必需的