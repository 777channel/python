import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 全局样式设置 (包括全局字体加粗) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.size'] = 22
plt.rcParams['font.weight'] = 'bold' # 设置全局默认粗体
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'


# --- 2. 数据准备 ---
# (数据与之前相同)
ph_values = [2, 3, 4, 5, 6, 7]
data_cd = {
    'pH': ph_values,
    'Cd$_{400}$': [0.0865, 0.598, 0.6265, 0.637, 0.6385, 0.706],
    'Cd$_{100}$As$_{200}$': [0.0895, 0.5620, 0.7410, 0.7345, 0.7610, 0.7430],
    'Cd$_{200}$As$_{200}$': [0.1435, 0.376, 0.493, 0.5645, 0.627, 0.693],
    'Cd$_{400}$As$_{200}$': [0.0705, 0.3355, 0.4190, 0.4390, 0.4430, 0.4855]
}
df_cd = pd.DataFrame(data_cd)
for col in df_cd.columns:
    if col != 'pH': df_cd[col] *= 100

data_as = {
    'pH': ph_values,
    'As$_{200}$': [0.08022, 0.16329, 0.19693, 0.23353, 0.279845, 0.20813],
    'Cd$_{100}$As$_{200}$': [0.19457, 0.3319, 0.344995, 0.3461075, 0.38043, 0.39387],
    'Cd$_{200}$As$_{200}$': [0.2123, 0.338055, 0.37169, 0.3904725, 0.409255, 0.385235],
    'Cd$_{400}$As$_{200}$': [0.210735, 0.335895, 0.353255, 0.3473434, 0.36816, 0.36109]
}
df_as = pd.DataFrame(data_as)
for col in df_as.columns:
    if col != 'pH': df_as[col] *= 100


# --- 3. 创建图表和子图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

# ==============================================================================
#                                子图 b: 镉 (Cd)
# ==============================================================================
colors_cd = {'Cd$_{400}$': '#37ad6b', 'Cd$_{100}$As$_{200}$': 'black', 'Cd$_{200}$As$_{200}$': '#f14004', 'Cd$_{400}$As$_{200}$': '#1a6fdf'}
markers = 'o'

for col in df_cd.columns:
    if col != 'pH':
        ax1.plot(df_cd['pH'], df_cd[col], marker=markers, color=colors_cd[col], label=col, linewidth=2.5)

# --- 坐标轴和标题设置 (明确添加 fontweight='bold') ---
ax1.set_xlabel('pH', fontweight='bold')
ax1.set_ylabel('Removal rate of Cd (%)', fontweight='bold')
ax1.set_xlim(1.5, 7.5)
ax1.set_ylim(0, 100)
ax1.set_xticks(range(2, 8))
ax1.tick_params(axis='both', labelsize=18)


# --- 图例设置 (使用 prop 参数强制加粗) ---
ax1.legend(loc='lower right', frameon=False, fontsize=16, prop={'weight': 'bold'})


# ==============================================================================
#                                子图 d: 砷 (As)
# ==============================================================================
colors_as = {'As$_{200}$': '#37ad6b', 'Cd$_{100}$As$_{200}$': 'black', 'Cd$_{200}$As$_{200}$': '#f14004', 'Cd$_{400}$As$_{200}$': '#1a6fdf'}

for col in df_as.columns:
    if col != 'pH':
        ax2.plot(df_as['pH'], df_as[col], marker=markers, color=colors_as[col], label=col, linewidth=2.5)

# --- 坐标轴和标题设置 (明确添加 fontweight='bold') ---
ax2.set_xlabel('pH', fontweight='bold')
ax2.set_ylabel('Removal rate of As (%)', fontweight='bold')
ax2.set_xlim(1.5, 7.5)
ax2.set_ylim(0, 50)
ax2.set_xticks(range(2, 8))
ax2.tick_params(axis='both', labelsize=18)


# --- 图例设置 (使用 prop 参数强制加粗) ---
ax2.legend(loc='lower right', frameon=False, fontsize=20, prop={'weight': 'bold'})


# --- 4. 最终调整和显示 ---
plt.subplots_adjust(wspace=0.3)
plt.savefig('removal_rate_plot_fully_bold.pdf', dpi=600)
plt.show()