import matplotlib.pyplot as plt
import numpy as np

# --- 全局字体设置 ---
# (这部分代码保持不变)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 设置 ---
# (这部分代码保持不变)
inner_colors = ['#0070C0']
outer_colors = ['#588E32']
inner_data = np.ones(5)
outer_data = [3, 2]

# --- 2. 核心绘图逻辑 ---
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(aspect="equal"))

# --- 步骤 A: 绘制三个同心圆环 ---

# 环 1: 最外圈的彩色环 (半径 1.0, 宽度 0.2)
ax.pie(
    outer_data,
    colors=outer_colors,
    radius=1.0,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.2, linewidth=0)
)

# 环 2: 中间的五色彩色环 (半径 0.8, 宽度 0.2)
ax.pie(
    inner_data,
    colors=inner_colors,
    radius=0.8,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.4, linewidth=0) # 宽度统一为0.2
)

# --- 新增步骤: 绘制最内部的白色圆环 ---
# 环 3: 最内圈的白色环 (半径 0.6, 宽度 0.2)
ax.pie(
    [1], # 只有一个部分，所以是完整的环
    colors=['white'], # 填充色为白色
    radius=0.6,
    startangle=90,
    wedgeprops=dict(width=0.4, edgecolor='black', linewidth=1.2) # 带有黑色边框
)
# 这样做完后，半径从0到0.4的区域就是最中心的空白区域。


# --- 步骤 B: 在背景之上叠加绘制虚线 ---
num_lines = 5
angles = [90 - i * (360 / num_lines) for i in range(num_lines)]

# 更新：虚线只跨越中间的彩色环 (环 2)
inner_radius_of_lines =0# 环2的内径 (0.6)
outer_radius_of_lines = 0.8       # 环2的外径

for angle in angles:
    rad = np.deg2rad(angle)
    start_x = inner_radius_of_lines * np.cos(rad)
    start_y = inner_radius_of_lines * np.sin(rad)
    end_x = outer_radius_of_lines * np.cos(rad)
    end_y = outer_radius_of_lines * np.sin(rad)
    ax.plot([start_x, end_x], [start_y, end_y], color='black', linestyle='--', linewidth=1.2)


# --- 4. 保存文件 ---
output_filename = 'final_ring_diagram.pdf'
plt.savefig(output_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.close()

print(f"图形已成功保存为 '{output_filename}'")