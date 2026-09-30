import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ==============================================================================
# 1. 全局设置 (Global Settings)
# ==============================================================================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# --- 新增参数：设置扇区的最小角度（单位：度） ---
MINIMUM_ANGLE_DEGREES = 15

# ==============================================================================
# 2. 数据与特征分类 (Data and Feature Categorization)
# ==============================================================================

# 从您的图表中提取的特征和对应的重要性值
features_list = [
    'BR', 'pH-B', 'BET', 'C-H-D', 'Cd-C', 'RM', 'C-L-D', 
    'pH-S', 'CEC', 'T-A', 'DW-D', 'FT-D', 'T'
]

# 对应的重要性值（单位：%）
values_list = [
    0.1684, 0.1229, 0.1093, 0.1024, 0.0979, 0.0891, 0.0842, 
    0.0828, 0.0625, 0.0458, 0.0165, 0.0107, 0.0075
]

category_map = {
    'Soil': ['pH-S', 'Cd-C', 'CEC'],
    'Biochar': ['pH-B', 'T', 'BET', 'RM'],
    'Experimental': ['C-L-D', 'C-H-D', 'DW-D', 'FT-D', 'T-A', 'BR']
}
category_colors = {
'Soil': '#ffb55f',        # 更淡的浅黄色系
'Biochar': '#fb7e70',      # 更淡的浅橙色/粉色系
'Experimental': '#7eb1d2' # 更淡的浅紫色系
}
feature_to_color = {f: c for cat, feats in category_map.items() for f in feats for c in [category_colors[cat]]}

# ==============================================================================
# 3. 数据准备 (Data Preparation)
# ==============================================================================
df = pd.DataFrame({'features': features_list, 'values': values_list})
df['percent_values'] = df['values'] * 100
df['category'] = df['features'].apply(lambda x: next((cat for cat, feats in category_map.items() if x in feats), None))
df['color'] = df['features'].map(feature_to_color)
df.sort_values(by='values', ascending=True, inplace=True)

# ==============================================================================
# 4. 绘制组合图 (Plotting the Combined Figure with Inset)
# ==============================================================================
fig, ax_bar = plt.subplots(figsize=(18, 12), dpi=600, facecolor='white')
bars = ax_bar.barh(df['features'], df['percent_values'], color=df['color'])
ax_bar.set_xlabel('Importance (%)', fontsize=26, fontweight='bold')
ax_bar.tick_params(axis='y', labelsize=26)
ax_bar.tick_params(axis='x', labelsize=26)
plt.setp(ax_bar.get_xticklabels(), fontweight="bold")
plt.setp(ax_bar.get_yticklabels(), fontweight="bold")
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['bottom'].set_linewidth(2.0) # 设置 X 轴（底部）线条宽度
ax_bar.spines['left'].set_linewidth(2.0)   # 设置 Y 轴（左侧）线条宽度
legend_elements = [Patch(facecolor=color, label=label) for label, color in category_colors.items()]
ax_bar.legend(handles=legend_elements, loc='lower right',
              prop={'weight': 'bold', 'size': 20}, frameon=False)

# --- 4.2 创建并绘制【带最小角度】的极坐标插图 ---
inset_pos = [0.53, 0.12, 0.6, 0.6] 
ax_polar = fig.add_axes(inset_pos, polar=True)
ax_polar.set_facecolor('none')
values_sorted = df['values'].values.tolist()
percent_values_sorted = df['percent_values'].values.tolist()
colors_sorted = df['color'].values.tolist()


# 1. 初始设置
total_angle_rad = 2 * np.pi
num_features = len(values_sorted)
gap_angle_rad = np.pi / 1000
min_angle_rad = np.deg2rad(MINIMUM_ANGLE_DEGREES)
total_value = sum(values_sorted)
total_drawable_angle = total_angle_rad - gap_angle_rad * num_features

# 2. 按比例计算初始角度
proportional_angles = [(v / total_value) * total_drawable_angle for v in values_sorted]

# 3. 找出过窄的扇区，并计算总共需要“补贴”多少角度
total_deficit = 0
indices_below_min = []
for i, angle in enumerate(proportional_angles):
    if angle < min_angle_rad:
        total_deficit += min_angle_rad - angle
        indices_below_min.append(i)

# 4. 从足够宽的扇区中，按比例“借出”角度来弥补亏空
final_angles = list(proportional_angles)
if total_deficit > 0:
    donor_indices = [i for i, angle in enumerate(proportional_angles) if angle >= min_angle_rad]
    total_donor_angle = sum(proportional_angles[i] for i in donor_indices)
    
    if total_donor_angle > 0:
        for i in range(num_features):
            if i in indices_below_min:
                final_angles[i] = min_angle_rad
            elif i in donor_indices:
                deduction = (proportional_angles[i] / total_donor_angle) * total_deficit
                final_angles[i] -= deduction

angles = final_angles

# 计算起始角度 (theta)
theta = np.zeros(len(values_sorted))
current_angle = np.pi / 2
for i in range(len(values_sorted)):
    theta[i] = current_angle
    current_angle += angles[i] + gap_angle_rad 

# ★★★ 修改点：手动扩展列表以包含13个数据点 ★★★
inner_radii = [0.93, 0.86, 0.8, 0.76, 0.70, 0.64, 0.58, 0.5, 0.44, 0.4, 0.35, 0.3, 0.25]
outer_radii = [1.23, 1.15, 1.1, 1.05, 1.00, 0.95, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.50]


for i in range(len(values_sorted)):
    bg_color = '#E5E5E5' if i % 2 == 0 else '#D0D0D0'
    ax_polar.bar(theta[i], inner_radii[i], width=angles[i], bottom=0,
                 color=bg_color, edgecolor='none', linewidth=0, align='edge', alpha=0.7)
for i in range(len(values_sorted)):
    height = outer_radii[i] - inner_radii[i]
    ax_polar.bar(theta[i], height, width=angles[i], bottom=inner_radii[i],
                 color=colors_sorted[i], edgecolor='white', linewidth=0.8, align='edge')
    angle_mid = theta[i] + angles[i] / 2
    radius = outer_radii[i] + 0.41

    angle_deg = angle_mid * 180 / np.pi
    ha = 'left' if 0 <= angle_deg <= 180 else 'center'
    ax_polar.text(angle_mid, radius, f"{percent_values_sorted[i]:.2f}%", ha=ha, va='center',
                  fontsize=20, fontweight='bold')
for angle in theta:
    ax_polar.plot([angle, angle], [0, 1.2], color='white', linewidth=1.2, alpha=0.9)
    
ax_polar.spines['polar'].set_visible(False)
ax_polar.grid(False)
ax_polar.set_xticklabels([])
ax_polar.set_yticklabels([])
ax_polar.set_xticks([])
ax_polar.set_yticks([])
ax_polar.set_ylim(0, max(outer_radii) + 0.7)

# ==============================================================================
# 5. 保存并显示最终图像
# ==============================================================================
plt.savefig('自然老化cd-ch特征图.pdf', dpi=600, bbox_inches='tight')
print("图已保存为 '自然老化cd-ch特征图.pdf'")
plt.close()