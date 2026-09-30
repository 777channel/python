import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 准备数据 - 使用您的图中的特征和数值
features = [
    'log Cdcontent',
    'pHsoil',
    'organicmatter',
    'Charcoal application ratio',
    'pH',
    'CEC',
    'temperature',
    'material type code',
    'variety binary',
    'irrigation type'
]

values = [
    0.178,
    0.160,
    0.155,
    0.131,
    0.121,
    0.095,
    0.092,
    0.043,
    0.014,
    0.010
]

# 转换为百分比值用于显示
percent_values = [val * 100 for val in values]

# 创建数据框并排序
df = pd.DataFrame({'features': features, 'values': values, 'percent_values': percent_values})
df.sort_values(by='values', ascending=True, inplace=True)

# 提取排序后的数据
features = df['features'].values.tolist()
values = df['values'].values.tolist()
percent_values = df['percent_values'].values.tolist()

# 设置图形大小，确保使用白色背景
plt.figure(figsize=(12, 12), dpi=300, facecolor='white')
ax = plt.subplot(111, polar=True)
ax.set_facecolor('white')

# 计算总数值和每个扇区的角度
total_value = sum(values)
total_angle = 2 * np.pi  # 圆的总角度

# 设置扇区之间的间隙
gap_angle = np.pi / 1000  # 极小的间隙
angles = [(value / total_value) * (total_angle - gap_angle * len(values)) for value in values]

# 计算起始角度
theta = np.zeros(len(values))
current_angle = np.pi / 2  # 90度起始，从顶部开始
for i in range(len(values)):
    theta[i] = current_angle
    current_angle += angles[i] + gap_angle

# 使用自定义颜色
blueColors = [
    '#3f5fd4', '#0059d2', '#005ed8', '#0065df', '#6659cd',
]

redColors = [
    '#c70073', '#d50064', '#e20063', '#ed005a', '#f70048'
]

# 组合为一个颜色列表 - 从蓝色到红色
# 由于我们有10个数据点，我们需要确保颜色足够
colors = blueColors + redColors
if len(colors) > len(values):
    colors = colors[:len(values)]  # 截取需要的部分
elif len(colors) < len(values):
    # 如果颜色不够，重复使用最后一个颜色
    colors = colors + [colors[-1]] * (len(values) - len(colors))

# 手动设置每个扇区的内半径（bottom值）- 值越大，内半径越小
inner_radii = [
    0.55,  # 0.010 (最小值)
    0.52,  # 0.014
    0.45,  # 0.043
    0.4,  # 0.092
    0.38,  # 0.095
    0.35,  # 0.121
    0.30,  # 0.131
    0.28,  # 0.155
    0.26,  # 0.160
    0.25   # 0.178 (最大值)
]

# 手动设置每个扇区的外半径 - 值越小，外半径越大
outer_radii = [
    0.95,   # 0.010 (最小值)
    0.9,  # 0.014
    0.85,  # 0.043
    0.78,  # 0.092
    0.74,  # 0.095
    0.7,  # 0.121
    0.65,  # 0.131
    0.6,  # 0.155
    0.55,  # 0.160
    0.5   # 0.178 (最大值)
]

# 首先绘制背景扇形 - 交替的灰色阴影
for i in range(len(values)):
    # 交替使用深浅灰色
    if i % 2 == 0:
        bg_color = '#E5E5E5'  # 浅灰色
    else:
        bg_color = '#D0D0D0'  # 深灰色
    
    # 绘制从中心到内半径的扇形
    ax.bar(theta[i], inner_radii[i], width=angles[i], bottom=0, 
           color=bg_color, edgecolor='white', linewidth=0.5, align='edge', alpha=0.7)

# 绘制主要扇形 - 使用不同的内外半径
for i in range(len(values)):
    # 计算此扇区的高度 (外半径 - 内半径)
    height = outer_radii[i] - inner_radii[i]
    
    # 绘制扇形
    bar = ax.bar(theta[i], height, width=angles[i], bottom=inner_radii[i], 
                color=colors[i], edgecolor='white', linewidth=0.8, align='edge')
    
    # 添加数值标签 - 将标签位置调整得更外部
    angle = theta[i] + angles[i] / 2  # 扇区中点角度
    # 标签位置设置在外半径之外的更远位置
    radius = outer_radii[i] + 0.2  # 增大此值以将标签放置得更外部
    
    # 计算角度，用于调整标签对齐方式
    angle_deg = angle * 180 / np.pi
    ha = 'left' if 0 <= angle_deg <= 180 else 'right'  # 水平对齐
    
    plt.text(angle, radius, f"{percent_values[i]:.1f}%", ha=ha, va='center', 
             fontsize=14, fontweight='bold', fontname='Times New Roman')

# 添加从中心到每个扇区的线条
for angle in theta:
    # 画从中心到外围的线
    ax.plot([angle, angle], [0, 1.2], color='white', linewidth=1.2, alpha=0.9)

# 隐藏所有轴线和标签
ax.spines['polar'].set_visible(False)  # 去掉外部黑色圆圈
ax.grid(False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

# 增大图形在极坐标系中的大小，确保有足够空间显示外部标签
ax.set_ylim(0, 1.7)  # 增大上限以适应更外部的标签

# 设置标题 - 使用Times New Roman字体
plt.title('Feature Importance for CD brown rice change\nModel: GBDT', 
          fontsize=16, pad=20, fontname='Times New Roman')

# 调整布局并保存图片
plt.tight_layout()
plt.savefig('feature_importance_cd_brown_rice.png', dpi=300, bbox_inches='tight', transparent=False)
plt.close()