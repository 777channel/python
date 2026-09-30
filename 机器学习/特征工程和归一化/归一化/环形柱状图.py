import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
features = ['organicmatter', 'pH', 'pHsoil', 'temperature', 'Charcoal_application_ratio', 
           'log_Cdcontent', 'CEC', 'variety_binary', 'material_type_code', 'irrigation_type']
values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建数据框
df = pd.DataFrame({'features': features, 'values': values})
# 升序排序
df.sort_values(by='values', ascending=True, inplace=True)

# 提取数据
values = df['values'].values.tolist()

# 设置图形大小
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 计算总数值
total_value = sum(values)
total_angle = 2 * np.pi  # 圆的总角度

gap_angle = np.pi / 50  # 设置每个扇区之间的间隙
angles = [(value / total_value) * (total_angle - gap_angle * len(values)) for value in values]

# 计算起始角度
theta = np.zeros(len(values))
current_angle = np.pi / 2  # 90 度起始，垂直向上
for i in range(len(values)):
    theta[i] = current_angle
    current_angle += angles[i] + gap_angle  # 添加间隙

# 创建颜色映射
cmap = plt.cm.coolwarm
normalized_values = (np.array(values) - min(values)) / (max(values) - min(values))
colors = [cmap(val) for val in normalized_values]

# 计算高度，数值越大，高度越小
max_value = max(values)
heights = [1.0 - (value / max_value) * 0.8 for value in values]

# 画图
bars = ax.bar(theta, heights, width=angles, bottom=0, color=colors, edgecolor='black')

# 隐藏标签
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_yticks([])

plt.show()
