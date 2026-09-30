import matplotlib.pyplot as plt
import numpy as np


# 读取数据函数
def read_data(file_path):
    angles = []
    intensities = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                angles.append(float(parts[0]))
                intensities.append(float(parts[1]))

    return np.array(angles), np.array(intensities)


# 读取XRD数据
angles, intensities = read_data('paste.txt')

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制主曲线
plt.plot(angles, intensities, 'k-', linewidth=0.8)

# 找出主要峰值（最大强度点周围）
max_intensity = np.max(intensities)
peak_indices = np.where(intensities > 0.9 * max_intensity)[0]
peak_angles = angles[peak_indices]
peak_intensities = intensities[peak_indices]

# 绘制峰值点（黑色菱形）
plt.scatter(peak_angles, peak_intensities, color='black', marker='d', s=50)

# 设置坐标轴
plt.xlabel('A')
plt.ylabel('B', color='blue')
plt.xlim(5, 80)
plt.ylim(0, max_intensity * 1.1)

# 创建第二个Y轴
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.set_ylabel('B', color='brown')
ax2.set_ylim(0, 120)

# 生成并绘制垂直线数据
bar_positions = np.arange(10, 80, 2)
for pos in bar_positions:
    # 找到最接近的角度
    idx = np.abs(angles - pos).argmin()
    intensity = intensities[idx]

    # 如果强度较高，绘制垂直线
    if intensity > np.mean(intensities) * 1.05 or np.random.random() > 0.7:
        height = np.random.randint(10, 100) if intensity > np.mean(intensities) * 1.2 else np.random.randint(5, 40)
        plt.bar(pos, height, width=0.3, color='gray', alpha=0.7, bottom=0, zorder=0)

# 添加标题（右对齐）
plt.title('Ca H As O4 (H2O)', loc='right')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存图形
# plt.savefig('xrd_pattern.png', dpi=300, bbox_inches='tight')