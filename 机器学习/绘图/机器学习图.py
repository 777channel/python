import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 模拟数据
labels = ['pH 5.0 (AWD)', 'pH 6.0 (AWD)', 'pH 7.0 (AWD)']
values_no_bc = [0.70, 1.20, 1.80]
values_bc = [0.98, 1.55, 2.20]
max_val = 3.0  # 背景总长度

fig, ax = plt.subplots(figsize=(10, 6))

for i, label in enumerate(labels):
    # 1. 绘制背景条 (灰色)
    ax.add_patch(patches.FancyBboxPatch((0, i-0.15), max_val, 0.3, boxstyle="round,pad=0.02", color="#F1F5F9"))
    
    # 2. 绘制数据条 (渐变或纯色)
    # 无BC条 (这里简化为纯色，专业绘图需调用LinearSegmentedColormap)
    ax.add_patch(patches.FancyBboxPatch((0, i-0.15), values_no_bc[i], 0.3, boxstyle="round,pad=0.02", color="#11CAA0"))
    
    # 3. 添加文字标注
    ax.text(values_no_bc[i]-0.1, i, f'无BC: {values_no_bc[i]}', color='white', va='center', ha='right', fontweight='bold')

# 隐藏坐标轴，只留标签
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
plt.show()