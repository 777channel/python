import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import matplotlib.patches as patches

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22

# 原始九个模型和数据
all_models = ['LR', 'Ridge', 'RF', 'GBDT', 'XGBoost', 'LightGBM', 
              'ExtraTrees', 'SVR', 'TabPFN']
all_train_R2 = [0.458, 0.524, 0.923, 0.991, 0.998, 0.583, 0.991, 0.503, 0.921]
all_test_R2 = [0.611, 0.469, 0.726, 0.708, 0.693, 0.442, 0.669, 0.713, 0.699]

# 根据测试集R²排名选择前6个模型
model_data = list(zip(all_models, all_train_R2, all_test_R2))
# 按测试集R²降序排序，选择前6个
top6_models = sorted(model_data, key=lambda x: x[2], reverse=True)[:6]

# 提取前6个模型的数据
models = [item[0] for item in top6_models]
train_R2 = [item[1] for item in top6_models]
test_R2 = [item[2] for item in top6_models]

# 计算六边形的角度（从顶部开始，顺时针）
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
# 调整角度，使第一个点在顶部
angles = angles + np.pi/2

# 创建图形
fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
ax.set_aspect('equal')

# 设置坐标轴范围
max_val = 1.0
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# 绘制六边形网格线（从内到外）
hex_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
for level in hex_levels:
    hex_x = level * np.cos(angles)
    hex_y = level * np.sin(angles)
    # 闭合六边形
    hex_x = np.append(hex_x, hex_x[0])
    hex_y = np.append(hex_y, hex_y[0])
    ax.plot(hex_x, hex_y, '--', color='gray', alpha=0.5, linewidth=0.8)

# 绘制从中心到各顶点的辐射线
for angle in angles:
    ax.plot([0, np.cos(angle)], [0, np.sin(angle)], '-', color='gray', alpha=0.5, linewidth=0.8)

# 将R²值转换为坐标（归一化到0-1范围）
def r2_to_coords(r2_values, angles):
    x_coords = []
    y_coords = []
    for r2, angle in zip(r2_values, angles):
        x = r2 * np.cos(angle)
        y = r2 * np.sin(angle)
        x_coords.append(x)
        y_coords.append(y)
    return x_coords, y_coords

# 计算训练集和测试集的坐标
train_x, train_y = r2_to_coords(train_R2, angles)
test_x, test_y = r2_to_coords(test_R2, angles)

# 闭合多边形
train_x_closed = train_x + [train_x[0]]
train_y_closed = train_y + [train_y[0]]
test_x_closed = test_x + [test_x[0]]
test_y_closed = test_y + [test_y[0]]

# 绘制填充区域
train_polygon = Polygon(list(zip(train_x_closed, train_y_closed)), 
                       facecolor='#ffd9d6', alpha=0.4, edgecolor='#ee6342', linewidth=2)
test_polygon = Polygon(list(zip(test_x_closed, test_y_closed)), 
                      facecolor='#bde0e2', alpha=0.4, edgecolor='#00718b', linewidth=2)

ax.add_patch(train_polygon)
ax.add_patch(test_polygon)

# 绘制数据点
ax.plot(train_x_closed, train_y_closed, 'o-', color='#ee6342', markersize=18, linewidth=2, label='Train')
ax.plot(test_x_closed, test_y_closed, 'o-', color='#00718b', markersize=18, linewidth=2, label='Test')

# 添加模型标签（在六边形外围）
label_distance = 1.2
for i, (angle, model) in enumerate(zip(angles, models)):
    x_label = label_distance * np.cos(angle)
    y_label = label_distance * np.sin(angle)
    ax.text(x_label, y_label, model, fontsize=20, fontweight='bold', 
            ha='center', va='center')

# 添加数值标签
for i, (angle, train_val, test_val) in enumerate(zip(angles, train_R2, test_R2)):
    # 训练集数值标签
    train_x_pos = (train_val - 0.15) * np.cos(angle)
    train_y_pos = (train_val + 0.05) * np.sin(angle)
    ax.text(train_x_pos, train_y_pos, f'{train_val:.3f}', 
            ha='center', va='center', fontsize=16, color='black', fontweight='bold')
    
    # 测试集数值标签
    test_x_pos = (test_val + 0.08) * np.cos(angle)
    test_y_pos = (test_val + 0.09) * np.sin(angle)
    ax.text(test_x_pos, test_y_pos, f'{test_val:.3f}', 
            ha='center', va='center', fontsize=16, color='black', fontweight='bold')


ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 创建图例
legend_elements = [
    Patch(facecolor='#FFE4E1', alpha=0.7, label='train'),
    Patch(facecolor='#66CDAA', alpha=0.7, label='test')
]

plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), 
           ncol=2,prop={'family': 'Times New Roman', 'weight': 'bold'}, fontsize=20)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig("hexagon_radar_chart.pdf", format='pdf', bbox_inches='tight', dpi=600)


# 输出模型性能分析
print("="*70)
print("Top 6 Natural Aging Model Performance Analysis (Higher R² indicates better performance)")
print("="*70)
print("Selected models based on testing R² ranking:")
for i, model in enumerate(models):
    overfitting = train_R2[i] - test_R2[i]
    print(f"{i+1}. {model:12s}: Training={train_R2[i]:6.3f}, Testing={test_R2[i]:6.3f}, Overfitting={overfitting:6.3f}")

print(f"\n{'='*70}")
print("Overfitting Analysis (Training R² - Testing R²):")
print("="*70)
overfitting_ranking = sorted(zip(models, [train_R2[i] - test_R2[i] for i in range(len(models))]), 
                            key=lambda x: x[1])
for i, (model, diff) in enumerate(overfitting_ranking, 1):
    status = "Mild overfitting" if diff < 0.1 else "Moderate overfitting" if diff < 0.3 else "Severe overfitting"
    print(f"{i:2d}. {model:12s}: {diff:6.3f} ({status})")

print(f"\n{'='*70}")
print("Excluded models (ranked 7-9 by testing R²):")
print("="*70)
excluded_models = sorted(model_data, key=lambda x: x[2], reverse=True)[6:]
for i, (model, train_val, test_val) in enumerate(excluded_models, 7):
    print(f"{i:2d}. {model:12s}: Testing R²={test_val:.3f}")