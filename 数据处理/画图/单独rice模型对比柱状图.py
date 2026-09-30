import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体属性
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 24
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 提取 rice 数据
rice_data = {
    'Model': ['GBDT', 'XGBoost', 'RF', 'ExtraTrees', 'SVR', 'LightGBM', 'Bayesian Ridge', 'Ridge', 'LR'],
    'R2_Train': [0.88305562, 0.90377686, 0.69388520, 0.62970333, 0.14399199, 0.34872229, 0.06335299, 0.05971451, 0.10372078],
    'RMSE_Train': [0.10588777, 0.09604968, 0.17132327, 0.18842177, 0.23088552, 0.24986444, 0.29967061, 0.30025209, 0.29314186],
    'MAE_Train': [0.07049621, 0.06108609, 0.12209700, 0.13817990, 0.15358075, 0.18682848, 0.22086940, 0.22125932, 0.22240223],
    'R2_Test': [0.75471199, 0.75396077, 0.68911015, 0.59387957, 0.45768478, 0.41094144, 0.11616963, 0.10929262, 0.07185485],
    'RMSE_Test': [0.11644415, 0.11662232, 0.13109912, 0.14983291, 0.17314319, 0.18045492, 0.22103649, 0.22189474, 0.22651050],
    'MAE_Test': [0.08300989, 0.08499017, 0.09700584, 0.11382181, 0.12280265, 0.14104705, 0.15696863, 0.15737366, 0.17742627],
}

# 模型名称映射到简称
model_name_mapping = {
    'GBDT': 'GBDT',
    'XGBoost': 'XGB',
    'RF': 'RF',
    'ExtraTrees': 'ET',
    'LR': 'LR',
    'LightGBM': 'LGB',
    'Bayesian Ridge': 'BR',
    'SVR': 'SVR',
    'Ridge': 'Ridge'
}

# 定义要使用的颜色
colors = ['#58659B', '#8A8CBF', '#B8A8CF', '#E7BCC6', '#FDCF9E', '#EFA484']
metrics = ['R2_Train', 'RMSE_Train', 'MAE_Train', 'R2_Test', 'RMSE_Test', 'MAE_Test']
metric_labels = ['Train R$^2$', 'Train RMSE', 'Train MAE', 'Test R$^2$', 'Test RMSE', 'Test MAE']

# 创建单个图表
fig, ax = plt.subplots(figsize=(18, 10))

# 获取模型数量和指标数量
num_models = len(rice_data['Model'])
num_metrics = len(metrics)

# 设置每个柱子的宽度和位置
bar_width = 0.14
index = np.arange(num_models)

# 绘制柱状图
for i, metric in enumerate(metrics):
    ax.bar(index + i * bar_width, rice_data[metric], bar_width,
           color=colors[i], label=metric_labels[i])

# 设置刻度标签样式
ax.set_xticks(index + bar_width * (num_metrics - 1) / 2)
ax.set_xticklabels([model_name_mapping[m] for m in rice_data['Model']], 
                   rotation=0, ha='center', fontweight='bold')

# 设置y轴刻度标签加粗
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# 设置y轴范围为0-1
ax.set_ylim(0, 1.0)

# 设置网格线
ax.grid(True, linestyle='--', alpha=0.7, axis='y')

# 设置边框线条加粗
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

# 设置刻度线长度
ax.tick_params(axis='both', which='major', length=10, width=2)

# 添加部位标题 (Rice)
ax.text(0.5, -0.15, 'Rice', transform=ax.transAxes,
        ha='center', va='center', fontweight='bold', fontsize=28)

# 添加图例
legend = ax.legend(loc='upper right', fontsize=18, 
                  frameon=True, fancybox=True, shadow=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_linewidth(1.0)

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('rice_model_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

print('Rice 图像绘制完成')