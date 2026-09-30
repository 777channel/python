import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.unicode_minus'] = False 
# 1. 准备数据
years = np.array([2020, 2040, 2060])
co2_levels = np.array([415, 473, 550])  # 微调中间值以获得平滑的增长曲线

# 2. 曲线平滑处理 (使用二次样条插值)
years_smooth = np.linspace(years.min(), years.max(), 300)
spl = make_interp_spline(years, co2_levels, k=2)
co2_smooth = spl(years_smooth)

# 3. 创建画布并设置白色背景
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')  # 整个图表的背景
ax.set_facecolor('white')        # 绘图区域的背景

# 4. 绘制平滑折线
ax.plot(years_smooth, co2_smooth, color='#6b4c82', linewidth=3, zorder=3)

# 5. 添加关键数值标注 (415 和 550)
ax.text(2020, 420, '415', color='#6b4c82', fontweight='bold', fontsize=22, ha='center')
ax.text(2060, 555, '550', color='#6b4c82', fontweight='bold', fontsize=22, ha='left')

# 6. 设置标题和文本 (使用 LaTeX 格式渲染下标
ax.tick_params(axis='both', which='major', direction='out', length=5, width=1.5)
# 7. 调整坐标轴刻度与范围
ax.set_xlim(2018, 2065)
ax.set_ylim(410, 560)
ax.set_xticks([2020, 2040, 2060])
ax.set_yticks([410, 450, 500, 550])

# 8. 隐藏上方和右侧边框，加粗剩余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 9. 自动调整布局并保存图片
plt.tight_layout()
plt.savefig('co2_chart_white.pdf', dpi=600, bbox_inches='tight', facecolor='white')

# 显示预览
plt.show()