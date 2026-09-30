import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 基础全局设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.unicode_minus'] = False 

# 1. 准备数据 (根据图片趋势，气温先略微波动再快速上升)
years = np.array([2020, 2030, 2045, 2060])
temp_change = np.array([0.0, -0.1, 0.8, 2.0]) 

# 2. 曲线平滑处理
years_smooth = np.linspace(years.min(), years.max(), 300)
spl = make_interp_spline(years, temp_change, k=2)
temp_smooth = spl(years_smooth)

# 3. 创建画布并设置白色背景
fig, ax = plt.subplots(figsize=(10, 6)) # 稍微加宽一点，视觉效果更接近原图
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# 4. 绘制平滑折线 (颜色参考图中较深的红褐色)
ax.plot(years_smooth, temp_smooth, color='#8a4e59', linewidth=4, zorder=3)

# 5. 添加关键数值标注 (+2.0°C)
# 这里的坐标根据数据点稍微偏移，确保不遮挡线段
ax.text(2060, 2.1, '+2.0°C', color='#8a4e59', fontweight='bold', fontsize=22, ha='center')

# 6. 设置刻度线参数
ax.tick_params(axis='both', which='major', direction='out', length=8, width=2)

# 7. 调整坐标轴刻度与范围
ax.set_xlim(2018, 2065)
ax.set_ylim(-0.5, 2.5) # 给顶部留出标注空间
ax.set_xticks([2020, 2040, 2060])
# 设置 Y 轴刻度，包含 0.0, +2.0 以及最高位的带单位标注
ax.set_yticks([0.0, 1.0, 2.0])
ax.set_yticklabels(['0.0', '+1.0', '+2.0'])

# 8. 辅助文本标签 (标题和右下角说明)
# 标题使用 LaTeX 渲染度数符号
ax.text(2022, 2.4, '2. PROJECTED GLOBAL TEMP\n   CHANGE (°C)', 
        fontsize=22, fontweight='bold', va='top', ha='left')

ax.set_ylabel('Temp Change', fontweight='bold', fontsize=22)

# 右下角标签
ax.text(2060, 0.3, 'Rising Temperature', fontsize=22, fontweight='bold', ha='right')

# Y轴最下方的 Baseline 标注
ax.text(2016, -0.1, 'Baseline', fontsize=18, ha='right', va='center')

# 9. 隐藏边框并加粗
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 10. 自动调整布局并保存
plt.tight_layout()
plt.savefig('temp_projection_white.pdf', dpi=600, bbox_inches='tight', facecolor='white')

# 显示预览
plt.show()