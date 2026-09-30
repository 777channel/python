import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 24
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['axes.unicode_minus'] = False 
# 读取Excel文件（修改为你的文件路径）
df = pd.read_excel(r"C:\Users\AMDYE\OneDrive\桌面\新建 XLSX 工作表.xlsx")

# 皮尔逊相关矩阵
corr = df.corr(method='pearson').round(2)

# # 掩盖上三角（只显示下三角+对角线）
# mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

# 配色：蓝 -> 白 -> 深红（参考图1）
colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
          '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

plt.rcParams['font.family'] = 'Times New Roman'

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr,
    # mask=mask,
    annot=True,
    fmt='.2f',
    cmap=cmap,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='white',
    square=True,
    annot_kws={'size': 18, 'fontfamily': 'Times New Roman', 'fontweight': 'bold', 'color': 'black'},
    cbar_kws={'shrink': 0.8},
    ax=ax
)
for text in ax.texts:
    text.set_color('black')
# 色条刻度字体
cbar = ax.collections[0].colorbar
cbar.set_ticks([-1.0, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00])
cbar.ax.tick_params(length=6, width=2)
for label in cbar.ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')
    label.set_fontsize(18)
# ax.set_title('Pearson Correlation Matrix', fontfamily='Times New Roman', fontsize=20, fontweight='bold', pad=10)
# 轴标签字体
ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Times New Roman', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Times New Roman', fontsize=18, rotation=0)
ax.tick_params(axis='both', length=0)

plt.tight_layout()
plt.savefig('第二年皮尔逊矩阵.png', dpi=600, bbox_inches='tight')
plt.show()