import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['font.size'] = 24
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

labels = ['DW-D', 'W', 'Cd', 'As-absorption']
corr = np.array([
    [ 1.00, -0.00,  0.00,  0.33],
    [-0.00,  1.00,  0.00,  0.49],
    [ 0.00,  0.00,  1.00, -0.69],
    [ 0.33,  0.49, -0.69,  1.00]
])

# ① 方法4：阈值归零，消除 -0.000...001 这类浮点误差
corr = np.where(np.abs(corr) < 1e-9, 0.0, corr)

cmap = LinearSegmentedColormap.from_list(
    'custom', ['#1070a8', '#e8e8e8', '#2e8b2e'], N=256
)

fig, ax = plt.subplots(figsize=(12, 10))

masked = np.ma.array(corr, mask=np.eye(4, dtype=bool))
im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

for i in range(4):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
                                color='#8c8c8c', zorder=2))
    ax.text(i, i, '1.00', ha='center', va='center',
            fontsize=22, fontweight='bold', color='black', zorder=3)
# for i in range(4):
#     ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1,
#                                 color='white', zorder=2))
# ② 方法3：格式化时强制零值显示为 0.00
def fmt(v):
    v = round(v, 10)             # 消除残余浮点误差
    return f'{0.0 if v == 0.0 else v:.2f}'

for i in range(4):
    for j in range(4):
        if i == j:
            continue
        val = corr[i, j]
        color = 'black'
        ax.text(j, i, fmt(val), ha='center', va='center',
                fontsize=22, fontweight='bold', color=color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(labels, fontsize=22, fontweight='bold')
ax.set_yticklabels(labels, fontsize=22, fontweight='bold', rotation=90, va='center')
ax.set_title('Pearson Correlation Matrix', fontsize=24, fontweight='bold', pad=10)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('corr_matrix.pdf', dpi=600, bbox_inches='tight')
plt.show()