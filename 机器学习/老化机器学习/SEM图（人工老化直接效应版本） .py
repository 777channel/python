import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 基于 image_0d55e5 的数据 ====================
# a路径系数：自变量 → ΔpH (基于 Indirect/0.03 的推算或对应比例)
a_coef = {
    'C-T': -14.53, 'C-W': 15.80, 'CEC': -1.63, 
    'BR': 1.20, 'Cd-C': -0.33, 'DW-D': 0.43
}
a_pvalue = {
    'C-T': 0.001, 'C-W': 0.001, 'CEC': 0.05, 
    'BR': 0.05, 'Cd-C': 0.05, 'DW-D': 0.05
}

# b路径系数：ΔpH → ΔCd (假设模型约为 0.03)
b_coef = 0.030
b_pvalue = 0.021

# c'路径系数：直接效应
c_prime_coef = {
    'C-T': -1.83, 'C-W': 1.60, 'CEC': -0.23, 
    'BR': 0.07, 'Cd-C': 0.06, 'DW-D': -0.02
}
c_prime_pvalue = {
    'C-T': 0.001, 'C-W': 0.001, 'CEC': 0.01, 
    'BR': 0.05, 'Cd-C': 0.297, 'DW-D': 0.789
}

# (绘图逻辑延用您提供的高清大字号风格)
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 20); ax.set_ylim(0, 14); ax.axis('off')

# 节点
names = ['C-T', 'C-W', 'CEC', 'BR', 'Cd-C', 'DW-D']
y_pos = [12, 10, 8, 6, 4, 2]
c_pre, c_med, c_out = '#ffb3b3', '#ffd699', '#99d6ff'

def draw_node(ax, x, y, width, height, text, color):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                        boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=24, fontweight='bold')

def draw_path(ax, start, end, coef, p, curv=0, style='solid', color='black'):
    sig = '***' if p < 0.001 else '*' if p < 0.05 else ''
    txt = f"{coef:.2f}{sig}"
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                            connectionstyle=f"arc3,rad={curv}", linewidth=2, color=color, linestyle=style)
    ax.add_patch(arrow)
    mid_x, mid_y = (start[0]+end[0])/2, (start[1]+end[1])/2 + (0.3 if curv>0 else -0.3)
    ax.text(mid_x, mid_y, txt, ha='center', va='center', fontsize=20, fontweight='bold', color=color)

# 绘图
for i, name in enumerate(names):
    draw_node(ax, 2, y_pos[i], 2.2, 1.1, name, c_pre)
    draw_path(ax, (3.1, y_pos[i]), (8.6, 7), a_coef[name], a_pvalue[name], curv=0.15 if y_pos[i]>7 else -0.15)
    draw_path(ax, (3.1, y_pos[i]), (16.1, 7), c_prime_coef[name], c_prime_pvalue[name], curv=0.35 if y_pos[i]>7 else -0.35, style='dashed', color='gray')

draw_node(ax, 10, 7, 2.8, 1.3, 'ΔpH', c_med)
draw_node(ax, 17.5, 7, 2.8, 1.3, 'ΔCd', c_out)
draw_path(ax, (11.4, 7), (16.1, 7), b_coef, b_pvalue)

plt.tight_layout()
plt.savefig('SEM_Mediation_Dataset1.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()