import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 核心数据输入 (根据 image_18b34a 更新) ====================
# 特征筛选：C-L-D, C-H-D, BET, BR, CEC, T-A
# a路径系数：自变量 → ΔpH (由 Indirect/0.03 推算近似)
a_coef = {
    'C-L-D': -17.50, 'C-H-D': 8.33, 'BET': -10.06, 
    'BR': 7.70, 'CEC': 0.60, 'T-A': -0.10
}
a_pvalue = {
    'C-L-D': 0.001, 'C-H-D': 0.001, 'BET': 0.01, 
    'BR': 0.001, 'CEC': 0.05, 'T-A': 0.05
}

# b路径系数：ΔpH → ΔCd
b_coef = 0.030
b_pvalue = 0.021

# c'路径系数：直接效应 (自变量 → ΔCd)
c_prime_coef = {
    'C-L-D': 0.184, 'C-H-D': -0.06, 'BET': -0.34, 
    'BR': 0.038, 'CEC': 0.14, 'T-A': -0.433
}
c_prime_pvalue = {
    'C-L-D': 0.01, 'C-H-D': 0.25, 'BET': 0.01, 
    'BR': 0.35, 'CEC': 0.01, 'T-A': 0.001
}

# ==================== 绘图辅助函数 ====================
def get_sig_marker(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return ''

def draw_node(ax, x, y, width, height, text, color):
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                        boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=2, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=24, fontweight='bold', zorder=4)

def draw_path(ax, start, end, coef, p, curv=0, style='solid', color='black', offset_text=(0,0)):
    sig = get_sig_marker(p)
    txt = f"{coef:.2f}{sig}"
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                            connectionstyle=f"arc3,rad={curv}", linewidth=2,
                            color=color, linestyle=style, zorder=2)
    ax.add_patch(arrow)
    mid_x = (start[0] + end[0]) / 2 + offset_text[0]
    mid_y = (start[1] + end[1]) / 2 + offset_text[1]
    ax.text(mid_x, mid_y, txt, ha='center', va='center', fontsize=24, fontweight='bold', color=color)

# ==================== 创建并绘制图形 ====================
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 20); ax.set_ylim(0, 14); ax.axis('off')

# 节点设置
nodes_x = 2
nodes_y = [12, 10, 8, 6, 4, 2]
names = ['C-L-D', 'C-H-D', 'BET', 'BR', 'CEC', 'T-A']
c_pre, c_med, c_out = '#ffb3b3', '#ffd699', '#99d6ff'

# 1. 绘制节点
for i, name in enumerate(names):
    draw_node(ax, nodes_x, nodes_y[i], 2.2, 1.1, name, c_pre)
draw_node(ax, 10, 7, 2.8, 1.3, 'ΔpH', c_med)
draw_node(ax, 17.5, 7, 2.8, 1.3, 'ΔCd', c_out)

# 2. 绘制路径 X -> ΔpH (a路径)
for i, name in enumerate(names):
    curv = 0.15 if nodes_y[i] > 7 else -0.15
    draw_path(ax, (nodes_x+1.1, nodes_y[i]), (8.6, 7), a_coef[name], a_pvalue[name], curv=curv)

# 3. 绘制路径 ΔpH -> ΔCd (b路径)
draw_path(ax, (11.4, 7), (16.1, 7), b_coef, b_pvalue, offset_text=(0, 0.4))

# 4. 绘制直接路径 X -> ΔCd (c'路径, 虚线)
for i, name in enumerate(names):
    curv = 0.35 if nodes_y[i] > 7 else -0.35
    draw_path(ax, (nodes_x+1.1, nodes_y[i]), (16.1, 7), 
              c_prime_coef[name], c_prime_pvalue[name], curv=curv, style='dashed', color='gray')

# ==================== 保存与展示 ====================
plt.tight_layout()
plt.savefig('SEM_Mediation_Analysis自然.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("成功！图形已保存为 'SEM_Mediation_Analysis自然.pdf'")
plt.show()