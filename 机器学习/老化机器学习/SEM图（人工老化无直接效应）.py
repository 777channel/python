import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 直接输入分析结果 ====================

# a路径系数：自变量 → ΔpH (从Model 1)
a_coef = {
    'pH-S': -3.28,
    'Cd-C': -0.183,
    'BR': 0.551,
    'DW-D': 0.446,
    'C-W': 16.34,
    'FT-D': -0.118
}

# a路径p值
a_pvalue = {
    'pH-S': 0.0001,
    'Cd-C': 0.052,
    'BR': 0.0001,
    'DW-D': 0.0001,
    'C-W': 0.0001,
    'FT-D': 0.136
}

# b路径系数：ΔpH → ΔCd (从Model 3)
b_coef = 0.029
b_pvalue = 0.021

# 自变量之间的相关系数
correlations = {
    ('pH-S', 'Cd-C'): (0.35, 0.02),      # (系数, p值)
    ('DW-D', 'FT-D'): (0.43, 0.0001),
    ('BR', 'C-W'): (0.20, 0.15)
}

# 间接效应和中介类型
mediation_info = {
    'pH-S': {'indirect': -0.095, 'agency': 0.172, 'type': 'No Mediation'},
    'Cd-C': {'indirect': -0.01, 'agency': -0.178, 'type': 'Fully Mediated'},
    'BR': {'indirect': 0.036, 'agency': 0.502, 'type': 'Partially Mediated'},
    'DW-D': {'indirect': 0.013, 'agency': -0.86, 'type': 'Fully Mediated'},
    'C-W': {'indirect': 0.474, 'agency': 0.296, 'type': 'Partially Mediated'},
    'FT-D': {'indirect': -0.003, 'agency': -7.295, 'type': 'Competitive Mediation'}
}

# ==================== 辅助函数 ====================

def get_sig_marker(p):
    """根据p值返回显著性标记"""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def draw_node(ax, x, y, width, height, text, color, fontsize=16):
    """绘制矩形节点"""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1", 
        edgecolor='black', 
        facecolor=color,
        linewidth=2.5,
        zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', zorder=4,
            family='Times New Roman')

def draw_arrow(ax, start, end, coef, pvalue, offset=(0, 0), curvature=0, color='black'):
    """绘制带系数标注的箭头（统一线宽2.0）"""
    x1, y1 = start
    x2, y2 = end
    
    sig = get_sig_marker(pvalue)
    coef_text = f'{coef:.2f}{sig}' if abs(coef) >= 0.01 else f'{coef:.3f}{sig}'
    
    if curvature == 0:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            mutation_scale=25,
            linewidth=2.0,
            color=color,
            zorder=2
        )
        ax.add_patch(arrow)
        mid_x = (x1 + x2) / 2 + offset[0]
        mid_y = (y1 + y2) / 2 + offset[1]
    else:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', 
            mutation_scale=25,
            connectionstyle=f"arc3,rad={curvature}",
            linewidth=2.0,
            color=color,
            zorder=2
        )
        ax.add_patch(arrow)
        mid_x = (x1 + x2) / 2 + curvature * (y2 - y1) * 0.3 + offset[0]
        mid_y = (y1 + y2) / 2 - curvature * (x2 - x1) * 0.3 + offset[1]
    
    # 纯文本标注，无边框
    ax.text(mid_x, mid_y, coef_text, ha='center', va='center',
            fontsize=12, fontweight='bold', zorder=5,
            family='Times New Roman', color='black')

def draw_correlation(ax, start, end, coef, pvalue, offset=(0, 0)):
    """绘制双向箭头表示相关性"""
    x1, y1 = start
    x2, y2 = end
    
    sig = get_sig_marker(pvalue)
    coef_text = f'{coef:.2f}{sig}'
    
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='<->', 
        mutation_scale=20,
        linewidth=2.0,
        color='#666',
        zorder=1
    )
    ax.add_patch(arrow)
    
    mid_x = (x1 + x2) / 2 + offset[0]
    mid_y = (y1 + y2) / 2 + offset[1]
    
    ax.text(mid_x, mid_y, coef_text, ha='center', va='center',
            fontsize=10, fontweight='bold', zorder=5, color='#666',
            family='Times New Roman')

# ==================== 创建图形 ====================

fig, ax = plt.subplots(1, 1, figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# 定义颜色
color_predictor = '#ffb3b3'
color_mediator = '#ffd699'
color_outcome = '#99d6ff'

# 定义节点位置和大小
node_width = 2.2
node_height = 1.2

# 自变量节点位置
predictors = {
    'pH-S': (1.5, 12),
    'Cd-C': (1.5, 10),
    'BR': (1.5, 8),
    'DW-D': (1.5, 6),
    'C-W': (1.5, 4),
    'FT-D': (1.5, 2)
}

# 中介变量位置
mediator_pos = (9, 7)

# 因变量位置
outcome_pos = (16.5, 7)

# ==================== 绘制自变量之间的相关性（先画，在底层）====================

# DW-D ↔ FT-D
draw_correlation(ax, 
                (predictors['DW-D'][0], predictors['DW-D'][1]), 
                (predictors['FT-D'][0], predictors['FT-D'][1]), 
                correlations[('DW-D', 'FT-D')][0], 
                correlations[('DW-D', 'FT-D')][1], 
                offset=(-0.6, 0))

# pH-S ↔ Cd-C
draw_correlation(ax, 
                (predictors['pH-S'][0], predictors['pH-S'][1]), 
                (predictors['Cd-C'][0], predictors['Cd-C'][1]), 
                correlations[('pH-S', 'Cd-C')][0], 
                correlations[('pH-S', 'Cd-C')][1], 
                offset=(-0.6, 0))

# BR ↔ C-W
draw_correlation(ax, 
                (predictors['BR'][0], predictors['BR'][1]), 
                (predictors['C-W'][0], predictors['C-W'][1]), 
                correlations[('BR', 'C-W')][0], 
                correlations[('BR', 'C-W')][1], 
                offset=(-0.6, 0))

# ==================== 绘制所有节点 ====================

# 绘制自变量
for name, (x, y) in predictors.items():
    draw_node(ax, x, y, node_width, node_height, name, color_predictor, fontsize=17)

# 绘制中介变量
draw_node(ax, mediator_pos[0], mediator_pos[1], 
          node_width * 1.2, node_height * 1.1, 'ΔpH', color_mediator, fontsize=18)

# 绘制因变量
draw_node(ax, outcome_pos[0], outcome_pos[1], 
          node_width * 1.2, node_height * 1.1, 'ΔCd', color_outcome, fontsize=18)

# ==================== 绘制间接路径：X → ΔpH（a路径）====================

# pH-S → ΔpH
draw_arrow(ax, 
          (predictors['pH-S'][0] + node_width/2, predictors['pH-S'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1] + node_height*0.3),
          a_coef['pH-S'], a_pvalue['pH-S'], 
          offset=(-1.5, 0.5), curvature=0.15)

# Cd-C → ΔpH
draw_arrow(ax, 
          (predictors['Cd-C'][0] + node_width/2, predictors['Cd-C'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1] + node_height*0.15),
          a_coef['Cd-C'], a_pvalue['Cd-C'], 
          offset=(-1, 0.3), curvature=0.08)

# BR → ΔpH
draw_arrow(ax, 
          (predictors['BR'][0] + node_width/2, predictors['BR'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1]),
          a_coef['BR'], a_pvalue['BR'], 
          offset=(0, 0.3))

# DW-D → ΔpH
draw_arrow(ax, 
          (predictors['DW-D'][0] + node_width/2, predictors['DW-D'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1] - node_height*0.15),
          a_coef['DW-D'], a_pvalue['DW-D'], 
          offset=(0, -0.3))

# C-W → ΔpH
draw_arrow(ax, 
          (predictors['C-W'][0] + node_width/2, predictors['C-W'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1] - node_height*0.3),
          a_coef['C-W'], a_pvalue['C-W'], 
          offset=(-1, -0.5), curvature=-0.15)

# FT-D → ΔpH
draw_arrow(ax, 
          (predictors['FT-D'][0] + node_width/2, predictors['FT-D'][1]),
          (mediator_pos[0] - node_width*0.6, mediator_pos[1] - node_height*0.4),
          a_coef['FT-D'], a_pvalue['FT-D'], 
          offset=(-1.5, -0.8), curvature=-0.2)

# ==================== ΔpH → ΔCd（b路径）====================

draw_arrow(ax, 
          (mediator_pos[0] + node_width*0.6, mediator_pos[1]),
          (outcome_pos[0] - node_width*0.6, outcome_pos[1]),
          b_coef, b_pvalue, 
          offset=(0, 0.3))

# ==================== 添加标题和注释 ====================

ax.text(10, 13.5, 'Structural Equation Model: Mediation Effects of ΔpH on Cd Transformation', 
        ha='center', va='top', fontsize=20, fontweight='bold', family='Times New Roman')

ax.text(1, 13.8, 'Standardized coefficients. *p<0.05, **p<0.01, ***p<0.001', 
        ha='left', va='top', fontsize=11, color='#666', family='Times New Roman')

ax.text(1, 13.4, 'Double-headed arrows (↔) indicate correlations between predictors', 
        ha='left', va='top', fontsize=11, color='#666', family='Times New Roman')

# ==================== 添加中介类型分类框 ====================

box_x, box_y = 17.5, 11
box_width, box_height = 2.3, 3.2

classification_box = FancyBboxPatch(
    (box_x, box_y), box_width, box_height,
    boxstyle="round,pad=0.1", 
    edgecolor='black', 
    facecolor='#f0f0f0',
    linewidth=2,
    zorder=3
)
# ax.add_patch(classification_box)

# # 标题
# ax.text(box_x + box_width/2, box_y + box_height - 0.3, 'Mediation Types', 
#         ha='center', va='top', fontsize=13, fontweight='bold', family='Times New Roman')

# # 内容
# y_start = box_y + box_height - 0.8

# ax.text(box_x + 0.15, y_start, 'Fully Mediated:', 
#         ha='left', va='top', fontsize=11, fontweight='bold', color='#4caf50', 
#         family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 0.25, 'DW-D (AS=-0.86)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 0.45, 'Cd-C (AS=-0.18)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')

# ax.text(box_x + 0.15, y_start - 0.85, 'Partially Mediated:', 
#         ha='left', va='top', fontsize=11, fontweight='bold', color='#ff9800', 
#         family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 1.1, 'C-W (AS=0.30)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 1.3, 'BR (AS=0.50)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')

# ax.text(box_x + 0.15, y_start - 1.7, 'No Mediation:', 
#         ha='left', va='top', fontsize=11, fontweight='bold', color='#f44336', 
#         family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 1.95, 'pH-S (AS=0.17)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')

# ax.text(box_x + 0.15, y_start - 2.35, 'Competitive:', 
#         ha='left', va='top', fontsize=11, fontweight='bold', color='#9c27b0', 
#         family='Times New Roman')
# ax.text(box_x + 0.2, y_start - 2.6, 'FT-D (AS=-7.30)', 
#         ha='left', va='top', fontsize=9, family='Times New Roman')

# ax.text(box_x + 0.15, y_start - 2.9, 'AS = Agency Share', 
#         ha='left', va='top', fontsize=8, color='#666', style='italic', 
#         family='Times New Roman')

# ==================== 添加图例 ====================

# legend_elements = [
#     mpatches.Rectangle((0, 0), 1, 1, facecolor=color_predictor, 
#                        edgecolor='black', linewidth=2, label='Predictor Variables'),
#     mpatches.Rectangle((0, 0), 1, 1, facecolor=color_mediator, 
#                        edgecolor='black', linewidth=2, label='Mediator (ΔpH)'),
#     mpatches.Rectangle((0, 0), 1, 1, facecolor=color_outcome, 
#                        edgecolor='black', linewidth=2, label='Outcome (ΔCd)'),
#     mpatches.FancyArrow(0, 0, 1, 0, width=0.3, 
#                         facecolor='black', edgecolor='none', label='Indirect paths (a×b)'),
#     mpatches.FancyArrow(0, 0, 1, 0, width=0.2, 
#                         facecolor='#666', edgecolor='none', label='Correlations'),
# ]

# ax.legend(handles=legend_elements, loc='lower left', 
#           bbox_to_anchor=(0.05, 0.02), fontsize=11, frameon=True, 
#           fancybox=True, shadow=False, ncol=5, 
#           prop={'family': 'Times New Roman'})

# ==================== 保存为PDF ====================

plt.tight_layout()
plt.savefig('SEM_Mediation_Analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
print("图形已保存为 'SEM_Mediation_Analysis.pdf'")

# 可选：显示图形
# plt.show()

# ==================== 打印中介效应结果 ====================

print("\n========== 中介效应结果 ==========")
print(f"{'Pathway':<25} {'a coef':<10} {'b coef':<10} {'Indirect':<12} {'Agency':<10} {'Type':<20}")
print("-" * 95)

for var in ['pH-S', 'Cd-C', 'BR', 'DW-D', 'C-W', 'FT-D']:
    pathway = f"{var} → ΔpH → ΔCd"
    a = a_coef[var]
    indirect = mediation_info[var]['indirect']
    agency = mediation_info[var]['agency']
    med_type = mediation_info[var]['type']
    
    print(f"{pathway:<25} {a:<10.3f} {b_coef:<10.3f} {indirect:<12.3f} {agency:<10.3f} {med_type:<20}")

print("\n图形生成完成！")