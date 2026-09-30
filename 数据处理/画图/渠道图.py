import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# --- 1. 字体设置 ---
try:
    font_prop = fm.FontProperties(family='SimSun')
    plt.rcParams['font.sans-serif'] = ['SimSun']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    font_prop = fm.FontProperties(family='sans-serif')
    plt.rcParams['font.sans-serif'] = ['sans-serif']
    print("SimSun font not found.")

# --- 2. 颜色定义 ---
color_background_main = '#E9F1E9'
color_plot_fill = '#FFFFE0'
color_plot_border = '#D2B48C'
color_irrigation_red = '#8B0000'
color_drainage_blue = '#ADD8E6'
color_dashed_box_center = '#000000'
color_text_default = '#000000'

# --- 3. 核心参数定义 ---
fig_width_inch = 12
fig_height_inch = 9

num_plot_rows = 3
num_plot_cols = 3

plot_width = 1.5
plot_height = 1.2
plot_border_linewidth = 0.02 # <<--- 添加这行定义 (可以调整粗细)

# --- 渠道线条粗细 ---
lw_main_irr = 0.08
lw_branch_irr = 0.05
lw_sub_branch_irr = 0.03
lw_drain = lw_branch_irr

# --- 间隙定义 ---
gap_plot_to_irr_h = 0.06
gap_plot_to_irr_v = 0.05
gap_between_irr_pair_h = 0.05
gap_between_irr_pair_v = 0.04
gap_irr_to_drain_h = 0.04
gap_irr_to_drain_v = 0.03
gap_main_irr_to_drain = 0.03

# --- 整体边距 ---
margin_left = 0.7
margin_bottom = 0.7
margin_right = 0.7
margin_top = 0.7

main_irr_x_center = margin_left + lw_main_irr / 2
drain_grid_x_start = main_irr_x_center + lw_main_irr / 2 + gap_main_irr_to_drain + lw_drain / 2 # Adjusted to include gap
current_y_base = margin_bottom

# --- 4. 创建画布 ---
fig, ax = plt.subplots(figsize=(fig_width_inch, fig_height_inch))
fig.patch.set_facecolor(color_background_main)
ax.set_facecolor(color_background_main)

# --- 5. 存储坐标 ---
plots_bl_coords = {}
irr_branches_y_centers = {}
irr_sub_branches_x_centers = {}
drain_branches_y_centers = []
drain_sub_branches_x_centers = []

# --- B. 计算蓝色排水斗渠的X坐标 ---
current_x = drain_grid_x_start # Start from the calculated drain_grid_x_start
for i in range(num_plot_cols + 1):
    drain_sub_branches_x_centers.append(current_x)
    if i < num_plot_cols:
        cell_width_for_drain = plot_width + 2*gap_plot_to_irr_v + \
                               2*lw_sub_branch_irr + gap_between_irr_pair_v + \
                               2*gap_irr_to_drain_v + lw_drain
        current_x += cell_width_for_drain

# --- C. 计算蓝色排水支渠的Y坐标 ---
current_y = margin_bottom + lw_drain / 2
for i in range(num_plot_rows + 1):
    drain_branches_y_centers.append(current_y)
    if i < num_plot_rows:
        cell_height_for_drain = plot_height + 2*gap_plot_to_irr_h + \
                                2*lw_branch_irr + gap_between_irr_pair_h + \
                                2*gap_irr_to_drain_h + lw_drain
        current_y += cell_height_for_drain

# --- D. 计算红色灌溉渠坐标 ---
# 水平红色支渠
for i in range(num_plot_rows + 1): # num_rows地块 -> num_rows+1 对红支渠
    # 每对红支渠位于两条蓝支渠的中间区域
    y_center_of_blue_gap = (drain_branches_y_centers[i] + drain_branches_y_centers[min(i + 1, num_plot_rows)]) / 2
    if i == num_plot_rows : #顶部的特殊处理，它在最后一条蓝线的下方
        y_center_of_blue_gap = drain_branches_y_centers[i] - lw_drain/2 - gap_irr_to_drain_h - gap_between_irr_pair_h/2 - lw_branch_irr
    elif i == 0: #底部的特殊处理，它在第一条蓝线的上方
        y_center_of_blue_gap = drain_branches_y_centers[i] + lw_drain/2 + gap_irr_to_drain_h + gap_between_irr_pair_h/2 + lw_branch_irr


    # 红支渠1 在蓝渠间隙中心线的下方 (或基于蓝线边缘)
    # y1 = drain_branches_y_centers[i] + lw_drain/2 + gap_irr_to_drain_h + lw_branch_irr/2
    # y2 = y1 + lw_branch_irr + gap_between_irr_pair_h
    # 更准确的：红支渠对应该在两条蓝支渠线定义的“单元格”内部
    if i < num_plot_rows: # 红支渠对在两条蓝支渠之间
        y_mid_between_blue = (drain_branches_y_centers[i] + drain_branches_y_centers[i+1]) / 2
        y1 = y_mid_between_blue - gap_between_irr_pair_h/2 - lw_branch_irr/2
        y2 = y_mid_between_blue + gap_between_irr_pair_h/2 + lw_branch_irr/2
    elif i == 0: # 最底部的一对红支渠（或者只有一条，根据模板图可能只有两条红线构成最底边界）
        y1 = drain_branches_y_centers[0] + lw_drain/2 + gap_irr_to_drain_h + lw_branch_irr/2
        y2 = y1 + lw_branch_irr + gap_between_irr_pair_h
    else: # i == num_plot_rows, 最顶部的一对红支渠
        y2 = drain_branches_y_centers[num_plot_rows] - lw_drain/2 - gap_irr_to_drain_h - lw_branch_irr/2
        y1 = y2 - lw_branch_irr - gap_between_irr_pair_h
    irr_branches_y_centers[i] = [y1,y2]

# 垂直红色斗渠
for i in range(num_plot_cols + 1):
    if i < num_plot_cols:
        x_mid_between_blue = (drain_sub_branches_x_centers[i] + drain_sub_branches_x_centers[i+1]) / 2
        x1 = x_mid_between_blue - gap_between_irr_pair_v/2 - lw_sub_branch_irr/2
        x2 = x_mid_between_blue + gap_between_irr_pair_v/2 + lw_sub_branch_irr/2
    elif i == 0:
        x1 = drain_sub_branches_x_centers[0] + lw_drain/2 + gap_irr_to_drain_v + lw_sub_branch_irr/2
        x2 = x1 + lw_sub_branch_irr + gap_between_irr_pair_v
    else: # i == num_plot_cols
        x2 = drain_sub_branches_x_centers[num_plot_cols] - lw_drain/2 - gap_irr_to_drain_v - lw_sub_branch_irr/2
        x1 = x2 - lw_sub_branch_irr - gap_between_irr_pair_v
    irr_sub_branches_x_centers[i] = [x1,x2]

# --- E. 计算地块坐标 ---
for r in range(num_plot_rows):
    lower_irr_branch_pair_top_y_edge = irr_branches_y_centers[r][1] + lw_branch_irr/2
    plot_y_bl = lower_irr_branch_pair_top_y_edge + gap_plot_to_irr_h
    for c in range(num_plot_cols):
        left_irr_sub_branch_pair_right_x_edge = irr_sub_branches_x_centers[c][1] + lw_sub_branch_irr/2
        plot_x_bl = left_irr_sub_branch_pair_right_x_edge + gap_plot_to_irr_v
        plots_bl_coords[(r,c)] = (plot_x_bl, plot_y_bl)

# --- 7. 开始绘制 ---
# 蓝色排水渠 (zorder=1)
min_x_drain_overall = drain_sub_branches_x_centers[0] - lw_drain/2
max_x_drain_overall = drain_sub_branches_x_centers[-1] + lw_drain/2
min_y_drain_overall = drain_branches_y_centers[0] - lw_drain/2
max_y_drain_overall = drain_branches_y_centers[-1] + lw_drain/2
for y_center in drain_branches_y_centers:
    ax.plot([min_x_drain_overall, max_x_drain_overall], [y_center, y_center], color=color_drainage_blue, linewidth=lw_drain, zorder=1, solid_capstyle='butt')
for x_center in drain_sub_branches_x_centers:
    ax.plot([x_center, x_center], [min_y_drain_overall, max_y_drain_overall], color=color_drainage_blue, linewidth=lw_drain, zorder=1, solid_capstyle='butt')

# 红色灌溉渠 (zorder=2, 3)
x_start_irr_branch = main_irr_x_center + lw_main_irr/2
x_end_irr_branch = irr_sub_branches_x_centers[num_plot_cols][0] - lw_sub_branch_irr/2 # End at the left line of the rightmost red pair
for r_pair_idx in range(num_plot_rows + 1):
    y1c, y2c = irr_branches_y_centers[r_pair_idx]
    ax.plot([x_start_irr_branch, x_end_irr_branch], [y1c, y1c], color=color_irrigation_red, linewidth=lw_branch_irr, zorder=2, solid_capstyle='butt')
    ax.plot([x_start_irr_branch, x_end_irr_branch], [y2c, y2c], color=color_irrigation_red, linewidth=lw_branch_irr, zorder=2, solid_capstyle='butt')
    # Simplified labels for now
    label_y_pos = (y1c + y2c) / 2
    if r_pair_idx == 0: label_y_pos = y1c - 0.1; txt = "支渠1"
    elif r_pair_idx == 1: label_y_pos = y1c - 0.1; txt = "支渠3" # Based on template visual
    elif r_pair_idx == 2: label_y_pos = y1c - 0.1; txt = "支渠5"
    else: txt = "" # No label for the topmost pair edge lines in template
    if txt:
        ax.text(main_irr_x_center - lw_main_irr - 0.1, label_y_pos, txt, ha='right', va='center', fontsize=7, fontproperties=font_prop, zorder=10)
        if txt == "支渠3": # Add 支渠2 and 支渠4 relative to 支渠3's blue line
             ax.text(main_irr_x_center - lw_main_irr - 0.1, drain_branches_y_centers[0] + (drain_branches_y_centers[1]-drain_branches_y_centers[0])/2 - gap_between_irr_pair_h , "支渠2", ha='right', va='center', fontsize=7, fontproperties=font_prop, zorder=10)
             ax.text(main_irr_x_center - lw_main_irr - 0.1, drain_branches_y_centers[1] + (drain_branches_y_centers[2]-drain_branches_y_centers[1])/2 - gap_between_irr_pair_h, "支渠4", ha='right', va='center', fontsize=7, fontproperties=font_prop, zorder=10)
        if txt == "支渠5":
             ax.text(main_irr_x_center - lw_main_irr - 0.1, drain_branches_y_centers[2] + (drain_branches_y_centers[3]-drain_branches_y_centers[2])/2 - gap_between_irr_pair_h, "支渠6", ha='right', va='center', fontsize=7, fontproperties=font_prop, zorder=10)


for c_pair_idx in range(num_plot_cols + 1): # Draw all red douqu pairs
    x1c, x2c = irr_sub_branches_x_centers[c_pair_idx]
    for r_plot_row_idx in range(num_plot_rows):
        y_bottom_edge_of_upper_irr_pair = irr_branches_y_centers[r_plot_row_idx+1][0] - lw_branch_irr/2
        y_top_edge_of_lower_irr_pair = irr_branches_y_centers[r_plot_row_idx][1] + lw_branch_irr/2
        ax.plot([x1c,x1c], [y_top_edge_of_lower_irr_pair+0.01, y_bottom_edge_of_upper_irr_pair-0.01], color=color_irrigation_red, linewidth=lw_sub_branch_irr, zorder=3, solid_capstyle='butt')
        ax.plot([x2c,x2c], [y_top_edge_of_lower_irr_pair+0.01, y_bottom_edge_of_upper_irr_pair-0.01], color=color_irrigation_red, linewidth=lw_sub_branch_irr, zorder=3, solid_capstyle='butt')

y_start_main_irr = min_y_drain_overall
y_end_main_irr = max_y_drain_overall
ax.plot([main_irr_x_center, main_irr_x_center], [y_start_main_irr, y_end_main_irr], color=color_irrigation_red, linewidth=lw_main_irr, zorder=3, solid_capstyle='butt')
ax.text(main_irr_x_center - lw_main_irr*1.5, (y_start_main_irr+y_end_main_irr)/2, '干\n渠',
        ha='center', va='center', fontsize=10, fontproperties=font_prop, rotation='vertical', zorder=10)

# --- 8. 绘制地块 ---
for r in range(num_plot_rows):
    for c in range(num_plot_cols):
        px, py = plots_bl_coords[(r,c)]
        ax.add_patch(patches.Rectangle((px,py), plot_width, plot_height, facecolor=color_plot_fill, edgecolor=color_plot_border, linewidth=plot_border_linewidth, zorder=4))

# --- 9. 中心标记 ---
center_r, center_c = 1, 1
if (center_r, center_c) in plots_bl_coords:
    plot_x_bl, plot_y_bl = plots_bl_coords[(center_r, center_c)]
    dash_box_x = plot_x_bl - 0.05; dash_box_y = plot_y_bl - 0.05
    dash_box_w = plot_width + 0.1; dash_box_h = plot_height + 0.1
    ax.add_patch(patches.Rectangle((dash_box_x, dash_box_y), dash_box_w, dash_box_h,
                                   facecolor='none', edgecolor=color_dashed_box_center,
                                   linewidth=0.7, linestyle='--', zorder=6))
    ax.text(plot_x_bl + plot_width / 2, plot_y_bl + plot_height / 2, '井灌区',
            ha='center', va='center', fontsize=9, fontproperties=font_prop, weight='bold', zorder=7)

# --- 10. 坐标轴和最终调整 ---
ax.set_xlim(margin_left -0.3, max_x_drain_overall + 0.3 + margin_right) # Adjusted xlim
ax.set_ylim(margin_bottom -0.3, max_y_drain_overall + 0.3 + margin_top) # Adjusted ylim
ax.set_aspect('equal', adjustable='box'); ax.axis('off'); plt.tight_layout(pad=0.05)

# 修改：只保存PDF版本
plt.savefig('灌溉排水网格图_最终模板_fixed.pdf', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
# plt.savefig('灌溉排水网格图_新模板.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor()) # 注释掉PNG保存
print("PDF图表已保存。")
plt.show()