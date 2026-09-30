import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# --- 1. 字体设置 (全程使用宋体) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimSun'] # 指定使用宋体
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    # 如果找不到宋体，则使用默认无衬线字体
    plt.rcParams['font.sans-serif'] = ['sans-serif']

# --- 2. 定义核心参数 ---
square_side = 2.5  # 井渠双灌区方块边长 (km)

# 核心数值计算 (用于标注)
area_black_sq = square_side**2  # 6.25 km^2
area_white_sq = area_black_sq / 3.0  # ~2.08 km^2

# 视觉尺寸定义 (为了美观)
douqu_side_visual = 4.5  # 将虚线框的视觉尺寸扩大到4.5km

# 颜色定义
color_background = '#98FB98'
color_well_canal = '#FFFACD'
color_douqu_border = '#E74C3C'
color_dim_line = '#1A5276'
color_text_on_light = '#34495E'

# --- 3. 创建画布与坐标系 ---
fig, ax = plt.subplots(figsize=(12, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor(color_background)

# --- 4. 定义3x3网格布局 ---
grid_spacing = 1.5
step_size = square_side + grid_spacing
start_coord = 2.0

# --- 5. 绘制3x3网格 ---
for i in range(3):  # 列
    for j in range(3):  # 行
        x = start_coord + i * step_size
        y = start_coord + j * step_size

        # 特殊处理中心方块 (i=1, j=1)
        if i == 1 and j == 1:
            center_x_grid = x + square_side / 2
            center_y_grid = y + square_side / 2
            douqu_x = center_x_grid - douqu_side_visual / 2
            douqu_y = center_y_grid - douqu_side_visual / 2
            center_square_x = center_x_grid - square_side / 2
            center_square_y = center_y_grid - square_side / 2

            # 绘制中心的井渠双灌区方块
            ax.add_patch(patches.Rectangle((center_square_x, center_square_y), square_side, square_side, facecolor=color_well_canal, edgecolor='#AAAAAA'))
            # 绘制扩大后的红色虚线框
            ax.add_patch(patches.Rectangle((douqu_x, douqu_y), douqu_side_visual, douqu_side_visual, facecolor='none', edgecolor=color_douqu_border, linewidth=2.5, linestyle='--'))
            
            # --- 6. 在框内添加面积标注 (已修改) ---
            ax.text(center_x_grid, center_y_grid, f'面积: {area_black_sq:.2f} km²',
                    ha='center', va='center', fontsize=11, color=color_text_on_light, weight='bold')
            ax.text(douqu_x + douqu_side_visual - 0.3, douqu_y + 0.3, f'面积: ~{area_white_sq:.2f} km²',
                    ha='right', va='bottom', fontsize=11, color=color_text_on_light, weight='bold')
        else:
            # 绘制其他8个常规方块
            ax.add_patch(patches.Rectangle((x, y), square_side, square_side, facecolor=color_well_canal, edgecolor='#AAAAAA'))

# --- 7. 添加外部尺寸标注 (已重新添加) ---
# 标注一个常规方块的边长
dim_y_pos1 = start_coord - 1.0
ax.arrow(start_coord, dim_y_pos1, square_side, 0, head_width=0.1, head_length=0.1, fc=color_dim_line, ec=color_dim_line, length_includes_head=True)
ax.arrow(start_coord + square_side, dim_y_pos1, -square_side, 0, head_width=0.1, head_length=0.1, fc=color_dim_line, ec=color_dim_line, length_includes_head=True)
ax.text(start_coord + square_side / 2, dim_y_pos1 - 0.4, '2.5 km', ha='center', va='center', fontsize=10, color=color_dim_line)

# 标注斗渠控制区的边长
center_x_pos_of_douqu = start_coord + step_size + square_side / 2
douqu_box_x_start_for_dim = center_x_pos_of_douqu - douqu_side_visual / 2
dim_y_pos3 = start_coord + 3*step_size - grid_spacing
ax.arrow(douqu_box_x_start_for_dim, dim_y_pos3, douqu_side_visual, 0, head_width=0.1, head_length=0.1, fc=color_dim_line, ec=color_dim_line, length_includes_head=True)
ax.arrow(douqu_box_x_start_for_dim + douqu_side_visual, dim_y_pos3, -douqu_side_visual, 0, head_width=0.1, head_length=0.1, fc=color_dim_line, ec=color_dim_line, length_includes_head=True)
ax.text(center_x_pos_of_douqu, dim_y_pos3 + 0.4, f'{douqu_side_visual:.1f} km', ha='center', va='center', fontsize=10, color=color_dim_line)

# --- 8. 添加标题、标签和图例 ---
ax.set_title('灌溉区分布示意图 (最终版)', fontsize=20, pad=20, weight='bold')
ax.set_xlabel('X 轴 / km (X-axis)', fontsize=14)
ax.set_ylabel('Y 轴 / km (Y-axis)', fontsize=14)

legend_patches = [
    patches.Patch(facecolor=color_well_canal, ec='#AAAAAA', label='井渠双灌区 (Well-Canal Zone)'),
    patches.Patch(facecolor=color_background, ec='#AAAAAA', label='纯渠灌区 (Canal-Only Zone)'),
    patches.Patch(color='none', ec=color_douqu_border, lw=2.5, ls='--', label='斗渠控制区 (数据按3:1比例标注)')
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.85)

# --- 9. 调整最终显示效果 ---
ax.set_aspect('equal', adjustable='box')
plot_limit_x = start_coord + 3 * step_size
plot_limit_y = start_coord + 3 * step_size + 1
ax.set_xlim(0, plot_limit_x)
ax.set_ylim(0, plot_limit_y)
plt.grid(False)
plt.tight_layout()

# --- 10. 显示图像 ---
plt.show()