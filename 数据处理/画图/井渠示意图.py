import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import numpy as np

# --- 1. 字体设置 (全程使用宋体) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimSun'] # 指定使用宋体
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("未找到宋体，可能无法正确显示中文，请安装SimSun字体或指定其他中文字体。")
    plt.rcParams['font.sans-serif'] = ['sans-serif']

# --- 2. 核心参数定义 ---
# 1万亩地块，边长2500m，6x6网格(5个间距)，井间距 = 2500/5 = 500m
TOTAL_SIDE_LENGTH_M = 2500
REAL_SPACING_METERS = 500

grid_size_x = 6
grid_size_y = 6
step_size = 3.5
well_radius = 0.4
pump_size = 0.5

# --- 颜色定义 ---
color_background = '#FFFFFF'
color_well = '#2980B9'
color_pump = '#7D3C98'
color_group_box = '#D32F2F'
color_dim_line = '#1A5276'
color_schematic_line = '#2C3E50' # 详图框和引导线的颜色 **<-- 这里是修正点**
color_schematic_fill = '#F7F9F9'
color_solar_panel = '#2C3E50'
color_wind_turbine = '#5D6D7E'
color_building = '#BDBDBD'
color_inverter = '#16A085'
color_battery = '#3498DB'
color_power_line = '#F39C12'
color_component_fill = '#FFFFFF'
color_hybrid_icon_solar = '#FFC300'
color_hybrid_icon_wind = '#5D6D7E'


# --- 3. 辅助函数 ---
def draw_turbine_icon(ax, cx, cy, scale=1.0):
    mast_height = 2.5 * scale
    blade_len = 0.8 * scale
    hub_radius = 0.1 * scale
    ax.plot([cx, cx], [cy, cy + mast_height], color=color_wind_turbine, linewidth=4*scale, zorder=11, solid_capstyle='round')
    ax.add_patch(patches.Circle((cx, cy + mast_height), hub_radius, color=color_wind_turbine, zorder=12))
    for angle in [90, 210, 330]:
        rad = np.deg2rad(angle)
        end_x = cx + blade_len * np.cos(rad)
        end_y = (cy + mast_height) + blade_len * np.sin(rad)
        ax.plot([cx, end_x], [cy + mast_height, end_y], color=color_wind_turbine, linewidth=3*scale, zorder=12)

def draw_cad_dimension(ax, p1, p2, offset, text, is_vertical=False):
    if not is_vertical:
        start_point, end_point = (p1, offset), (p2, offset)
        text_pos = ((p1 + p2) / 2, offset)
        rotation, va = 0, 'center'
    else:
        start_point, end_point = (offset, p1), (offset, p2)
        text_pos = (offset, (p1 + p2) / 2)
        rotation, va = 90, 'center'
    arrow = patches.FancyArrowPatch(start_point, end_point, arrowstyle='<|-|>', mutation_scale=10, color=color_dim_line, linewidth=1.2, zorder=20)
    ax.add_patch(arrow)
    ax.text(text_pos[0], text_pos[1], text, ha='center', va=va, rotation=rotation, fontsize=11, color=color_dim_line, weight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.8), zorder=21)

def draw_small_hybrid_icon(ax, cx, cy, scale=0.4):
    ax.add_patch(patches.Rectangle((cx - 0.7*scale, cy), 0.6*scale, 0.6*scale, facecolor=color_hybrid_icon_solar, zorder=5))
    ax.plot([cx + 0.3*scale, cx + 0.3*scale], [cy, cy + 1.2*scale], color=color_hybrid_icon_wind, linewidth=2*scale, zorder=5)
    for angle in [90, 210, 330]:
        rad = np.deg2rad(angle)
        end_x = (cx + 0.3*scale) + 0.4*scale * np.cos(rad)
        end_y = (cy + 1.2*scale) + 0.4*scale * np.sin(rad)
        ax.plot([cx + 0.3*scale, end_x], [cy + 1.2*scale, end_y], color=color_hybrid_icon_wind, linewidth=1.5*scale, zorder=5)

# --- 4. 创建画布与坐标系 ---
fig, ax = plt.subplots(figsize=(22, 18))
fig.patch.set_facecolor('white')
ax.set_facecolor(color_background)

# --- 5. 绘制井群主体 ---
start_coord_x = 4.0
start_coord_y = 4.0
for i in range(grid_size_x):
    for j in range(grid_size_y):
        well_x = start_coord_x + i * step_size
        well_y = start_coord_y + j * step_size
        ax.add_patch(patches.Circle((well_x, well_y), well_radius, facecolor=color_well, zorder=3))
        ax.add_patch(patches.Rectangle((well_x + well_radius, well_y - pump_size/2), pump_size, pump_size, facecolor=color_pump, zorder=4))
        if i % 2 == 0 and j % 2 == 0:
            box_x, box_y = well_x - step_size*0.5, well_y - step_size*0.5
            box_size = step_size * 2
            ax.add_patch(patches.Rectangle((box_x, box_y), box_size, box_size, facecolor='none', edgecolor=color_group_box, linewidth=2, linestyle='--', zorder=2))
            draw_small_hybrid_icon(ax, box_x + box_size/2, box_y + box_size/2)

# --- 6. 绘制尺寸标注 ---
draw_cad_dimension(ax, start_coord_x, start_coord_x + step_size, start_coord_y - 1.5, f'{REAL_SPACING_METERS} m')
draw_cad_dimension(ax, start_coord_y, start_coord_y + step_size, start_coord_x - 1.5, f'{REAL_SPACING_METERS} m', is_vertical=True)
total_start_x, total_end_x = start_coord_x - well_radius, start_coord_x + (grid_size_x - 1) * step_size + well_radius
total_start_y, total_end_y = start_coord_y - well_radius, start_coord_y + (grid_size_y - 1) * step_size + well_radius
draw_cad_dimension(ax, total_start_x, total_end_x, total_end_y + 1.5, f'总边长: {TOTAL_SIDE_LENGTH_M} m')

# --- 7. 绘制引导线和详图框 ---
source_group_i, source_group_j = 2, 2
source_center_x = start_coord_x + (source_group_i + 0.5) * step_size
source_center_y = start_coord_y + (source_group_j + 0.5) * step_size
detail_box_x = start_coord_x + grid_size_x * step_size + 1.5
detail_box_y = start_coord_y - 1.0
detail_box_w, detail_box_h = 12.0, 12.0
leader_line = patches.FancyArrowPatch((source_center_x, source_center_y), (detail_box_x, detail_box_y + detail_box_h * 0.7), connectionstyle="arc3,rad=-0.2", color=color_schematic_line, linewidth=2, arrowstyle="->,head_length=10,head_width=5")
ax.add_patch(leader_line)
ax.add_patch(patches.Rectangle((detail_box_x, detail_box_y), detail_box_w, detail_box_h, facecolor=color_schematic_fill, edgecolor=color_schematic_line, linewidth=2))
ax.text(detail_box_x + detail_box_w/2, detail_box_y + detail_box_h + 0.4, '典型供电单元详图', ha='center', va='bottom', fontsize=16, weight='bold')

# --- 8. 在详图框内绘制详细设备 ---
bx, by = detail_box_x, detail_box_y
turbine_x, turbine_y = bx + 2.5, by + 7.5
draw_turbine_icon(ax, turbine_x, turbine_y, scale=1.4)
ax.text(turbine_x, turbine_y - 0.5, '风力发电', ha='center', va='top', fontsize=12)
panel_x, panel_y = bx + 6.5, by + 8.5
panel_w, panel_h, panel_gap_x, panel_gap_y = 0.8, 0.5, 0.2, 0.3
for row in range(4):
    for col in range(6):
        ax.add_patch(patches.Rectangle((panel_x + col * (panel_w + panel_gap_x), panel_y + row * (panel_h + panel_gap_y)), panel_w, panel_h, facecolor=color_solar_panel, zorder=10))
panel_array_center_x = panel_x + (6 * (panel_w + panel_gap_x) - panel_gap_x) / 2
ax.text(panel_array_center_x, panel_y - 0.5, '光伏发电 (24片)', ha='center', va='top', fontsize=12)
building_x, building_y = bx + 1.5, by + 3.0
building_w, building_h = 9.0, 4.0
ax.add_patch(patches.Rectangle((building_x, building_y), building_w, building_h, facecolor=color_building, ec='black', zorder=8))
ax.text(building_x + building_w/2, building_y + building_h + 0.3, '厂房 / 设备间', ha='center', va='bottom', fontsize=12)
battery_start_x, battery_start_y = building_x + 0.5, building_y + 0.5
b_size = 0.6
for row in range(4):
    for col in range(4):
        ax.add_patch(patches.Rectangle((battery_start_x + col * (b_size+0.2), battery_start_y + row * (b_size+0.2)), b_size, b_size, facecolor=color_battery, ec='white', zorder=9))
ax.text(battery_start_x + 1.5, battery_start_y - 0.4, '蓄电池组 (16块)', ha='center', va='top', fontsize=10)
inverter_x, inverter_y = building_x + 5.0, building_y + 0.75
inverter_w, inverter_h = 3.5, 2.5
ax.add_patch(patches.Rectangle((inverter_x, inverter_y), inverter_w, inverter_h, facecolor=color_inverter, ec='black', zorder=9))
ax.text(inverter_x + inverter_w/2, inverter_y + inverter_h/2, '逆控一体机', ha='center', va='center', fontsize=11, color='white', weight='bold')
motor_x, motor_y = bx + 6.0, by + 1.0
ax.add_patch(patches.Circle((motor_x, motor_y), 0.8, facecolor=color_component_fill, ec=color_schematic_line, lw=2, zorder=10))
ax.text(motor_x, motor_y, 'M', ha='center', va='center', fontsize=20, weight='bold', zorder=11, fontfamily='serif')
ax.text(motor_x, motor_y - 1.2, '水泵电机负载', ha='center', va='top', fontsize=12)
arrow_props = dict(arrowstyle="->,head_length=8,head_width=5", lw=2.5, color=color_power_line, zorder=7)
ax.add_patch(patches.FancyArrowPatch((turbine_x, turbine_y), (inverter_x, inverter_y + 2.0), connectionstyle="arc3,rad=0.3", **arrow_props))
ax.add_patch(patches.FancyArrowPatch((panel_array_center_x, panel_y), (inverter_x + inverter_w, inverter_y + 1.5), connectionstyle="arc3,rad=-0.2", **arrow_props))
ax.add_patch(patches.FancyArrowPatch((inverter_x, inverter_y + 0.5), (battery_start_x + 4*(b_size+0.2), inverter_y + 0.5), arrowstyle="<->", connectionstyle="arc3,rad=0", lw=2.0, color=color_power_line, zorder=7))
ax.add_patch(patches.FancyArrowPatch((inverter_x + inverter_w/2, inverter_y), (motor_x, motor_y + 0.8), connectionstyle="arc3,rad=0.2", **arrow_props))

# --- 9. 添加标题和图例 ---
ax.set_title('1万亩井群规划及典型供电单元详图', fontsize=20, pad=20, weight='bold')
legend_patches = [
    patches.Patch(color=color_well, label='水井 (Well)'),
    patches.Patch(color=color_pump, label='水泵 (Water Pump)'),
    patches.Patch(facecolor='none', ec=color_group_box, lw=2, ls='--', label='四井一组控制区'),
    patches.Patch(color=color_hybrid_icon_solar, label='分布式供电标记'),
    patches.Patch(facecolor=color_schematic_fill, ec=color_schematic_line, lw=2, label='典型单元详图框'),
    patches.Patch(facecolor='none', ec=color_dim_line, lw=1.2, ls='-', label=f'尺寸标注')
]
ax.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(0.01, 0.01), fontsize=12, frameon=True, facecolor='white', framealpha=0.9)

# --- 10. 调整最终显示效果 ---
plot_limit_x = detail_box_x + detail_box_w + 1.0
plot_limit_y = start_coord_y + (grid_size_y - 1) * step_size + 3.0
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, plot_limit_x)
ax.set_ylim(0, plot_limit_y)
ax.set_xticks([])
ax.set_yticks([])
plt.grid(False)
plt.tight_layout()

# --- 11. 保存并显示图像 ---
pdf_filename = '1万亩井群最终规划详图.pdf'
plt.savefig(pdf_filename, format='pdf', dpi=300, bbox_inches='tight')
print(f"图像已成功保存为：{pdf_filename}")

plt.show()