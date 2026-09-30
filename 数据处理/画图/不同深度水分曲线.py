import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from scipy.interpolate import griddata, NearestNDInterpolator
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# 设置全局字体为宋体，并增大字号
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 28  # 增大全局字体大小

# 1. 数据准备（包含5-10月数据）
boundary_coords = [
    (108.47823143, 41.20293957), (108.47016335, 41.1880205),
    (108.55835438, 41.15516717), (108.64156723, 41.19583568),
    (108.67040634, 41.20862214), (108.64654541, 41.233413),
    (108.56363297, 41.21533726)  # 严格保持原始边界坐标
]

wells_data = [
    [219, 108.45872746, 41.16203699, 0.6339, 1.0361, 1.4647, 2.0740, 2.4387, 2.65],
    [234, 108.63428183, 41.21257039, 5.4450, 5.8922, 6.1653, 6.2793, 6.1120, 5.95],
    [236, 108.53601155, 41.22523243, 9.8456, 10.0828, 10.2856, 10.6183, 10.3217, 10.1],
    [237, 108.60379315, 41.20704091, 3.2500, 3.3175, 4.1783, 3.8842, 4.4233, 4.2],
    [238, 108.56188144, 41.15790088, 1.4436, 1.4427, 1.8290, 2.3423, 2.7713, 3.0],
    [239, 108.64128986, 41.17935376, 0.6339, 1.0361, 1.4647, 2.0740, 2.4387, 2.7]
]

sample_points = [
    ["区1", 108.530, 41.195],
    ["区2", 108.535, 41.180],
    ["区3", 108.555, 41.205],
    ["区4", 108.560, 41.190],
    ["区5", 108.625, 41.210]
]

# 2. 创建精确的研究区域边界（不进行平滑处理）
study_area = Polygon(boundary_coords)  # 原始边界
x_min, y_min, x_max, y_max = study_area.bounds
from matplotlib import gridspec
fig = plt.figure(figsize=(18, 9))
fig.suptitle('民生支渠地下水埋深等值线图 (5-10月)', fontsize=20, y=0.9)
# 创建更紧凑的网格
gs = gridspec.GridSpec(2, 3, figure=fig,
                     width_ratios=[1, 1, 1],    
                     height_ratios=[1, 1],     
                     left=0.01, right=0.99,      
                     bottom=0.08, top=0.92,      
                     wspace=-0.10, hspace=-0.2) 

axes = []
for i in range(6):
    ax = fig.add_subplot(gs[i//3, i%3])
    axes.append(ax)

plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.1, 
                   wspace=-0.2, hspace=-0.25)

# 4. 处理每个月份
months = ['5月', '6月', '7月', '8月', '9月', '10月']
for idx, (ax, month) in enumerate(zip(axes, months)):
    # 准备观测井数据
    month_idx = months.index(month) + 3
    obs_x = np.array([well[1] for well in wells_data])
    obs_y = np.array([well[2] for well in wells_data])
    obs_z = np.array([well[month_idx] for well in wells_data])
    
    # 生成密集网格（适当扩大范围确保边界覆盖）
    grid_x, grid_y = np.mgrid[x_min-0.01:x_max+0.01:300j, 
                             y_min-0.01:y_max+0.01:300j]
    
    # 线性插值（确保边界填充）
    grid_z = griddata((obs_x, obs_y), obs_z, (grid_x, grid_y), 
                     method='linear', fill_value=np.nan)
    
    # 精确边界掩膜
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    mask = np.array([study_area.contains(Point(p)) for p in points])
    mask = mask.reshape(grid_x.shape)
    
    # 边界处特殊处理（确保完全填充）
    grid_z_masked = np.where(mask, grid_z, np.nan)
    
    # 对边缘NaN进行最近邻填充
    valid_mask = ~np.isnan(grid_z_masked)
    interp = NearestNDInterpolator(
        np.column_stack((grid_x[valid_mask], grid_y[valid_mask])),
        grid_z_masked[valid_mask])
    grid_z_filled = interp(grid_x, grid_y)
    
    # 最终掩膜处理
    grid_z_final = np.where(mask, grid_z_filled, np.nan)
    
    # 绘制等值线（使用反转的viridis颜色映射）
    levels = np.linspace(np.nanmin(grid_z_final)*0.98, 
                        np.nanmax(grid_z_final)*1.02, 15)
    contour = ax.contourf(grid_x, grid_y, grid_z_final, 
                         levels=levels, cmap='viridis_r', alpha=0.8, 
                         extend='both')
    
    # 添加原始边界线（不加平滑）
    boundary_path = Path(np.array(study_area.exterior.coords))
    patch = PathPatch(boundary_path, facecolor='none', 
                     edgecolor='black', linewidth=1.5, linestyle='-')
    ax.add_patch(patch)
    
    # 标注采样点和地下水埋深数据
    for point in sample_points:
        # 如果是区1或区2，进一步向左移动
        if point[0] == "区1":
            point = [point[0], point[1] - 0.01, point[2]]  # 向左移动
        elif point[0] == "区2":
            point = [point[0], point[1] - 0.01, point[2]]  # 向左移动
        
        # 如果是区3或区4，向下移动
        elif point[0] == "区3":
            point = [point[0], point[1], point[2] - 0.005]  # 向下移动
        elif point[0] == "区4":
            point = [point[0], point[1], point[2] - 0.005]  # 向下移动
        
        # 计算该点的地下水埋深值
        point_value = interp(point[1], point[2])
        
        # 绘制红色采样点
        ax.plot(point[1], point[2], 'o', color='red', 
                markersize=8, markeredgecolor='k')
        
        # 标注地下水埋深值（右侧）
        ax.text(point[1] + 0.01, point[2], f'{point_value:.2f}m', 
                fontsize=18, ha='left', va='center', color='black')

        # 标注区编号（上方）
        ax.text(point[1], point[2] + 0.003, point[0], 
               fontsize=18, ha='center', va='bottom', color='black')
    
    # 设置图形属性
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)  # 关闭边框
    
    # 添加月份标注（增大字号）
    ax.text(0.5, -0.10, month, transform=ax.transAxes, 
           ha='center', va='center', fontsize=18)


# 添加颜色条（增大字号）
cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])  # 调整颜色条位置
cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal')
cbar.set_label('地下水埋深 (m)', fontsize=18)  # 增大颜色条标签字号
cbar.ax.tick_params(labelsize=16)  # 增大颜色条刻度字号

plt.savefig('民生支渠地下水埋深等值线图_精确边界版_颜色反转_红点标注_无边框_调整位置_区3区4下移.pdf', 
           dpi=300, bbox_inches='tight')
print("图片已输出")

