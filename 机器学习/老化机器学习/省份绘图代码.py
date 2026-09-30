import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取中国省份地图数据
china_provinces = gpd.read_file(r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\2. Province\province.shp")

# 查看pr_name列的具体内容
print("pr_name列的所有省份：")
print(china_provinces['pr_name'].unique())

# 定义东北三省名称
northeast_provinces = ['辽宁省', '吉林省', '黑龙江省']

# 检查并调整省份名称
print("\n检查东北三省名称：")
found_provinces = []
for province in northeast_provinces:
    if province in china_provinces['pr_name'].values:
        print(f"✓ 找到：{province}")
        found_provinces.append(province)
    else:
        print(f"✗ 未找到：{province}")

# 如果没找到带"省"的，尝试不带"省"的
if len(found_provinces) < 3:
    alternative_names = ['辽宁', '吉林', '黑龙江']
    print("\n尝试其他名称格式...")
    found_provinces = []
    for province in alternative_names:
        if province in china_provinces['pr_name'].values:
            print(f"✓ 找到：{province}")
            found_provinces.append(province)
    if found_provinces:
        northeast_provinces = found_provinces

# 创建东北地区标识列
china_provinces['is_northeast'] = china_provinces['pr_name'].isin(northeast_provinces)

print(f"\n最终使用的东北三省名称：{northeast_provinces}")
print(f"匹配到的省份数量：{china_provinces['is_northeast'].sum()}")

# ============ 绘制全国地图 ============
fig, ax = plt.subplots(figsize=(15, 10))

# 绘制所有省份（浅灰色）
china_provinces[~china_provinces['is_northeast']].plot(
    ax=ax,
    color='lightgray',
    edgecolor='black',
    linewidth=0.5,
    alpha=0.7
)

# 突出显示东北三省（蓝色）
northeast_gdf = china_provinces[china_provinces['is_northeast']]
northeast_gdf.plot(
    ax=ax,
    color='blue',
    edgecolor='darkblue',
    linewidth=2,
    alpha=0.8
)

# 在东北三省中心添加省份名称标签
for idx, row in northeast_gdf.iterrows():
    # 计算省份几何中心
    centroid = row.geometry.centroid
    province_name = row['pr_name']

    # 添加省份名称
    ax.annotate(
        province_name,
        xy=(centroid.x, centroid.y),
        xytext=(0, 0),
        textcoords='offset points',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='darkblue', alpha=0.8)
    )

# 设置标题和样式
plt.title('中国东北三省地图', fontsize=20, fontweight='bold', pad=20)

# 移除坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 添加图例
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='blue', alpha=0.8, label='东北三省'),
    Patch(facecolor='lightgray', alpha=0.7, label='其他省份')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('东北三省地图.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 全国地图已保存为 '东北三省地图.png'")
plt.close()  # 关闭图形释放内存


# ============ 绘制东北地区详细地图 ============
def create_detailed_northeast_map_robust():
    """创建更详细的东北地区地图 - 处理MultiPolygon问题"""

    # 筛选出东北三省
    northeast_only = china_provinces[china_provinces['is_northeast']].copy()

    if len(northeast_only) == 0:
        print("未找到东北三省数据")
        return

    print(f"\n创建详细地图，包含 {len(northeast_only)} 个省份")

    # 创建东北地区详细地图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 定义三种蓝色和对应的省份名称
    colors = ['#1f77b4', '#3787c0', '#5ba3d0']
    province_names = []

    # 逐个绘制每个省份
    for i, (idx, row) in enumerate(northeast_only.iterrows()):
        # 创建单行GeoDataFrame进行绘制
        single_province = gpd.GeoDataFrame([row])
        single_province.plot(
            ax=ax,
            color=colors[i % len(colors)],
            alpha=0.8,
            edgecolor='white',
            linewidth=2
        )

        province_names.append(row['pr_name'])
        print(f"✓ 绘制了 {row['pr_name']}")

    # 添加省份标签
    for idx, row in northeast_only.iterrows():
        try:
            # 计算几何中心
            if isinstance(row.geometry, MultiPolygon):
                # 对于MultiPolygon，取最大的多边形的中心
                largest_polygon = max(row.geometry.geoms, key=lambda p: p.area)
                centroid = largest_polygon.centroid
            else:
                # 对于单个Polygon
                centroid = row.geometry.centroid

            # 添加标签
            ax.annotate(
                row['pr_name'],
                xy=(centroid.x, centroid.y),
                ha='center', va='center',
                fontsize=14, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='darkblue', alpha=0.9)
            )
            print(f"✓ 添加了 {row['pr_name']} 的标签")

        except Exception as e:
            print(f"✗ 处理 {row['pr_name']} 时出错: {e}")

    plt.title('东北三省详细地图', fontsize=18, fontweight='bold')

    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 手动创建图例 - 避免PatchCollection警告
    from matplotlib.patches import Patch
    legend_elements = []
    for i, name in enumerate(province_names):
        legend_elements.append(
            Patch(facecolor=colors[i % len(colors)], alpha=0.8, label=name)
        )

    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig('东北三省详细地图.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ 详细地图已保存为 '东北三省详细地图.png'")
    plt.close()


# 创建详细地图
create_detailed_northeast_map_robust()

print("\n🎉 所有地图创建完成！")
print("生成的文件：")
print("1. 东北三省地图.png - 全国视角")
print("2. 东北三省详细地图.png - 东北地区特写")

# 可选：显示省份的几何类型信息
print("\n省份几何类型信息：")
for idx, row in china_provinces[china_provinces['is_northeast']].iterrows():
    geom_type = type(row.geometry).__name__
    print(f"{row['pr_name']}: {geom_type}")