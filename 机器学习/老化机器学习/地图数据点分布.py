import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from shapely.geometry import Point

# 使用 'Agg' 后端
matplotlib.use('Agg')

# ==================== 全局字体设置 (确保为 Times New Roman) ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def create_dispersed_dot_map(country_shp_path, province_shp_path, data_provinces, output_dir='./'):
    """
    绘制一幅完整的中国地图，每个数据点表示为一个独立的小气泡，
    并在相应省份的一个核心区域内适度分散。图中所有文字均为英文。
    
    参数:
    - country_shp_path: 国家边界shapefile文件路径。
    - province_shp_path: 省份边界shapefile文件路径。
    - data_provinces: 包含省份名称的列表，每个名称代表一个数据点。
    - output_dir: 图片保存目录。
    """
    print("🎨 Creating a dispersed dot map of China with English labels...")
    
    try:
        # 1. 读取 shapefile
        china_country = gpd.read_file(country_shp_path, encoding='utf-8')
        china_provinces = gpd.read_file(province_shp_path, encoding='utf-8')
        print("✅ Successfully loaded both country and province shapefiles.")
    except Exception as e:
        print(f"❌ Error loading shapefiles: {e}")
        return

    # 2. 开始绘图
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 3. 绘制地图底板
    china_country.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=3)
    china_provinces.plot(ax=ax, facecolor='#f0f0f0', edgecolor='white', linewidth=0.8, zorder=2)
    print("✅ Plotted map base layers.")

    # 4. 在省份核心区域内绘制每一个数据点
    # --- 参数设置 ---
    bubble_size = 35
    bubble_color = '#d62728'
    
    # ★★★ 核心参数：分散系数 ★★★
    # 控制点在省份内的分散程度。
    # 1.0 表示在整个省份完全随机分散。
    # 0.7 表示在省份中心70%的核心区域内分散。
    # 0.1 表示点非常集中于中心。
    # 推荐值: 0.75
    dispersion_factor = 0.75
    
    print(f"⏳ Plotting {len(data_provinces)} individual data points...")
    for province_name in data_provinces:
        province_geo = china_provinces[china_provinces['pr_name'].str.contains(province_name)]
        
        if province_geo.empty:
            print(f"⚠️ Warning: Could not find province '{province_name}'. Skipping.")
            continue
        
        polygon = province_geo.geometry.iloc[0]
        
        # ★★★ 核心改动：定义一个用于随机取点的核心区域 ★★★
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny
        
        # 计算核心区域的边界
        core_minx = minx + (1 - dispersion_factor) / 2 * width
        core_maxx = maxx - (1 - dispersion_factor) / 2 * width
        core_miny = miny + (1 - dispersion_factor) / 2 * height
        core_maxy = maxy - (1 - dispersion_factor) / 2 * height

        # 循环直到在多边形内部找到一个点
        while True:
            # 在计算出的核心区域内生成随机点
            p = Point(random.uniform(core_minx, core_maxx), random.uniform(core_miny, core_maxy))
            
            # 最终检查，确保点一定在不规则的省份边界内
            if polygon.contains(p):
                ax.scatter(p.x, p.y, 
                           s=bubble_size, 
                           c=bubble_color, 
                           alpha=0.7, 
                           edgecolors='none',
                           zorder=4)
                break
    print("✅ All data points plotted.")

    # 5. 创建纯英文图例
    legend_handle = plt.scatter([], [], s=bubble_size, c=bubble_color, alpha=0.7, edgecolors='none')
    legend_font = {'family': 'Times New Roman', 'size': 20}
    ax.legend(handles=[legend_handle], 
              labels=['Data Point'],
              loc='upper center', 
              
              prop=legend_font,
              title_fontproperties=legend_font,
              frameon=False,
              labelspacing=1.0)

    # 6. 美化和保存
    ax.set_axis_off()
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "China_Dispersed_Dot_Map_EN.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n✅ English dispersed dot map saved as: {save_path}")
    plt.close()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # --- 请确认这两个文件的路径 ---
    country_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\1. Country\country.shp"
    province_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\2. Province\province.shp"
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图'

    # --- 数据源 ---
    data_points = (
        ['广东'] * 10 +
        ['天津'] * 5 +
        ['江苏'] * 18 +
        ['河南'] * 4 +
        ['浙江'] * 5 +
        ['湖南'] * 35 +
        ['四川'] * 9 +
        ['福建'] * 1 +
        ['辽宁'] * 9 +
        ['江西'] * 4 +
        ['河北'] * 5 
    )
    
    create_dispersed_dot_map(
        country_shp_path=country_shapefile_path,
        province_shp_path=province_shapefile_path,
        data_provinces=data_points,
        output_dir=output_directory
    )
    
    print("\n🎉 Figure generation completed!")