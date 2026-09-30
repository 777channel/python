import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import os

# 使用 'Agg' 后端
matplotlib.use('Agg')

# ==================== 全局字体设置 ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['font.sans-serif'] = ['SimHei']

def create_definitive_china_map(country_shp_path, province_shp_path, output_dir='./'):
    """
    使用“国界底图+省份着色图”的叠加方案，绘制一幅绝对完整的中国地图。
    """
    print("🎨 Creating the definitive, complete map of China using a two-layer approach...")
    
    try:
        # 1. 读取两个关键的shapefile
        # 国界文件，用于设定完整的、不可裁剪的边界
        china_country = gpd.read_file(country_shp_path, encoding='utf-8')
        # 省份文件，用于内部区域的划分和着色
        china_provinces = gpd.read_file(province_shp_path, encoding='utf-8')
        print("✅ Successfully loaded both country and province shapefiles.")
    except Exception as e:
        print(f"❌ Error loading shapefiles: {e}")
        return

    # 2. 定义区域划分和颜色
    regions = {
        'Northeast': ['辽宁省', '吉林省', '黑龙江省'], 
        'North China': ['北京市', '天津市', '河北省', '山西省', '内蒙古自治区'],
        'East China': ['上海市', '江苏省', '浙江省', '安徽省', '福建省', '江西省', '山东省', '台湾省'],
        'Central China': ['河南省', '湖北省', '湖南省'], 
        'South China': ['广东省', '广西壮族自治区', '海南省', '香港特别行政区', '澳门特别行政区'],
        'Southwest': ['重庆市', '四川省', '贵州省', '云南省', '西藏自治区'], 
        'Northwest': ['陕西省', '甘肃省', '青海省', '宁夏回族自治区', '新疆维吾尔自治区']
    }
    region_colors = {
        'Northeast': '#8193d5', 'North China': '#dbeb7c', 'East China': '#9ac2fb', 
        'Central China': '#fadc4b', 'Southwest': '#ddaaea', 'Northwest': '#ffcc79', 
        'South China': '#ff8c8c',
    }
    
    # 3. 为省份数据分配区域
    def get_region(province_name):
        province_name_str = str(province_name)
        for region, provinces in regions.items():
            if any(p in province_name_str for p in provinces):
                return region
        return 'Other'
    
    china_provinces['region'] = china_provinces['pr_name'].apply(get_region)
    
    # 4. 开始绘图
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # ★★★ 核心步骤 ★★★
    # 第一步：先将国界文件(country.shp)绘制为底图。
    # 这将自动设定一个包含所有领土（包括南海诸岛）的正确绘图范围。
    china_country.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=1, zorder=3)
    print("✅ Plotted national boundary to establish the full map extent.")
    
    # 第二步：在底图之上，绘制并为省份图层(province.shp)着色。
    china_provinces.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, zorder=1)
    for region_name, color in region_colors.items():
        region_gdf = china_provinces[china_provinces['region'] == region_name]
        if not region_gdf.empty:
            region_gdf.plot(ax=ax, color=color, edgecolor='white', linewidth=0.8, alpha=0.9, zorder=2)
    print("✅ Plotted and colored provincial regions on top of the boundary.")
    
    # 5. 美化和保存
    ax.set_axis_off()
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "China_Map_Definitive.png")
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    print(f"\n✅ Definitive map saved as: {save_path}")
    plt.close()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # --- 请确认这两个文件的路径 ---
    # 1. 国家完整边界文件
    country_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\1. Country\country.shp"
    
    # 2. 省份划分文件
    province_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\2. Province\province.shp"
    
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图'
    
    create_definitive_china_map(
        country_shp_path=country_shapefile_path,
        province_shp_path=province_shapefile_path,
        output_dir=output_directory
    )
    
    print("\n🎉 Figure generation completed!")