import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import os

# 使用 'Agg' 后端
matplotlib.use('Agg')

# ==================== 全局字体设置 ====================
# 虽然此图基本不用文字，但保留设置以备不时之需
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.sans-serif'] = ['SimHei']

def create_china_map_template(country_shp_path, province_shp_path, output_dir='./'):
    """
    使用国界和省份文件，创建一幅灰白色的、仅含边界的中国地图模板。
    """
    print("🎨 Creating a grey base map template of China...")
    
    try:
        # 1. 读取国界和省份的shapefile
        china_country = gpd.read_file(country_shp_path, encoding='utf-8')
        china_provinces = gpd.read_file(province_shp_path, encoding='utf-8')
        print("✅ Successfully loaded both country and province shapefiles.")
    except Exception as e:
        print(f"❌ Error loading shapefiles: {e}")
        return

    # 2. 开始绘图
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    
    # 3. 绘制地图图层
    # 第一层：将省份图层作为底图，填充为非常浅的灰色，并用白色细线描边
    china_provinces.plot(
        ax=ax, 
        facecolor='#f0f0f0',  # 非常浅的灰色填充
        edgecolor='white',   # 白色的省份分界线
        linewidth=0.8,       # 省份分界线宽度
        zorder=1             # 将其置于底层
    )
    
    # 第二层：在顶层绘制国界线，使用深灰色，使其清晰可见
    china_country.plot(
        ax=ax, 
        edgecolor='#666666', # 深灰色国界线
        facecolor='none',    # 不填充颜色
        linewidth=1.2,       # 国界线宽度
        zorder=2             # 将其置于顶层
    )
    
    print("✅ Plotted the base layers for the map template.")

    # 4. 美化和保存
    ax.set_axis_off() # 关闭所有坐标轴和边框
    
    # 可选：如果需要，可以添加一个简单的标题
    # ax.set_title('Base Map of China', fontsize=28, fontweight='bold')
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "China_Map_Template_Grey.pdf")
    
    # 使用 bbox_inches='tight' 和 pad_inches=0 来去除多余的白边
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
    
    print(f"\n✅ Grey map template saved as: {save_path}")
    plt.close()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 1. 国家完整边界文件路径
    country_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\1. Country\country.shp"
    
    # 2. 省份划分文件路径
    province_shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\2. Province\province.shp"
    
    # 3. 输出目录路径
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图'
    
    create_china_map_template(
        country_shp_path=country_shapefile_path,
        province_shp_path=province_shapefile_path,
        output_dir=output_directory
    )
    
    print("\n🎉 Figure generation completed!")