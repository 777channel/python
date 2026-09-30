import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 避免显示问题
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')  # 使用更美观的样式

def create_sample_data_for_demo():
    """Create sample data for demonstration (replace with your real data)"""
    
    ph_s_values = [5.0, 5.5, 6.0, 6.5, 7.0]
    cd_c_values = [0.4, 1.0, 2.0, 3.0, 4.0]
    
    data = []
    for ph_s in ph_s_values:
        for cd_c in cd_c_values:
            # 模拟一些合理的优化结果
            base_ft = np.random.randint(5, 25)
            base_dw = np.random.randint(5, 25) 
            base_c = np.random.randint(40, 120)
            
            data.append({
                'PH-S': ph_s,
                'Cd-C': cd_c,
                '推荐人工FT-D': base_ft,
                '推荐人工DW-D': base_dw,
                '推荐人工C-D': base_c,
                '推荐人工C-T': np.random.randint(25, 35),
                '推荐人工C-W': np.random.randint(65, 90),
                '推断人工模式': np.random.choice(['Mixed Mode', 'Pure Constant T&H Mode', 'Pure FT-DW Mode'])
            })
    
    return pd.DataFrame(data)

def create_merged_map_and_chart(results_df, china_provinces_path, save_path='合并地图老化参数图.png'):
    """
    Create combined map and aging parameter comparison chart
    
    Parameters:
    results_df: DataFrame containing optimization results
    china_provinces_path: Path to China provinces shapefile
    save_path: Save path for the output image
    """
    
    # 确保数据按照土壤条件排序
    results_df = results_df.sort_values(['PH-S', 'Cd-C']).reset_index(drop=True)
    
    # 创建情景标签 - 使用下标格式，保留小数点
    scenario_labels = []
    for _, row in results_df.iterrows():
        # 保留小数点格式
        ph_val = f"{row['PH-S']:.1f}"
        cd_val = f"{row['Cd-C']:.1f}"
        label = f"pH$_{{{ph_val}}}$C$_{{{cd_val}}}$"
        scenario_labels.append(label)
    
    # 创建图形 - 使用更大的尺寸
    fig = plt.figure(figsize=(24, 14))
    
    # 设置整体标题
    fig.suptitle('Comprehensive Analysis of Biochar Artificial Aging Optimization in Northeast China', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 创建网格布局：调整高度比例和间距
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.2, 1.2], 
                         hspace=0.15, wspace=0.12)  # 减少上下间距，调整左右比例
    
    # ==================== 左侧：东北三省地图 ====================
    ax_map = fig.add_subplot(gs[:, 0])  # 跨越两行
    
    # 读取真实地图数据
    china_provinces = gpd.read_file(china_provinces_path)
    
    # 定义东北三省名称（根据实际数据调整）
    northeast_provinces = ['辽宁省', '吉林省', '黑龙江省']
    
    # 检查并调整省份名称
    found_provinces = []
    for province in northeast_provinces:
        if province in china_provinces['pr_name'].values:
            found_provinces.append(province)
    
    # 如果没找到带"省"的，尝试不带"省"的
    if len(found_provinces) < 3:
        alternative_names = ['辽宁', '吉林', '黑龙江']
        for province in alternative_names:
            if province in china_provinces['pr_name'].values:
                found_provinces.append(province)
        if found_provinces:
            northeast_provinces = found_provinces
    
    print(f"Found Northeast provinces: {northeast_provinces}")
    
    # 创建东北地区标识列
    china_provinces['is_northeast'] = china_provinces['pr_name'].isin(northeast_provinces)
    
    # 绘制所有省份（浅灰色）
    china_provinces[~china_provinces['is_northeast']].plot(
        ax=ax_map,
        color='lightgray',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7
    )
    
    # 突出显示东北三省（蓝色）
    northeast_gdf = china_provinces[china_provinces['is_northeast']]
    northeast_gdf.plot(
        ax=ax_map,
        color='#1f77b4',
        edgecolor='darkblue',
        linewidth=2,
        alpha=0.8
    )
    
    # 在东北三省中心添加省份名称标签
    for idx, row in northeast_gdf.iterrows():
        try:
            # 计算省份几何中心
            if isinstance(row.geometry, MultiPolygon):
                largest_polygon = max(row.geometry.geoms, key=lambda p: p.area)
                centroid = largest_polygon.centroid
            else:
                centroid = row.geometry.centroid
            
            province_name = row['pr_name']
            
            # 省份名称英文映射
            name_mapping = {
                '黑龙江省': 'Heilongjiang', '黑龙江': 'Heilongjiang',
                '吉林省': 'Jilin', '吉林': 'Jilin', 
                '辽宁省': 'Liaoning', '辽宁': 'Liaoning'
            }
            english_name = name_mapping.get(province_name, province_name)
            
            # 添加省份名称
            ax_map.annotate(
                english_name,
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
        except Exception as e:
            print(f"Error processing province label: {e}")
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', alpha=0.8, label='Northeast China Study Area'),
        Patch(facecolor='lightgray', alpha=0.7, label='Other Provinces')
    ]
    ax_map.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 设置地图标题和样式
    ax_map.set_title('🗺️ Northeast China Study Area', fontsize=16, fontweight='bold', pad=15)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    
    # ==================== 右上：堆叠柱状图 ====================
    ax_stack = fig.add_subplot(gs[0, 1])
    
    # 准备数据
    x_pos = np.arange(len(scenario_labels))
    ft_data = results_df['推荐人工FT-D'].values
    dw_data = results_df['推荐人工DW-D'].values
    c_data = results_df['推荐人工C-D'].values
    
    # 创建堆叠柱状图
    bar_width = 0.8
    
    # 绘制堆叠柱状图 - 使用更协调的颜色
    bars1 = ax_stack.bar(x_pos, ft_data, bar_width, 
                        label='Freeze-Thaw Days (FT-D)', color='#87CEEB', alpha=0.8, 
                        edgecolor='white', linewidth=0.5)
    bars2 = ax_stack.bar(x_pos, dw_data, bar_width, bottom=ft_data,
                        label='Dry-Wet Days (DW-D)', color='#98FB98', alpha=0.8, 
                        edgecolor='white', linewidth=0.5)
    bars3 = ax_stack.bar(x_pos, c_data, bar_width, bottom=ft_data + dw_data,
                        label='Constant T&H Days (C-D)', color='#FFA500', alpha=0.8, 
                        edgecolor='white', linewidth=0.5)
    
    # 设置标题和标签
    ax_stack.set_xlabel('', fontsize=11)  # 不显示x轴标签
    ax_stack.set_ylabel('Treatment Days', fontsize=11)
    
    # 不显示x轴标签和刻度
    ax_stack.set_xticks([])
    ax_stack.set_xticklabels([])
    
    # 添加网格
    ax_stack.grid(True, alpha=0.3, axis='y')
    ax_stack.set_axisbelow(True)
    
    # 设置图例
    ax_stack.legend(loc='upper left', fontsize=10)
    
    # 在部分柱子上添加总天数标签 - 每隔5个显示避免拥挤
    total_days = ft_data + dw_data + c_data
    for i in range(0, len(x_pos), 5):  # 每隔5个显示
        ax_stack.text(x_pos[i], total_days[i] + max(total_days)*0.02, f'{total_days[i]:.0f}', 
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ==================== 右下：温湿度双轴图 ====================
    ax_temp = fig.add_subplot(gs[1, 1])
    ax_humidity = ax_temp.twinx()
    
    # 准备温湿度数据
    temp_data = results_df['推荐人工C-T'].values
    humidity_data = results_df['推荐人工C-W'].values
    
    # 绘制温度折线图
    line1 = ax_temp.plot(x_pos, temp_data, 'o-', linewidth=2.5, markersize=5, 
                        color='#FF6B6B', label='Recommended Temperature (℃)', markerfacecolor='white', 
                        markeredgecolor='#FF6B6B', markeredgewidth=1.5)
    
    # 绘制湿度折线图
    line2 = ax_humidity.plot(x_pos, humidity_data, 's-', linewidth=2.5, markersize=5, 
                            color='#4ECDC4', label='Recommended Humidity (%)', markerfacecolor='white',
                            markeredgecolor='#4ECDC4', markeredgewidth=1.5)
    
    # 设置标题和标签
    ax_temp.set_xlabel('Soil Scenarios', fontsize=11)
    ax_temp.set_ylabel('Temperature (℃)', fontsize=11, color='#FF6B6B')
    ax_humidity.set_ylabel('Humidity (%)', fontsize=11, color='#4ECDC4')
    
    # 设置轴颜色
    ax_temp.tick_params(axis='y', labelcolor='#FF6B6B', labelsize=10)
    ax_humidity.tick_params(axis='y', labelcolor='#4ECDC4', labelsize=10)
    
    # 设置x轴 - 显示所有25个标签，垂直显示
    ax_temp.set_xticks(x_pos)
    ax_temp.set_xticklabels(scenario_labels, rotation=90, ha='center', fontsize=8)
    
    # 添加网格
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_axisbelow(True)
    
    # 移除图例（根据用户要求）
    # 原来的图例代码被注释掉：
    # lines1, labels1 = ax_temp.get_legend_handles_labels()
    # lines2, labels2 = ax_humidity.get_legend_handles_labels()
    # ax_temp.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # ==================== 添加统计信息 ====================
    # 计算统计数据
    avg_ft = ft_data.mean()
    avg_dw = dw_data.mean()
    avg_c = c_data.mean()
    avg_total = total_days.mean()
    temp_range = f"{temp_data.min():.0f}-{temp_data.max():.0f}"
    humidity_range = f"{humidity_data.min():.0f}-{humidity_data.max():.0f}"
    
    # 在右下图添加统计信息文本框
    stats_text = f"""📈 Statistical Summary
• Avg. FT: {avg_ft:.1f} days
• Avg. DW: {avg_dw:.1f} days  
• Avg. Constant T&H: {avg_c:.1f} days
• Avg. Total: {avg_total:.1f} days
• Temp. Range: {temp_range}℃
• Humidity Range: {humidity_range}%"""
    
    # 添加文本框到右下角
    ax_humidity.text(0.98, 0.02, stats_text, transform=ax_humidity.transAxes, 
                    fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    
    # ==================== 调整布局和保存 ====================
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.12)  # 调整顶部和底部边距
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Combined chart saved as: {save_path}")
    
    # 显示一些关键信息
    print(f"\n📊 Data Statistics:")
    print(f"   • Total soil scenarios processed: {len(results_df)}")
    print(f"   • Average recommended total days: {avg_total:.1f} days")
    print(f"   • Recommended temperature range: {temp_range} ℃")
    print(f"   • Recommended humidity range: {humidity_range} %")
    
    plt.close()  # 释放内存
    
    return fig



# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🎨 Generating combined map and aging parameter comparison chart...")
    
    # 方法1: 读取真实数据
    try:
        results_df = pd.read_csv('./optimization_results_hyperopt_final.csv')
        print("✅ Successfully loaded real data")
    except FileNotFoundError:
        print("⚠️ Real data file not found, using simulated data for demonstration")
        results_df = create_sample_data_for_demo()
    
    # 检查数据
    print(f"📊 Data overview: {len(results_df)} rows of data")
    print("First 5 rows:")
    print(results_df[['PH-S', 'Cd-C', '推荐人工FT-D', '推荐人工DW-D', '推荐人工C-D']].head())
    
    # 生成合并图表
    # Shapefile path
    shapefile_path = r"C:\Users\AMDYE\OneDrive\桌面\ChinaAdminDivisonSHP-24.02.06\2. Province\province.shp"
    
    create_merged_map_and_chart(
        results_df, 
        china_provinces_path=shapefile_path,
        save_path='Northeast_China_Soil_Aging_Analysis.png'
    )
    
    print("\n🎉 Combined chart generation completed!")
    print("Generated file: Northeast_China_Soil_Aging_Analysis.png")
    print("\n📋 Chart description:")
    print("• Left: Geographic map of Northeast China")
    print("• Top right: Recommended treatment days comparison for 25 soil scenarios")
    print("• Bottom right: Recommended temperature & humidity trends with statistical summary")