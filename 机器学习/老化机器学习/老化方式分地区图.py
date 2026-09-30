import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# ★★★ 关键的第一步 ★★★
# 使用 'Agg' 后端，确保脚本只会生成文件。
matplotlib.use('Agg')

# ==================== 全局字体设置 (保持不变) ====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 26
plt.rcParams['ytick.labelsize'] = 26
plt.rcParams['legend.fontsize'] = 26
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.0
plt.rcParams['ytick.minor.width'] = 1.0
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6

def draw_single_region_plot(ax, region_data, title):
    """
    在指定的Axes上绘制单个区域的三Y轴组合图 (核心绘图逻辑不变)。
    注意：函数签名已简化，移除了网格布局相关的参数。
    """
    # 确保年份按数字排序
    region_data = region_data.sort_values('Year').reset_index(drop=True)
    years = sorted(region_data['Year'].unique())
    
    # 根据年份获取数据
    dw_data = [region_data[region_data['Year'] == y].iloc[0]['Equivalent_DW-D'] for y in years]
    ft_data = [region_data[region_data['Year'] == y].iloc[0]['Equivalent_FT-D'] for y in years]
    c_data = [region_data[region_data['Year'] == y].iloc[0]['Equivalent_C-D'] for y in years]
    temp_data = [region_data[region_data['Year'] == y].iloc[0]['Equivalent_C-T'] for y in years]
    humidity_data = [region_data[region_data['Year'] == y].iloc[0]['Equivalent_C-W'] for y in years]
    
    ax.bar(years, ft_data, 0.6, label='FT-D', color='#E0E0CE', edgecolor='None')
    ax.bar(years, dw_data, 0.6, bottom=ft_data, label='DW-D', color='#EABD96', edgecolor='None')
    ax.bar(years, c_data, 0.6, bottom=[ft + dw for ft, dw in zip(ft_data, dw_data)], label='C-D', color='#73ABBC', edgecolor='None')
    
    ax.set_ylabel('Treatment Days', fontweight='bold', fontsize=28)
    ax.tick_params(axis='y', labelsize=28)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax2 = ax.twinx()
    p2, = ax2.plot(years, temp_data, color='#5B9EC9', marker='o', linestyle='-', 
                   alpha=0.8, linewidth=3.5, markersize=8, label='T (°C)')
    ax2.set_ylim(20, 45)
    ax2.set_ylabel('Temperature (°C)', color='black', fontweight='bold', fontsize=28)
    ax2.tick_params(axis='y', labelcolor='black', color='black', labelsize=28)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')

    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 80))
    p3, = ax3.plot(years, humidity_data, color='#F68E8B', marker='s', linewidth=3.5, 
                   alpha=0.8, linestyle='-', markersize=8, label='W (%)')
    ax3.set_ylim(50, 100)
    ax3.set_ylabel('Humidity (%)', color='black', fontweight='bold', fontsize=28)
    ax3.tick_params(axis='y', labelcolor='black', color='black', labelsize=28)
    for label in ax3.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xticks(years)
    ax.set_xticklabels([f'{int(y)}' for y in years], fontweight='bold', fontsize=28)
    ax.set_xlabel('Aging Years', fontweight='bold', fontsize=28)
    
    ax.set_title(title, fontweight='bold', pad=20, fontsize=28) # 增加标题和图表的间距
    ax.margins(x=0.1, y=0.1)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')

    h1, l1 = ax.get_legend_handles_labels()
    return h1 + [p2, p3]

def generate_individual_regional_plots(data_filepath, output_dir='./'):
    """
    读取CSV文件，并为每个区域生成一张独立的PNG图。
    """
    print("🎨 Generating individual plot for each region...")
    
    try:
        results_df = pd.read_csv(data_filepath, encoding='utf-8-sig')
        results_df['Year'] = pd.to_numeric(results_df['Year'], errors='coerce')
        print(f"✅ Successfully loaded data from {data_filepath}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 定义区域ID到名称的映射
    region_map = {
        1: 'Northeast', 2: 'Southwest', 3: 'Central China',
        4: 'North China', 5: 'East China', 6: 'South China'
    }
    
    # 获取数据中所有唯一的区域ID
    regions_to_plot = sorted(results_df['Region'].unique())
    print(f"Found regions with IDs: {regions_to_plot}. Creating a plot for each.")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 循环遍历每个区域ID
    for region_id in regions_to_plot:
        region_name = region_map.get(region_id, f"{region_id}")
        region_data = results_df[results_df['Region'] == region_id]
        
        if region_data.empty:
            print(f"  ⚠️ Skipping {region_name}, no data found.")
            continue
            
        print(f"  -> Drawing plot for {region_name}...")
        
        # ★★★ 核心修改：为每个区域创建一个全新的、独立的图表 ★★★
        fig, ax = plt.subplots(figsize=(13, 9)) # 设定适合单个图表的尺寸
        
        # 调用绘图函数
        legend_handles = draw_single_region_plot(
            ax=ax, 
            region_data=region_data, 
            title=region_name
        )

        # 为当前图表添加图例
        if legend_handles:
            fig.legend(
                handles=legend_handles, 
                loc='upper center',          # 放置在图表顶部
                bbox_to_anchor=(0.5, 1.05), # 微调位置，使其在标题之上
                ncol=len(legend_handles),    # 水平排列
                frameon=False,
                prop={'size': 26}
            )

        # 调整布局，防止标签被截断
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # 留出顶部空间给图例

        # ★★★ 核心修改：定义独立的文件名并保存为PNG ★★★
        filename = f"{region_name}.png"
        save_path = os.path.join(output_dir, filename)
        
        # 保存为高分辨率的PNG文件
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        
        print(f"     ✅ Saved as: {save_path}")
        
        # ★★★ 关键的第三步 ★★★
        # 关闭当前图表，释放内存，为下一个图表做准备
        plt.close(fig)

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    data_file = r'C:\Users\AMDYE\OneDrive\桌面\aging days results.csv'
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图\Individual_Plots' # 建议为独立图片创建一个子目录
    
    generate_individual_regional_plots(
        data_filepath=data_file,
        output_dir=output_directory
    )
    
    print("\n🎉 All individual plots have been generated!")