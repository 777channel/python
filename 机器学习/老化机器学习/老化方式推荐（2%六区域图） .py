import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# ★★★ 关键的第一步 ★★★
# 使用 'Agg' 后端，确保脚本只生成文件，不弹出预览窗口。
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

def draw_region_subplot(ax, region_data, title, is_last_row, is_first_col, is_last_col):
    """
    在指定的Axes上绘制单个区域的三Y轴组合图
    """
    # 确保年份按数字排序
    region_data = region_data.sort_values('Year').reset_index(drop=True)
    years = sorted(region_data['Year'].unique())
    
    dw_data = []
    ft_data = []
    c_data = []
    temp_data = []
    humidity_data = []
    
    for year in years:
        year_data = region_data[region_data['Year'] == year].iloc[0]
        dw_data.append(year_data['Equivalent_DW-D'])
        ft_data.append(year_data['Equivalent_FT-D'])
        c_data.append(year_data['Equivalent_C-D'])
        temp_data.append(year_data['Equivalent_C-T'])
        humidity_data.append(year_data['Equivalent_C-W'])
    
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
    ax3.set_ylim(40, 100)
    ax3.set_ylabel('Humidity (%)', color='black', fontweight='bold', fontsize=28)
    ax3.tick_params(axis='y', labelcolor='black', color='black', labelsize=28)
    for label in ax3.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xticks(years)
    ax.set_xticklabels([f'{int(y)}' for y in years], fontweight='bold', fontsize=28)
    ax.set_xlabel('Aging Years', fontweight='bold', fontsize=28)
    
    ax.set_title(title, fontweight='bold', pad=10, fontsize=28)
    ax.margins(x=0.1, y=0.1)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3, axis='y')

    h1, l1 = ax.get_legend_handles_labels()
    return h1 + [p2, p3]

def create_regional_overview_figure(data_filepath, output_dir='./'):
    """
    读取Excel，将6个区域的数据绘制到一个2x3的网格图中。
    """
    print("🎨 Creating a single 2x3 regional overview figure...")
    
    try:
        # ★★★ 修改点 1：使用 read_excel 并指定 sheet_name ★★★
        results_df = pd.read_excel(data_filepath, sheet_name='2%天数')
        
        # 确保年份列是数字类型
        results_df['Year'] = pd.to_numeric(results_df['Year'], errors='coerce')
        print(f"✅ Successfully loaded data from sheet '2%数据' in {data_filepath}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    
    region_map = {
        1: 'Northeast', 2: 'North China', 3: 'Southwest', 4: 'Central China',
        5: 'East China', 6: 'South China'
    }
    regions_to_plot = sorted(results_df['Region'].unique())
    print(f"Found regions: {regions_to_plot}. Plotting the first 6.")
    
    legend_handles = None

    for i, region_id in enumerate(regions_to_plot[:6]):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        region_name = region_map.get(region_id, f"{region_id}")
        region_data = results_df[results_df['Region'] == region_id]
        
        if region_data.empty:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(region_name)
            continue
            
        print(f"   Drawing plot for {region_name} at ({row}, {col})")
        handles = draw_region_subplot(
            ax=ax, region_data=region_data, title=region_name,
            is_last_row=(row == 1), is_first_col=(col == 0), is_last_col=(col == 2)
        )
        if legend_handles is None:
            legend_handles = handles

    if legend_handles:
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.88),
                    ncol=len(legend_handles), frameon=False, prop={'size': 28})

    plt.subplots_adjust(
        left=0.05, bottom=0.1, right=0.95, top=0.85, 
        wspace=0.7, hspace=0.3
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = "Regional_Overview_2%.pdf"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n✅ Single overview figure saved as: {save_path}")
    plt.close()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # ★★★ 修改点 2：更新 Excel 文件路径 ★★★
    data_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\2%数据（部分）.xlsx"
    output_directory = r'C:\Users\AMDYE\OneDrive\python\老化方式图'
    
    create_regional_overview_figure(
        data_filepath=data_file,
        output_dir=output_directory
    )
    
    print("\n🎉 Figure generation completed!")