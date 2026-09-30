import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import os
from matplotlib.ticker import FuncFormatter  # ✅ 新增导入

# ==================== 全局字体设置 (Global Font Settings) ====================
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
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28
# 设置坐标轴线条宽度
plt.rcParams['axes.linewidth'] = 2
# 加粗坐标轴线条（增加数值）
plt.rcParams['axes.linewidth'] = 3          # 从2改为3或更大
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3

# 调整数字字号
plt.rcParams['xtick.labelsize'] = 28       # 根据需要调整
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

def load_and_process_simple_excel(excel_filepath):
    """
    读取Excel文件，根据Region列的地区名称来分组数据
    """
    try:
        print(f"📂 读取Excel文件: {excel_filepath}")
        
        # 读取Excel文件
        df = pd.read_excel(excel_filepath)
        print(f"✅ 成功读取: {df.shape}")
        print(f"📋 列名: {list(df.columns)}")
        
        # 检查是否存在Region列
        if 'Region' not in df.columns:
            print("❌ 错误: 未找到Region列!")
            return {}
        
        # 地区名称按代码中的顺序
        region_names = ['Central China','East China','North China', 'Northeast', 'South China' , 'Southwest']
        
        # 处理数据
        processed_data = {}
        
        # 查看数据中实际存在的地区
        unique_regions = df['Region'].unique()
        print(f"📍 数据中的地区: {list(unique_regions)}")
        
        for region_name in region_names:
            print(f"\n🔄 处理地区: {region_name}")
            
            # 根据Region列筛选该地区的数据
            region_mask = df['Region'] == region_name
            region_df = df[region_mask]
            
            if region_df.empty:
                print(f"  ⚠️ 未找到地区数据: {region_name}")
                continue
            
            print(f"  📊 找到 {len(region_df)} 行数据")
            
            # 提取每列的数据（去除NaN值）
            region_data = {}
            data_columns = ['Cd_Natural_Decay_Rates', 'Cd_Artificial_Decay_Rates', 
                           'pH_Natural_Decay_Rates', 'pH_Artificial_Decay_Rates']
            
            for col in data_columns:
                if col in df.columns:
                    # 提取该列的数值，转换为绝对值并去除NaN
                    col_data = region_df[col].dropna().tolist()
                    # 转换为绝对值
                    col_data = [abs(float(x)) for x in col_data if pd.notna(x)]
                    region_data[col] = col_data
                    print(f"    {col}: {len(col_data)} 个数据点")
                else:
                    region_data[col] = []
                    print(f"    {col}: 列不存在")
            
            processed_data[region_name] = region_data
        
        print(f"\n✅ 总共处理了 {len(processed_data)} 个地区")
        return processed_data
        
    except Exception as e:
        print(f"❌ 文件读取错误: {e}")
        return {}

def draw_single_subplot(ax, region_data, region_name, years=None, is_bottom_row=False, is_left_col=False):
    """
    在单个子图上绘制4条曲线 (保持原有绘制方法)
    """
    print(f"    🎨 绘制子图: '{region_name}'")
    
    # 检查数据长度，自动确定年份 (从2年开始)
    if years is None:
        data_length = 0
        for col in ['Cd_Natural_Decay_Rates', 'pH_Natural_Decay_Rates', 
                   'Cd_Artificial_Decay_Rates', 'pH_Artificial_Decay_Rates']:
            if len(region_data.get(col, [])) > 0:
                data_length = len(region_data[col])
                break
        
        if data_length == 0:
            years = list(range(2, 11))  # 默认2-10年
        else:
            years = list(range(2, data_length + 2))  # 从2年开始
    
    # 4条线的配置 (保持原有配置)
    line_configs = [
        {'color': '#2E8B57', 'marker': 'o', 'linestyle': '-', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Natural ΔCd'},
        {'color': '#4682B4', 'marker': 's', 'linestyle': '-', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Natural ΔpH'},
        {'color': '#DC143C', 'marker': '^', 'linestyle': '--', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Artificial ΔCd'},
        {'color': '#FF8C00', 'marker': 'D', 'linestyle': '--', 'linewidth': 3, 'markersize': 8, 'label': 'Simulated Artificial  ΔpH'}
    ]
    
    # 对应的数据列
    data_columns = ['Cd_Natural_Decay_Rates', 'pH_Natural_Decay_Rates', 
                   'Cd_Artificial_Decay_Rates', 'pH_Artificial_Decay_Rates']
    
    lines_drawn = 0
    legend_handles = []
    
    # 绘制4条曲线
    for i, col in enumerate(data_columns):
        data = region_data.get(col, [])
        if len(data) == len(years):  # 确保数据点数量正确
            line = ax.plot(years, data, alpha=0.9, **line_configs[i])
            legend_handles.extend(line)
            lines_drawn += 1
            print(f"    📈 绘制曲线 {i+1}: {col} = {len(data)}个数据点")
        else:
            print(f"    ⚠️ 跳过 {col}: 数据点不匹配 (有{len(data)}个，需要{len(years)}个)")
    
    # 计算RMSE (均方根误差) - 保持原有逻辑
    rmse_results = {}
    
    # Cd RMSE
    cd_natural = region_data.get('Cd_Natural_Decay_Rates', [])
    cd_artificial = region_data.get('Cd_Artificial_Decay_Rates', [])
    if len(cd_natural) == len(years) and len(cd_artificial) == len(years):
        try:
            cd_natural_array = np.array(cd_natural)
            cd_artificial_array = np.array(cd_artificial)
            cd_rmse = np.sqrt(np.mean((cd_natural_array - cd_artificial_array) ** 2))
            rmse_results['cd'] = cd_rmse
        except:
            pass
    
    # pH RMSE
    ph_natural = region_data.get('pH_Natural_Decay_Rates', [])
    ph_artificial = region_data.get('pH_Artificial_Decay_Rates', [])
    if len(ph_natural) == len(years) and len(ph_artificial) == len(years):
        try:
            ph_natural_array = np.array(ph_natural)
            ph_artificial_array = np.array(ph_artificial)
            ph_rmse = np.sqrt(np.mean((ph_natural_array - ph_artificial_array) ** 2))
            rmse_results['ph'] = ph_rmse
        except:
            pass
    
    # 显示RMSE结果 (保持原有样式)
    text_lines = []
    if 'cd' in rmse_results:
        text_lines.append(f"Cd RMSE={rmse_results['cd']:.4f}")
    if 'ph' in rmse_results:
        text_lines.append(f"pH RMSE={rmse_results['ph']:.4f}")
    
    if text_lines:
        rmse_text = '\n'.join(text_lines)
        ax.text(0.95, 0.95, rmse_text, transform=ax.transAxes, 
                fontsize=26, fontweight='bold', va='top', ha='right')
    
    # ✅ 设置Y轴数字乘以100显示
    def y_formatter(x, pos):
        return f'{x*100:.1f}'  # 乘以100，保留1位小数
    
    ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    
    # 设置坐标轴标签
    if is_left_col:
        ax.set_ylabel('Attenuation Rate (%)', fontweight='bold', fontsize=28)  # ✅ 添加%单位
    
    if is_bottom_row:
        ax.set_xlabel('Aging Years', fontweight='bold', fontsize=28)
    
    # 设置x轴刻度
    ax.set_xticks(years)
    ax.set_xticklabels([f'{int(y)}' for y in years], fontweight='bold')
    
    # 设置标题
    ax.set_title(region_name, fontweight='bold', pad=15, fontsize=28)
    
    # 格式化 (保持原有格式)
    ax.tick_params(axis='both', which='major', labelsize=28)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 美化 (保持原有样式 + 加粗坐标轴)
    ax.margins(x=0.1, y=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', width=2, length=6)
    ax.grid(True, linestyle="--", alpha=0.3)
    
    print(f"    ✅ 子图完成: 绘制了{lines_drawn}条曲线")
    return legend_handles, lines_drawn

def create_2x3_regional_degradation_figure(excel_filepath, output_dir='./figures'):
    """
    创建2×3布局的地区衰减分析图表
    """
    print("🎨 创建2×3地区衰减分析图表...")
    print("=" * 60)
    
    # 读取和处理数据
    processed_data = load_and_process_simple_excel(excel_filepath)
    
    if not processed_data:
        print("❌ 没有可处理的数据!")
        return
    
    print(f"\n🌍 处理 {len(processed_data)} 个地区:")
    for region_name, region_data in processed_data.items():
        print(f"  📍 {region_name}")
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建输出目录: {output_dir}")
    
    # 定义区域顺序 (保持代码中的顺序)
    region_order = ['Central China','East China','North China', 'Northeast', 'South China' , 'Southwest']
    
    # 创建2×3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)
    
    all_legend_handles = []
    total_subplots = 0
    total_lines = 0
    
    # 按指定顺序绘制每个区域
    for idx, region_name in enumerate(region_order):
        row = idx // 3  # 行索引 (0或1)
        col = idx % 3   # 列索引 (0,1,2)
        
        # 获取当前区域的数据
        region_data = processed_data.get(region_name, {})
        
        if not region_data:
            # 如果没有该区域的数据，隐藏该子图
            axes[row, col].set_visible(False)
            print(f"⚠️ No data found for region: {region_name}")
            continue
        
        print(f"  Processing region [{row+1}, {col+1}]: {region_name}")
        
        # 确定是否为边缘位置
        is_bottom_row = (row == 1)
        is_left_col = (col == 0)
        
        # 绘制该区域的子图
        handles, lines_count = draw_single_subplot(
            axes[row, col], region_data, region_name,
            is_bottom_row=is_bottom_row, is_left_col=is_left_col
        )
        
        total_lines += lines_count
        total_subplots += 1
        
        if handles and not all_legend_handles:  # 只收集第一个有效子图的图例
            all_legend_handles = handles
    
    print(f"✅ 创建了 {total_subplots} 个子图，总共 {total_lines} 条曲线")
    
    # 添加主标题 (保持原有样式)
    # main_title = 'Degradation Analysis (RMSE)'
    # fig.suptitle(main_title, fontsize=24, fontweight='bold', y=0.97)
    
    # 添加统一图例 (保持原有样式)
    if all_legend_handles:
        fig.legend(handles=all_legend_handles[:4],
                  loc='center', 
                  bbox_to_anchor=(0.5, 0.93),
                  ncol=4,
                  frameon=False,
                  prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 28})
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    # 保存图表
    filename = "Regional_Degradation_Analysis_RMSE"
    png_path = os.path.join(output_dir, f"{filename}.png")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"✅ 保存图表:")
    print(f"   📁 PNG: {png_path}")
    print(f"   📁 PDF: {pdf_path}")
    print(f"   📊 子图数量: {total_subplots}")
    print(f"   📊 总曲线数: {total_lines}")
    
    plt.close()

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🎨 2×3地区衰减分析图表生成器 (使用RMSE)")
    print("=" * 60)
    print("📋 处理流程:")
    print("  ✅ 读取Excel文件 (4列数据)")
    print("  ✅ 每9行为一个地区: Central China, East China, North China, Northeast, South China, Southwest")
    print("  ✅ 2×3网格布局: 每个地区一个子图")
    print("  ✅ 每个子图4条曲线: Natural/Artificial × Cd/pH")
    print("  ✅ 自动处理9个数据点 (2-10年)")
    print("  ✅ 使用RMSE衡量曲线相似性")
    print("=" * 60)
    
    # 文件路径
    excel_file = r'C:\Users\AMDYE\OneDrive\桌面\衰减率和误差.xlsx'
    output_directory = './老化方式图'
    
    # 生成图表
    create_2x3_regional_degradation_figure(excel_file, output_directory)
    
    print("\n" + "=" * 60)
    print("🎉 2×3地区衰减分析完成! (使用RMSE)")
    print("📁 请检查 ./figures 目录中的生成图表")
    print("📝 图表中显示的RMSE值: 值越小表示Natural和Artificial曲线越接近")
    print("=" * 60)