import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 配置区域
# ==============================================================================
# 这里填你生成的 CSV 所在的文件夹路径
# 例如: .../MPs_Grid_Scan_0to3
base_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\MPs_Grid_Scan_0to3"

# 想要画的目标变量 (文件夹名)
targets = ['deltaCd', 'deltapH'] 

# ==============================================================================
# 绘图逻辑
# ==============================================================================

def plot_interaction_trends(data_path, output_dir, target_name):
    """
    绘制交互趋势图：展示 MPR (X轴) 对 抑制效应 (Y轴) 的影响，并按 BR (颜色) 分组
    """
    # 1. 读取数据
    df = pd.read_csv(data_path)
    
    # 2. 数据美化 (为了图例显示好看)
    # 将数值型的 BR 转换为 字符串，比如 0.5 -> "0.5%"
    df['Biochar Ratio'] = df['BR'].apply(lambda x: f"{x:.1f}%")
    
    # 3. 设置绘图风格 (学术论文风格)
    sns.set_theme(style="ticks", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei'] # 优先用Arial，支持中文防止乱码
    plt.rcParams['axes.unicode_minus'] = False
    
    # 4. 定义Y轴标签和标题
    if "Cd" in target_name:
        y_label = "Abs. Inhibition on Cd Immobilization"
        title_suffix = "(Cd)"
    else:
        y_label = "Abs. Inhibition on pH Improvement"
        title_suffix = "(pH)"

    # 5. 绘图 (FacetGrid 分面图)
    # 行=Cd浓度，列=pH值
    g = sns.relplot(
        data=df,
        x="MPR", 
        y="Abs_Effect",
        hue="Biochar Ratio",      # 不同的施炭量显示不同颜色
        style="Biochar Ratio",    # 不同的施炭量同时改变线型（便于黑白打印区分）
        col="pH-S", 
        row="Cd-C",
        kind="line",
        markers=True,             # 显示数据点
        dashes=False,             # 禁用虚线，全部实线（或者设为True区分）
        palette="viridis",        # 颜色渐变
        height=3.5, 
        aspect=1.2,
        linewidth=2
    )
    
    # 6. 细节调整
    g.set_axis_labels("Microplastics Ratio (MPR %)", y_label)
    g.fig.subplots_adjust(top=0.92) # 调整顶部空间
    g.fig.suptitle(f"MPs Inhibition Effect: Dose-Response Interaction {title_suffix}", fontsize=14)
    
    # 添加网格线
    for ax in g.axes.flat:
        ax.grid(True, linestyle='--', alpha=0.5)
        # 反转Y轴？通常抑制效应是负值还是正值？
        # 你的CSV里 Abs_Effect 是正数（代表变化的幅度），所以不需要反转
        # 如果 Abs_Effect 是负数，可能需要 ax.invert_yaxis()
    
    # 7. 保存图片
    save_filename = f"{target_name}_Interaction_Plot.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"[{target_name}] 图表已保存: {save_path}")

# ==============================================================================
# 执行
# ==============================================================================
if __name__ == "__main__":
    for target in targets:
        # 构造文件路径
        # 假设结构是: .../MPs_Grid_Scan_0to3/deltaCd/1_full_grid_data.csv
        data_file = os.path.join(base_dir, target, "1_full_grid_data.csv")
        output_dir = os.path.join(base_dir, target)
        
        if os.path.exists(data_file):
            print(f"正在处理: {target} ...")
            plot_interaction_trends(data_file, output_dir, target)
        else:
            print(f"跳过: 找不到文件 {data_file}")
            
    print("\n绘图完成！")