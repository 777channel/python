# ---------------------------------------------------------------------------
# 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. 全局绘图参数
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# ---------------------------------------------------------------------------
# 2. 计算与绘图逻辑
# ---------------------------------------------------------------------------
def run_correlation_analysis(file_path, output_dir):
    
    # 定义所有需要计算的列（包含特征和目标，直接放在一个列表里）
    # 顺序即为矩阵中显示的顺序
    all_columns = [
        'pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'type', 
        'pH-B', 'BET', 'BR', 'MPs', 'Size', 'MPR', 
        'ΔCd', 'ΔpH'
    ]
    
    print(f"正在读取文件: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # --- 1. 计算皮尔逊相关性 ---
    try:
        # 提取指定列
        df_subset = df[all_columns]
        # 计算矩阵
        corr = df_subset.corr(method='pearson')
        print("皮尔逊相关性矩阵计算完成。")
    except KeyError as e:
        print(f"错误: 数据表中缺少指定的列，请检查列名拼写: {e}")
        return

    # --- 2. 绘图 (使用您的指定代码) ---
    # 设置画布大小
    fig, ax = plt.subplots(figsize=(18, 16), dpi=600)
    
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=-1, vmax=1)

    # 字体参数
    text_in_figure_size, axis_label_size, title_size = 22, 22, 24

    # 循环绘制：上三角气泡，下三角数值
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if i > j:
                # 下三角：数值
                color = cmap(norm(corr.iloc[i, j]))
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=text_in_figure_size,
                        fontweight='bold', color=color)
            elif i < j:
                # 上三角：气泡
                color = cmap(norm(corr.iloc[i, j]))
                # 气泡大小系数，如果觉得气泡太小可以调大这个 500
                ax.scatter(j, i, s=np.abs(corr.iloc[i, j]) * 600, color=color, alpha=0.7)

    # 坐标轴设置
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, ha='center', fontsize=axis_label_size)
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=axis_label_size)

    # 范围与方向
    ax.set_xlim(-0.5, len(corr.columns) - 0.5)
    ax.set_ylim(-0.5, len(corr.columns) - 0.5)
    ax.invert_yaxis() # 确保 (0,0) 在左上角

    # 颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pearson Correlation Coefficient', fontsize=26, fontweight='bold')
    cbar.ax.tick_params(labelsize=24)

    # 保存
    plt.tight_layout()
    save_name = 'Pearson_Matrix_All_Features.pdf'
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"绘图完成！图片已保存至: {save_path}")

# ---------------------------------------------------------------------------
# 3. 执行入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "Correlation_Analysis"

    # 创建输出目录
    save_dir = os.path.join(output_base_path, output_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    
    run_correlation_analysis(file_path, save_dir)