import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# --- 1. 全局样式设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18


# --- 2. 创建一个通用的绘图函数 ---
# ==================== 函数参数修改处 ====================
# 函数现在接受两套标签：一套用于坐标轴，一套用于内部注释
def create_full_heatmap(data, axis_labels, annot_labels, title):
    """
    一个通用的函数，用于创建和绘制一个完整的、方形的相关性热力图。
    """
    # 2a. 创建用于着色的纯数字DataFrame (使用坐标轴标签来创建)
    corr_matrix = pd.DataFrame(data, index=axis_labels, columns=axis_labels)
    color_matrix = corr_matrix.copy()
    
    color_matrix.fillna(-999, inplace=True)
    
    custom_color_list = ['#0d87a4', '#FFFFFF', '#ff8756']
    cmap = LinearSegmentedColormap.from_list("my_custom_cmap", custom_color_list)
    
    cmap.set_bad(color='#E0E0E0')
    
    masked_data = np.ma.masked_where(color_matrix == -999, color_matrix)
    
    np.fill_diagonal(color_matrix.values, 0)

    # 2b. 创建用于注释的文本DataFrame
    annot_matrix = corr_matrix.map(lambda val: f"{val:.2f}" if pd.notna(val) else "*")
    
    # ==================== 内部注释标签修改处 ====================
    # 将对角线上的注释替换为带换行符的 annot_labels
    for i, label in enumerate(annot_labels):
        annot_matrix.iloc[i, i] = label
    # ==========================================================
        
    # 3. 开始绘图
    plt.figure(figsize=(10, 8.5), dpi=600)
    ax = sns.heatmap(
        masked_data,
        annot=annot_matrix,
        fmt='',
        cmap=cmap,
        linewidths=1.5,
        linecolor='white',
        vmin=-1, vmax=1,
        center=0,
        cbar_kws={'shrink': 0.82},
        annot_kws={"size": 20},
        # ==================== 坐标轴标签修改处 ====================
        # 明确使用不带换行符的 axis_labels 来设置坐标轴
        xticklabels=axis_labels,
        yticklabels=axis_labels
        # ==========================================================
    )

    # 4. 美化图表
    for i, label in enumerate(axis_labels):
         ax.texts[i * len(axis_labels) + i].set_color("black")

    ax.set_title(title, fontsize=26, pad=20)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2)

    # 保存图像
    filename = title.replace(' ', '_').lower() + '_split_labels.pdf'
    plt.savefig(filename, dpi=600)
    print(f"图表已保存为: {filename}")
    plt.show()


# --- 5. 准备数据和标签 ---
# (Cd absorption)
# 准备两套标签：一套用于坐标轴，一套用于内部注释
labels_cd_axis = ['pH', 'Cd', 'As', 'Cd Absorption']      # 不带换行，用于坐标轴
labels_cd_annot = ['pH', 'Cd', 'As', 'Cd\nAbsorption']    # 带换行，用于对角线注释
data_cd = [
    [1.00, np.nan, -0.19, 0.33],
    [np.nan, 1.00, -0.01, 0.84],
    [-0.19, -0.01, 1.00, -0.30],
    [0.33, 0.84, -0.30, 1.00]
]

# (As absorption)
labels_as_axis = ['pH', 'Cd', 'As', 'As Absorption']      # 不带换行，用于坐标轴
labels_as_annot = ['pH', 'Cd', 'As', 'As\nAbsorption']    # 带换行，用于对角线注释
data_as = [
    [1.00, -0.09, np.nan, 0.23],
    [-0.09, 1.00, -0.01, 0.19],
    [np.nan, -0.01, 1.00, 0.88],
    [0.23, 0.19, 0.88, 1.00]
]

# --- 6. 调用函数绘制两个图 ---
create_full_heatmap(data_cd, labels_cd_axis, labels_cd_annot, "Cd absorption Correlation")
create_full_heatmap(data_as, labels_as_axis, labels_as_annot, "As absorption Correlation")