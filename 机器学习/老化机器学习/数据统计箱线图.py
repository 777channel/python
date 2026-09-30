import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.impute import KNNImputer
import os

# --- 全局绘图与字体配置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.unicode_minus'] = False 
plt.ioff()  # 关闭交互模式
mpl.use('Agg')  # 使用非交互式后端

# --- 绘图辅助函数 ---
def select_representative_points(data, n_points=10):
    """选择最具代表性的点用于在箱线图上展示"""
    if len(data) <= n_points:
        return data
    
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    if len(outliers) >= n_points:
        return np.random.choice(outliers, size=n_points, replace=False)
    
    remaining = n_points - len(outliers)
    normal_range = data[(data >= lower_bound) & (data <= upper_bound)]
    
    if len(normal_range) == 0:
        return np.concatenate([outliers, np.random.uniform(np.min(data), np.max(data), remaining)])
    
    percentiles = np.linspace(0, 100, remaining + 2)[1:-1]
    selected_normal = np.percentile(normal_range, percentiles)
    
    return np.concatenate([outliers, selected_normal])


# --- 主执行块 ---
if __name__ == '__main__':
    # 1. 加载数据
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv" 
    
    try:
        df = pd.read_csv(file_path)
        print(f"步骤 1: 数据成功从 '{file_path}' 加载。")
    except FileNotFoundError:
        print(f"错误: 在 '{file_path}' 未找到文件。程序将退出。")
        exit()

    # 2. 初步数据清洗
    print("步骤 2: 正在进行初步数据清洗...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print("数据类型转换完成。")

    # 3. 执行KNN插补
    print("\n步骤 3: 正在使用 KNN 方法填充缺失值...")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    knn_imputer = KNNImputer(n_neighbors=5)
    df_imputed_numeric = knn_imputer.fit_transform(df[numeric_cols])
    df_imputed_knn = pd.DataFrame(df_imputed_numeric, columns=numeric_cols, index=df.index)
    for col in df.columns:
        if col not in numeric_cols:
            df_imputed_knn[col] = df[col]
    print("KNN 填充完成。")

    # 4. 对填充后的数据进行统计分析
    print("\n步骤 4: KNN填充后数据的统计摘要如下：")
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(df_imputed_knn.describe())
    print("="*60)
    
    # 5. 使用填充后的数据进行箱线图绘制
    print("\n步骤 5: 正在使用填充后的数据生成箱线图...")
    
    # !! 重要提醒: 请确保您的CSV文件列名与这里的名称完全一致 !!
    category_vars = {
        'Soil Properties': ['pH-S', 'CEC', 'Cd-C'],
        'Biochar Properties': ['pH-B', 'T', 'BET'],
        'Experimental Condition': ['BR'],
        'Output Variables': ['pH-S-CH', 'Cd-CH']
    }
    colors = ['#58659B', '#8A8CBF', '#B8A8CF']

    # --- 开始绘图 ---
    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    axs = axs.flatten()

    for idx, (category_name, var_keys) in enumerate(category_vars.items()):
        ax = axs[idx]
        
        try:
            data_to_plot = [df_imputed_knn[var_key].dropna().values for var_key in var_keys]
        except KeyError as e:
            print(f"\n错误: 找不到列 {e}。请检查您的CSV文件列名是否与 'category_vars' 中的定义完全匹配。")
            exit()
            
        box_width = 0.4 if category_name == 'Experimental Condition' else 0.6
        bp = ax.boxplot(data_to_plot, positions=range(1, len(var_keys) + 1), 
                        patch_artist=True, showfliers=False, widths=box_width,  
                        boxprops=dict(linewidth=2.5))
        
        labels = var_keys
        
        for i, box in enumerate(bp['boxes']):
            box_color = colors[i % len(colors)]
            box.set(facecolor=box_color, edgecolor='black', linewidth=2.5)
            bp['medians'][i].set(color='black', linewidth=2.5)
            for j in range(2):
                bp['whiskers'][i*2+j].set(color='black', linewidth=2.5)
                bp['caps'][i*2+j].set(color='black', linewidth=2.5)
            
            var_data = df_imputed_knn[var_keys[i]].dropna().values
            selected_points = select_representative_points(var_data)
            
            x_pos = [i + 1 + (np.random.random() - 0.5) * 0.3 for _ in range(len(selected_points))]
            ax.scatter(x_pos, selected_points, color=box_color, alpha=0.7, s=50, 
                       edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=0) 
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.tick_params(axis='both', which='major', length=10, width=2)
        
        subplot_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(0.01, 0.98, subplot_labels[idx], transform=ax.transAxes,
                ha='left', va='top', fontweight='bold', fontsize=26)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            
        # ***** 新增代码: 为 'pH-S' 创建插图 *****
        if category_name == 'Soil Properties':
            # 定义插图位置和大小 [left, bottom, width, height]
            inset_ax_ph_s = ax.inset_axes([0.05, 0.3, 0.25, 0.6]) 
            
            ph_s_data = df_imputed_knn['pH-S'].dropna().values
            
            inset_bp_s = inset_ax_ph_s.boxplot([ph_s_data], patch_artist=True, 
                                               showfliers=False, widths=0.6,
                                               boxprops=dict(linewidth=2.5))
            
            # 使用与主图'pH-S'相同的颜色 (colors[0])
            inset_bp_s['boxes'][0].set(facecolor=colors[0], edgecolor='black', linewidth=2.5)
            inset_bp_s['medians'][0].set(color='black', linewidth=2.5)
            for j in range(2):
                inset_bp_s['whiskers'][j].set(color='black', linewidth=2.5)
                inset_bp_s['caps'][j].set(color='black', linewidth=2.5)
            
            selected_points_s = select_representative_points(ph_s_data, n_points=5)
            x_pos_s = [1 + (np.random.random()-0.5)*0.3 for _ in range(len(selected_points_s))]
            inset_ax_ph_s.scatter(x_pos_s, selected_points_s, color=colors[0], alpha=0.7, s=30, 
                                  edgecolor='black', linewidth=0.5)
            
            inset_ax_ph_s.text(0.5, -0.1, 'pH-S', transform=inset_ax_ph_s.transAxes, 
                               ha='center', va='center', fontweight='bold', fontsize=18)
            inset_ax_ph_s.set_xticks([]) # 移除x轴刻度
            
            for spine in inset_ax_ph_s.spines.values():
                spine.set_linewidth(2.0)
        # *******************************************

        if category_name == 'Biochar Properties':
            inset_ax_ph_b = ax.inset_axes([0.08, 0.3, 0.2, 0.6])
            
            ph_b_data = df_imputed_knn['pH-B'].dropna().values
            
            inset_bp_b = inset_ax_ph_b.boxplot([ph_b_data], patch_artist=True, 
                                               showfliers=False, widths=0.6,
                                               boxprops=dict(linewidth=2.5))
            
            # 'pH-B' 是 Biochar Properties 类别中的第一个变量, 所以颜色也是 colors[0]
            inset_bp_b['boxes'][0].set(facecolor=colors[0], edgecolor='black', linewidth=2.5)
            inset_bp_b['medians'][0].set(color='black', linewidth=2.5)
            for j in range(2):
                inset_bp_b['whiskers'][j].set(color='black', linewidth=2.5)
                inset_bp_b['caps'][j].set(color='black', linewidth=2.5)
            
            selected_points_b = select_representative_points(ph_b_data, n_points=5)
            x_pos_b = [1 + (np.random.random()-0.5)*0.3 for _ in range(len(selected_points_b))]
            inset_ax_ph_b.scatter(x_pos_b, selected_points_b, color=colors[0], alpha=0.7, s=30, 
                                  edgecolor='black', linewidth=0.5)
            
            inset_ax_ph_b.text(0.5, -0.1, 'pH-B', transform=inset_ax_ph_b.transAxes, 
                               ha='center', va='center', fontweight='bold', fontsize=18)
            inset_ax_ph_b.set_xticks([])
            
            for spine in inset_ax_ph_b.spines.values():
                spine.set_linewidth(2.0)
        
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title('')
        ax.text(0.5, -0.1, category_name, transform=ax.transAxes, 
                ha='center', va='center', fontweight='bold', fontsize=26)

    plt.tight_layout(pad=1.5)

    output_filename = 'final_plot_with_two_insets.png'
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"\n成功: 包含两个插图的最终图表已保存为 '{output_filename}'")