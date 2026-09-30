import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体属性 - 只使用Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22

# 设置matplotlib不显示图形窗口
plt.ioff()  # 关闭交互模式
mpl.use('Agg')  # 使用非交互式后端

# 读取CSV文件
df = pd.read_csv(r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv")

# 使用您提供的变量名称映射
variable_mapping = {
    # 土壤特性
    'pHsoil': 'pH-S',
    'organicmatter': 'OM',
    'CEC': 'CEC',
    'Cdcontent': 'Cd-I',
    
    # 生物炭特性
    'pH': 'pH-B',
    'temperature': 'T',
    
    # 实验变量
    'Charcoal_application_ratio': 'BR',
    
    # 输出变量
    'CD_soil_change': 'Cd-S',
    'CD_brown_rice_change': 'Cd-R',
    'CD_stem_change': 'Cd-ST',
    'CD_leaf_change': 'Cd-L',
    'CD_shell_change': 'Cd-SH',
    'CD_root_change': 'Cd-RO'
}

# 根据类别分组变量
category_vars = {
    'Soil Properties': ['pHsoil', 'organicmatter', 'CEC', 'Cdcontent'],
    'Biochar Properties': ['pH', 'temperature'],
    'Experimental Condition': ['Charcoal_application_ratio'],
    'Output Variables': ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change', 
               'CD_leaf_change', 'CD_shell_change', 'CD_root_change']
}

# 参考图片中的颜色
colors = ['#58659B', '#8A8CBF', '#B8A8CF', '#E7BCC6', '#FDCF9E', '#EFA484', '#B6766C']

# 创建2x2的大图
fig, axs = plt.subplots(2, 2, figsize=(24, 18))
axs = axs.flatten()  # 将2x2数组展平为1维数组，便于索引

# 函数：选择最具代表性的点（包括离群值和分布代表性点）
def select_representative_points(data, n_points=8):  # 减少为8个点
    if len(data) <= n_points:
        return data
    
    # 计算四分位数
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 找出离群值
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    # 如果离群值足够多，直接返回
    if len(outliers) >= n_points:
        return np.random.choice(outliers, size=n_points, replace=False)
    
    # 如果离群值不够，添加均匀分布的点
    remaining = n_points - len(outliers)
    normal_range = data[(data >= lower_bound) & (data <= upper_bound)]
    
    if len(normal_range) == 0:
        # 如果没有正常范围内的点，返回所有离群值和随机点
        return np.concatenate([outliers, np.random.uniform(np.min(data), np.max(data), remaining)])
    
    # 选择分布均匀的点
    percentiles = np.linspace(0, 100, remaining+2)[1:-1]  # 去掉0和100的端点
    selected_normal = np.percentile(normal_range, percentiles)
    
    return np.concatenate([outliers, selected_normal])

# 遍历每个类别，创建子图
for idx, (category_name, var_keys) in enumerate(category_vars.items()):
    ax = axs[idx]
    
    # 准备数据
    data_to_plot = []
    for var_key in var_keys:
        data_to_plot.append(df[var_key].dropna().values)
    # 在这里添加条件判断，只为实验变量设置不同的宽度
    if category_name == 'Experimental Condition':
        bp = ax.boxplot(data_to_plot, positions=range(1, len(var_keys) + 1), 
                       patch_artist=True, showfliers=False, widths=0.4,  
                       boxprops=dict(linewidth=2.5))
    else:
        bp = ax.boxplot(data_to_plot, positions=range(1, len(var_keys) + 1), 
                       patch_artist=True, showfliers=False, widths=0.6,  
                       boxprops=dict(linewidth=2.5))
    # 准备标签
    labels = [variable_mapping[key] for key in var_keys]
    
    # 自定义箱体 - 使用新的颜色方案
    for i, box in enumerate(bp['boxes']):
        box_color = colors[i % len(colors)]
        box.set(facecolor=box_color, edgecolor='black', linewidth=2.5)  # 增加线宽
        bp['medians'][i].set(color='black', linewidth=2.5)  # 增加线宽
        bp['whiskers'][i*2].set(color='black', linewidth=2.5)  # 增加线宽
        bp['whiskers'][i*2+1].set(color='black', linewidth=2.5)  # 增加线宽
        bp['caps'][i*2].set(color='black', linewidth=2.5)  # 增加线宽
        bp['caps'][i*2+1].set(color='black', linewidth=2.5)  # 增加线宽
        
        # 添加有限数量的散点（只保留最具代表性的点）
        var_data = df[var_keys[i]].dropna().values
        selected_points = select_representative_points(var_data, n_points=10)  # 减少为8个点
        
        # 为选定的点添加散点图
        x_pos = [i+1 + (np.random.random()-0.5)*0.3 for _ in range(len(selected_points))]
        ax.scatter(x_pos, selected_points, color=box_color, alpha=0.7, s=50, 
                  edgecolor='black', linewidth=0.5)
    
    # 设置图表属性
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    # 设置整个图形的边框线条加粗
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.tick_params(axis='both', which='major', length=10, width=2)  # 增加主刻度线的长度和宽度
    # 定义标识符
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    # 在右上角添加标识符
    ax.text(0.01, 0.98, subplot_labels[idx], transform=ax.transAxes,
        ha='left', va='top', fontweight='bold', fontsize=26)
    # 设置次刻度线
    if len(labels) > 1:  # 确保有多个标签才设置次刻度
        major_ticks = range(1, len(labels) + 1)
        minor_ticks = [i + 0.5 for i in range(0, len(labels))]
        ax.set_xticks(minor_ticks, minor=True)
        
    # 为Y轴添加次刻度线
    y_major_ticks = ax.get_yticks()
    if len(y_major_ticks) > 1:
        y_tick_interval = y_major_ticks[1] - y_major_ticks[0]
        y_minor_ticks = [tick + y_tick_interval/2 for tick in y_major_ticks[:-1]]
        ax.set_yticks(y_minor_ticks, minor=True)

    # 设置次刻度线样式
    ax.tick_params(axis='both', which='minor', length=5, width=1.5, 
                labelsize=0, labelcolor='none')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    # 对于输出变量，设置Y轴范围为-100到100
    if category_name == 'Output Variables':
        ax.set_ylim(-100, 100)
    if category_name == 'Biochar Properties':
        # 创建插入的小图
        inset_ax = ax.inset_axes([0.1, 0.3, 0.3, 0.6])  # [x, y, width, height] 相对于主图的位置和大小
        
        # 获取pH-B的数据
        ph_b_data = df['pH'].dropna().values
        
        # 绘制pH-B的箱线图
        inset_bp = inset_ax.boxplot([ph_b_data], patch_artist=True, 
                                  showfliers=False, widths=0.6,
                                  boxprops=dict(linewidth=2.5))
        
        # 自定义插图的箱体
        inset_bp['boxes'][0].set(facecolor=colors[0], edgecolor='black', linewidth=2.5)
        inset_bp['medians'][0].set(color='black', linewidth=2.5)
        inset_bp['whiskers'][0].set(color='black', linewidth=2.5)
        inset_bp['whiskers'][1].set(color='black', linewidth=2.5)
        inset_bp['caps'][0].set(color='black', linewidth=2.5)
        inset_bp['caps'][1].set(color='black', linewidth=2.5)
        
        # 添加代表性的点
        selected_points = select_representative_points(ph_b_data, n_points=5)  # 减少为5个点
        x_pos = [1 + (np.random.random()-0.5)*0.3 for _ in range(len(selected_points))]
        inset_ax.scatter(x_pos, selected_points, color=colors[0], alpha=0.7, s=30, 
                       edgecolor='black', linewidth=0.5)
        
        # 设置插图属性
        inset_ax.text(0.5, -0.1, 'pH-B', transform=inset_ax.transAxes, 
             ha='center', va='center', fontweight='bold', fontsize=18)
        inset_ax.set_xticks([])  
        inset_ax.set_ylim(7, 11)
        # 设置插图边框加粗
        for spine in inset_ax.spines.values():
            spine.set_linewidth(2.0)
    
    # 移除Y轴标签和X轴标签
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    # 将标题放在图表底部中心位置
    ax.set_title('')  # 清除默认标题位置
    ax.text(0.5, -0.1, category_name, transform=ax.transAxes, 
           ha='center', va='center', fontweight='bold', fontsize=26)

# 调整布局
plt.tight_layout()

# 保存整个2x2图表
plt.savefig('数据统计箱线图.pdf', dpi=600, bbox_inches='tight')
plt.close(fig)
   