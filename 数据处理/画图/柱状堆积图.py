import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rcParams

# 设置全局字体和样式，让图表更接近Nature风格
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 11
rcParams['font.weight'] = 'normal'
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8

# ===== 数据配置 =====
data = {
    'categories': ['Sample 1', 'Sample 2', 'Sample 3'],
    # 每个堆积部分的数据
    'stack_parts': {
        'Part A': np.array([20, 20, 20]),  # 底部部分 - 蓝色
        'Part B': np.array([50, 60, 40]),  # 中间部分 - 黄色
        'Part C': np.array([30, 20, 40])  # 顶部部分 - 灰色
    },
    'colors': {
        'Part A': '#6BAED6',  # 蓝色 (Nature风格)
        'Part B': '#FED976',  # 黄色 (Nature风格)
        'Part C': '#969696'  # 灰色 (Nature风格)
    },
    # 误差线数据
    'errors': {
        'Part A': np.array([2, 3, 2]),
        'Part B': np.array([3, 4, 3]),
        'Part C': np.array([2, 1, 2])
    }
}


def create_refined_nature_bar(data_dict):
    """创建精细版Nature期刊风格堆积柱状图"""
    # 创建图表，白色背景，增加分辨率
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 提取数据
    categories = data_dict['categories']
    stack_parts = data_dict['stack_parts']
    colors = data_dict['colors']
    errors = data_dict.get('errors', None)

    # 堆积部分的名称（从底部到顶部排列）
    part_names = list(stack_parts.keys())

    # 创建x轴位置
    x = np.arange(len(categories))
    width = 0.45  # 更细的柱子宽度

    # 初始化底部位置
    bottoms = np.zeros(len(categories))

    # 绘制每个堆积部分
    bars = []
    for part_name in part_names:
        # 获取当前部分的数据和颜色
        part_data = stack_parts[part_name]
        color = colors[part_name]

        # 绘制当前部分
        if errors and part_name in errors:
            error_data = errors[part_name]
            bar = plt.bar(x, part_data, width, bottom=bottoms, color=color,
                          edgecolor='black', linewidth=0.4, yerr=error_data,
                          capsize=2.5, error_kw={'ecolor': 'black', 'capthick': 0.4, 'elinewidth': 0.4})
        else:
            bar = plt.bar(x, part_data, width, bottom=bottoms, color=color,
                          edgecolor='black', linewidth=0.4)

        # 更精细的数值标签
        for i, v in enumerate(part_data):
            # 只有当值大于某个阈值时才显示标签，避免过于拥挤
            if v >= 10:  # 只标注值大于等于10的部分
                plt.text(x[i], bottoms[i] + v / 2, str(v),
                         color='black', ha='center', va='center',
                         fontweight='normal', fontsize=9)

        # 更新底部位置
        bottoms += part_data
        bars.append(bar[0])

    # 设置坐标轴和标签
    plt.xticks(x, categories, fontsize=10)
    plt.xlabel('Samples', fontsize=11, labelpad=7)
    plt.ylabel('Values', fontsize=11, labelpad=7)

    # 添加更精细的图例
    legend_elements = [Patch(facecolor=colors[part], edgecolor='black', linewidth=0.4,
                             label=part) for part in part_names]
    legend = plt.legend(handles=legend_elements, loc='upper right', frameon=False,
                        fontsize=9, ncol=1, handlelength=1.2, handleheight=1.2)

    # 设置图表样式 - 精细版Nature期刊风格
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.4)

    # 调整y轴范围，留出适当的顶部空间
    y_max = bottoms.max() * 1.05  # 略微减少顶部空间使图更紧凑
    plt.ylim(0, y_max)

    # 设置更精细的刻度样式
    plt.tick_params(axis='both', direction='out', length=3, width=0.4, pad=3)

    # 添加坐标轴标签
    ax.text(-0.1, 1.02, 'a', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='right')

    # 对齐和调整布局
    plt.tight_layout(pad=1.0)

    # 显示图表
    plt.show()

    # 返回图表对象用于保存
    return fig


# ===== 创建图表 =====
fig = create_refined_nature_bar(data)

# 可以保存为高分辨率图片
# fig.savefig('nature_style_chart.png', dpi=300, bbox_inches='tight', transparent=False)
# fig.savefig('nature_style_chart.pdf', bbox_inches='tight', transparent=False)