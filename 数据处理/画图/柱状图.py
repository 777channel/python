import numpy as np
import matplotlib.pyplot as plt

# ===== 数据配置 =====
# 可以直接修改这里的数据和颜色
data = {
    'categories': ['Control', 'Alhpa', 'Beta', 'Gamma'],
    'values': np.array([45, 70, 85, 78]),
    'errors': np.array([5, 6, 4, 5]),
    'colors': {
        'Control': {'hex': '#6BAED6'},
        'Alhpa': {'hex': '#FED976'},
        'Beta': {'hex': '#969696'},
        'Gamma': {'hex': '#FFFFCC'}
    },
    'significance': {
        'positions': [1, 2, 3],  # 需要标记的位置（0表示第一个柱子）
        'symbols': ['*', '**', '**'],  # 显著性标记符号
        'height_offset': 2  # 标记距离误差线的距离
    }
}


def create_bar_chart(data_dict):
    """创建单个柱状图"""
    # 创建图表
    plt.figure(figsize=(8, 6))

    # 提取数据
    categories = data_dict['categories']
    values = data_dict['values']
    errors = data_dict['errors']
    colors = data_dict['colors']
    sig = data_dict['significance']

    # 绘制柱状图
    bars = plt.bar(categories, values,
                   yerr=errors,
                   capsize=5,  # 误差线末端的横线长度
                   color=[colors[cat]['hex'] for cat in categories],
                   width=0.6)  # 柱子宽度

    # 添加显著性标记
    for pos, symbol in zip(sig['positions'], sig['symbols']):
        plt.text(pos, values[pos] + errors[pos] + sig['height_offset'],
                 symbol, ha='center', va='bottom')

    # 设置图表样式
    plt.ylim(0, 100)  # y轴范围
    plt.ylabel('Fold Change')  # y轴标签
    plt.title('Your Chart Title')  # 图表标题

    # 移除顶部和右侧边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 添加图例
    plt.legend(categories, loc='upper right')

    # 添加颜色信息到图表底部
    y_pos = -0.3  # 颜色圆圈的垂直位置
    for i, (cat, color_info) in enumerate(colors.items()):
        # 绘制颜色圆圈
        plt.plot(i, y_pos, 'o',
                 color=color_info['hex'],
                 markersize=15,
                 transform=plt.gca().get_xaxis_transform())

        # 添加颜色代码和RGB值
        rgb = color_info['rgb']
        plt.text(i, y_pos - 0.1,
                 f'#{color_info["hex"].lstrip("#")}\nR:{rgb[0]}\nG:{rgb[1]}\nB:{rgb[2]}',
                 ha='center', va='top',
                 transform=plt.gca().get_xaxis_transform())

    # 调整布局以显示颜色信息
    plt.subplots_adjust(bottom=0.2)

    # 显示图表
    plt.show()




# ===== 创建图表 =====
create_bar_chart(data)

"""
# 修改特定柱子颜色的示例：
data['colors']['Control'] = {
    'hex': '#7FACCC',
    'rgb': (127, 172, 204)
}

# 修改数据的示例：
data['values'] = np.array([50, 75, 90, 80])
data['errors'] = np.array([4, 5, 3, 4])

# 修改显著性标记的示例：
data['significance'] = {
    'positions': [1, 2],
    'symbols': ['*', '**'],
    'height_offset': 3
}
"""