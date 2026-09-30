import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm



plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 基于Excel文件的数据
data = {
    'sample': ['As200', 'As200Cd100', 'As200Cd200', 'As200Cd400'],
    
    # 吸附量 (mg/kg)
    'hydroxyl': [14.522, 11.92767, 11.947, 12.77467],
    'hydroxyl_err': [2.858, 0.104, 1.016, 2.701],
    
    'carboxyl': [9.418, 11.92767, 11.947, 12.31311],
    'carboxyl_err': [2.122, 1.783, 0.249, 1.105],
    
    'other_nonmineral': [15.656, 17.53667, 17.494, 20.21622],
    'other_nonmineral_err': [3.495, 1.699, 0.016, 1.996],
    
    'mineral': [22.682, 23.85533, 23.894, 24.16467],
    'mineral_err': [3.495, 1.699, 0.016, 1.996],
    
    # 百分比贡献
    'hydroxyl_pct': [0.2332, 0.1828, 0.183, 0.1839],
    'carboxyl_pct': [0.1512, 0.1828, 0.183, 0.1772],
    'other_nonmineral_pct': [0.2514, 0.2688, 0.268, 0.291],
    'mineral_pct': [0.3642, 0.3656, 0.366, 0.3478]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 根据您提供的图片设置颜色（从下到上的顺序）
# 对应您图片中从上到下的颜色
colors = {
    'hydroxyl': '#BD2251',  # 1 - 黄色
    'carboxyl': '#F58B34',  # 2 - 橙色
    'other_nonmineral': '#DC4959',  # 3 - 红色
    'mineral': '#97D7F2'  # 4 - 紫色
}

# 百分比条形图的浅色版本
light_colors = {
    'hydroxyl': '#BD2251',  # 1 - 黄色
    'carboxyl': '#F58B34',  # 2 - 橙色
    'other_nonmineral': '#DC4959',  # 3 - 红色
    'mineral': '#97D7F2'  # 4 - 紫色
}

# 设置图形和坐标轴
fig, ax1 = plt.subplots(figsize=(12, 8))
ax2 = ax1.twinx()  # 创建第二个y轴

# 柱状图的宽度
width = 0.35

# 柱状图的位置
positions = np.arange(len(df['sample']))
pos1 = positions - width/2  # 左侧柱状图（吸附量）
pos2 = positions + width/2  # 右侧柱状图（百分比）

# 初始化堆叠的底部位置
bottom1 = np.zeros(len(df['sample']))
bottom2 = np.zeros(len(df['sample']))

# 创建带有误差棒的堆叠柱状图（左轴 - 吸附量）
# 和堆叠柱状图（右轴 - 百分比）
components = ['hydroxyl', 'carboxyl', 'other_nonmineral', 'mineral']
for component in components:
    # 左侧柱状图（带有误差棒）- 吸附量
    ax1.bar(pos1, df[component], width, bottom=bottom1, 
            color=colors[component], label=component)
    
    # 添加误差棒
    ax1.errorbar(pos1, bottom1 + df[component], yerr=df[f'{component}_err'], 
                fmt='none', color='black', capsize=5)
    
    # 更新下一个堆叠的底部位置
    bottom1 += df[component]
    
    # 右侧柱状图 - 百分比
    ax2.bar(pos2, df[f'{component}_pct'], width, bottom=bottom2, 
            color=light_colors[component], alpha=0.7)
    
    # 更新下一个堆叠的底部位置
    bottom2 += df[f'{component}_pct']

# 设置x轴
ax1.set_xticks(positions)
ax1.set_xticklabels(df['sample'])

# 设置标签
ax1.set_xlabel('样品类型')
ax1.set_ylabel('吸附量 (mg/kg)')
ax2.set_ylabel('百分比')

# 设置标题
plt.title('不同机制对砷的吸附量及其贡献')

# 设置y轴范围
ax1.set_ylim(0, 85)
ax2.set_ylim(0, 1.1)

# 将右侧y轴格式化为百分比
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))

# 添加网格
ax1.grid(True, linestyle='--', alpha=0.7)

# 添加图例
handles, labels = ax1.get_legend_handles_labels()
# 反转顺序以匹配从下到上的堆叠
handles = handles[::-1]
labels = labels[::-1]
# 自定义标签（中文）
custom_labels = ['羟基', '羧基', '其他非矿物', '矿物机制']
ax1.legend(handles, custom_labels, loc='upper right')

# 调整布局
plt.tight_layout()

# 保存图片为PNG格式（高分辨率）
plt.savefig('砷吸附机制分析.png', dpi=300, bbox_inches='tight')