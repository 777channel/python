import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置字体属性
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 28
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# 创建数据字典
data = {
    'soil': {
        'Model': ['GBDT', 'XGBoost', 'RF', 'ExtraTrees', 'LR', 'LightGBM', 'Bayesian Ridge', 'SVR', 'Ridge'],
        'R2_Train': [0.89835520, 0.90416150, 0.89377851, 0.82014022, 0.28212686, 0.57687922, 0.25169816, 0.33778301, 0.24885302],
        'RMSE_Train': [0.10069402, 0.09777574, 0.10293600, 0.13394551, 0.26776964, 0.20546852, 0.27321185, 0.25701667, 0.27372075],
        'MAE_Train': [0.07768955, 0.07422759, 0.07622663, 0.10345795, 0.19917577, 0.15326184, 0.19806898, 0.18176370, 0.19823675],
        'R2_Test': [0.86478491, 0.86012516, 0.82294351, 0.71449542, 0.66298991, 0.63938391, 0.56990302, 0.56863192, 0.56164775],
        'RMSE_Test': [0.11040184, 0.11228805, 0.12683324, 0.16024235, 0.17429666, 0.18055701, 0.19690256, 0.19719131, 0.19873123],
        'MAE_Test': [0.08980182, 0.09234823, 0.10248369, 0.13214986, 0.15466849, 0.15996588, 0.17794120, 0.17441003, 0.17816099],
    },
    'rice': {
        'Model': ['GBDT', 'XGBoost', 'RF', 'ExtraTrees', 'SVR', 'LightGBM', 'Bayesian Ridge', 'Ridge', 'LR'],
        'R2_Train': [0.88305562, 0.90377686, 0.69388520, 0.62970333, 0.14399199, 0.34872229, 0.06335299, 0.05971451, 0.10372078],
        'RMSE_Train': [0.10588777, 0.09604968, 0.17132327, 0.18842177, 0.23088552, 0.24986444, 0.29967061, 0.30025209, 0.29314186],
        'MAE_Train': [0.07049621, 0.06108609, 0.12209700, 0.13817990, 0.15358075, 0.18682848, 0.22086940, 0.22125932, 0.22240223],
        'R2_Test': [0.75471199, 0.75396077, 0.68911015, 0.59387957, 0.45768478, 0.41094144, 0.11616963, 0.10929262, 0.07185485],
        'RMSE_Test': [0.11644415, 0.11662232, 0.13109912, 0.14983291, 0.17314319, 0.18045492, 0.22103649, 0.22189474, 0.22651050],
        'MAE_Test': [0.08300989, 0.08499017, 0.09700584, 0.11382181, 0.12280265, 0.14104705, 0.15696863, 0.15737366, 0.17742627],
    },
    'shell': {
        'Model': ['XGBoost', 'GBDT', 'RF', 'SVR', 'LR', 'ExtraTrees', 'Bayesian Ridge', 'LightGBM', 'Ridge'],
        'R2_Train': [0.97309568, 0.98227531, 0.96073529, 0.24715950, 0.25470465, 0.38871056, 0.23099819, 0.33124944, 0.09146083],
        'RMSE_Train': [0.03256484, 0.02643183, 0.03934031, 0.17262620, 0.17139660, 0.15522492, 0.17410116, 0.16235665, 0.18925059],
        'MAE_Train': [0.01779691, 0.01307122, 0.01521064, 0.05414203, 0.09805482, 0.06424605, 0.09134685, 0.07146941, 0.07900363],
        'R2_Test': [0.9668731, 0.96653214, 0.94829407, 0.41905165, 0.30178305, 0.27558086, 0.22310381, 0.16799326, 0.03737878],
        'RMSE_Test': [0.04136046, 0.04279009, 0.05318615, 0.17827776, 0.19544468, 0.19907804, 0.20616194, 0.21233496, 0.22948596],
        'MAE_Test': [0.01632491, 0.01827303, 0.02451242, 0.05837883, 0.08955209, 0.07103519, 0.07992253, 0.07602636, 0.09609965],
    },
    'stem': {
        'Model': ['XGBoost', 'GBDT', 'LightGBM', 'SVR', 'RF', 'Ridge', 'ExtraTrees', 'Bayesian Ridge', 'LR'],
        'R2_Train': [0.97919314, 0.97885804, 0.43661658, 0.18625961, 0.76945735, 0.13910966, 0.15448118, 0.44379209, 0.45543333],
        'RMSE_Train': [0.03055860, 0.03073556, 0.15876541, 0.19080914, 0.10156194, 0.19625818, 0.19448182, 0.15775111, 0.15609154],
        'MAE_Train': [0.01804155, 0.01822135, 0.07250308, 0.07799288, 0.04439705, 0.10642476, 0.10381083, 0.11074744, 0.11381981],
        'R2_Test': [0.85782786, 0.85006100, 0.25660981, 0.20983926, 0.12182386, 0.05062040, 0.01338502, -0.29549420, -0.32458645],
        'RMSE_Test': [0.05809620, 0.05953648, 0.13257111, 0.13667737, 0.14408916, 0.14981677, 0.15272914, 0.17500813, 0.17696226],
        'MAE_Test': [0.03726344, 0.03808363, 0.04588077, 0.04315795, 0.07723143, 0.07726654, 0.06675257, 0.13153534, 0.13418191],
    },
    'leaf': {
        'Model': ['XGBoost', 'GBDT', 'RF', 'ExtraTrees', 'SVR', 'LightGBM', 'Ridge', 'Bayesian Ridge', 'LR'],
        'R2_Train': [0.88695129, 0.86849452, 0.85575732, 0.85087192, 0.75045238, 0.33831382, 0.11770706, 0.15483420, 0.17996041],
        'RMSE_Train': [0.10693724, 0.11533685, 0.12079335, 0.12291982, 0.15888133, 0.25871535, 0.29874632, 0.29293913, 0.28801103],
        'MAE_Train': [0.06256818, 0.06863494, 0.08178805, 0.08164401, 0.10872087, 0.19084943, 0.23077644, 0.22578926, 0.21978784],
        'R2_Test': [0.76299080, 0.75630550, 0.74424738, 0.72786719, 0.67303635, 0.42763988, 0.19321343, 0.27579577, 0.34191371],
        'RMSE_Test': [0.10035423, 0.10174213, 0.10422886, 0.10751843, 0.11784949, 0.15592389, 0.18512169, 0.17539148, 0.16719350],
        'MAE_Test': [0.07990017, 0.08209684, 0.08989639, 0.08897567, 0.09866945, 0.12484403, 0.14246464, 0.12901064, 0.12040184],
    },
    'root': {
        'Model': ['ExtraTrees', 'XGBoost', 'GBDT', 'RF', 'SVR', 'LightGBM', 'LR', 'Ridge', 'Bayesian Ridge'],
        'R2_Train': [0.90275925, 0.85249176, 0.88671041, 0.85411019, 0.66871006, 0.48767548, 0.12667191, 0.09816533, 0.04074874],
        'RMSE_Train': [0.10305538, 0.12697196, 0.11123501, 0.12622996, 0.18586141, 0.23654790, 0.30884122, 0.31384123, 0.32367767],
        'MAE_Train': [0.07226494, 0.09477778, 0.07719846, 0.09175416, 0.16260742, 0.17728559, 0.23066674, 0.23330004, 0.23970494],
        'R2_Test': [0.85228620, 0.83027065, 0.82818816, 0.77521626, 0.68931896, 0.44430639, 0.22051719, 0.10625631, -0.04576519],
        'RMSE_Test': [0.10450362, 0.11200206, 0.11270608, 0.12891443, 0.15155771, 0.20269292, 0.24006226, 0.25705675, 0.27805918],
        'MAE_Test': [0.07699116, 0.08561117, 0.08461648, 0.09612540, 0.13116811, 0.14102637, 0.17658811, 0.18496289, 0.19009217],
    }
}
# 模型名称映射到简称
model_name_mapping = {
    'GBDT': 'GBDT',
    'XGBoost': 'XGB',
    'RF': 'RF',
    'ExtraTrees': 'ET',
    'LR': 'LR',
    'LightGBM': 'LGB',
    'Bayesian Ridge': 'BR',
    'SVR': 'SVR',
    'Ridge': 'Ridge'
}
# 定义要使用的颜色
colors = ['#58659B', '#8A8CBF', '#B8A8CF', '#E7BCC6', '#FDCF9E', '#EFA484']

# 创建3x2的图表网格
fig, axs = plt.subplots(3, 2, figsize=(25, 30))
axs = axs.flatten()

# 定义要绘制的指标（数据字典中的键名）
metrics = ['R2_Train', 'RMSE_Train', 'MAE_Train', 'R2_Test', 'RMSE_Test', 'MAE_Test']

# 定义用于显示在图例中的标签名称
metric_labels = ['Train R$^2$', 'Train RMSE', 'Train MAE', 'Test R$^2$', 'Test RMSE', 'Test MAE']
# 遍历每个部位
for idx, (part, part_data) in enumerate(data.items()):
    ax = axs[idx]
    
    # 获取模型数量和指标数量
    num_models = len(part_data['Model'])
    num_metrics = len(metrics)
    
    # 设置每个柱子的宽度和位置
    bar_width = 0.16
    index = np.arange(num_models)

    # 绘制柱状图
    for i, metric in enumerate(metrics):
        bars = ax.bar(index + i*bar_width, part_data[metric], bar_width,
                    color=colors[i], label=metric_labels[i]) 
    # 在绘图循环中手动设置刻度标签样式
    ax.set_xticks(index + bar_width * (num_metrics-1) / 2)
    ax.set_xticklabels([model_name_mapping[m] for m in part_data['Model']], 
                    rotation=0, ha='center', fontweight='bold')

    # 设置y轴刻度标签加粗
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # 设置y轴范围为0-1
    ax.set_ylim(0, 1.0)
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 设置边框线条加粗
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
    
    # 设置刻度线长度
    ax.tick_params(axis='both', which='major', length=10, width=2)
    
    # 添加部位标题（放在图的下方）
    ax.set_title('')
    ax.text(0.5, -0.15, part.capitalize(), transform=ax.transAxes,
           ha='center', va='center', fontweight='bold', fontsize=28)
    
    # 添加子图标识符
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    ax.text(0.005, 0.98, subplot_labels[idx], transform=ax.transAxes,
           ha='left', va='top', fontweight='bold', fontsize=28)
    # 添加图例（只在第一个子图中显示，垂直排列）
    legend = ax.legend(loc='upper right', labels=metric_labels, fontsize=20, 
                    frameon=True, fancybox=True, shadow=True)
    # 自定义图例样式
    frame = legend.get_frame()
    frame.set_facecolor('white')  # 白色背景
    frame.set_edgecolor('black')  # 黑色边框
    frame.set_linewidth(1.0)      # 设置边框宽度


# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 0.98], pad=3.0)
plt.subplots_adjust(bottom=0.15)

# 保存图表
plt.savefig('model_comparison_barplot.pdf', dpi=600, bbox_inches='tight')
plt.close(fig)
print('图像绘制完成')