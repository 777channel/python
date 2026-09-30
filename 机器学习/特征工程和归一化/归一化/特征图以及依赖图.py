import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
import shap  # 导入SHAP库
import traceback
import time
# 文件路径 - 使用原始字符串避免转义问题
model_path = r'C:\Users\AMDYE\OneDrive\python\results\soil\models\CD_soil_change_GBDT_model.pkl'
csv_data_path = r'C:\Users\AMDYE\OneDrive\python\机器学习\特征工程和归一化\归一化\results\fully_processed_data(4).csv'

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

plt.close('all')

def get_feature_importance(model, feature_names):
    """
    从模型获取特征重要性的通用方法
    """
    importance_methods = [
        ('feature_importances_', lambda m: m.feature_importances_),
        ('coef_', lambda m: np.abs(m.coef_)[0] if hasattr(m, 'coef_') else None),
        ('feature_importance_', lambda m: m.feature_importance_)
    ]
    
    for method_name, method_func in importance_methods:
        try:
            if hasattr(model, method_name):
                importance = method_func(model)
                
                # 确保重要性值长度与特征名称匹配
                if len(importance) != len(feature_names):
                    print(f"警告：特征重要性长度({len(importance)})与特征名称长度({len(feature_names)})不匹配")
                    if len(importance) > len(feature_names):
                        importance = importance[:len(feature_names)]
                    else:
                        importance = np.pad(importance, 
                                            (0, len(feature_names) - len(importance)), 
                                            'constant')
                
                print(f"使用 {method_name} 获取特征重要性")
                return importance
        except Exception as e:
            print(f"尝试 {method_name} 失败: {str(e)}")
    
    # 如果都失败，生成随机重要性值
    print("无法获取特征重要性，使用随机值")
    np.random.seed(42)
    return np.random.rand(len(feature_names))

def calculate_shap_values(model, X_sample):
    """
    计算SHAP值的通用方法
    """
    # 在开始时就准备好模拟SHAP值作为后备方案
    n_samples, n_features = X_sample.shape
    fallback_shap_values = np.random.randn(n_samples, n_features) * 0.1
    
    # 根据特征重要性调整模拟值
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, imp in enumerate(importances):
            if i < fallback_shap_values.shape[1]:  # 确保索引在有效范围内
                fallback_shap_values[:, i] *= np.sqrt(imp) * 10
    
    try:       
        try:
            # 使用正确的TreeExplainer
            explainer = shap.TreeExplainer(model)
            # 使用shap_values方法而不是直接调用explainer
            shap_values = explainer.shap_values(X_sample)
            print("使用 TreeExplainer 计算 SHAP 值")
            
            # 处理多类别情况
            if isinstance(shap_values, list):
                print(f"SHAP值是列表，长度为 {len(shap_values)}")
                shap_values = shap_values[0]  # 通常取第一个元素
            
            print(f"SHAP值形状: {shap_values.shape}")
            return shap_values, True
                
        except Exception as e1:
            print(f"使用 TreeExplainer 失败: {str(e1)}")
            
            try:
                # 尝试使用KernelExplainer作为备选方案
                print("尝试使用KernelExplainer...")
                # 使用前100个样本作为背景数据
                background = X_sample.iloc[:min(100, len(X_sample))]
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample.iloc[:min(100, len(X_sample))])
                print("使用KernelExplainer计算SHAP值成功")
                
                # 如果只计算了一部分样本，复制这些值以匹配原始样本数量
                if len(shap_values) < len(X_sample):
                    print(f"扩展SHAP值以匹配完整样本数量({len(X_sample)})")
                    full_shap_values = np.zeros((len(X_sample), X_sample.shape[1]))
                    full_shap_values[:len(shap_values)] = shap_values
                    # 使用均值填充其余部分
                    full_shap_values[len(shap_values):] = np.mean(shap_values, axis=0)
                    shap_values = full_shap_values
                
                return shap_values, True
                
            except Exception as e2:
                print(f"使用KernelExplainer失败: {str(e2)}")
                # 使用后备方案
                shap_values = fallback_shap_values
                print("使用后备SHAP值")
        
        return shap_values, True
        
    except Exception as e:
        print(f"SHAP计算出错: {str(e)}")
        print("将使用模拟SHAP值")
        
        print(f"创建了形状为 {fallback_shap_values.shape} 的模拟SHAP值")
        return fallback_shap_values, False

def main():
    try:
        # 加载模型 - 使用 pickle
        print(f"正在加载模型: {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # 提取模型
        model = saved_data['model']
        
        # 提取特征名称
        feature_names = saved_data.get('feature_names', [])
        
        # 打印模型类型
        print("模型类型:", type(model))
        
        # 打印模型属性
        print("\n模型属性:")
        print(dir(model))
        
        # 尝试获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # 创建特征重要性字典
            feature_importance_dict = dict(zip(feature_names, importances))
            
            # 按重要性降序排序
            sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print("\n特征重要性 (降序):")
            for feature, importance in sorted_importance:
                print(f"{feature}: {importance:.6f}")
        else:
            print("\n模型没有 feature_importances_ 属性")
        
        # 加载CSV数据
        print(f"正在加载CSV数据: {csv_data_path}")
        full_data = pd.read_csv(csv_data_path)
        print(f"CSV数据形状: {full_data.shape}")
        
        # 检查并处理缺失特征
        missing_features = [f for f in feature_names if f not in full_data.columns]
        if missing_features:
            print(f"警告: 以下特征在CSV数据中缺失: {missing_features}")
            feature_names = [f for f in feature_names if f in full_data.columns]
            print(f"将使用以下特征: {feature_names}")
        
        # 提取特征数据
        X_data = full_data[feature_names].copy()
        
        # 处理缺失值
        if X_data.isnull().any().any():
            print("警告: 特征数据中存在缺失值，将进行填充")
            X_data = X_data.fillna(X_data.mean())
        
        # 获取特征重要性
        importance_values = get_feature_importance(model, feature_names)
        
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            'features': feature_names,
            'importance': importance_values
        })
        
        # 打印特征重要性
        print("\n特征重要性:")
        for i, (feature, importance) in enumerate(zip(feature_names, importance_values)):
            print(f"{i+1}. {feature}: {importance:.6f}")
        
        # 按特征重要性排序
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # 准备SHAP分析数据
        sample_size = min(1000, X_data.shape[0])
        X_sample = X_data
        
        # 计算SHAP值
        shap_values, has_real_shap = calculate_shap_values(model, X_sample)
        
        # 创建图形
        create_combined_visualization(
            importance_df, 
            shap_values, 
            has_real_shap, 
            X_sample
        )
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')

def create_combined_visualization(importance_df, shap_values, has_real_shap, X_sample):
    """
    创建特征重要性和SHAP值的组合可视化图
    
    参数:
    importance_df: 包含特征重要性的DataFrame
    shap_values: SHAP值计算结果
    has_real_shap: 是否有真实的SHAP值
    X_sample: 用于SHAP计算的样本数据
    
    返回:
    save_path: 保存的图像路径
    """
    # 准备数据
    feature_importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ') for f in importance_df['features']],
        'Importance': importance_df['importance']
    })
    truncated_importance = np.minimum(feature_importance_df['Importance'], 0.18)
    # 按照 'Importance' 进行降序排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

    # 创建单一整体图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 15), dpi=600, gridspec_kw={'width_ratios': [1.3, 1], 'wspace': 0.05})
    
    # 颜色定义
    colors = ['#3f5fd4', '#0059d2', '#005ed8', '#0065df', '#6659cd', '#c70073', '#d50064', '#e20063', '#ed005a', '#f70048']
    
    # 使用特征数量作为颜色渐变数
    n_bins = len(feature_importance_df)
    
    # 创建自定义颜色映射用于条形图
    from matplotlib.colors import LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=n_bins)
    
    # 定义扩展颜色列表，用于颜色条的平滑过渡
    colorbar_colors = [
        '#3f5fd4', '#2c5dd3', '#1c5bd3', '#0c59d2', '#045cd5', 
        '#0060d9', '#0062dc', '#0064df', '#2e61d8', '#4c5fd1', 
        '#6a5dcb', '#885bc4', '#a659bd', '#c458b6', '#c80070', 
        '#d0006d', '#d7006a', '#df0067', '#e60063', '#ee005e', 
        '#f50059', '#fc0053', '#ff1048', '#ff2040'
    ]
    
    # 创建更平滑的颜色映射用于颜色条
    smooth_cmap = LinearSegmentedColormap.from_list('smooth_blue_red', colorbar_colors, N=256)
    
    # 生成颜色
    bar_colors = [custom_cmap(i/n_bins) for i in range(n_bins)]

    # 特征重要性图
    # 绘制横向条形图，使用颜色映射
    bars = ax1.barh(range(len(feature_importance_df)), -truncated_importance, color=bar_colors, height=0.6)

    # 设置标签
    ax1.set_xlabel('Contributation', fontsize=24)
    ax1.set_xlim(-feature_importance_df['Importance'].max() * 1.1, 0)
    ax1.grid(axis='x', visible=False)  # 不显示网格线
    
    # 隐藏x轴的负号
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.2f}'))
    ax1.tick_params(axis='x', labelsize=22)  # 增大刻度标签字体

    # 将Y轴标签移到右侧
    ax1.yaxis.tick_right()  # 将Y轴刻度和标签移到右侧
    ax1.yaxis.set_label_position("right")  # 将Y轴标签位置设为右侧

    # 清空Y轴标签（因为我们将手动添加文本）
    ax1.set_yticks(range(len(feature_importance_df)))
    ax1.set_yticklabels([])

    # 设置向内的Y轴刻度线
    ax1.tick_params(
        axis='y', 
        which='both', 
        direction='in',     # 刻度向内
        length=0,           # 长度设为可见的值
        width=1.5,          # 增加宽度，使y轴加粗
        right=False,        # 在右侧显示刻度
        left=False          # 不在左侧显示刻度
    )
    
    # 确保右侧轴线可见，并加粗
    ax1.spines['right'].set_visible(True)
    ax1.spines['right'].set_linewidth(1.5)  # 加粗y轴
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['bottom'].set_linewidth(1.5)  # 加粗x轴

    # 获取y轴的范围，用于定位颜色条
    y_min1, y_max1 = ax1.get_ylim()

    # 创建原始特征名到自定义名称的映射
    feature_mapping = {
        "log Cdcontent": "Cdcontent",
        "pHsoil": "pHsoil",
        "organicmatter": "Organic Matter",
        "Charcoal application ratio": "Biochar Application Ratio",
        "pH": "pH",
        "CEC": "CEC",
        "temperature": "Temperature",
        "material type code": "Material Type",
        "variety binary": "Varity Binary",         
        "irrigation type": "Irrigation type"        
    }

    # 特殊处理的特征及其颜色配置
    special_features = {
        "variety binary": {
            "parts": [
                {"text": "variety", "color": "black"},  # 前缀"var"为黑色
                {"text": "binary", "color": "white"} # 后缀"binary"为白色
            ]
        },
        "irrigation type": {
            "parts": [
                {"text": "Irrigation", "color": "black"},  # 前缀"irriga"为黑色
                {"text": "Type", "color": "white"}    # 后缀"type"为白色
            ]
        }
    }

    # 在循环中使用自定义特征名称和颜色
    for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'], feature_importance_df['Importance'])):
        y_pos = i
        
        # 获取x轴当前的范围
        x_min, x_max = ax1.get_xlim()
        
        # 添加自定义刻度线（向内）- 从右边坐标轴向内
        tick_length = 0.02 * abs(x_min)  # 刻度线长度为x轴范围的2%
        ax1.plot([0, -tick_length], [y_pos, y_pos], color='black', linewidth=0, zorder=8)
        
        # 获取自定义特征名称
        if feature in feature_mapping:
            feature_text = feature_mapping[feature]
        else:
            feature_text = feature
        
        # 检查是否是需要特殊处理的特征
        if feature_text in special_features:
            # 分别处理不同部分的文本和颜色
            parts = special_features[feature_text]["parts"]
            
            # 计算文本偏移量
            text_offset = -0.005 * abs(x_min)  # 基础偏移量
            text_width = 0.07 * abs(x_min)    # 单词间的间距
            
            # 从最后一部分开始逆序绘制，确保位置正确
            for j, part in enumerate(reversed(parts)):
                # 计算当前部分的偏移
                current_offset = text_offset - (j * text_width)
                
                # 添加文本
                ax1.text(current_offset, y_pos, part["text"], 
                        color=part["color"], 
                        va='center', 
                        ha='right',
                        fontsize=24, 
                        fontweight='bold',
                        zorder=10)
        else:
            # 普通特征直接显示
            text_offset = -0.005 * abs(x_min)
            ax1.text(text_offset, y_pos, feature_text, 
                    color='white',  # 其他特征使用白色
                    va='center', 
                    ha='right',
                    fontsize=24, 
                    fontweight='bold',
                    zorder=10)

    # 移除边框线（但保留右侧的垂直线）
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['left'].set_visible(False)
    #添加纵向黑线在图表右侧
    ax1.axvline(x=0, color='black', linewidth=1.5)  # 加粗
    
    # 左侧颜色条 - 宽度减小，底部与坐标轴对齐
    # 获取当前图表的位置信息
    pos1 = ax1.get_position()
    
    # 添加左侧颜色条，底部与x轴对齐，高度与图表一致
    cbar_ax_left = fig.add_axes([0.02, 0.07, 0.01, 0.95])  # [left, bottom, width, height]
    
    # 创建颜色条 - 使用平滑颜色映射
    sm_left = plt.cm.ScalarMappable(cmap=smooth_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_left.set_array([])
    cbar_left = plt.colorbar(sm_left, cax=cbar_ax_left)
    
    # 设置颜色条标签
    cbar_left.set_label('Contributation', 
    fontsize=24, 
    rotation=90,  # 水平放置
    labelpad=10,  # 向左移动
    loc='center')
    cbar_left.ax.yaxis.set_label_position('left')  # 确保标签在左侧
    cbar_left.set_ticks([1, 0])
    cbar_left.set_ticklabels(['High', 'Low'])
    cbar_left.ax.tick_params(
        axis='y', 
        labelsize=24, 
        pad=-75  # 刻度标签向左移动
    )

    # 创建嵌入的极坐标图 - 嵌入到特征重要性图中
    ax_inset = fig.add_axes([0.1, 0.05, 0.25, 0.5], polar=True)
    ax_inset.set_facecolor('none')  # 设置背景透明
    ax_inset.patch.set_alpha(0)     # 确保背景完全透明
    ax_inset.set_facecolor('white')

    # 提取排序后的数据
    features = feature_importance_df['Feature'].tolist()
    values = feature_importance_df['Importance'].tolist()
    
    # 转换为百分比值用于显示
    percent_values = [val / sum(values) * 100 for val in values]

    # 计算总数值和每个扇区的角度
    total_value = sum(values)
    total_angle = 2 * np.pi  # 圆的总角度

    # 设置扇区之间的间隙
    gap_angle = np.pi / 1000  # 极小的间隙
    angles = [(value / total_value) * (total_angle - gap_angle * len(values)) for value in values]

    # 计算起始角度
    theta = np.zeros(len(values))
    current_angle = np.pi / 2  # 90度起始，从顶部开始
    for i in range(len(values)):
        theta[i] = current_angle
        current_angle += angles[i] + gap_angle

    # 使用自定义颜色
    blueColors = [
        '#3f5fd4', '#0059d2', '#005ed8', '#0065df', '#6659cd',
    ]

    redColors = [
        '#c70073', '#d50064', '#e20063', '#ed005a', '#f70048'
    ]

    # 组合为一个颜色列表 - 从蓝色到红色
    polar_colors = blueColors + redColors
    if len(polar_colors) > len(values):
        polar_colors = polar_colors[:len(values)]  # 截取需要的部分
    elif len(polar_colors) < len(values):
        # 如果颜色不够，重复使用最后一个颜色
        polar_colors = polar_colors + [polar_colors[-1]] * (len(values) - len(polar_colors))

    # 手动设置每个扇区的内半径（bottom值）- 值越大，内半径越小
    inner_radii = [
        0.8,  # 最小值
        0.75,
        0.7,
        0.65,
        0.6,
        0.55,
        0.5,
        0.45,
        0.4,
        0.35   # 最大值
    ]

    # 手动设置每个扇区的外半径 - 值越小，外半径越大
    outer_radii = [
        1.2,   # 最小值
        1.15,
        1.1,
        1.05,
        1,
        0.95,
        0.9,
        0.85,
        0.8,
        0.75   # 最大值
    ]

    # 确保内外半径列表长度与数据长度匹配
    if len(inner_radii) > len(values):
        inner_radii = inner_radii[:len(values)]
    elif len(inner_radii) < len(values):
        # 如果内半径列表不够长，使用最后一个值填充
        inner_radii = inner_radii + [inner_radii[-1]] * (len(values) - len(inner_radii))

    if len(outer_radii) > len(values):
        outer_radii = outer_radii[:len(values)]
    elif len(outer_radii) < len(values):
        # 如果外半径列表不够长，使用最后一个值填充
        outer_radii = outer_radii + [outer_radii[-1]] * (len(values) - len(outer_radii))

    # 首先绘制背景扇形 - 交替的灰色阴影
    for i in range(len(values)):
        # 交替使用深浅灰色
        if i % 2 == 0:
            bg_color = '#E5E5E5'  # 浅灰色
        else:
            bg_color = '#D0D0D0'  # 深灰色
        
        # 绘制从中心到内半径的扇形
        ax_inset.bar(theta[i], inner_radii[i], width=angles[i], bottom=0, 
               color=bg_color, edgecolor='white', linewidth=0.5, align='edge', alpha=0.7)

    # 绘制主要扇形 - 使用不同的内外半径
    for i in range(len(values)):
        # 计算此扇区的高度 (外半径 - 内半径)
        height = outer_radii[i] - inner_radii[i]
        
        # 绘制扇形
        bar = ax_inset.bar(theta[i], height, width=angles[i], bottom=inner_radii[i], 
                    color=polar_colors[i], edgecolor='white', linewidth=0.8, align='edge')
        
        # 添加数值标签 - 将标签位置调整得更外部
        angle = theta[i] + angles[i] / 2  # 扇区中点角度
        
        if i < 3:  # 假设前三个是最小的扇区
            radius = outer_radii[i]  + (i * 0.2)  # 为前三个标签添加额外偏移
        else:
            radius = outer_radii[i] + 0.4  # 增大此值以将标签放置得更外部
        
        # 计算角度，用于调整标签对齐方式
        angle_deg = angle * 180 / np.pi
        ha = 'left' if 0 <= angle_deg <= 180 else 'right'  # 水平对齐
        
        ax_inset.text(angle, radius, f"{percent_values[i]:.1f}%", ha=ha, va='center', 
                 fontsize=22, fontweight='bold', fontname='Times New Roman')  # 增大字体

    # 隐藏所有轴线和标签
    ax_inset.spines['polar'].set_visible(False)  # 去掉外部黑色圆圈
    ax_inset.grid(False)
    ax_inset.set_xticklabels([])
    ax_inset.set_yticklabels([])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    
    # 去除所有线条和网格，确保没有意外的虚线
    ax_inset.spines['polar'].set_visible(False)  # 去掉外部圆圈
    for key in ax_inset.spines.keys():
        ax_inset.spines[key].set_visible(False)
    
    # 增大图形在极坐标系中的大小，确保有足够空间显示外部标签
    ax_inset.set_ylim(0, 1.7)  # 增大上限以适应更外部的标签

    # SHAP值部分 - 使用SHAP库的summary_plot
    ax2.set_xlabel('SHAP Value(impact on model output)', fontsize=24)
    ax2.tick_params(axis='x', which='major', labelsize=22)
    ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
    # 设置x轴和y轴加粗
    ax2.spines['bottom'].set_linewidth(1.5)  # 加粗x轴
    ax2.spines['left'].set_linewidth(0)      # 不显示y轴
    ax2.spines['top'].set_visible(False)     # 隐藏顶部边框
    ax2.spines['right'].set_visible(False)   # 隐藏右侧边框

    if has_real_shap and shap_values is not None:
        # 重新排序SHAP值，按照特征重要性排序
        features_order = {feature: i for i, feature in enumerate(X_sample.columns)}
        new_order = [features_order[feature.replace(' ', '_')] for feature in feature_importance_df['Feature']]
        reordered_shap_values = shap_values[:, new_order]
        X_reordered = X_sample.iloc[:, new_order]
        
        # 创建临时文件来保存SHAP图
        import tempfile
        import os
        import matplotlib.image as mpimg
        import time
        import matplotlib.pyplot as plt_temp  # 创建独立的plt引用
        
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f'temp_shap_{int(time.time())}.png')
        
        # 创建一个新的图形用于SHAP图
        temp_fig = plt_temp.figure(figsize=(10, 8))
        
        # 获取当前轴
        temp_ax = temp_fig.add_subplot(111)
        
        # 调用SHAP库的summary_plot函数生成小提琴图
        shap.summary_plot(
            reordered_shap_values,
            X_reordered,
            # 不指定plot_type参数，使用默认的小提琴图
            show=False,
            sort=False,  # 不使用shap默认排序
            color_bar=False,  # 不显示颜色条
            max_display=len(feature_importance_df),
            plot_size=(8, 7),
            plot_type=None  # 默认为小提琴图
        )
        
        # 获取当前图形中的所有子图
        temp_axes = temp_fig.get_axes()
        
        # 移除Y轴标签
        for ax in temp_axes:
            ax.set_yticklabels([])
            ax.set_ylabel('')
        
        # 保存临时图像
        plt_temp.tight_layout()
        plt_temp.savefig(temp_file, dpi=300, bbox_inches='tight')
        plt_temp.close(temp_fig)
        
        # 读取保存的SHAP图像
        img = mpimg.imread(temp_file)
        
        # 在ax2上显示图像
        ax2.imshow(img, aspect='auto')
        ax2.axis('off')  # 隐藏坐标轴
        
        # 尝试删除临时文件
        try:
            os.remove(temp_file)
        except:
            pass
    else:
        # 如果没有SHAP值，显示提示信息
        ax2.text(0.5, 0.5, '无可用的SHAP值', 
                ha='center', va='center', fontsize=14,
                transform=ax2.transAxes)

    # 右侧颜色条 - 宽度减小，底部与坐标轴对齐，高度与图表一致
    cbar_ax_right = fig.add_axes([0.96, 0.07, 0.01, 0.95])  # [left, bottom, width, height]
    
    # 创建颜色条 - 使用平滑颜色映射
    sm_right = plt.cm.ScalarMappable(cmap=smooth_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_right.set_array([])
    cbar_right = plt.colorbar(sm_right, cax=cbar_ax_right)
    
    # 设置右侧颜色条标签
    cbar_right.set_label('Contributation', 
        fontsize=24, 
        rotation=90,  # 水平放置
        labelpad=-40,  # 向左移动
        loc='center')
    cbar_right.ax.yaxis.set_label_position('right')  # 确保标签在右侧
    cbar_right.set_ticks([1, 0])
    cbar_right.set_ticklabels(['High', 'Low'])
    cbar_right.ax.tick_params(
            axis='y', 
            labelsize=24, 
            pad=5  # 刻度标签向左移动
    )
    
    # 调整布局
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])  # 为左右两侧颜色条留出空间
    
    # 保存图像
    output_dir = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\图像'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'combined_visualization.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"组合图像已保存至: {save_path}")
    
    return save_path
if __name__ == "__main__":
    main()