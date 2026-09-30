import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import pickle
import warnings
warnings.filterwarnings('ignore')
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

def create_feature_importance_visualization(importance_df, output_path=None):
    """
    创建特征重要性可视化图
    
    参数:
    importance_df: 包含特征重要性的DataFrame（包含'features'和'importance'列）
    output_path: 保存图像的路径
    
    返回:
    save_path: 保存的图像路径
    """
    # 关闭之前的图形
    plt.close('all')
    
    # 准备数据 - 替换特征名称中的下划线为空格
    feature_importance_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ') for f in importance_df['features']],
        'Importance': importance_df['importance']
    })
    
    # 按照 'Importance' 进行降序排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
    
    # 如果需要，可以截断过大的值，使图表更美观
    truncated_importance = np.minimum(feature_importance_df['Importance'], 0.3)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    
    # 颜色定义 - 从蓝色到红色的渐变
    colors = ['#3f5fd4', '#0059d2', '#005ed8', '#0065df', '#6659cd', '#c70073', '#d50064', '#e20063', '#ed005a', '#f70048']
    
    # 使用特征数量作为颜色渐变数
    n_bins = len(feature_importance_df)
    
    # 创建自定义颜色映射用于条形图
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

    # 绘制横向条形图，使用颜色映射
    bars = ax.barh(range(len(feature_importance_df)), -truncated_importance, color=bar_colors, height=0.6)

    # 设置标签和格式
    ax.set_xlabel('Contribution', fontsize=22)
    ax.set_xlim(-feature_importance_df['Importance'].max() * 1.1, 0)
    ax.grid(axis='x', visible=False)  # 不显示网格线
    
    # 隐藏x轴的负号
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.2f}'))
    ax.tick_params(axis='x', labelsize=20)  # 增大刻度标签字体
    # 设置 X 轴标签为粗体
    ax.xaxis.label.set_fontweight('bold')  # 这会将 X 轴的标题标签设置为粗体
    # 设置 X 轴刻度标签为粗体
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')  # 这会将所有 X 轴的刻度标签设置为粗体
    # 设置X轴刻度线长度
    ax.tick_params(
        axis='x', 
        which='both',
        direction='out',     # 刻度向内
        length=5,           # 长度设为5个点(或其他你想要的值)
        width=1.5,          # 刻度线宽度
        labelsize=20        # 保持之前设置的标签大小
    )
    # 将Y轴标签移到右侧
    ax.yaxis.tick_right()  # 将Y轴刻度和标签移到右侧
    ax.yaxis.set_label_position("right")  # 将Y轴标签位置设为右侧

    # 清空Y轴标签（因为我们将手动添加文本）
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels([])

    # 设置向内的Y轴刻度线
    ax.tick_params(
        axis='y', 
        which='both', 
        direction='in',     # 刻度向内
        length=0,           # 长度设为可见的值
        width=1.5,          # 增加宽度，使y轴加粗
        right=False,        # 在右侧显示刻度
        left=False          # 不在左侧显示刻度
    )
    
    # 确保右侧轴线可见，并加粗
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(1.5)  # 加粗y轴
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1.5)  # 加粗x轴

    # 创建原始特征名到自定义名称的映射
    feature_mapping = {
        "log Cdcontent": "Cd-C",
        "pHsoil": "pH-S",
        "organicmatter": "OM",
        "Charcoal application ratio": "BR",
        "pH": "pH-B",
        "CEC": "CEC",
        "temperature": "T",
        "material type code": "BM",
        "variety binary": "RS",         
        "irrigation type": "I"        
    }

    # 特殊处理的特征及其颜色配置
    special_features = {
        "variety binary": {
            "parts": [
                {"text": "Variety", "color": "black"},  # 前缀为黑色
                {"text": "Binary", "color": "white"}    # 后缀为白色
            ]
        },
        "irrigation type": {
            "parts": [
                {"text": "Irrigation", "color": "black"}, # 前缀为黑色
                {"text": "Type", "color": "white"}        # 后缀为白色
            ]
        }
    }

    # 在循环中使用自定义特征名称和颜色
    for i, (feature, importance) in enumerate(zip(feature_importance_df['Feature'], feature_importance_df['Importance'])):
        y_pos = i
        
        # 获取x轴当前的范围
        x_min, x_max = ax.get_xlim()
        
        # 添加自定义刻度线（向内）- 从右边坐标轴向内
        tick_length = 0.02 * abs(x_min)  # 刻度线长度为x轴范围的2%
        ax.plot([0, -tick_length], [y_pos, y_pos], color='black', linewidth=0, zorder=8)
        
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
            text_width = 0.07 * abs(x_min)     # 单词间的间距
            
            # 从最后一部分开始逆序绘制，确保位置正确
            for j, part in enumerate(reversed(parts)):
                # 计算当前部分的偏移
                current_offset = text_offset - (j * text_width)
                
                # 添加文本
                ax.text(current_offset, y_pos, part["text"], 
                        color=part["color"], 
                        va='center', 
                        ha='right',
                        fontsize=24, 
                        fontweight='bold',
                        zorder=10)
        else:
            # 普通特征直接显示
            text_offset = -0.005 * abs(x_min)
            ax.text(text_offset, y_pos, feature_text, 
                    color='white',  # 其他特征使用白色
                    va='center', 
                    ha='right',
                    fontsize=24, 
                    fontweight='bold',
                    zorder=10)

    # 移除边框线（但保留右侧的垂直线）
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    
    # 添加纵向黑线在图表右侧
    ax.axvline(x=0, color='black', linewidth=1.5)  # 加粗
    
    # 添加左侧颜色条
    cbar_ax_left = fig.add_axes([0.02, 0.1, 0.01, 0.8])  # [left, bottom, width, height]
    
    # 创建颜色条 - 使用平滑颜色映射
    sm_left = plt.cm.ScalarMappable(cmap=smooth_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm_left.set_array([])
    cbar_left = plt.colorbar(sm_left, cax=cbar_ax_left)
    
    # 设置颜色条标签
    cbar_left.set_label('Contribution', 
        fontsize=24, 
        rotation=90,  # 垂直放置
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
    
    # 添加插入极坐标图（饼图）
    ax_inset = fig.add_axes([0.1, 0.05, 0.5, 0.5], polar=True)
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
            radius = outer_radii[i] + (i * 0.2)  # 为前三个标签添加额外偏移
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
    
    # 调整布局
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])  # 为左侧颜色条留出空间
    
    # 保存图像
    if output_path:
        save_path = output_path
    else:
        # 默认保存路径
        output_dir = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\图像\rice'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'model_soil.pdf')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"特征重要性图像已保存至: {save_path}")
    
    return save_path

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
        
        # 创建特征重要性可视化
        create_feature_importance_visualization(importance_df)
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')

if __name__ == "__main__":
    main()