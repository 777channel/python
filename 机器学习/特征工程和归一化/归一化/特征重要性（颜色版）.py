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
    
    # 特征分类映射 - 原始特征名到类别
    feature_categories = {
        # 土壤特性
        'pHsoil': 'soil',
        'organicmatter': 'soil',
        'log_Cdcontent': 'soil',
        'CEC': 'soil',
        
        # 生物炭特性
        'pH': 'biochar',
        'temperature': 'biochar',
        'material_type_code': 'biochar',
        
        # 水稻特性
        'variety_binary': 'rice',
        
        # 实验条件
        'Charcoal_application_ratio': 'experiment',
        'irrigation_type': 'experiment'
    }
    
    # 类别颜色映射
    category_colors = {
        'soil': '#7fb2d3',      # 土壤特性 - 蓝色
        'rice': '#8dd3c9',      # 水稻特性 - 绿色
        'experiment': '#ffb55f', # 实验条件 - 橙色
        'biochar': '#fc7f71'    # 生物炭特性 - 红色
    }
    
    # 添加类别和颜色到数据框
    feature_importance_df['Category'] = feature_importance_df['Feature'].apply(
        lambda x: feature_categories.get(x.replace(' ', '_'), 'other'))
    feature_importance_df['Color'] = feature_importance_df['Category'].map(category_colors)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), dpi=600)
    
    # 生成条形图
    bars = ax.barh(range(len(feature_importance_df)), -feature_importance_df['Importance'], 
                  color=feature_importance_df['Color'], height=0.6)

    # 设置标签和格式
    ax.set_xlabel('Contribution', fontsize=24)
    ax.set_xlim(-feature_importance_df['Importance'].max() * 1.1, 0)
    ax.grid(axis='x', visible=False)  # 不显示网格线
    
    # 隐藏x轴的负号
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.2f}'))
    ax.tick_params(axis='x', labelsize=24)  # 增大刻度标签字体
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
        labelsize=24        # 保持之前设置的标签大小
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

    # 计算总的重要性值以计算百分比
    total_importance = feature_importance_df['Importance'].sum()
    
    # 在条形图左侧（紧贴条形图）添加百分比值，在条形图右侧添加特征名称（用白色加粗）
    for i, (idx, row) in enumerate(feature_importance_df.iterrows()):
        feature = row['Feature']
        importance = row['Importance']
        # 计算百分比
        percent = importance / total_importance * 100
        
        # 左边条形图起始位置
        bar_left = -importance
        
        # 添加百分比值到条形图最左侧（紧贴条形图）
        ax.text(bar_left - 0.003, i, f"{percent:.1f}%", 
                va='center', ha='right', fontsize=24, fontweight='bold')
        
        # 获取自定义特征名称
        if feature in feature_mapping:
            feature_text = feature_mapping[feature]
        else:
            feature_text = feature
        
        # 添加特征名称到条形图右侧，使用白色加粗字体
        text_offset = -0.005 * abs(ax.get_xlim()[0])
        ax.text(text_offset, i, feature_text, 
                color='black',  # 使用白色
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
    
    # 计算每个类别的重要性总和
    category_importance = feature_importance_df.groupby('Category')['Importance'].sum()
    category_percent = category_importance / category_importance.sum() * 100
    
    # 创建扇形图代替极坐标图
    ax_pie = fig.add_axes([0.1, 0.15, 0.35, 0.35])  # 调整位置
    
    # 准备扇形图数据
    pie_labels = []
    pie_sizes = []
    pie_colors = []
    
    # 按类别名称对应
    legend_labels = {
        'soil': 'Soil',
        'rice': 'Rice',
        'experiment': 'Experiment',
        'biochar': 'Biochar'
    }
    
    for category, color in category_colors.items():
        if category in category_percent:
            pie_labels.append(legend_labels[category])  # 使用图例标签
            pie_sizes.append(category_percent[category])
            pie_colors.append(color)
    
    # 绘制扇形图 - 不使用autopct参数，我们将手动添加标签
    wedges, texts = ax_pie.pie(
        pie_sizes, 
        labels=None,
        colors=pie_colors,
        startangle=90,
        wedgeprops=dict(width=0.5, edgecolor='white')
    )
    
    # 手动添加百分比标签到扇形图外围
    for i, p in enumerate(wedges):
        # 获取扇形的角度和半径
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        ang_rad = np.deg2rad(ang)
        radius = 1.2  # 调整此值以改变标签距离圆心的距离
        
        # 计算标签位置
        x = radius * np.cos(ang_rad)
        y = radius * np.sin(ang_rad)
        
        # 根据位置调整对齐方式
        ha = 'left' if x > 0 else 'right'
        
        # 添加百分比标签，使用黑色加粗字体
        percent_text = f"{pie_sizes[i]:.1f}%"
        ax_pie.text(x, y, percent_text, ha=ha, va='center', 
                   fontsize=24, fontweight='bold', color='black')
    
    # 创建自定义图例，放在扇形图上方，并去掉边框
    legend_elements = []
    for i, (category, label) in enumerate(legend_labels.items()):
        if category in category_importance.index:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', 
                        markerfacecolor=category_colors[category], markersize=15, 
                        label=label)
            )

    # 将图例放在扇形图上方，去掉边框
    legend = ax_pie.legend(handles=legend_elements, loc='upper center', 
                        bbox_to_anchor=(0.5, 1.3),
                        ncol=2, fontsize=24,  # 使用两列布局并调整字体大小
                        frameon=False)  # 去掉边框
    # 设置图例文本为粗体
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontname('Times New Roman')  # 确保字体一致性
    # 调整布局
    plt.tight_layout(rect=[0.05, 0, 0.95, 1])
    
    # 保存图像
    if output_path:
        save_path = output_path
    else:
        # 默认保存路径
        output_dir = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\图像\soil'
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'model_soil.pdf')
    
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
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