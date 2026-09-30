import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建原始特征名到自定义名称的映射
feature_mapping = {
    "log_Cdcontent": "Cd-C",
    "pHsoil": "pH-S",
    "organicmatter": "OM",
    "Charcoal_application_ratio": "BR",
    "pH": "pH-B",
    "CEC": "CEC",
    "temperature": "T",
    "material_type_code": "BM",
    "variety_binary": "RS",         
    "irrigation_type": "I"        
}

def get_feature_importance(model, feature_names):
    """从模型获取特征重要性的通用方法"""
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

def create_feature_importance_subplot(ax, model, feature_names, data, title, fig_index):
    """创建特征重要性子图，包含扇形图"""
    # 获取特征重要性
    importance_values = get_feature_importance(model, feature_names)
    
    # 创建特征重要性数据框
    importance_df = pd.DataFrame({
        'features': feature_names,
        'importance': importance_values
    })
    
    # 按特征重要性排序
    importance_df = importance_df.sort_values('importance', ascending=True)
    
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
        'biochar': '#fc7f71',    # 生物炭特性 - 红色
        'other': '#cccccc'      # 其他 - 灰色
    }
    
    # 添加类别和颜色到数据框
    importance_df['Category'] = importance_df['features'].apply(
        lambda x: feature_categories.get(x, 'other'))
    importance_df['Color'] = importance_df['Category'].map(category_colors)
    
    # 生成条形图
    bars = ax.barh(range(len(importance_df)), -importance_df['importance'], 
                  color=importance_df['Color'], height=0.6)

    # 设置标签和格式
    ax.set_xlabel('Contribution', fontsize=22, fontweight='bold')
    ax.set_xlim(-importance_df['importance'].max() * 1.1, 0)
    ax.grid(axis='x', visible=False)  # 不显示网格线
    
    # 隐藏x轴的负号
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{abs(x):.2f}'))
    ax.tick_params(axis='x', labelsize=20)  # 增大刻度标签字体
    
    # 设置 X 轴标签为粗体
    ax.xaxis.label.set_fontweight('bold')
    
    # 设置 X 轴刻度标签为粗体
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    # 设置X轴刻度线长度
    ax.tick_params(
        axis='x', 
        which='both',
        direction='out',
        length=5,
        width=1.5,
        labelsize=20
    )
    
    # 将Y轴标签移到右侧
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # 清空Y轴标签
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels([])

    # 设置向内的Y轴刻度线
    ax.tick_params(
        axis='y', 
        which='both', 
        direction='in',
        length=0,
        width=1.5,
        right=False,
        left=False
    )
    
    # 确保右侧轴线可见，并加粗
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1.5)

    # 计算总的重要性值以计算百分比
    total_importance = importance_df['importance'].sum()
    
    # 在条形图左侧添加百分比值，在条形图右侧添加特征名称
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        feature = row['features']
        importance = row['importance']
        # 计算百分比
        percent = importance / total_importance * 100
        
        # 左边条形图起始位置
        bar_left = -importance
        
        # 添加百分比值到条形图最左侧
        ax.text(bar_left - 0.003, i, f"{percent:.1f}%", 
                va='center', ha='right', fontsize=18, fontweight='bold')
        
        # 获取自定义特征名称
        feature_text = feature_mapping.get(feature, feature)
        
        # 添加特征名称到条形图右侧
        text_offset = -0.005 * abs(ax.get_xlim()[0])
        ax.text(text_offset, i, feature_text, 
                color='black',
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
    ax.axvline(x=0, color='black', linewidth=1.5)
    
    # 添加标题
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # 计算每个类别的重要性总和
    category_importance = importance_df.groupby('Category')['importance'].sum()
    category_percent = category_importance / category_importance.sum() * 100
    
    # 为每个子图创建一个不同位置的扇形图区域
    pie_id = f"pie_{fig_index}"  # 创建唯一标识符
        
    # 计算每个子图不同的扇形图位置
    left_offset = 0.1
    bottom_offset = 0.1
    
    # 创建扇形图 - 使用唯一ID和位置
    pie_rect = [left_offset, bottom_offset, 0.25, 0.25]  # [left, bottom, width, height]
    
    # 关键修改：确保每个扇形图使用唯一的rect和ID
    ax_pie = ax.figure.add_axes(pie_rect, label=pie_id)
    
    # 准备扇形图数据
    pie_labels = []
    pie_sizes = []
    pie_colors = []
    
    # 按类别名称对应
    legend_labels = {
        'soil': 'Soil',
        'rice': 'Rice',
        'experiment': 'Experiment',
        'biochar': 'Biochar',
        'other': 'Other'
    }
    
    for category, color in category_colors.items():
        if category in category_percent:
            pie_labels.append(legend_labels[category])  # 使用图例标签
            pie_sizes.append(category_percent[category])
            pie_colors.append(color)
    
    # 绘制扇形图
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
                   fontsize=18, fontweight='bold', color='black')
    
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
                        ncol=2, fontsize=14,  # 使用两列布局并调整字体大小
                        frameon=False)  # 去掉边框
    
    # 移除扇形图的坐标轴
    ax_pie.axis('off')
    
    # 在扇形图中间添加标题
    part_name = title.split()[0]  # 提取部位名称
    ax_pie.text(0, 0, f"{part_name} Feature", ha='center', va='center', 
               fontsize=14, fontweight='bold')
    
    return ax

def create_shap_subplot(ax, model, feature_names, data, title):
    """创建SHAP子图，按照特征重要性排序，使用临时文件方法并移除左侧特征名称"""
    # 获取特征重要性并排序
    importance_values = get_feature_importance(model, feature_names)
    
    # 创建特征重要性数据框并排序
    importance_df = pd.DataFrame({
        'features': feature_names,
        'importance': importance_values
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 按照重要性排序的特征列表
    ordered_features = importance_df['features'].tolist()
    
    # 重新排序特征数据
    X = data[ordered_features].copy()
    
    # 计算SHAP值
    try:
        # 尝试使用TreeExplainer
        if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # 处理多输出情况
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # 使用第一个输出
        else:
            raise AttributeError("模型没有predict或predict_proba方法")
    except Exception as e:
        print(f"计算SHAP值时出错: {str(e)}")
        ax.text(0.5, 0.5, "SHAP计算错误", 
               ha='center', va='center', fontsize=20, fontweight='bold')
        return ax
    
    # 创建临时文件名
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    temp_filename = temp_file.name
    temp_file.close()
    
    # 创建一个单独的图形并绘制SHAP摘要图
    plt.figure(figsize=(8, 8))  # 调整大小以更好地控制比例
    
    # 绘制SHAP图，但不显示左侧特征名称
    # 首先保存当前rc参数
    original_params = plt.rcParams.copy()
    
    # 设置SHAP图的参数
    plt.rcParams.update({
        'ytick.left': False,    # 不显示左侧Y轴刻度
        'ytick.labelleft': False  # 不显示左侧Y轴标签
    })
    
    # 绘制SHAP摘要图
    shap.summary_plot(
        shap_values, 
        X,
        feature_names=None,  # 不显示特征名称
        plot_type="dot",
        show=False,
        sort=False,  # 已经排序了，不需要再排序
        max_display=len(ordered_features)
    )
    
    # 调整图形以移除左侧空白
    plt.tight_layout(pad=0.5, rect=[0.1, 0, 1, 0.95])  # 左边留一点空间
    
    # 保存图形到临时文件
    plt.savefig(temp_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 恢复原始rc参数
    plt.rcParams.update(original_params)
    
    # 将图像加载为背景图片
    from matplotlib.image import imread
    img = imread(temp_filename)
    
    # 在ax中显示图像，调整比例以与左侧图对齐
    ax.imshow(img, aspect='auto', extent=[0, 1, 0, 1], origin='upper',
             transform=ax.transAxes)
    ax.axis('off')  # 隐藏坐标轴
    
    # 添加标题
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # 删除临时文件
    import os
    os.unlink(temp_filename)
     
    return ax

def create_combined_plots(model_paths, csv_path, output_path):
    """创建组合图表"""
    # 定义部位列表
    parts = ['soil', 'rice', 'shell', 'stem', 'leaf', 'root']
    
    # 创建4x2的大图
    fig = plt.figure(figsize=(25, 40))
    gs = gridspec.GridSpec(4, 2, figure=fig)
    
    # 加载CSV数据
    print(f"正在加载CSV数据: {csv_path}")
    full_data = pd.read_csv(csv_path)
    
    # 循环处理每个部位
    for i, part in enumerate(parts):
        model_path = model_paths[part]
        print(f"正在处理 {part} 部位，模型路径: {model_path}")
        
        # 加载模型
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            model = saved_data['model']
            feature_names = saved_data.get('feature_names', [])
            
            print(f"{part} 模型类型: {type(model)}")
        except Exception as e:
            print(f"加载 {part} 模型时出错: {str(e)}")
            continue
        
        # 设置子图位置
        if i < 2:  # soil和rice，需要特征重要性图和SHAP图
            ax_importance = fig.add_subplot(gs[i, 0])
            ax_shap = fig.add_subplot(gs[i, 1])
            
            # 创建特征重要性子图 - 包含扇形图
            # 创建特征重要性子图 - 包含扇形图，传递唯一索引
            create_feature_importance_subplot(
                ax_importance, model, feature_names, full_data, 
                f"{part.capitalize()} Feature Importance", i  # 使用i作为唯一索引
            )
            
            # 创建SHAP子图
            create_shap_subplot(
                ax_shap, model, feature_names, full_data,
                f"{part.capitalize()} SHAP Values"
            )
        else:  # 其他部位，只需要特征重要性图
            row = (i+2) // 2  # 计算行号 (shell和stem在第2行，leaf和root在第3行)
            col = (i+2) % 2   # 计算列号
            
            ax_importance = fig.add_subplot(gs[row, col])
            
            # 创建特征重要性子图 - 包含扇形图
            create_feature_importance_subplot(
                ax_importance, model, feature_names, full_data, 
                f"{part.capitalize()} Feature Importance", i
            )
    
    # 调整布局 - 为扇形图留出足够空间
    plt.tight_layout(pad=5.0)
    
    # 保存图表
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"组合图表已保存至: {output_path}")
    plt.close(fig)
    
    return output_path

if __name__ == "__main__":
    # 定义模型路径
    model_paths = {
        'soil': r'C:\Users\AMDYE\OneDrive\python\results\soil\models\CD_soil_change_GBDT_model.pkl',
        'rice': r'C:\Users\AMDYE\OneDrive\python\results\rice\models\CD_brown_rice_change_GBDT_model.pkl',
        'shell': r'C:\Users\AMDYE\OneDrive\python\results\shell\models\CD_shell_change_XGBoost_model.pkl',
        'stem': r'C:\Users\AMDYE\OneDrive\python\results\stem\models\CD_stem_change_XGBoost_model.pkl',
        'leaf': r'C:\Users\AMDYE\OneDrive\python\results\leaf\models\CD_leaf_change_XGBoost_model.pkl',
        'root': r'C:\Users\AMDYE\OneDrive\python\results\root\models\CD_root_change_ExtraTrees_model.pkl'
    }
    
    # 定义CSV数据路径
    csv_path = r'C:\Users\AMDYE\OneDrive\python\机器学习\特征工程和归一化\归一化\results\fully_processed_data(4).csv'
    
    # 定义输出路径
    output_path = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\图像\combined_feature_importance.png'
    
    # 创建组合图表
    create_combined_plots(model_paths, csv_path, output_path)