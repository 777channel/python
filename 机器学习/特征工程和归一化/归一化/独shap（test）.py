import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
import os
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def spxy_train_test_split(X, y, test_size=0.3, random_state=42, distance_weight=0.5, distance_metric='cosine'):
    """
    改进的SPXY方法划分训练集和测试集，允许自定义距离度量

    参数:
    X -- 特征矩阵
    y -- 目标变量
    test_size -- 测试集比例，默认0.3
    random_state -- 随机种子
    distance_weight -- X和Y距离的权重比例，0.5表示等权重
    distance_metric -- 特征空间的距离度量方式，可选值：'euclidean', 'cityblock', 'cosine'等
    """
    from sklearn.preprocessing import StandardScaler
    from scipy.spatial.distance import pdist, squareform

    np.random.seed(random_state)

    # 标准化特征
    X_scaled = StandardScaler().fit_transform(X)

    # 处理y可能为零或常数的情况
    if y.std() == 0:
        y_scaled = y.copy()
    else:
        y_scaled = (y - y.mean()) / y.std()

    # 计算特征空间的距离，使用指定的度量方式
    print(f"使用{distance_metric}距离来计算特征空间距离")
    Dx = squareform(pdist(X_scaled, metric=distance_metric))

    # 计算目标空间的距离
    Dy = squareform(pdist(y_scaled.values.reshape(-1, 1), metric='euclidean'))

    # 调整X和Y距离的权重
    D = distance_weight * Dx + (1 - distance_weight) * Dy

    n_samples = len(X)
    test_size_n = int(n_samples * test_size)
    train_size_n = n_samples - test_size_n

    # 初始化
    train_indexes = []
    test_indexes = list(range(n_samples))

    # 选择第一个样本 - 可以选择距离其他样本最远的点
    sum_dist = D.sum(axis=1)
    max_dist_idx = np.argmax(sum_dist)
    train_indexes.append(max_dist_idx)
    test_indexes.remove(max_dist_idx)

    # 逐个选择剩余的样本
    while len(train_indexes) < train_size_n:
        min_dist = np.zeros(len(test_indexes))
        for i, idx in enumerate(test_indexes):
            d = min([D[idx, j] for j in train_indexes])
            min_dist[i] = d

        # 选择距离训练集最远的样本
        max_idx = np.argmax(min_dist)
        selected_idx = test_indexes[max_idx]
        train_indexes.append(selected_idx)
        test_indexes.remove(selected_idx)

    # 根据索引划分数据集
    X_train = X.iloc[train_indexes]
    X_test = X.iloc[test_indexes]
    y_train = y.iloc[train_indexes]
    y_test = y.iloc[test_indexes]

    print(f"SPXY划分完成，训练集大小: {len(X_train)}，测试集大小: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def generate_standalone_shap_plot(model_path, csv_path, output_dir=None, distance_metric='cosine', distance_weight=0.5, test_size=0.3, random_state=42):
    """
    生成SHAP图，并按照模型重要性排序显示特征名称
    
    参数:
    model_path: 模型pickle文件路径
    csv_path: 数据CSV文件路径
    output_dir: 输出目录，默认为当前目录
    distance_metric: SPXY划分中使用的距离度量方式，可选值：'euclidean', 'cityblock', 'cosine'等
    distance_weight: X和Y距离的权重比例，0.5表示等权重，值越大越重视特征空间距离
    test_size: 测试集比例，默认0.3
    random_state: 随机种子，默认42
    
    返回:
    save_path: 保存的图像路径
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        # 提取模型
        if isinstance(saved_data, dict) and 'model' in saved_data:
            model = saved_data['model']
            feature_names = saved_data.get('feature_names', [])
        else:
            # 如果模型不是字典格式，假设它直接是模型对象
            model = saved_data
            feature_names = getattr(model, 'feature_names_in_', [])
        
        print(f"模型类型: {type(model)}")
        
        # 检查特征名称
        if not feature_names:
            print("警告: 模型中没有特征名称信息")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None
    
    # 加载CSV数据
    print(f"正在加载CSV数据: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"加载CSV数据时出错: {str(e)}")
        return None
    
    # 检查并筛选特征
    if feature_names:
        # 确保所有特征都在数据中
        available_features = [f for f in feature_names if f in data.columns]
        missing_features = [f for f in feature_names if f not in data.columns]
        
        if missing_features:
            print(f"警告: 以下特征在CSV数据中缺失: {missing_features}")
        
        if not available_features:
            print("错误: 没有可用的特征")
            return None
        
        # 提取特征数据
        X = data[available_features].copy()
    else:
        # 如果没有特征名称，尝试使用所有非目标列
        # 假设目标列名以"CD_"开头
        feature_cols = [col for col in data.columns if not col.startswith("CD_")]
        if not feature_cols:
            print("错误: 无法确定特征列")
            return None
        
        X = data[feature_cols].copy()
        available_features = feature_cols
    
    # 处理缺失值
    if X.isnull().any().any():
        print("警告: 特征数据中存在缺失值，将进行填充")
        X = X.fillna(X.mean())
    
    # 确定目标变量（假设模型文件名包含目标变量名）
    model_name = os.path.basename(model_path)
    if "CD_" in model_name:
        target_col = next((col for col in data.columns if col.startswith("CD_") and col in model_name), None)
    else:
        # 如果文件名中没有目标变量，则尝试找到第一个以CD_开头的列
        target_col = next((col for col in data.columns if col.startswith("CD_")), None)
    
    if target_col is None:
        print("警告: 无法确定目标变量，将使用全部数据进行SHAP分析")
        X_ordered = X.copy()
    else:
        print(f"目标变量: {target_col}")
        y = data[target_col].copy()
        
        # 使用SPXY方法划分训练集和测试集
        X_train, X_test, y_train, y_test = spxy_train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            distance_weight=distance_weight, distance_metric=distance_metric
        )
        
        # 获取特征重要性并排序
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                raise AttributeError("模型没有feature_importances_或coef_属性")
            
            # 确保长度匹配
            if len(importances) != len(available_features):
                print(f"警告: 特征重要性长度({len(importances)})与特征名称长度({len(available_features)})不匹配")
                if len(importances) > len(available_features):
                    importances = importances[:len(available_features)]
                else:
                    importances = np.pad(importances, (0, len(available_features) - len(importances)), 'constant')
            
            # 创建特征重要性字典和列表
            importance_dict = dict(zip(available_features, importances))
            
            # 按照重要性排序
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            ordered_features = [item[0] for item in sorted_features]
            
            print("特征按重要性排序:")
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.6f}")
        except Exception as e:
            print(f"获取特征重要性时出错: {str(e)}")
            # 如果无法获取特征重要性，使用原始顺序
            ordered_features = available_features
            print("使用原始特征顺序")
        
        # 重新排序特征数据
        X_ordered = X_test[ordered_features].copy()
    
    # 计算SHAP值
    try:
        # 尝试使用TreeExplainer
        if hasattr(model, 'predict_proba') or hasattr(model, 'predict'):
            print("使用TreeExplainer计算SHAP值")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_ordered)
            
            # 处理多输出情况
            if isinstance(shap_values, list):
                print(f"SHAP值是列表，长度为 {len(shap_values)}")
                shap_values = shap_values[0]  # 使用第一个输出
            
            print(f"SHAP值形状: {shap_values.shape}")
        else:
            raise AttributeError("模型没有predict或predict_proba方法")
    except Exception as e:
        print(f"计算SHAP值时出错: {str(e)}")
        return None
    

   # 创建SHAP图
    try:
        plt.figure(figsize=(10, 10), dpi=600)
        
        # 生成SHAP摘要图，显示特征名称
        shap.summary_plot(
            shap_values, 
            X_ordered,
            feature_names=ordered_features,  # 明确提供排序后的特征名称
            plot_type="dot",  # 使用原始的点图样式
            show=False,
            sort=False,  # 保持我们的自定义排序
            max_display=len(ordered_features)
        )
        
        # 获取当前坐标轴
        ax = plt.gca()
        
        # 清空Y轴标签（因为我们将手动添加文本）
        ax.set_yticks(range(len(ordered_features)))
        ax.set_yticklabels([])
        
        # 设置X轴标签
        plt.xlabel('SHAP Value (impact on model output)', fontsize=22, fontname='Times New Roman', fontweight='bold')
        plt.xticks(fontsize=20, fontname='Times New Roman', fontweight='bold')
        
        # 设置刻度线参数，保持与参考代码一致
        plt.tick_params(
            axis='x', 
            which='both',
            direction='out',     # 刻度向外
            length=5,            # 长度设为5点
            width=1.5,           # 刻度线宽度1.5
            labelsize=20         # 保持标签大小一致
        )
        
        # 设置Y轴刻度线
        plt.tick_params(
            axis='y', 
            which='both',
            direction='in',      # 刻度向内
            length=0,            # 与参考代码一致
            width=1.5,           # 宽度1.5
            right=False,         # 不在右侧显示刻度
            left=False           # 不在左侧显示刻度
        )
        
        # 设置坐标轴线宽度
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)  # 设置所有轴线宽度为1.5
        
        # 隐藏顶部和左侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
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
        
        # 在右侧添加自定义特征名称标签
        for i, feature in enumerate(ordered_features):
            # 获取自定义特征名称
            if feature in feature_mapping:
                feature_text = feature_mapping[feature]
            else:
                feature_text = feature
            
            # 添加特征名称到Y轴右侧，使用黑色加粗字体
            text_offset = ax.get_xlim()[1] * 1.05  # 根据X轴范围调整偏移量
            ax.text(text_offset, i, feature_text, 
                    color='black',
                    va='center', 
                    ha='left',
                    fontsize=22, 
                    fontweight='bold',
                    zorder=10)
        
        # 保存图像
        model_name = os.path.basename(model_path).split('_')[0]  # 提取模型名称
        save_path = os.path.join(output_dir, f'shap_{model_name}11.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP图已保存至: {save_path}")
        
        plt.close()
        return save_path
    except Exception as e:
        print(f"生成SHAP图时出错: {str(e)}")
        plt.close()
        return None

# 示例使用
if __name__ == "__main__":
    # 替换为你的实际路径
    model_path = r'C:\Users\AMDYE\OneDrive\python\results\soil\models\CD_soil_change_GBDT_model.pkl'
    csv_path = r'C:\Users\AMDYE\OneDrive\python\机器学习\特征工程和归一化\归一化\results\fully_processed_data(4).csv'
    output_dir = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\图像\soil'
    
    # 可以在这里设置SPXY方法的参数
    # distance_metric可选值: 'euclidean', 'cityblock', 'cosine' 等
    distance_metric = 'cityblock'  # 修改这里更改距离度量方式
    distance_weight = 0.5          # 修改这里更改特征空间和目标空间的权重比例
    test_size = 0.3                # 修改这里更改测试集比例
    random_state = 42              # 修改这里更改随机种子
    
    generate_standalone_shap_plot(
        model_path, 
        csv_path, 
        output_dir, 
        distance_metric=distance_metric,
        distance_weight=distance_weight,
        test_size=test_size,
        random_state=random_state
    )