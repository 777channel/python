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

def generate_standalone_shap_plot(model_path, csv_path, output_dir=None):
    """
    生成SHAP图，并按照模型重要性排序显示特征名称
    
    参数:
    model_path: 模型pickle文件路径
    csv_path: 数据CSV文件路径
    output_dir: 输出目录，默认为当前目录
    
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
    X_ordered = X[ordered_features].copy()
    
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
        
        # 设置Y轴标签字体
        plt.yticks(fontsize=14, fontname='Times New Roman', fontweight='bold')
        
        # 设置X轴标签
        plt.xlabel('SHAP Value (impact on model output)', fontsize=16, fontname='Times New Roman', fontweight='bold')
        plt.xticks(fontsize=14, fontname='Times New Roman', fontweight='bold')
        
        # 保存图像
        model_name = os.path.basename(model_path).split('_')[0]  # 提取模型名称
        save_path = os.path.join(output_dir, f'shap_soil.pdf')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
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
    
    generate_standalone_shap_plot(model_path, csv_path, output_dir)