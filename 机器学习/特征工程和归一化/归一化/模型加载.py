import pickle
import numpy as np
import os

def load_model_with_details(model_path):
    try:
        # 加载模型
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        # 提取模型
        model = saved_data['model']
        feature_names = saved_data.get('feature_names', [])
        
        # 打印模型类型
        print("模型类型:", type(model))

        # 打印模型属性
        print("\n模型所有属性:")
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

        return model, feature_names

    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


model_path = r'C:\Users\AMDYE\OneDrive\python\results\soil\models\CD_soil_change_XGBoost_model.pkl'
model, features = load_model_with_details(model_path)