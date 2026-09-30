import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import shap
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  
plt.rcParams['axes.unicode_minus'] = False

def load_data_with_materials_label_encoding(file_path):
    """
    读取数据文件，保留原料和改性信息并进行标签编码
    """
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(file_path)
    except Exception as e:
        try:
            # 如果CSV读取失败，尝试读取Excel文件
            df = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e2:
            print(f"无法读取文件: {e}\n尝试作为Excel读取也失败: {e2}")
            raise
    
    # 保留原料和改性列以及数值列
    cd_data = df[['原料', '改性', 'ph', 'Cd(II) (mg/L)', 'As³⁺ (mg/L)', 'Cd²⁺ 吸附量']].copy()
    cd_data.columns = ['material', 'modification', 'ph', 'cd_initial', 'as_initial', 'cd_adsorption']
    
    as_data = df[['原料_1', '改性_1', 'ph_1', 'Cd(II) (mg/L)_1', 'As³⁺ (mg/L)_1', 'As³⁺ 吸附量']].copy()
    as_data.columns = ['material', 'modification', 'ph', 'cd_initial', 'as_initial', 'as_adsorption']
    
    # 清理数据 - 数值列
    for dataset in [cd_data, as_data]:
        for col in ['ph', 'cd_initial', 'as_initial', 'cd_adsorption' if 'cd_adsorption' in dataset.columns else 'as_adsorption']:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
    
    # 使用标签编码处理原料和改性
    from sklearn.preprocessing import LabelEncoder
    
    # 对镉吸附数据进行编码
    le_material_cd = LabelEncoder()
    le_modification_cd = LabelEncoder()
    cd_data['material'] = le_material_cd.fit_transform(cd_data['material'])
    cd_data['modification'] = le_modification_cd.fit_transform(cd_data['modification'])
    
    # 对砷吸附数据进行编码
    le_material_as = LabelEncoder()
    le_modification_as = LabelEncoder()
    as_data['material'] = le_material_as.fit_transform(as_data['material'])
    as_data['modification'] = le_modification_as.fit_transform(as_data['modification'])
    
    # 打印编码映射，方便后续解释
    print("\n原料编码映射 (镉吸附数据):")
    for i, material in enumerate(le_material_cd.classes_):
        print(f"{material}: {i}")
    
    print("\n改性编码映射 (镉吸附数据):")
    for i, modification in enumerate(le_modification_cd.classes_):
        print(f"{modification}: {i}")
    
    # 移除含有NaN的行
    cd_data.dropna(inplace=True)
    as_data.dropna(inplace=True)
    
    return cd_data, as_data


def build_xgboost_model(X, y):
    """
    构建XGBoost模型
    """
    # 创建XGBoost回归模型
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # 拟合模型
    model.fit(X, y)
    
    # 计算R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    return model, r2

def generate_equation(model, poly, feature_names, model_name):
    """
    生成回归方程的可读表示
    """
    coeffs = model.coef_
    intercept = model.intercept_
    
    feature_names_out = poly.get_feature_names_out(feature_names)
    
    equation = f"{model_name} = {intercept:.4f}"
    for i, name in enumerate(feature_names_out[1:], 1):  # 跳过第一个特征（常数项）
        if abs(coeffs[i-1]) > 1e-6:  # 只显示大于阈值的系数
            sign = "+" if coeffs[i-1] > 0 else "-"
            equation += f" {sign} {abs(coeffs[i-1]):.4f}*{name}"
    
    return equation

def analyze_margin_effects(model, poly, feature_names, target_feature_idx, fixed_values=None):
    """
    分析某个特征对目标变量的边际效应
    target_feature_idx: 目标特征的索引（在原始特征中）
    fixed_values: 其他特征的固定值，用于计算特定条件下的边际效应
    """
    # 获取多项式特征的名称
    feature_names_out = poly.get_feature_names_out(feature_names)
    coeffs = model.coef_
    
    # 找出所有包含目标特征的项
    target_feature = feature_names[target_feature_idx]
    related_terms = []
    
    for i, name in enumerate(feature_names_out[1:], 1):  # 跳过常数项
        if target_feature in name.split():
            related_terms.append((name, coeffs[i-1]))
    
    # 分析边际效应方程
    margin_effect = f"∂(y)/∂({target_feature}) = "
    terms = []
    
    for name, coef in related_terms:
        if "^2" in name and target_feature in name:  # 平方项
            terms.append(f"{2*coef:.4f}*{target_feature}")
        elif target_feature in name:
            # 获取交互项中的其他特征
            other_features = [f for f in feature_names if f != target_feature and f in name]
            if other_features:
                terms.append(f"{coef:.4f}*{' '.join(other_features)}")
            else:
                terms.append(f"{coef:.4f}")
                
    margin_effect += " + ".join(terms) if terms else "0"
    
    # 如果提供了固定值，计算在这些条件下的具体边际效应
    if fixed_values is not None:
        total_effect = 0
        for name, coef in related_terms:
            if "^2" in name and target_feature in name:  # 平方项
                # 如果是目标特征的平方，系数乘以2，再乘以特征的值
                total_effect += 2 * coef * fixed_values.get(target_feature, 0)
            elif target_feature in name:
                # 如果是交互项，需要乘以其他特征的值
                other_features = [f for f in feature_names if f != target_feature and f in name]
                if other_features:
                    product = coef
                    for f in other_features:
                        product *= fixed_values.get(f, 0)
                    total_effect += product
                else:
                    # 如果是一次项，直接加系数
                    total_effect += coef
                    
        margin_effect += f"\n在给定条件下的边际效应值: {total_effect:.4f}"
    
    return margin_effect

def calculate_critical_points(model, poly, feature_names, target_idx, fixed_values):
    """
    计算临界点
    target_idx: 目标特征的索引（原始特征）
    fixed_values: 其他特征的固定值
    """
    # 获取多项式特征的名称
    feature_names_out = poly.get_feature_names_out(feature_names)
    coeffs = model.coef_
    
    target_feature = feature_names[target_idx]
    
    # 收集涉及目标特征的所有项
    linear_coef = 0  # 一次项系数
    squared_coef = 0  # 二次项系数
    interaction_coefs = {}  # 交互项系数
    
    for i, name in enumerate(feature_names_out[1:], 1):  # 跳过常数项
        if target_feature in name:
            if "^2" in name and target_feature in name:  # 平方项
                squared_coef = coeffs[i-1]
            elif name == target_feature:  # 一次项
                linear_coef = coeffs[i-1]
            else:  # 交互项
                for f in feature_names:
                    if f != target_feature and f in name:
                        interaction_coefs[f] = coeffs[i-1]
    
    # 计算边际效应为零时的临界点
    # 边际效应: linear_coef + 2*squared_coef*x + sum(interaction_coefs[f]*fixed_values[f])
    # 令其等于零解出x
    
    # 计算交互项的总贡献
    interaction_sum = sum(coef * fixed_values.get(f, 0) for f, coef in interaction_coefs.items())
    
    # 解方程: linear_coef + interaction_sum + 2*squared_coef*x = 0
    if abs(squared_coef) > 1e-6:  # 确保二次项系数不为零
        critical_point = -(linear_coef + interaction_sum) / (2 * squared_coef)
        return critical_point
    else:
        return None  # 如果二次项系数为零，没有临界点

def create_shap_analysis(X, y, model, poly, feature_names, title):
    """
    创建SHAP分析
    """
    # 转换为多项式特征
    X_poly = poly.transform(X)
    
    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_poly, 100))
    shap_values = explainer.shap_values(X_poly)
    
    # 创建SHAP摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_poly, feature_names=poly.get_feature_names_out(feature_names), 
                     show=False, plot_size=(10, 6))
    plt.title(f"{title} - SHAP特征重要性")
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}_SHAP_importance.png", dpi=300)
    plt.close()
    
    # 对原始特征的SHAP分析 - 使用KernelExplainer直接对模型进行解释
    explainer_orig = shap.KernelExplainer(lambda x: model.predict(poly.transform(x)), 
                                         shap.sample(X, 100))
    shap_values_orig = explainer_orig.shap_values(X)
    
    # 为每个特征创建依赖图
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values_orig, X, feature_names=feature_names, 
                            show=False, interaction_index=None)
        plt.title(f"{title} - {feature}的SHAP依赖图")
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}_SHAP_{feature}.png", dpi=300)
        plt.close()
    
    # 为两两特征组合创建交互图
    for i, feature1 in enumerate(feature_names):
        for j, feature2 in enumerate(feature_names):
            if i < j:  # 避免重复
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature1, shap_values_orig, X, feature_names=feature_names,
                                    interaction_index=feature2, show=False)
                plt.title(f"{title} - {feature1}和{feature2}的SHAP交互图")
                plt.tight_layout()
                plt.savefig(f"{title.replace(' ', '_')}_SHAP_{feature1}_{feature2}.png", dpi=300)
                plt.close()

def analyze_data_with_materials(file_path):
    """
    使用包含原料和改性信息的XGBoost模型分析数据
    """
    print("开始分析镉砷吸附数据 (包含原料和改性)...")
    
    # 加载数据并进行One-Hot编码
    cd_data, as_data = load_data_with_materials_label_encoding(file_path)
    
    print(f"数据加载完成。镉吸附数据: {len(cd_data)}行, 砷吸附数据: {len(as_data)}行")
    print(f"镉吸附数据包含的特征: {cd_data.columns.tolist()}")
    
    # 分析镉吸附
    print("\n=== 镉吸附分析 (XGBoost, 包含原料/改性) ===")
    X_cd = cd_data.drop('cd_adsorption', axis=1)
    y_cd = cd_data['cd_adsorption']
    
    # 构建XGBoost模型
    cd_model, cd_r2 = build_xgboost_model(X_cd, y_cd)
    
    print(f"XGBoost模型拟合度: R² = {cd_r2:.4f}")
    
    # 输出特征重要性
    feature_importance = cd_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X_cd.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\n特征重要性 (前10个):")
    print(feature_imp_df.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15))
    plt.title('镉吸附 - XGBoost特征重要性')
    plt.tight_layout()
    plt.savefig('Cd吸附量_XGBoost_特征重要性.png', dpi=300)
    plt.close()
    
    # 创建SHAP分析
    explainer = shap.TreeExplainer(cd_model)
    shap_values = explainer.shap_values(X_cd)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_cd, show=False)
    plt.title("镉吸附 - SHAP特征重要性")
    plt.tight_layout()
    plt.savefig("Cd吸附量_XGBoost_SHAP_importance.png", dpi=300)
    plt.close()
    
    # 为主要数值特征创建依赖图
    for feature in ['cd_initial', 'as_initial', 'ph']:
        if feature in X_cd.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_cd, show=False)
            plt.title(f"镉吸附 - {feature}的SHAP依赖图")
            plt.tight_layout()
            plt.savefig(f"Cd吸附量_XGBoost_SHAP_{feature}.png", dpi=300)
            plt.close()
    
    # 分析砷吸附
    print("\n=== 砷吸附分析 (XGBoost, 包含原料/改性) ===")
    X_as = as_data.drop('as_adsorption', axis=1)
    y_as = as_data['as_adsorption']
    
    # 构建XGBoost模型
    as_model, as_r2 = build_xgboost_model(X_as, y_as)
    
    print(f"XGBoost模型拟合度: R² = {as_r2:.4f}")
    
    # 输出特征重要性
    feature_importance = as_model.feature_importances_
    feature_imp_df = pd.DataFrame({
        'Feature': X_as.columns,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)
    
    print("\n特征重要性 (前10个):")
    print(feature_imp_df.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(15))
    plt.title('砷吸附 - XGBoost特征重要性')
    plt.tight_layout()
    plt.savefig('As吸附量_XGBoost_特征重要性.png', dpi=300)
    plt.close()
    
    # 创建SHAP分析
    explainer = shap.TreeExplainer(as_model)
    shap_values = explainer.shap_values(X_as)
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_as, show=False)
    plt.title("砷吸附 - SHAP特征重要性")
    plt.tight_layout()
    plt.savefig("As吸附量_XGBoost_SHAP_importance.png", dpi=300)
    plt.close()
    
    # 为主要数值特征创建依赖图
    for feature in ['cd_initial', 'as_initial', 'ph']:
        if feature in X_as.columns:
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature, shap_values, X_as, show=False)
            plt.title(f"砷吸附 - {feature}的SHAP依赖图")
            plt.tight_layout()
            plt.savefig(f"As吸附量_XGBoost_SHAP_{feature}.png", dpi=300)
            plt.close()
    
    # 比较模型性能
    print("\n=== 比较模型性能 ===")
    
    # 不含原料和改性的基础模型
    X_cd_basic = cd_data[['cd_initial', 'as_initial', 'ph']]
    X_as_basic = as_data[['cd_initial', 'as_initial', 'ph']]
    
    cd_model_basic, cd_r2_basic = build_xgboost_model(X_cd_basic, y_cd)
    as_model_basic, as_r2_basic = build_xgboost_model(X_as_basic, y_as)
    
    print("镉吸附模型性能比较:")
    print(f"基础模型 (仅浓度和pH): R² = {cd_r2_basic:.4f}")
    print(f"完整模型 (含原料和改性): R² = {cd_r2:.4f}")
    print(f"改进: {(cd_r2 - cd_r2_basic) * 100:.2f}%")
    
    print("\n砷吸附模型性能比较:")
    print(f"基础模型 (仅浓度和pH): R² = {as_r2_basic:.4f}")
    print(f"完整模型 (含原料和改性): R² = {as_r2:.4f}")
    print(f"改进: {(as_r2 - as_r2_basic) * 100:.2f}%")
    
    print("\n分析完成！所有图表已保存。")
    return "完整分析成功"

if __name__ == "__main__":
    # 指定Excel文件路径
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\cd+as data.csv"  # 替换为实际文件路径
    
    # 执行分析
    analyze_data_with_materials(file_path)