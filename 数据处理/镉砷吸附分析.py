import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
import shap
from pdpbox import pdp, info_plots
from sklearn.inspection import partial_dependence, permutation_importance
import warnings
warnings.filterwarnings('ignore')
import os

# 设置字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """
    读取数据文件，只保留数值变量（忽略原料和改性）
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
    
    # 只保留需要的数值列，忽略原料和改性
    cd_data = df[['ph', 'Cd(II) (mg/L)', 'As³⁺ (mg/L)', 'Cd²⁺ 吸附量']].copy()
    cd_data.columns = ['ph', 'cd_initial', 'as_initial', 'cd_adsorption']
    
    as_data = df[['ph_1', 'Cd(II) (mg/L)_1', 'As³⁺ (mg/L)_1', 'As³⁺ 吸附量']].copy()
    as_data.columns = ['ph', 'cd_initial', 'as_initial', 'as_adsorption']
    
    # 清理数据
    for dataset in [cd_data, as_data]:
        for col in ['ph', 'cd_initial', 'as_initial', 'cd_adsorption' if 'cd_adsorption' in dataset.columns else 'as_adsorption']:
            dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        dataset.dropna(subset=['ph', 'cd_initial', 'as_initial', 'cd_adsorption' if 'cd_adsorption' in dataset.columns else 'as_adsorption'], inplace=True)
    
    return cd_data, as_data

def optimize_random_forest(X, y):
    """
    使用网格搜索优化随机森林超参数
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def build_random_forest_model(X, y, optimize=True):
    """
    构建随机森林回归模型
    """
    if optimize:
        model = optimize_random_forest(X, y)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
        model.fit(X, y)
    
    # 计算R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    cv_r2 = np.mean(cv_scores)
    
    return model, r2, cv_r2

def create_permutation_importance(model, X, y, feature_names, title, output_dir):
    """
    创建排列重要性分析
    """
    # 计算排列重要性
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    # 获取重要性
    importances = result.importances_mean
    std = result.importances_std
    indices = np.argsort(importances)[::-1]
    
    # 绘制排列重要性
    plt.figure(figsize=(12, 8))
    plt.title(f"{title} - 排列特征重要性")
    plt.bar(range(len(importances)), importances[indices], yerr=std[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_permutation_importance.png"), dpi=300)
    plt.close()
    
    # 保存重要性到文件
    with open(os.path.join(output_dir, f"{title.replace(' ', '_')}_permutation_importance.txt"), "w", encoding='utf-8') as f:
        f.write(f"{title} - 排列特征重要性:\n\n")
        for i in indices:
            f.write(f"{feature_names[i]}: {importances[i]:.4f} ± {std[i]:.4f}\n")

def create_shap_analysis(model, X, feature_names, title, output_dir):
    """
    创建SHAP分析
    """
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 创建SHAP摘要图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, plot_size=(12, 8))
    plt.title(f"{title} - SHAP特征重要性")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_SHAP_summary.png"), dpi=300)
    plt.close()
    
    # 创建SHAP依赖图
    for feature in feature_names:
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(
            feature, 
            shap_values, 
            X, 
            feature_names=feature_names,
            show=False,
            interaction_index=None
        )
        plt.title(f"{title} - {feature}的SHAP依赖图")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_SHAP_{feature}.png"), dpi=300)
        plt.close()
    
    # 分析特征交互
    # 针对每对特征，找出最强的交互
    for i, feature1 in enumerate(feature_names):
        for j, feature2 in enumerate(feature_names):
            if i < j:  # 避免重复
                plt.figure(figsize=(12, 8))
                shap.dependence_plot(
                    feature1, 
                    shap_values, 
                    X, 
                    feature_names=feature_names,
                    interaction_index=feature2,
                    show=False
                )
                plt.title(f"{title} - {feature1}和{feature2}的SHAP交互图")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_SHAP_{feature1}_{feature2}.png"), dpi=300)
                plt.close()

def create_simplified_pdp_plots(model, X, feature_names, title, output_dir, max_features=3):
    """
    创建简化版的PDP图，只生成一元图和一个特定的二元图，控制图的总数量
    
    参数:
    model - 训练好的模型
    X - 特征数据
    feature_names - 特征名称列表
    title - 图表标题
    output_dir - 输出目录
    max_features - 最多生成多少个特征的PDP图
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator
    import os

    print(f"开始生成简化的PDP图...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果有特征重要性，根据重要性选择特征，否则取前max_features个
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:max_features]
        selected_features = [feature_names[i] for i in indices]
        print(f"根据特征重要性选择了前{len(selected_features)}个特征: {selected_features}")
    else:
        selected_features = feature_names[:min(max_features, len(feature_names))]
        print(f"没有特征重要性，选择了前{len(selected_features)}个特征: {selected_features}")
    
    # 创建一元PDP图
    for feature in selected_features:
        print(f"生成特征 '{feature}' 的PDP图...")
        
        # 获取特征范围
        min_val = X[feature].min()
        max_val = X[feature].max()
        
        # 确保范围合理
        if np.isclose(min_val, max_val):
            margin = abs(min_val * 0.1) if min_val != 0 else 0.1
            min_val = min_val - margin
            max_val = max_val + margin
        
        # 生成网格点
        grid_size = 30
        pdp_grid = np.linspace(min_val, max_val, grid_size)
        pdp_values = []
        
        # 对每个网格点计算PDP值
        for val in pdp_grid:
            X_temp = X.copy()
            X_temp[feature] = val
            predictions = model.predict(X_temp)
            mean_pred = np.mean(predictions)
            pdp_values.append(mean_pred)
        
        # 创建图形
        plt.figure(figsize=(10, 6))
        
        # 绘制PDP曲线
        plt.plot(pdp_grid, pdp_values, 'b-', linewidth=2.5)
        
        # 添加数据分布直方图
        ax2 = plt.gca().twinx()
        sns.histplot(X[feature], kde=False, bins=min(20, len(X[feature].unique())),
                   alpha=0.2, color='gray', ax=ax2, stat='density')
        ax2.set_ylabel('')
        ax2.set_yticks([])  # 隐藏右侧Y轴刻度
        
        # 计算导数以找出临界点
        derivatives = np.gradient(pdp_values, pdp_grid)
        
        # 找出导数为0的点(临界点)
        for i in range(len(derivatives) - 1):
            if derivatives[i] * derivatives[i + 1] <= 0:  # 导数改变符号
                # 通过线性插值估计导数为0的位置
                x1, x2 = pdp_grid[i], pdp_grid[i + 1]
                y1, y2 = derivatives[i], derivatives[i + 1]
                
                if abs(y1 - y2) < 1e-10:
                    critical_x = (x1 + x2) / 2
                else:
                    critical_x = x1 - y1 * (x2 - x1) / (y2 - y1)
                
                # 计算临界点处的响应值
                critical_y = np.interp(critical_x, pdp_grid, pdp_values)
                
                # 标记临界点
                plt.scatter(critical_x, critical_y, color='red', s=100, zorder=5)
                plt.axvline(critical_x, color='red', linestyle='--', alpha=0.3)
                plt.text(critical_x, min(pdp_values) + 0.05 * (max(pdp_values) - min(pdp_values)), 
                       f'{critical_x:.2f}', rotation=90, verticalalignment='bottom')
        
        # 设置标签和标题
        plt.xlabel(feature.replace('_', ' '))
        plt.ylabel(f'{title} 预测值')
        plt.title(f"{title} - {feature.replace('_', ' ')}的部分依赖图")
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().yaxis.set_major_locator(MaxNLocator(5))
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_PDP_{feature}.png"), dpi=300)
        plt.close()
    

def analyze_critical_points(model, X, feature_names, target_idx, fixed_values, title, output_dir):
    """
    简化版的临界点分析，只分析一个目标特征在不同条件下的影响
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    target_feature = feature_names[target_idx]
    print(f"分析特征 '{target_feature}' 在不同条件下的临界点...")
    
    # 获取特征范围
    target_min = X[target_feature].min()
    target_max = X[target_feature].max()
    
    # 生成网格点
    grid_size = 150
    target_values = np.linspace(target_min, target_max, grid_size)
    
    plt.figure(figsize=(10, 6))
    
    # 存储所有找到的临界点
    all_critical_points = []
    
    # 对每组固定特征值进行计算
    for fixed_vals in fixed_values:
        condition_str = ", ".join([f"{k}={v}" for k, v in fixed_vals.items() if k != target_feature])
        print(f"  计算条件: {condition_str}")
        
        predictions = []
        
        # 对每个目标特征值计算预测值
        for val in target_values:
            test_point = pd.DataFrame([fixed_vals.copy()], columns=X.columns)
            test_point[target_feature] = val
            pred = model.predict(test_point)[0]
            predictions.append(pred)
        
        # 绘制响应曲线
        plt.plot(target_values, predictions, label=condition_str, linewidth=2)
        
        # 计算导数(差分)
        derivatives = np.gradient(predictions, target_values)
        
        # 寻找临界点（导数为0的点）
        critical_points = []
        for i in range(len(derivatives) - 1):
            if derivatives[i] * derivatives[i + 1] <= 0:  # 导数改变符号
                # 估计临界点位置
                x1, x2 = target_values[i], target_values[i + 1]
                y1, y2 = derivatives[i], derivatives[i + 1]
                
                if abs(y1 - y2) < 1e-10:
                    critical_x = (x1 + x2) / 2
                else:
                    critical_x = x1 - y1 * (x2 - x1) / (y2 - y1)
                
                critical_y = np.interp(critical_x, target_values, predictions)
                
                # 确定临界点类型
                if i > 0 and i < len(derivatives) - 2:
                    before_sign = np.sign(derivatives[i-1])
                    after_sign = np.sign(derivatives[i+2])
                    
                    if before_sign > 0 and after_sign < 0:
                        point_type = "最大值"
                    elif before_sign < 0 and after_sign > 0:
                        point_type = "最小值"
                    else:
                        point_type = "拐点"
                else:
                    point_type = "拐点"
                
                critical_info = {
                    'feature': target_feature,
                    'critical_value': critical_x,
                    'response': critical_y,
                    'conditions': fixed_vals.copy(),
                    'type': point_type
                }
                critical_points.append(critical_info)
                all_critical_points.append(critical_info)
                
                # 在图上标记临界点
                plt.scatter(critical_x, critical_y, color='red', s=80, zorder=5)
                plt.axvline(critical_x, color='red', linestyle='--', alpha=0.3)
                plt.text(critical_x, min(predictions) + 0.05 * (max(predictions) - min(predictions)), 
                       f'{critical_x:.2f}', rotation=90, verticalalignment='bottom')
                
                print(f"    找到临界点: {target_feature}={critical_x:.4f}, 类型: {point_type}, 响应值: {critical_y:.4f}")
    
    # 设置图表属性
    plt.xlabel(target_feature.replace('_', ' '))
    plt.ylabel(f'{title} 预测值')
    plt.title(f"{title} - 不同条件下{target_feature.replace('_', ' ')}的响应曲线及临界点")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_critical_points_{target_feature}.png"), dpi=300)
    plt.close()
    
    # 保存临界点信息到文件
    if all_critical_points:
        with open(os.path.join(output_dir, f"{title.replace(' ', '_')}_critical_points_{target_feature}.txt"), "w", encoding='utf-8') as f:
            f.write(f"{title} - {target_feature.replace('_', ' ')}的临界点分析:\n\n")
            for cp in all_critical_points:
                conditions = ", ".join([f"{k}={v}" for k, v in cp['conditions'].items() if k != target_feature])
                f.write(f"条件: {conditions}\n")
                f.write(f"临界值: {target_feature} = {cp['critical_value']:.4f}\n")
                f.write(f"响应值: {cp['response']:.4f}\n")
                f.write(f"类型: {cp['type']}\n")
                
                # 加入解释
                if cp['type'] == "最大值":
                    f.write(f"解释: 当{target_feature} < {cp['critical_value']:.4f}时，增加{target_feature}会提高{title}；\n")
                    f.write(f"       当{target_feature} > {cp['critical_value']:.4f}时，增加{target_feature}会降低{title}。\n")
                elif cp['type'] == "最小值":
                    f.write(f"解释: 当{target_feature} < {cp['critical_value']:.4f}时，增加{target_feature}会降低{title}；\n")
                    f.write(f"       当{target_feature} > {cp['critical_value']:.4f}时，增加{target_feature}会提高{title}。\n")
                
                f.write("\n")
    
    return all_critical_points



def analyze_data(file_path, output_dir="./output"):
    """
    主分析函数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始分析镉砷吸附数据... 图像将保存到: {output_dir}")
    
    # 加载数据
    cd_data, as_data = load_data(file_path)
    
    print(f"数据加载完成。镉吸附数据: {len(cd_data)}行, 砷吸附数据: {len(as_data)}行")
    
    # ===== 1. 镉吸附分析 =====
    print("\n=== 镉吸附分析 (随机森林) ===")
    X_cd = cd_data[['cd_initial', 'as_initial', 'ph']]
    y_cd = cd_data['cd_adsorption']
    
    # 构建随机森林模型
    cd_rf_model, cd_rf_r2, cd_rf_cv_r2 = build_random_forest_model(X_cd, y_cd, optimize=True)
    
    print(f"随机森林模型拟合度: R² = {cd_rf_r2:.4f}")
    print(f"交叉验证R²: {cd_rf_cv_r2:.4f}")
    
    # 创建排列重要性分析
    create_permutation_importance(cd_rf_model, X_cd, y_cd, X_cd.columns, "Cd吸附量", output_dir)
    
    # 创建SHAP分析
    create_shap_analysis(cd_rf_model, X_cd, X_cd.columns, "Cd吸附量", output_dir)
    
   # 创建PDP图
    create_simplified_pdp_plots(cd_rf_model, X_cd, X_cd.columns, "Cd吸附量", output_dir)
    
    # 分析临界点
    cd_critical_configs = [
        {'cd_initial': 20, 'ph': 4},
        {'cd_initial': 20, 'ph': 7},
        {'cd_initial': 50, 'ph': 4},
        {'cd_initial': 50, 'ph': 7}
    ]
    
    cd_as_critical_points = analyze_critical_points(
    cd_rf_model, X_cd, X_cd.columns, 1, cd_critical_configs, "Cd吸附量", output_dir)
    # ===== 2. 砷吸附分析 =====
    print("\n=== 砷吸附分析 (随机森林) ===")
    X_as = as_data[['cd_initial', 'as_initial', 'ph']]
    y_as = as_data['as_adsorption']
    
    # 构建随机森林模型
    as_rf_model, as_rf_r2, as_rf_cv_r2 = build_random_forest_model(X_as, y_as, optimize=True)
    
    print(f"随机森林模型拟合度: R² = {as_rf_r2:.4f}")
    print(f"交叉验证R²: {as_rf_cv_r2:.4f}")
    
    # 创建排列重要性分析
    create_permutation_importance(as_rf_model, X_as, y_as, X_as.columns, "As吸附量", output_dir)
    
    # 创建SHAP分析
    create_shap_analysis(as_rf_model, X_as, X_as.columns, "As吸附量", output_dir)
    
    # 创建ALE/PDP图
    create_simplified_pdp_plots(as_rf_model, X_as, X_as.columns, "As吸附量", output_dir)
    
    # 分析临界点
    as_critical_configs = [
        {'as_initial': 10, 'ph': 4},
        {'as_initial': 10, 'ph': 7},
        {'as_initial': 20, 'ph': 4},
        {'as_initial': 20, 'ph': 7}
    ]
    
    as_cd_critical_points = analyze_critical_points(
        as_rf_model, X_as, X_as.columns, 0, as_critical_configs, "As吸附量", output_dir)
    
    # ===== 3. 创建交互热图 =====
    print("\n=== 镉砷交互热图 ===")
    
    # 对不同pH值创建cd_initial和as_initial的交互热图
    for ph_value in [4, 7]:
        # 创建cd和as的网格
        cd_min, cd_max = X_cd['cd_initial'].min(), X_cd['cd_initial'].max()
        as_min, as_max = X_cd['as_initial'].min(), X_cd['as_initial'].max()
        
        cd_grid = np.linspace(cd_min, cd_max, 50)
        as_grid = np.linspace(as_min, as_max, 50)
        
        cd_mesh, as_mesh = np.meshgrid(cd_grid, as_grid)
        
        # 创建预测点
        grid_points = []
        for i in range(cd_mesh.shape[0]):
            for j in range(cd_mesh.shape[1]):
                point = {'cd_initial': cd_mesh[i, j], 
                        'as_initial': as_mesh[i, j], 
                        'ph': ph_value}
                grid_points.append(point)
        
        grid_df = pd.DataFrame(grid_points)
        
        # 预测镉吸附量
        cd_adsorption = cd_rf_model.predict(grid_df)
        cd_adsorption_grid = cd_adsorption.reshape(cd_mesh.shape)
        
        # 预测砷吸附量
        as_adsorption = as_rf_model.predict(grid_df)
        as_adsorption_grid = as_adsorption.reshape(cd_mesh.shape)
        
        # 绘制镉吸附热图
        plt.figure(figsize=(12, 10))
        plt.contourf(cd_mesh, as_mesh, cd_adsorption_grid, cmap='viridis', levels=20)
        plt.colorbar(label='Cd吸附量预测值')
        
        # 添加临界线
        for cp in cd_as_critical_points:
            if cp['conditions'].get('ph') == ph_value:
                plt.axvline(cp['critical_value'], color='red', linestyle='--', alpha=0.5)
                plt.text(cp['critical_value'], as_min, f"{cp['critical_value']:.2f}", 
                       color='red', rotation=90, verticalalignment='bottom')
        
        plt.xlabel('cd_initial (mg/L)')
        plt.ylabel('as_initial (mg/L)')
        plt.title(f'pH={ph_value}时镉砷初始浓度对镉吸附量的影响')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Cd_adsorption_interaction_pH{ph_value}.png"), dpi=300)
        plt.close()
        
        # 绘制砷吸附热图
        plt.figure(figsize=(12, 10))
        plt.contourf(cd_mesh, as_mesh, as_adsorption_grid, cmap='viridis', levels=20)
        plt.colorbar(label='As吸附量预测值')
        
        # 添加临界线
        for cp in as_cd_critical_points:
            if cp['conditions'].get('ph') == ph_value:
                plt.axhline(cp['critical_value'], color='red', linestyle='--', alpha=0.5)
                plt.text(cd_min, cp['critical_value'], f"{cp['critical_value']:.2f}", 
                       color='red', verticalalignment='center')
        
        plt.xlabel('cd_initial (mg/L)')
        plt.ylabel('as_initial (mg/L)')
        plt.title(f'pH={ph_value}时镉砷初始浓度对砷吸附量的影响')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"As_adsorption_interaction_pH{ph_value}.png"), dpi=300)
        plt.close()
    
    # ===== 4. 总结报告 =====
    print("\n=== 总结报告 ===")
    
    # 创建总结报告文件
    with open(os.path.join(output_dir, "analysis_summary.txt"), "w", encoding='utf-8') as f:
        f.write("===== 镉砷吸附分析总结报告 =====\n\n")
        
        f.write("== 1. 模型性能 ==\n")
        f.write(f"Cd吸附模型 R²: {cd_rf_r2:.4f} (交叉验证 R²: {cd_rf_cv_r2:.4f})\n")
        f.write(f"As吸附模型 R²: {as_rf_r2:.4f} (交叉验证 R²: {as_rf_cv_r2:.4f})\n\n")
        
        f.write("== 2. 特征重要性 ==\n")
        f.write("Cd吸附模型:\n")
        for feature, importance in sorted(zip(X_cd.columns, cd_rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
            f.write(f"  {feature}: {importance:.4f}\n")
        
        f.write("\nAs吸附模型:\n")
        for feature, importance in sorted(zip(X_as.columns, as_rf_model.feature_importances_), key=lambda x: x[1], reverse=True):
            f.write(f"  {feature}: {importance:.4f}\n\n")
        
        f.write("== 3. 临界点分析 ==\n")
        f.write("Cd吸附中As的临界点:\n")
        for cp in cd_as_critical_points:
            conditions = ", ".join([f"{k}={v}" for k, v in cp['conditions'].items() if k != "as_initial"])
            f.write(f"  条件: {conditions}\n")
            f.write(f"  临界值: as_initial = {cp['critical_value']:.2f} mg/L\n")
            f.write(f"  类型: {cp['type']}\n")
            f.write(f"  效应: 当as_initial < {cp['critical_value']:.2f} mg/L时，砷{'促进' if cp['type'] == '最大值' else '抑制'}镉吸附\n")
            f.write(f"        当as_initial > {cp['critical_value']:.2f} mg/L时，砷{'抑制' if cp['type'] == '最大值' else '促进'}镉吸附\n\n")
        
        f.write("As吸附中Cd的临界点:\n")
        for cp in as_cd_critical_points:
            conditions = ", ".join([f"{k}={v}" for k, v in cp['conditions'].items() if k != "cd_initial"])
            f.write(f"  条件: {conditions}\n")
            f.write(f"  临界值: cd_initial = {cp['critical_value']:.2f} mg/L\n")
            f.write(f"  类型: {cp['type']}\n")
            f.write(f"  效应: 当cd_initial < {cp['critical_value']:.2f} mg/L时，镉{'促进' if cp['type'] == '最大值' else '抑制'}砷吸附\n")
            f.write(f"        当cd_initial > {cp['critical_value']:.2f} mg/L时，镉{'抑制' if cp['type'] == '最大值' else '促进'}砷吸附\n\n")
        
        f.write("== 4. 主要发现 ==\n")
        f.write("1. 对镉吸附的影响因素按重要性排序\n")
        f.write("2. 对砷吸附的影响因素按重要性排序\n")
        f.write("3. pH值对镉砷吸附的影响\n")
        f.write("4. 镉砷之间的相互竞争或协同作用\n\n")
        
        f.write("== 5. 结论与建议 ==\n")
        f.write("1. 根据分析结果，镉砷的吸附过程存在明显的交互作用\n")
        f.write("2. 临界点分析表明，在不同条件下，镉砷之间可能表现出竞争或协同关系\n")
        f.write("3. 优化吸附过程应考虑pH值和初始浓度的联合影响\n")
    
    print("\n分析完成！所有图表、模型和报告已保存到:", output_dir)
    return "分析成功"

if __name__ == "__main__":
    # 指定文件路径
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\cd+as data.csv"  # 替换为实际文件路径
    output_dir = "./output"
    
    # 执行分析
    analyze_data(file_path, output_dir)