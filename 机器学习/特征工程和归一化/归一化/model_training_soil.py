import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import warnings
import traceback
import time
from sklearn.ensemble import ExtraTreesRegressor
from data_normalization import full_preprocessing_pipeline
warnings.filterwarnings('ignore')
import pickle
import shap
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


def initialize_models():
    """
    初始化所有要测试的模型及其超参数网格。
    您可根据需要修改搜索范围。
    """
    models = {
        'LR': {
            'model': LinearRegression(),
            'params': {}
        },
        'Bayesian Ridge': {
            'model': BayesianRidge(),
            'params': {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [200, 300],
                'max_depth': [4, 5, 6],          # 限制深度
                'learning_rate': [0.02, 0.03],   # 降低学习率
                'reg_alpha': [0.1, 0.2],         # 增强L1正则化
                'reg_lambda': [1.0],        # 增强L2正则化
                'subsample': [0.8],         # 减少样本使用比例
                'colsample_bytree': [0.7, 0.8],  # 减少特征使用比例
                'min_child_weight': [3, 5]       # 增加最小子权重
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [300],
                'num_leaves': [31, 47],          # 减少叶子节点数量
                'max_depth': [5, 6],             # 限制深度
                'learning_rate': [0.02],   # 降低学习率
                'min_child_samples': [20],   # 增加叶节点最小样本数
                'min_split_gain': [0.05, 0.1],   # 增加分裂所需的最小增益
                'reg_alpha': [0.1, 0.2],         # 增强L1正则化
                'reg_lambda': [0.5, 1.0],        # 增强L2正则化
                'subsample': [0.7],         # 减少样本使用比例
                'colsample_bytree': [0.7],  # 减少特征使用比例
                'path_smooth': [1, 3]            # 增加平滑参数
            }
        },
        'RF': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [200, 300, 400],
                'max_depth': [None, 20, 30],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [1, 2],
                'max_features': [0.8, 'sqrt', 'log2']
            }
        },
        'GBDT': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [200, 300],  # 增加树的数量
                'max_depth': [4, 5, 6],  # 增加树的深度
                'learning_rate': [0.02, 0.03],  # 调整学习率
                'subsample': [0.7, 0.8],  # 增加样本使用比例
                'min_samples_split': [5, 8],  # 减少分裂所需的最小样本数
                'min_samples_leaf': [3, 5],  # 减少叶节点所需的最小样本数
                'max_features': [0.6, 0.7], # 增加特征使用比例
                'alpha': [0.8, 0.9],  # 注意这里是 'alpha' 而不是 'alpha'
                'init': [None]  # 显式设置init为None，避免SHAP分析时出错
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [400, 500],          # 回归到中等数量的树
                'max_depth': [None, 10],             # 尝试不限制深度或适度限制
                'min_samples_split': [2, 3],         # 减少分裂限制
                'min_samples_leaf': [1, 2],          # 降低叶节点限制
                'max_features': ['auto', 0.8, 0.9],  # 尝试自动选择或较高比例
                'bootstrap': [False, True],          # 测试两种方式
                'ccp_alpha': [0.0, 0.001]            # 减小或移除剪枝
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
        }
    }
    return models


def find_best_params(model_name, model_info, X_train, y_train, cv=5):
    """
    使用网格搜索找到模型的最佳参数并返回已拟合的模型
    """
    model = model_info['model']
    param_grid = model_info['params']

    if not param_grid:  # 若无需要调优的参数
        # 确保模型已经拟合
        model.fit(X_train, y_train)
        return model

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=0
    )

    try:
        grid_search.fit(X_train, y_train)
        print(f"\n{model_name} 最佳参数:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"最佳R²得分: {grid_search.best_score_:.4f}")

        # 返回已拟合的最佳模型
        best_model = grid_search.best_estimator_

        # 额外检查：对于GBDT模型，确保init属性已设置
        if model_name == 'GBDT' and not hasattr(best_model, 'init_'):
            print("注意: GBDT模型未正确初始化，这可能导致SHAP分析失败")
            # 尝试再次拟合模型
            best_model.fit(X_train, y_train)

        return best_model
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认参数继续")
        # 确保返回拟合后的模型
        model.fit(X_train, y_train)
        return model


def plot_all_models_comparison(results, target_col, output_dir='.'):
    """
    为目标变量创建一个大图，包含所有模型的预测比较
    """
    if not results:
        print(f"没有可用于绘图的结果: {target_col}")
        return

    n_rows = 3
    n_cols = 3
    model_order = ['LR', 'Bayesian Ridge', 'Ridge', 'XGBoost', 'LightGBM',
                   'RF', 'GBDT', 'ExtraTrees', 'SVR']

    # 增大图片尺寸
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 25))
    axes = axes.flatten()

    train_color = 'red'
    test_color = 'blue'

    model_results = {r['Model']: r for r in results}
    best_model = max(results, key=lambda x: x['R2_Test'])['Model']

    for i, model_name in enumerate(model_order):
        ax = axes[i]
        if model_name in model_results:
            r = model_results[model_name]
            y_train = r['Train_Values']
            y_train_pred = r['Train_Predictions']
            y_test = r['Test_Values']
            y_test_pred = r['Test_Predictions']

            r2_train = r['R2_Train']
            r2_test = r['R2_Test']
            rmse_train = r['RMSE_Train']
            rmse_test = r['RMSE_Test']
            mae_train = r['MAE_Train']
            mae_test = r['MAE_Test']

            # 增大点大小并增加透明度
            ax.scatter(y_train, y_train_pred, color=train_color, marker='o', alpha=0.6, s=50)
            ax.scatter(y_test, y_test_pred, color=test_color, marker='^', alpha=0.6, s=50)

            all_y_values = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
            min_val = np.min(all_y_values)
            max_val = np.max(all_y_values)

            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.0)

            # 增大文字大小和调整位置
            train_text = (f"Train\nR²: {r2_train:.3f}\nRMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}")
            ax.text(0.05, 0.95, train_text, transform=ax.transAxes, fontsize=24,fontweight='bold', verticalalignment='top',
                    bbox=dict(facecolor=train_color, alpha=0.1, edgecolor=train_color, linestyle='--'))

            test_text = (f"Test\nR²: {r2_test:.3f}\nRMSE: {rmse_test:.3f}\nMAE: {mae_test:.3f}")
            ax.text(0.68, 0.25, test_text, transform=ax.transAxes, fontsize=24,fontweight='bold', verticalalignment='top',
                    bbox=dict(facecolor=test_color, alpha=0.1, edgecolor=test_color, linestyle='--'))

            model_label = chr(97 + i)
            is_best = " (Best)" if model_name == best_model else ""
            # 增大标题字体
            ax.set_title(f'({model_label}) {model_name}{is_best}\nR² = {r2_test:.3f}', fontsize=26,fontweight='bold')

            padding = (max_val - min_val) * 0.05
            ax.set_xlim(min_val - padding, max_val + padding)
            ax.set_ylim(min_val - padding, max_val + padding)
            # 增加网格线透明度
            ax.grid(False)
            # 增大坐标轴刻度字体并设置为粗体
            for label in ax.get_xticklabels():
                label.set_fontsize(24)
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontsize(24)
                label.set_fontweight('bold')
        else:
            model_label = chr(97 + i)
            ax.text(0.5, 0.5, f'({model_label}) {model_name}\n无数据',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=24)
            ax.axis('off')

    # 增大坐标轴标题字体
    fig.text(0.5, 0.04, 'Observed Values', ha='center', fontsize=26,fontweight='bold')
    fig.text(0.04, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=26,fontweight='bold')

    plt.suptitle(f'Soil-Cd', fontsize=28, y=0.93,fontweight='bold')

    # 调整紧凑布局，给标题留出更多空间
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_all_models_comparison.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"所有模型比较图已保存至 {save_path}")
    plt.close()


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import shap
from sklearn.inspection import partial_dependence


def plot_enhanced_feature_importance(importance_dict, target_col, model_name, output_dir='.', top_n=10):
    """
    绘制特征重要性条形图和饼图，按照新的四类分类方式
    """
    # 创建图形和网格
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # 对特征名称进行处理，将下划线替换为空格
    cleaned_importance = {k.replace('_', ' '): v for k, v in importance_dict.items()}

    # 排序特征，从高到低
    sorted_features = sorted(cleaned_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]

    # 按照新的四类进行分类
    biochar_features = []  # 生物炭特性
    soil_features = []  # 土壤特性
    rice_features = []  # 水稻特性
    exp_features = []  # 实验条件

    # 特征分类映射 - 原始特征名到类别
    feature_categories = {
        # 生物炭特性
        'pH': 'biochar',
        'temperature': 'biochar',
        'material_type_code': 'biochar',

        # 土壤特性
        'pHsoil': 'soil',
        'organicmatter': 'soil',
        'log_Cdcontent': 'soil',
        'CEC': 'soil',

        # 水稻特性
        'variety_binary': 'rice',

        # 实验条件
        'Charcoal_application_ratio': 'exp',
        'irrigation_type': 'exp',
        'Charcoal x pHsoil': 'exp',
        'irrigation_x_pHsoil': 'exp',
        'pHsoil_x_variety': 'exp'
    }

    # 分类特征
    for feature, value in sorted_features:
        # 还原特征名（去除空格）
        original_feature = feature.replace(' ', '_')

        # 根据特征类别进行分类
        category = feature_categories.get(original_feature, 'unknown')

        if category == 'biochar':
            biochar_features.append((feature, value))
        elif category == 'soil':
            soil_features.append((feature, value))
        elif category == 'rice':
            rice_features.append((feature, value))
        else:  # 默认为实验条件
            exp_features.append((feature, value))

    # 计算每个类别的总重要性
    biochar_total = sum(v for _, v in biochar_features)
    soil_total = sum(v for _, v in soil_features)
    rice_total = sum(v for _, v in rice_features)
    exp_total = sum(v for _, v in exp_features)

    # 左侧子图：特征重要性条形图
    ax1 = fig.add_subplot(gs[0])

    # 提取前N个特征并按重要性排序
    features = []
    importance = []
    colors = []

    # 类别颜色
    category_colors = {
        'biochar': '#5DA5DA',  # 蓝色 - 生物炭特性
        'soil': '#60BD68',  # 绿色 - 土壤特性
        'rice': '#F15854',  # 红色 - 水稻特性
        'exp': '#FAA43A'  # 橙色 - 实验条件
    }

    # 将top_n特征按重要性排序，并添加颜色
    for feature, value in top_features:
        features.append(feature)
        importance.append(value)

        # 还原特征名（去除空格）
        original_feature = feature.replace(' ', '_')

        # 根据特征类别分配颜色
        category = feature_categories.get(original_feature, 'unknown')
        colors.append(category_colors.get(category, '#BBBBBB'))  # 默认灰色

    # 反转顺序，使得最重要的在顶部
    features = features[::-1]
    importance = importance[::-1]
    colors = colors[::-1]

    # 创建水平条形图
    bars = ax1.barh(range(len(features)), importance, align='center', height=0.6, color=colors)

    # 设置Y轴标签为特征名称，增大字体
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features, fontsize=14)

    # 设置X轴和标题，增大字体
    ax1.set_xlabel('Importance', fontsize=16)
    ax1.set_ylabel('Feature', fontsize=16)

    # 将target_col中的下划线替换为空格
    formatted_target = target_col.replace('_', ' ')
    ax1.set_title(f'Top {top_n} Feature Importance for {formatted_target}', fontsize=18)

    # 添加类别图例 - 使用英文标签
    import matplotlib.patches as mpatches
    biochar_patch = mpatches.Patch(color=category_colors['biochar'], label='Biochar Properties')
    soil_patch = mpatches.Patch(color=category_colors['soil'], label='Soil Properties')
    rice_patch = mpatches.Patch(color=category_colors['rice'], label='Rice Properties')
    exp_patch = mpatches.Patch(color=category_colors['exp'], label='Experimental Conditions')
    ax1.legend(handles=[biochar_patch, soil_patch, rice_patch, exp_patch],
               loc='lower right', fontsize=12)

    # 右侧子图：饼图
    ax2 = fig.add_subplot(gs[1])

    # 饼图数据
    pie_labels = ['Biochar Properties', 'Soil Properties', 'Rice Properties', 'Experimental Conditions']
    pie_sizes = [biochar_total, soil_total, rice_total, exp_total]

    # 计算百分比，最多保留两位小数
    percentages = [100 * size / sum(pie_sizes) for size in pie_sizes]

    # 饼图颜色 - 与条形图对应
    colors = [category_colors['biochar'], category_colors['soil'],
              category_colors['rice'], category_colors['exp']]

    # 绘制饼图
    wedges, texts, autotexts = ax2.pie(
        pie_sizes,
        labels=None,  # 不在饼图上直接显示标签
        colors=colors,
        autopct='%1.1f%%',  # 在饼图上显示百分比
        startangle=90,  # 从顶部开始
        wedgeprops=dict(width=0.5, edgecolor='w'),  # 设置为环形
        pctdistance=0.85  # 百分比文字的位置
    )

    # 设置自动文本的样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_weight('bold')

    # 在饼图中央添加一个标签来代替图例 - 使用英文标签
    legend_text = '\n'.join([f'{label} ({perc:.2f}%)' for label, perc in zip(pie_labels, percentages)])
    ax2.text(0, 0, legend_text, horizontalalignment='center',
             verticalalignment='center', fontsize=12,
             transform=ax2.transAxes)

    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_{model_name}_enhanced_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"增强特征重要性图已保存至 {save_path}")
    plt.close()


def plot_shap_values(model, X_test, feature_names, target_col, model_name, output_dir='.'):
    """
    绘制SHAP值分析图，包括汇总图和详细影响图，去除标签中的下划线
    """
    try:
        # 检查模型是否已拟合
        if not hasattr(model, 'predict'):
            print(f"错误: 模型 {model_name} 没有 'predict' 方法，可能未经过拟合")
            return

        # 尝试调用预测方法，看是否会引发错误
        try:
            model.predict(X_test.iloc[:1])
        except Exception as e:
            print(f"预测测试失败，模型可能未经过拟合: {str(e)}")
            return

        # 替换特征名称中的下划线为空格，使标签更美观
        clean_feature_names = [name.replace('_', ' ') for name in feature_names]

        # 特征标签映射 - 使用更好的标签
        feature_labels_map = {
            'organicmatter': 'Organic Matter',
            'pHsoil': 'Soil pH',
            'Charcoal application ratio': 'Biochar Application Ratio',
            'log Cdcontent': 'Initial Cd Content',
            'pHsoil x variety': 'Soil pH × Rice Variety',
            'Charcoal x pHsoil': 'Biochar × Soil pH',
            'irr type 0 x pHsoil': 'AWD Irrigation × Soil pH',
            'irr type 0': 'AWD Irrigation',
            'irr type 1': 'Continuous Flooding',
            'variety binary': 'Rice Variety'
        }

        # 应用映射到特征名称
        display_feature_names = []
        for name in clean_feature_names:
            if name in feature_labels_map:
                display_feature_names.append(feature_labels_map[name])
            else:
                display_feature_names.append(name)

        # 创建SHAP解释器
        import shap
        if model_name in ['XGBoost', 'LightGBM', 'RF', 'GBDT', 'ExtraTrees']:
            # 对于树模型，使用TreeExplainer
            explainer = shap.TreeExplainer(model)
        else:
            # 对于其他模型，使用KernelExplainer，通过抽样减少计算量
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_test, min(50, len(X_test))))
        # 计算SHAP值
        shap_values = explainer.shap_values(X_test)

        # 如果shap_values是列表，取第一个元素（针对一些模型可能会返回多个目标的SHAP值）
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # 创建图形
        plt.figure(figsize=(14, 10))

        # 汇总图 - 显示每个特征对模型输出的影响
        shap.summary_plot(
            shap_values,
            X_test,
            feature_names=display_feature_names,  # 使用美化后的特征名称
            show=False
        )

        # 将target_col中的下划线替换为空格
        formatted_target = target_col.replace('_', ' ')
        plt.title(f'SHAP Value Impact on {formatted_target} - {model_name}', fontsize=16)

        # 保存汇总图
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f'{target_col}_{model_name}_shap_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHAP汇总图已保存至 {summary_path}")

        # 绘制SHAP依赖图 - 为前三个最重要的特征
        # 计算特征重要性
        importance = np.abs(shap_values).mean(0)
        idx = np.argsort(importance)
        top_features_idx = idx[-3:]  # 获取前三个最重要特征的索引

        for i in reversed(top_features_idx):
            plt.figure(figsize=(10, 8))
            feature_name = display_feature_names[i]  # 使用美化后的特征名称

            # 绘制依赖图
            shap.dependence_plot(
                i,
                shap_values,
                X_test,
                feature_names=display_feature_names,  # 使用美化后的特征名称
                show=False
            )

            # 设置标题
            plt.title(f'SHAP Dependence for {feature_name} - {model_name}', fontsize=16)

            # 保存依赖图
            feature_filename = feature_names[i].replace(' ', '_')  # 文件名使用原始特征名（替换空格为下划线）
            dep_path = os.path.join(output_dir, f'{target_col}_{model_name}_shap_dependence_{feature_filename}.png')
            plt.savefig(dep_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"SHAP依赖图({feature_name})已保存至 {dep_path}")

    except Exception as e:
        print(f"绘制SHAP值时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_pdp_with_saved_model(model, X_train, feature_names, target_col, model_name, output_dir='.', n_features=12):
    """
    使用已训练好的模型生成偏依赖图(PDP)

    参数:
    -----------
    model : 已训练的模型对象
        必须是已经训练好的模型对象，具有predict方法
    X_train : pandas DataFrame
        训练数据特征
    feature_names : list
        要生成PDP的特征名称列表
    target_col : str
        目标列名称(用于图表标题)
    model_name : str
        模型名称(用于图表标题)
    output_dir : str
        保存图表的目录
    n_features : int
        要绘制的顶级特征数量
    """
    try:
        # 引入必要的库
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        import seaborn as sns
        from matplotlib.ticker import MaxNLocator
        from sklearn.inspection import partial_dependence
        import matplotlib
        matplotlib.rcParams['font.size'] = 12

        print(f"开始为{model_name}模型生成PDP图...")

        # 验证模型对象是否有效
        if model is None:
            print("错误: 模型对象为None")
            return

        # 检查模型是否已训练
        if not hasattr(model, 'predict'):
            print(f"错误: 模型 {model_name} 没有'predict'方法")
            return

        # 测试模型预测功能
        try:
            test_sample = X_train.iloc[:1].copy()
            test_pred = model.predict(test_sample)
            print(f"模型预测测试成功 - 输出值: {test_pred[0]:.6f}")

            # 再测试一次，稍微修改输入，确认模型对输入变化有响应
            if len(X_train.columns) > 0:
                first_feature = X_train.columns[0]
                test_sample[first_feature] = test_sample[first_feature] * 1.1  # 增加10%
                test_pred_modified = model.predict(test_sample)
                print(f"修改输入后的预测值: {test_pred_modified[0]:.6f}")
                print(f"输入变化响应检查: {'通过' if test_pred[0] != test_pred_modified[0] else '失败'}")
        except Exception as e:
            print(f"模型预测测试失败: {str(e)}")
            return

        # 确定最重要的特征
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:n_features]
            top_features = [feature_names[i] for i in indices]
            print(f"根据特征重要性选择了前{len(top_features)}个特征")
        else:
            top_features = feature_names[:min(n_features, len(feature_names))]
            print(f"模型没有feature_importances_属性，选择了前{len(top_features)}个特征")

        # 特征分类
        feature_categories = {
            'biochar_properties': ['pH', 'temperature', 'material_type_code'],  # 生物炭特性
            'soil_properties': ['pHsoil', 'organicmatter', 'log_Cdcontent', 'CEC'],  # 土壤特性
            'rice_properties': ['variety_binary'],  # 水稻特性
            'experimental_conditions': ['Charcoal_application_ratio', 'irrigation_type']  # 实验条件
        }

        # 创建从特征到类别的反向映射
        feature_to_category = {}
        for category, features in feature_categories.items():
            for feature in features:
                feature_to_category[feature] = category

        # 特征标签映射
        feature_labels = {
            'organicmatter': 'Organic Matter (mg/kg)',
            'pHsoil': 'Soil pH',
            'pH': 'Biochar pH',
            'temperature': 'Temperature (°C)',
            'Charcoal_application_ratio': 'Biochar Application Ratio (%)',
            'log_Cdcontent': 'Initial Cd Content (mg/kg)',
            'CEC': 'CEC (cmol(+)/kg)',
            'pHsoil_x_variety': 'Soil pH × Rice Variety',
            'Charcoal x pHsoil': 'Biochar × Soil pH',
            'irrigation_x_pHsoil': 'Irrigation × Soil pH',
            'variety_binary': 'Rice Variety',
            'material_type_code': 'Biochar Material Type',
            'irrigation_type': 'Irrigation Method'
        }

        # 分类特征
        categorical_features = ['material_type_code', 'irrigation_type', 'variety_binary']

        # 分类特征标签
        categorical_labels = {
            'material_type_code': {
                0: 'Rice Straw',
                1: 'Wheat Straw',
                2: 'Corn Straw',
                3: 'Other Crop',
                4: 'Nutshell',
                5: 'Wood',
                6: 'Animal',
                7: 'Waste',
                8: 'Mixed'
            },
            'irrigation_type': {
                0: 'AWD',
                1: 'Continuous'
            },
            'variety_binary': {
                0: 'Indica',
                1: 'Japonica'
            }
        }

        # 类别的颜色映射
        category_colors = {
            'biochar_properties': '#5DA5DA',  # 蓝色 - 生物炭特性
            'soil_properties': '#60BD68',  # 绿色 - 土壤特性
            'rice_properties': '#F15854',  # 红色 - 水稻特性
            'experimental_conditions': '#FAA43A'  # 橙色 - 实验条件
        }

        # 设置网格布局
        n_cols = 3
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(20, 6 * n_rows))

        # 添加左侧Y轴标题
        fig.text(0.02, 0.5, 'Cd Soil Change (%)', va='center', rotation='vertical', fontsize=18, fontweight='bold')

        # 为每个特征生成PDP
        for i, feature in enumerate(top_features):
            if i >= n_rows * n_cols:
                break

            # 创建子图
            ax = fig.add_subplot(n_rows, n_cols, i + 1)

            # 获取特征类别
            category = feature_to_category.get(feature, 'unknown')
            line_color = category_colors.get(category, '#1E88E5')  # 默认蓝色

            # 检查是否为分类特征
            is_categorical = feature in categorical_features

            print(f"\n计算特征 '{feature}' 的PDP...")

            # 计算PDP值
            if is_categorical:
                if feature == 'material_type_code':
                    # 获取训练数据中存在的类别
                    unique_values = sorted(X_train[feature].unique())
                    print(f"  发现的类别值: {unique_values}")
                    pdp_grid = np.array(unique_values)

                    # 计算每个类别的PDP值
                    pdp_values = []
                    for val in pdp_grid:
                        # 创建临时数据集，只修改目标特征
                        X_temp = X_train.copy()
                        X_temp[feature] = val

                        # 使用模型进行预测
                        predictions = model.predict(X_temp)
                        mean_pred = np.mean(predictions)
                        pdp_values.append(mean_pred)
                        print(f"  类别 {val} -> 平均预测值: {mean_pred:.6f}")

                    # 获取类别标签
                    category_labels_list = []
                    for cat in pdp_grid:
                        if cat in categorical_labels.get(feature, {}):
                            category_labels_list.append(categorical_labels[feature][cat])
                        else:
                            category_labels_list.append(f"Type {int(cat)}")

                    # 绘制柱状图
                    bars = ax.bar(range(len(pdp_grid)), pdp_values, color=line_color, alpha=0.7)

                    # 在柱子上显示数值
                    for j, v in enumerate(pdp_values):
                        ax.text(j, v + 0.01, f"{v:.3f}",
                                ha='center', va='bottom', fontsize=9, color='black',
                                rotation=0)

                    # 设置刻度
                    ax.set_xticks(range(len(pdp_grid)))
                    ax.set_xticklabels(category_labels_list, rotation=90)
                else:
                    # 二分类特征
                    values = [0, 1]
                    pdp_values = []

                    for val in values:
                        # 创建临时数据集
                        X_temp = X_train.copy()
                        X_temp[feature] = val

                        # 使用模型进行预测
                        predictions = model.predict(X_temp)
                        mean_pred = np.mean(predictions)
                        pdp_values.append(mean_pred)
                        print(f"  二分类值 {val} -> 平均预测值: {mean_pred:.6f}")

                    # 获取标签
                    x_labels = [categorical_labels[feature].get(0),
                                categorical_labels[feature].get(1)]

                    # 绘制线图
                    ax.plot(values, pdp_values, '-o', color=line_color, linewidth=2.5)
                    ax.set_xticks(values)
                    ax.set_xticklabels(x_labels)

                    # 在点上显示数值
                    for j, v in enumerate(pdp_values):
                        ax.text(values[j], v + 0.01, f"{v:.3f}",
                                ha='center', va='bottom', fontsize=10, color='black')
            else:
                # 数值特征
                try:
                    # 获取特征范围
                    min_val = X_train[feature].min()
                    max_val = X_train[feature].max()
                    print(f"  特征范围: [{min_val}, {max_val}]")

                    # 确保范围合理
                    if np.isclose(min_val, max_val):
                        print(f"  警告: 特征 {feature} 在训练数据中几乎是常数 ({min_val})")
                        margin = abs(min_val * 0.1) if min_val != 0 else 0.1
                        min_val = min_val - margin
                        max_val = max_val + margin
                        print(f"  扩展范围到 [{min_val}, {max_val}]")

                    # 生成网格点 - 使用合理数量的点
                    grid_size = 20
                    pdp_grid = np.linspace(min_val, max_val, grid_size)
                    pdp_values = []

                    # 对每个网格点计算PDP值
                    for val in pdp_grid:
                        # 创建临时数据集
                        X_temp = X_train.copy()
                        X_temp[feature] = val

                        # 使用模型预测
                        predictions = model.predict(X_temp)
                        mean_pred = np.mean(predictions)
                        pdp_values.append(mean_pred)

                        # 只打印部分点以减少输出
                        if len(pdp_values) <= 3 or len(pdp_values) > grid_size - 3:
                            print(f"  值 {val:.4f} -> 平均预测值: {mean_pred:.6f}")

                    # 计算PDP值范围
                    pdp_min = min(pdp_values)
                    pdp_max = max(pdp_values)
                    pdp_range = pdp_max - pdp_min
                    print(f"  PDP值范围: [{pdp_min:.6f}, {pdp_max:.6f}], 差异: {pdp_range:.6f}")

                    # 如果PDP值变化微小，打印警告
                    if pdp_range < 1e-5:
                        print(f"  警告: 特征 {feature} 的PDP值几乎不变，模型对此特征不敏感")

                    # 绘制曲线
                    ax.plot(pdp_grid, pdp_values, '-', color=line_color, linewidth=2.5)

                    # 添加数据分布直方图
                    ax2 = ax.twinx()
                    sns.histplot(X_train[feature], kde=False, bins=min(20, len(X_train[feature].unique())),
                                 alpha=0.2, color='gray', ax=ax2, stat='density')
                    ax2.set_ylabel('')
                    ax2.set_yticks([])  # 隐藏右侧Y轴刻度

                except Exception as e:
                    print(f"  计算PDP时出错: {str(e)}")
                    continue

            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # 设置标签
            feature_label = feature_labels.get(feature, feature.replace('_', ' '))
            ax.set_xlabel(feature_label, fontsize=14)

            # 在左上角添加类别标签
            category_names = {
                'biochar_properties': 'Biochar Properties',
                'soil_properties': 'Soil Properties',
                'rice_properties': 'Rice Properties',
                'experimental_conditions': 'Experimental Conditions'
            }
            category_name = category_names.get(category, 'Other')
            ax.annotate(category_name, xy=(0.02, 0.98), xycoords='axes fraction',
                        fontsize=10, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', fc=line_color, alpha=0.2))

            # 使用MaxNLocator确保Y轴有合理数量的刻度
            ax.yaxis.set_major_locator(MaxNLocator(5))

        # 设置主标题
        formatted_target = target_col.replace('_', ' ')
        fig.suptitle(f'Partial Dependence Plots for {formatted_target} - {model_name}',
                     fontsize=20, y=0.98)

        # 调整布局
        plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])

        # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{target_col}_{model_name}_pdp_model_based.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"基于已训练模型的偏依赖图已保存至 {save_path}")
        plt.close()

    except Exception as e:
        print(f"生成偏依赖图时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def generate_pdp_with_best_model(results, X_train, feature_names, target_col, output_dir='.'):
    """
    使用保存在results中的最佳模型生成偏依赖图

    参数:
    -----------
    results : list
        模型训练结果列表，每个元素应包含'Model'和'Model_Object'键
    X_train : pandas DataFrame
        训练数据特征
    feature_names : list
        要生成PDP的特征名称列表
    target_col : str
        目标列名称(用于图表标题)
    output_dir : str
        保存图表的目录
    """
    import os

    # 导入PDP绘图函数
    # 确保plot_pdp_with_saved_model函数已定义

    # 根据测试集R²找出最佳模型
    if not results:
        print("错误: 结果列表为空")
        return

    best_result = max(results, key=lambda x: x.get('R2_Test', -float('inf')))
    best_model_name = best_result.get('Model', 'Unknown')
    best_model = best_result.get('Model_Object', None)

    if best_model is None:
        print(f"错误: 在结果中找不到名为'{best_model_name}'的模型对象")
        return

    print(f"找到最佳模型: {best_model_name}，R²: {best_result.get('R2_Test', 'N/A')}")

    # 创建模型专属的输出目录
    model_output_dir = os.path.join(output_dir, f"{best_model_name.lower()}_pdp")
    os.makedirs(model_output_dir, exist_ok=True)

    # 调用PDP绘图函数
    print(f"使用{best_model_name}模型为{target_col}生成偏依赖图...")
    plot_pdp_with_saved_model(
        model=best_model,
        X_train=X_train,
        feature_names=feature_names,
        target_col=target_col,
        model_name=best_model_name,
        output_dir=model_output_dir
    )

    print(f"PDP图表已保存至 {model_output_dir}")
    return model_output_dir


def plot_complete_analysis(best_model, X_train, X_test, feature_names, target_col, model_name, importance_dict,
                           output_dir='.'):
    """
    完整的分析绘图函数，包括特征重要性、SHAP值和部分依赖图
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 1. 绘制增强的特征重要性图
    plot_enhanced_feature_importance(importance_dict, target_col, model_name, output_dir)

    try:
        # 2. 绘制SHAP值分析
        plot_shap_values(best_model, X_test, feature_names, target_col, model_name, output_dir)
    except Exception as e:
        print(f"SHAP分析出错 (这可能需要额外安装shap库): {str(e)}")
        import traceback
        traceback.print_exc()

    try:
        # 3. 绘制部分依赖图
        plot_pdp_with_saved_model(best_model, X_train, feature_names, target_col, model_name, output_dir)
    except Exception as e:
        print(f"部分依赖图分析出错: {str(e)}")
        import traceback
        traceback.print_exc()

    print(f"完整的分析图表已生成并保存至 {output_dir}")

def train_model_for_rice(df, test_size=0.3, random_state=42, output_dir='.'):
    """
    只针对目标列 'CD_rice_change' 进行训练和评估
    假设 df 已经经过 full_preprocessing_pipeline 预处理
    """
    target_col = 'CD_soil_change'
    if target_col not in df.columns:
        print(f"警告: {target_col} 不在数据中，无法训练")
        return None, None

    print(f"\n对目标变量 {target_col} 进行模型训练和评估...")

    # 1. 分离特征与目标
    #   所有非目标列作为特征列
    feature_cols = [c for c in df.columns if not c.startswith('CD_')]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    print("\n执行特征选择...")

    important_features = [
        # 强相关特征
        'organicmatter',  # OM
        'pH',
        'pHsoil',  # pH_soil
        'temperature',  # Temp
        'Charcoal_application_ratio',  # Char_ratio
        # 其他可能相关的特征
        'log_Cdcontent',  # Cd_content
        'CEC',
        # 品种特征
        'variety_binary',
        # 新的材料类型特征
        'material_type_code',
        # 新的灌溉方式特征
        'irrigation_type',  # 代替 'irr_type_0', 'irr_type_1'
    ]

    # 检查哪些重要特征在当前数据集中存在
    available_features = [f for f in important_features if f in X.columns]
    missing_features = [f for f in important_features if f not in X.columns]

    if missing_features:
        print(f"以下重要特征不在数据集中: {missing_features}")

    print(f"基于相关性分析选择了 {len(available_features)} 个特征: {available_features}")

    # 如果找到的特征太少，退回到原始特征集
    if len(available_features) < 5:
        print("可用的重要特征太少，使用所有特征")
    else:
        X = X[available_features]
        feature_cols = available_features



    print(f"\n最终用于建模的特征 ({len(feature_cols)}): {feature_cols}")
    # 2. 检查数据量
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    # 3. 使用 SPXY 划分训练集和测试集
    X_train, X_test, y_train, y_test = spxy_train_test_split(
        X, y, test_size=0.3, random_state=random_state, distance_weight=0.5,distance_metric='cityblock'
    )

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")



    # 5. 初始化模型并开始训练
    model_infos = initialize_models()
    results = []

    # 这里用 RepeatedKFold 做交叉验证
    from sklearn.model_selection import RepeatedKFold, cross_val_score

    for name, model_info in model_infos.items():
        try:
            print(f"\n训练模型: {name}")
            # 对于树模型可直接用 X_train, 对于线性模型若想再次标准化可在此处做
            # 这里简单示例，全部直接用 X_train 即可
            X_train_model = X_train
            X_test_model = X_test

            best_model = find_best_params(name, model_info, X_train_model, y_train, cv=5)

            cv_obj = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)
            try:
                cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=cv_obj, scoring='r2', error_score=-1)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                print(f"交叉验证 R² (平均 ± 标准差): {cv_mean:.4f} ± {cv_std:.4f}")
            except Exception as e:
                print(f"交叉验证失败: {str(e)}")
                cv_mean, cv_std = -999, 0

            # 最终在完整训练集上训练
            best_model.fit(X_train_model, y_train)

            # 预测
            y_train_pred = best_model.predict(X_train_model)
            y_test_pred = best_model.predict(X_test_model)

            # 评估指标
            r2_train = r2_score(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            mae_train = mean_absolute_error(y_train, y_train_pred)

            r2_test = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae_test = mean_absolute_error(y_test, y_test_pred)

            print(f"训练集 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
            print(f"测试集 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                fi = best_model.feature_importances_
                feature_importance = dict(zip(X_train_model.columns, fi))

            results.append({
                'Model': name,
                'R2_Train': r2_train,
                'RMSE_Train': rmse_train,
                'MAE_Train': mae_train,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test,
                'MAE_Test': mae_test,
                'CV_R2_Mean': cv_mean,
                'CV_R2_Std': cv_std,
                'Train_Predictions': y_train_pred,
                'Test_Predictions': y_test_pred,
                'Train_Values': y_train,
                'Test_Values': y_test,
                'Feature_Importance': feature_importance,
                'Model_Object': best_model
            })

        except Exception as e:
            print(f"训练模型 {name} 时出错: {str(e)}")

    if not results:
        print("所有模型训练失败，无法生成结果")
        return None, None

    # 根据测试集R²排序
    import pandas as pd
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['Train_Predictions', 'Test_Predictions', 'Train_Values', 'Test_Values', 'Feature_Importance']}
        for r in results
    ])
    results_df = results_df.sort_values('R2_Test', ascending=False)

    print("\n模型性能 (按测试集 R² 排序):")
    print(results_df[['Model', 'R2_Train', 'R2_Test', 'CV_R2_Mean']])

    # 绘图 - 使用原有的所有模型比较图
    plot_all_models_comparison(results, target_col, output_dir)

    # 若最佳模型有特征重要性，则绘制
    best_model_idx = results_df['R2_Test'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_result = next(r for r in results if r['Model'] == best_model_name)

    if best_result['Feature_Importance'] is not None:

        pass
        # 使用增强的特征重要性图（按四类进行分类）
        try:
            enhanced_viz_dir = os.path.join(output_dir, "enhanced_viz")
            os.makedirs(enhanced_viz_dir, exist_ok=True)

            plot_enhanced_feature_importance(
                best_result['Feature_Importance'],
                target_col,
                best_model_name,
                enhanced_viz_dir
            )

            # 获取最佳模型对象 - 假设已经在训练循环中存储了模型对象
            best_model = best_result.get('Model_Object')  # 确保在训练过程中将模型对象存储在结果中
            if best_model is None:
                print(f"未找到已训练的{best_model_name}模型对象，尝试重新训练")
                # 重新训练最佳模型
                for name, model_info in model_infos.items():
                    if name == best_model_name:
                        best_model = model_info['model']
                        # 显式拟合模型
                        print(f"重新拟合{best_model_name}模型...")
                        best_model.fit(X_train, y_train)
                        # 验证模型是否已拟合
                        try:
                            best_model.predict(X_test.iloc[:1])
                            print(f"模型已成功拟合")
                        except Exception as e:
                            print(f"模型拟合失败: {str(e)}")
                            best_model = None
                        break

            if best_model is not None:
                feature_names = X_train.columns.tolist()

                # 尝试生成SHAP值分析
                try:
                    shap_dir = os.path.join(enhanced_viz_dir, "shap")
                    os.makedirs(shap_dir, exist_ok=True)
                    plot_shap_values(
                        best_model,
                        X_test,
                        feature_names,
                        target_col,
                        best_model_name,
                        shap_dir
                    )
                except Exception as e:
                    print(f"SHAP分析生成错误: {str(e)}")
                    traceback.print_exc()

                # 尝试生成部分依赖图
                try:
                    generate_pdp_with_best_model(
                        results=results,
                        X_train=X_train,
                        feature_names=feature_cols,
                        target_col=target_col,
                        output_dir=os.path.join(output_dir, "pdp")
                    )
                except Exception as e:
                    print(f"部分依赖图生成错误: {str(e)}")
                    traceback.print_exc()
        except Exception as e:
            print(f"生成增强可视化图表时出错: {str(e)}")
            traceback.print_exc()
     # 根据测试集R²指标找出性能最佳的模型
    best_model_idx = results_df['R2_Test'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_result = next(r for r in results if r['Model'] == best_model_name)

    if best_result is not None:
        # 使用从results中获取的已训练模型对象
        best_model = best_result['Model_Object']  # 这里使用已训练的模型对象

        # 准备保存的字典
        model_save_dict = {
            'model': best_model,
            'feature_names': feature_cols,
            'feature_importance': {
                'values': best_model.feature_importances_.tolist() if hasattr(best_model, 'feature_importances_') else None,
                'feature_names': feature_cols
            },
            'metrics': {  # 保存模型的评估指标
                'R2_Test': best_result['R2_Test'],
                'R2_Train': best_result['R2_Train'],
                'MAE_Test': best_result['MAE_Test'],
                'RMSE_Test': best_result['RMSE_Test']
            }
        }

        # 创建模型保存目录
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # 保存模型路径
        model_path = os.path.join(models_dir, f"{target_col}_{best_model_name}_model.pkl")

        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(model_save_dict, f)

        print(f"最佳模型({best_model_name})已保存至: {model_path}")
        print(f"最佳模型性能指标: R²={best_result['R2_Test']:.4f}")
    else:
        print("未找到有效的模型结果，无法保存")
    return results, results_df

def main(data_path=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\temperature.csv', force_preprocessing=False, output_dir='results'):
    """
    主函数 - 针对目标列 'CD_rice_change' 进行数据预处理和模型训练
    参数:
      data_path: 原始数据CSV文件路径
      force_preprocessing: 是否强制重新执行预处理 (即使存在预处理后的文件)
      output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        processed_data_path = os.path.join(output_dir, 'fully_processed_data(4).csv')

        # 记录开始时间
        start_time = time.time()

        print(f"{'=' * 70}")
        print("开始数据预处理和CD_brown_rice_change模型训练流程")
        print(f"{'=' * 70}")
        print(f"数据文件: {data_path}")
        print(f"输出目录: {output_dir}")

        # 步骤1: 数据预处理
        if force_preprocessing or not os.path.exists(processed_data_path):
            print("\n执行数据预处理...")
            if not os.path.exists(data_path):
                print(f"错误: 找不到输入文件 {data_path}")
                return
            try:
                processed_df, target_cols = full_preprocessing_pipeline(data_path)
                print("\n预处理完成。检查结果...")
                print(f"处理后数据形状: {processed_df.shape}")
                print(f"目标变量列: {target_cols}")

                processed_df.to_csv(processed_data_path, index=False)
                print(f"预处理完成，处理后的数据已保存到 {processed_data_path}")
            except Exception as e:
                print(f"预处理过程中出错: {str(e)}")
                traceback.print_exc()
                return
        else:
            print(f"\n找到已处理的数据文件 {processed_data_path}，跳过预处理步骤")
            try:
                processed_df = pd.read_csv(processed_data_path)
                target_cols = [col for col in processed_df.columns if col.startswith('CD_')]
                print(f"已加载处理后的数据，形状: {processed_df.shape}")
                print(f"目标变量列: {target_cols}")
            except Exception as e:
                print(f"读取处理后的数据文件时出错: {str(e)}")
                traceback.print_exc()
                return

        # 步骤2: 针对 'CD_rice_change' 进行模型训练和评估
        print(f"\n{'=' * 70}")
        print("开始针对 CD_soil_change 的模型训练和评估")
        print(f"{'=' * 70}")
        # 调用您专门为rice编写的训练函数
        results, results_df = train_model_for_rice(processed_df, test_size=0.3,
                                                   output_dir=os.path.join(output_dir, "soil"))
        if results_df is not None:
            results_path = os.path.join(output_dir, "soil", 'CD_soil_change_model_results.csv')
            os.makedirs(os.path.join(output_dir, "soil"), exist_ok=True)
            results_df.to_csv(results_path, index=False)
            print(f"\n模型结果已保存至 {results_path}")
        else:
            print("没有生成有效的模型训练结果！")

        # 计算总运行时间
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\n总运行时间: {int(hours)}时 {int(minutes)}分 {int(seconds)}秒")
        print(f"\n完整流程已完成! 所有结果文件已保存在 {output_dir} 目录。")

    except Exception as e:
        print(f"运行过程中发生未捕获的错误: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据预处理和仅针对CD_soil_change列的模型训练流程')
    parser.add_argument('--data', type=str, default=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv',
                        help='原始数据CSV文件路径 (默认: lr_imputed_data_new.csv)')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='强制重新执行预处理 (即使已存在预处理后的文件)')
    args = parser.parse_args()

    main(args.data, args.force_preprocess, args.output)

