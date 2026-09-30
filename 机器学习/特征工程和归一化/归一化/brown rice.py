import os
import argparse
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
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from data_normalization import full_preprocessing_pipeline

warnings.filterwarnings('ignore')
import pickle
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold


def spxy_train_test_split(X, y, test_size=0.3, random_state=42, distance_weight=0.7, distance_metric='cityblock'):
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
            'params': {
                'fit_intercept': [True],  # 保持不变
                'positive': [False, True]  # 保持不变
            }
        },
        'Bayesian Ridge': {
            'model': BayesianRidge(),
            'params': {
                'alpha_1': [1e-5, 1e-4],  # 增加先验精度
                'alpha_2': [1e-5, 1e-4],  # 增加先验精度
                'lambda_1': [1e-5, 1e-4],  # 增加先验精度
                'lambda_2': [1e-5, 1e-4],  # 增加先验精度
                'n_iter': [500],  # 保持不变
                'alpha_init': [None],  # 保持不变
                'lambda_init': [None]  # 保持不变
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [50.0, 150.0],  # 进一步增加L2正则化强度
                'solver': ['auto'],  # 保持不变
                'max_iter': [2000],  # 保持不变
                'tol': [1e-4],  # 保持不变
                'fit_intercept': [True]  # 保持不变
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [400],  # 设置一个合适的中间值
                'max_depth': [4],  # 设置一个合适的中间值
                'learning_rate': [0.06],  # 设置一个合适的中间值
                'reg_alpha': [0.1],  # 保持已增强的L1正则化
                'reg_lambda': [1],  # 设置适中的L2正则化
                'subsample': [0.75],  # 设置一个平衡的子采样率
                'colsample_bytree': [0.7],  # 设置一个平衡的特征采样率
                'min_child_weight': [3]  # 增加最小子节点权重以控制过拟合
            }
        },

        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [400],  # 保持固定值
                'num_leaves': [15],  # 保持固定值
                'max_depth': [4],  # 保持固定值
                'learning_rate': [0.1],  # 保持固定值
                'min_child_samples': [15],  # 保持固定值
                'min_split_gain': [0.1],  # 保持固定值
                'reg_alpha': [0.3],  # 保持固定值
                'reg_lambda': [1],  # 保持固定值
                'subsample': [0.7],  # 保持固定值
                'colsample_bytree': [0.7]  # 保持固定值
            }
        },
        'RF': {
            'model': RandomForestRegressor(random_state=42, bootstrap=True, oob_score=True),
            'params': {
                'n_estimators': [300, 500],  # 恢复到之前的值
                'max_depth': [12, 20],  # 恢复到之前的值
                'min_samples_split': [2, 4],  # 恢复到之前的值
                'min_samples_leaf': [2, 3],  # 恢复到之前的值
                'max_features': [0.7, 0.8],  # 恢复到之前的值
                'max_samples': [0.7, 0.8]  # 恢复到之前的值
            }
        },
        'GBDT': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [400],  # 设置一个合适的中间值
                'max_depth': [4],  # 设置一个适中的深度
                'learning_rate': [0.2],  # 设置一个适中的学习率
                'subsample': [0.7],  # 设置一个平衡的子采样率
                'min_samples_split': [7],  # 保持适当的分裂样本数
                'min_samples_leaf': [8],  # 保持适当的叶子节点样本数
                'max_features': [0.75]  # 设置一个平衡的特征采样率
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42, bootstrap=True, oob_score=True),
            'params': {
                'n_estimators': [600],  # 恢复到之前的值
                'max_depth': [15],  # 恢复到之前的值
                'min_samples_split': [2],  # 恢复到之前的值
                'min_samples_leaf': [1],  # 恢复到之前的值
                'max_features': [0.7],  # 恢复到之前的值
                'max_samples': [0.7]  # 恢复到之前的值
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [5.0, 20.0],  # 恢复到之前的值
                'epsilon': [0.02, 0.05],  # 恢复到之前的值
                'gamma': ['scale', 0.01],  # 恢复到之前的值
                'kernel': ['rbf'],  # 保持不变
                'degree': [3]  # 保持不变
            }
        }
    }
    return models


def find_best_params(model_name, model_info, X_train, y_train, cv=8):
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
    import matplotlib.pyplot as plt
    import numpy as np

    if not results:
        print(f"没有可用于绘图的结果: {target_col}")
        return

    n_rows = 3
    n_cols = 3
    model_order = ['LR', 'Bayesian Ridge', 'Ridge', 'XGBoost', 'LightGBM',
                   'RF', 'GBDT', 'ExtraTrees', 'SVR']

    # 增大图片尺寸
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 20))
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
            ax.text(0.05, 0.95, train_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(facecolor=train_color, alpha=0.1, edgecolor=train_color, linestyle='--'))

            test_text = (f"Test\nR²: {r2_test:.3f}\nRMSE: {rmse_test:.3f}\nMAE: {mae_test:.3f}")
            ax.text(0.70, 0.25, test_text, transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(facecolor=test_color, alpha=0.1, edgecolor=test_color, linestyle='--'))

            model_label = chr(97 + i)
            is_best = " (Best)" if model_name == best_model else ""
            # 增大标题字体
            ax.set_title(f'({model_label}) {model_name}{is_best}\nR² = {r2_test:.3f}', fontsize=14)

            padding = (max_val - min_val) * 0.05
            ax.set_xlim(min_val - padding, max_val + padding)
            ax.set_ylim(min_val - padding, max_val + padding)
            # 增加网格线透明度
            ax.grid(True, alpha=0.2)
            # 增大坐标轴刻度字体
            ax.tick_params(axis='both', which='major', labelsize=12)
        else:
            model_label = chr(97 + i)
            ax.text(0.5, 0.5, f'({model_label}) {model_name}\n无数据',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')

    # 增大坐标轴标题字体
    fig.text(0.5, 0.04, 'Observed Values', ha='center', fontsize=18)
    fig.text(0.04, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=18)

    # 调整总标题位置 - 将y值从0.995降低到0.97，使标题更居中
    plt.suptitle(f'Cd rice change model comparison', fontsize=20, y=0.92)

    # 调整紧凑布局，给标题留出更多空间
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_all_models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"所有模型比较图已保存至 {save_path}")
    plt.close()


def plot_feature_importance(importance_dict, target_col, model_name, output_dir='.', top_n=10):
    """
    绘制特征重要性条形图，优化显示效果，并优化标题格式
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 对特征名称进行处理，将下划线替换为空格
    cleaned_importance = {k.replace('_', ' '): v for k, v in importance_dict.items()}

    # 排序特征，从高到低
    top_features = sorted(cleaned_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in top_features]
    importance = [f[1] for f in top_features]

    # 创建更大的图形
    plt.figure(figsize=(14, 10))

    # 反转特征顺序，使得最重要的在顶部
    features = features[::-1]
    importance = importance[::-1]

    # 创建水平条形图
    bars = plt.barh(range(len(features)), importance, align='center', height=0.6)

    # 使用改进的渐变颜色，蓝色到红色渐变
    cmap = plt.cm.coolwarm
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i / len(features)))

    # 设置Y轴标签为特征名称，增大字体
    plt.yticks(range(len(features)), features, fontsize=14)

    # 添加特征重要性值的标签
    for i, v in enumerate(importance):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=12)

    # 设置X轴和标题，增大字体
    plt.xlabel('Importance', fontsize=16)
    plt.ylabel('Feature', fontsize=16)

    # 将target_col中的下划线替换为空格
    formatted_target = target_col.replace('_', ' ')

    # 使用格式化后的目标变量名称
    plt.title(f'Top {top_n} Feature Importance for {formatted_target}\nModel: {model_name}', fontsize=18)

    # 添加网格线以提高可读性
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图已保存至 {save_path}")
    plt.close()


def train_model_for_rice(df, test_size=0.2, random_state=42, output_dir='.'):
    """
    只针对目标列 'CD_rice_change' 进行训练和评估
    假设 df 已经经过 full_preprocessing_pipeline 预处理
    """
    target_col = 'CD_brown_rice'
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

    # 基于相关性分析的高相关特征 - 根据热图确定的重要特征
    important_features = [
        # 强相关特征
        'organicmatter',  # OM
        'pHsoil',  # pH_soil
        # 其他可能相关的特征
        'log_Cdcontent',  # Cd_content
        'CEC',
        # 品种特征
        'variety_binary',
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

    # 打印最终使用的特征
    print(f"\n最终用于建模的特征 ({len(feature_cols)}): {feature_cols}")

    print(f"特征工程后的特征数量: {len(feature_cols)}")
    # 2. 检查数据量
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]

    # 3. 使用 SPXY 划分训练集和测试集
    X_train, X_test, y_train, y_test = spxy_train_test_split(
        X, y, test_size=0.2, random_state=random_state, distance_weight=0.5, distance_metric='cityblock'
    )

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # 5. 初始化模型并开始训练
    model_infos = initialize_models()
    results = []

    # 5.1 先训练所有基础模型，获取他们的性能和预测结果
    all_models = {}
    all_models_train_pred = {}
    all_models_test_pred = {}

    for name, model_info in model_infos.items():
        try:
            print(f"\n训练模型: {name}")
            # 对于树模型可直接用 X_train, 对于线性模型若想再次标准化可在此处做
            # 这里简单示例，全部直接用 X_train 即可
            X_train_model = X_train
            X_test_model = X_test

            best_model = find_best_params(name, model_info, X_train_model, y_train, cv=8)

            cv_obj = RepeatedKFold(n_splits=8, n_repeats=3, random_state=random_state)
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

            # 存储模型和预测结果用于后面的集成学习
            all_models[name] = best_model
            all_models_train_pred[name] = y_train_pred
            all_models_test_pred[name] = y_test_pred

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
                'Feature_Importance': feature_importance
            })

        except Exception as e:
            print(f"训练模型 {name} 时出错: {str(e)}")

    # 5.2 在所有基础模型训练完成后，创建混合模型(堆叠集成)
    print("\n创建堆叠集成模型...")

    if len(all_models) >= 3:  # 确保有足够的基础模型
        try:
            # 选择表现最好的3-5个模型进行堆叠
            results_df = pd.DataFrame([
                {k: v for k, v in r.items() if
                 k not in ['Train_Predictions', 'Test_Predictions', 'Train_Values', 'Test_Values',
                           'Feature_Importance']}
                for r in results
            ])

            # 按测试集R²降序排序
            results_df = results_df.sort_values('R2_Test', ascending=False)
            top_model_names = results_df.head(5)['Model'].tolist()

            print(f"选择的基础模型: {top_model_names}")
            base_models = [(name, all_models[name]) for name in top_model_names if name in all_models]

            # 选择合适的元模型
            meta_models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'LightGBM': lgb.LGBMRegressor(verbosity=-1, random_state=42, n_estimators=100)
            }

            best_stacking_meta = None
            best_r2_stacking = -float('inf')

            # 使用交叉验证尝试不同的元模型
            for meta_name, meta_model in meta_models.items():
                # 创建堆叠模型
                stacking_model = StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_model,
                    cv=8,
                    n_jobs=-1,
                    verbose=0
                )

                print(f"\n尝试堆叠集成 (元模型: {meta_name})...")

                # 训练堆叠模型
                stacking_model.fit(X_train, y_train)

                # 预测
                y_train_pred_stack = stacking_model.predict(X_train)
                y_test_pred_stack = stacking_model.predict(X_test)

                # 评估
                r2_train_stack = r2_score(y_train, y_train_pred_stack)
                rmse_train_stack = np.sqrt(mean_squared_error(y_train, y_train_pred_stack))
                mae_train_stack = mean_absolute_error(y_train, y_train_pred_stack)

                r2_test_stack = r2_score(y_test, y_test_pred_stack)
                rmse_test_stack = np.sqrt(mean_squared_error(y_test, y_test_pred_stack))
                mae_test_stack = mean_absolute_error(y_test, y_test_pred_stack)

                print(f"堆叠集成 ({meta_name}) - 训练集 R²: {r2_train_stack:.4f}, RMSE: {rmse_train_stack:.4f}")
                print(f"堆叠集成 ({meta_name}) - 测试集 R²: {r2_test_stack:.4f}, RMSE: {rmse_test_stack:.4f}")

                # 如果这个元模型更好，保存它
                if r2_test_stack > best_r2_stacking:
                    best_r2_stacking = r2_test_stack
                    best_stacking_meta = meta_name

                    # 将堆叠模型结果添加到结果列表
                    results.append({
                        'Model': f'Stacking_{meta_name}',
                        'R2_Train': r2_train_stack,
                        'RMSE_Train': rmse_train_stack,
                        'MAE_Train': mae_train_stack,
                        'R2_Test': r2_test_stack,
                        'RMSE_Test': rmse_test_stack,
                        'MAE_Test': mae_test_stack,
                        'CV_R2_Mean': -999,  # 不需要CV值
                        'CV_R2_Std': 0,
                        'Train_Predictions': y_train_pred_stack,
                        'Test_Predictions': y_test_pred_stack,
                        'Train_Values': y_train,
                        'Test_Values': y_test,
                        'Feature_Importance': None  # 堆叠模型没有直接的特征重要性
                    })

                    # 保存最佳堆叠模型
                    models_dir = os.path.join(output_dir, "models")
                    os.makedirs(models_dir, exist_ok=True)
                    model_path = os.path.join(models_dir, f"{target_col}_Stacking_{meta_name}_model.pkl")
                    with open(model_path, 'wb') as f:
                        pickle.dump(stacking_model, f)
                    print(f"最佳堆叠模型 (Stacking_{meta_name}) 已保存至: {model_path}")

            print(f"\n最佳堆叠集成元模型: {best_stacking_meta}, 测试集 R²: {best_r2_stacking:.4f}")

        except Exception as e:
            print(f"创建堆叠集成模型时出错: {str(e)}")
            traceback.print_exc()
    else:
        print("可用的基础模型不足，无法创建堆叠集成模型")

    if not results:
        print("所有模型训练失败，无法生成结果")
        return None, None

    # 根据测试集R²排序
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if
         k not in ['Train_Predictions', 'Test_Predictions', 'Train_Values', 'Test_Values', 'Feature_Importance']}
        for r in results
    ])
    results_df = results_df.sort_values('R2_Test', ascending=False)

    print("\n模型性能 (按测试集 R² 排序):")
    print(results_df[['Model', 'R2_Train', 'R2_Test', 'CV_R2_Mean']])

    # 绘图
    plot_all_models_comparison(results, target_col, output_dir)

    # 若最佳模型有特征重要性，则绘制
    best_model_idx = results_df['R2_Test'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    best_result = next(r for r in results if r['Model'] == best_model_name)
    if best_result['Feature_Importance'] is not None:
        plot_feature_importance(best_result['Feature_Importance'], target_col, best_model_name, output_dir)
    # 保存最佳模型和特征列表
    try:
        # 创建模型保存目录
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # 保存模型
        model_path = os.path.join(models_dir, f"{target_col}_{best_model_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"最佳模型({best_model_name})已保存至: {model_path}")

        # 保存特征列表
        feature_list_path = os.path.join(models_dir, f"{target_col}_feature_list.pkl")
        with open(feature_list_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"特征列表已保存至: {feature_list_path}")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        traceback.print_exc()
    return results, results_df


def main(data_path=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv', force_preprocessing=False,
         output_dir='results'):
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
        processed_data_path = os.path.join(output_dir, 'fully_processed_data(5).csv')

        # 记录开始时间
        start_time = time.time()

        print(f"{'=' * 70}")
        print("开始数据预处理和CD_brown_rice模型训练流程")
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
        print("开始针对 CD_rice_change 的模型训练和评估")
        print(f"{'=' * 70}")
        # 调用您专门为rice编写的训练函数
        results, results_df = train_model_for_rice(processed_df, test_size=0.2,
                                                   output_dir=os.path.join(output_dir, "brown rice"))
        if results_df is not None:
            results_path = os.path.join(output_dir, " brown rice", 'CD_rice_change_model_results.csv')
            os.makedirs(os.path.join(output_dir, " brown rice"), exist_ok=True)
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
    parser = argparse.ArgumentParser(description='数据预处理和仅针对CD_rice_change列的模型训练流程')
    parser.add_argument('--data', type=str,
                        default=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new(2).csv',
                        help='原始数据CSV文件路径 (默认: lr_imputed_data_new(2).csv)')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='强制重新执行预处理 (即使已存在预处理后的文件)')
    args = parser.parse_args()

    main(args.data, args.force_preprocess, args.output)

