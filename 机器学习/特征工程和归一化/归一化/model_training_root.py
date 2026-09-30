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
                'reg_lambda': [1.0, 1.5],        # 增强L2正则化
                'subsample': [0.7, 0.8],         # 减少样本使用比例
                'colsample_bytree': [0.7, 0.8],  # 减少特征使用比例
                'min_child_weight': [3, 5]       # 增加最小子权重
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [200],
                'num_leaves': [31],          # 减少叶子节点数量
                'max_depth': [5],             # 限制深度
                'learning_rate': [0.02],   # 降低学习率
                'min_child_samples': [15, 20],   # 增加叶节点最小样本数
                'min_split_gain': [0.05, 0.1],   # 增加分裂所需的最小增益
                'reg_alpha': [0.1, 0.2],         # 增强L1正则化
                'reg_lambda': [0.5],        # 增强L2正则化
                'subsample': [0.7],         # 减少样本使用比例
                'colsample_bytree': [0.7],  # 减少特征使用比例
                'path_smooth': [3]            # 增加平滑参数
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
                'max_depth': [4, 5,6],  # 增加树的深度
                'learning_rate': [0.02, 0.03],  # 调整学习率
                'subsample': [0.7, 0.8],  # 增加样本使用比例
                'min_samples_split': [5, 8],  # 减少分裂所需的最小样本数
                'min_samples_leaf': [3, 5],  # 减少叶节点所需的最小样本数
                'max_features': [0.6, 0.7], # 增加特征使用比例
                'alpha':[0.8,0.9]
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42),
            'params': {
                'n_estimators': [200],          # 回归到中等数量的树
                'max_depth': [10],             # 尝试不限制深度或适度限制
                'min_samples_split': [2, 3],         # 减少分裂限制
                'min_samples_leaf': [1, 2],          # 降低叶节点限制
                'max_features': ['auto', 0.8, 0.9,1],  # 尝试自动选择或较高比例
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
    使用网格搜索找到模型的最佳参数
    """
    model = model_info['model']
    param_grid = model_info['params']

    if not param_grid:  # 若无需要调优的参数
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
        return grid_search.best_estimator_
    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认参数继续")
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

    plt.suptitle(f'Root-Cd', fontsize=28, y=0.93,fontweight='bold')

    # 调整紧凑布局，给标题留出更多空间
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_all_models_comparison.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"所有模型比较图已保存至 {save_path}")
    plt.close()




def plot_feature_importance(importance_dict, target_col, model_name, output_dir='.', top_n=10):
    """
    绘制特征重要性条形图，优化显示效果，并优化标题格式
    """
    import matplotlib.pyplot as plt

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

    # 使用渐变颜色
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(features)))

    # 设置Y轴标签为特征名称，增大字体
    plt.yticks(range(len(features)), features, fontsize=14)

    # 设置X轴和标题，增大字体
    plt.xlabel('Importance', fontsize=16)
    plt.ylabel('Feature', fontsize=16)

    # 将target_col中的下划线替换为空格
    formatted_target = target_col.replace('_', ' ')

    # 使用格式化后的目标变量名称
    plt.title(f'Top {top_n} Feature Importance for {formatted_target}\nModel: {model_name}', fontsize=18)

    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col}_{model_name}_feature_importance.png')
    plt.savefig(save_path, dpi=300)
    print(f"特征重要性图已保存至 {save_path}")
    plt.close()


def train_model_for_rice(df, test_size=0.3, random_state=42, output_dir='.'):
    """
    只针对目标列 'CD_rice_change' 进行训练和评估
    假设 df 已经经过 full_preprocessing_pipeline 预处理
    """
    target_col = 'CD_root_change'
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
        'material_type_code',  # 代替 'material_plant origin', 'material_Non-plant origin'
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
                'Model_Object':best_model
            })

        except Exception as e:
            print(f"训练模型 {name} 时出错: {str(e)}")

    if not results:
        print("所有模型训练失败，无法生成结果")
        return None, None

    # 根据测试集R²排序
    import pandas as pd
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



def main(data_path=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\data_finall version.csv', force_preprocessing=False, output_dir='results'):
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
        print("开始针对 CD_root_change 的模型训练和评估")
        print(f"{'=' * 70}")
        # 调用您专门为rice编写的训练函数
        results, results_df = train_model_for_rice(processed_df, test_size=0.2,
                                                   output_dir=os.path.join(output_dir, "root"))
        if results_df is not None:
            results_path = os.path.join(output_dir, "root", 'CD_root_change_model_results.csv')
            os.makedirs(os.path.join(output_dir, "root"), exist_ok=True)
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
    parser = argparse.ArgumentParser(description='数据预处理和仅针对CD_root_change列的模型训练流程')
    parser.add_argument('--data', type=str, default=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv',
                        help='原始数据CSV文件路径 (默认:lr_imputed_data_new.csv)')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='强制重新执行预处理 (即使已存在预处理后的文件)')
    args = parser.parse_args()

    main(args.data, args.force_preprocess, args.output)

