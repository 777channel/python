import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings('ignore')

# 配置图形样式
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def spxy_train_test_split(X, y, test_size=0.3, random_state=42):
    """使用SPXY方法划分训练集和测试集

    SPXY方法结合了样本特征空间和目标空间的信息，确保训练集和测试集在两个空间中都具有代表性。
    这有助于减少模型训练中的过拟合问题。

    参数：
    X -- 特征矩阵
    y -- 目标变量
    test_size -- 测试集比例，默认0.3
    random_state -- 随机种子，用于结果复现

    返回：
    X_train, X_test, y_train, y_test -- 划分后的训练集和测试集
    """
    print("使用SPXY方法划分训练集和测试集...")

    # 设置随机种子
    np.random.seed(random_state)

    X_scaled = StandardScaler().fit_transform(X)
    y_scaled = (y - y.mean()) / y.std()

    # 计算样本间的距离矩阵
    # 特征空间的距离
    Dx = squareform(pdist(X_scaled, metric='euclidean'))
    # 目标空间的距离 - 将Series转为numpy数组
    Dy = squareform(pdist(y_scaled.values.reshape(-1, 1), metric='euclidean'))
    # 合并距离矩阵，给予X和y相同的权重
    D = Dx + Dy

    n_samples = len(X)
    test_size_n = int(n_samples * test_size)
    train_size_n = n_samples - test_size_n

    # 初始化训练集和测试集索引
    train_indexes = []
    test_indexes = list(range(n_samples))

    # 首先选择第一个样本（距离其他所有样本最远的样本）
    sum_dist = D.sum(axis=1)
    max_dist_idx = np.argmax(sum_dist)

    # 将其添加到训练集
    train_indexes.append(max_dist_idx)
    test_indexes.remove(max_dist_idx)

    # 逐个选择剩余的样本
    while len(train_indexes) < train_size_n:
        # 计算测试集中的每个样本到训练集的最小距离
        min_dist = np.zeros(len(test_indexes))
        for i, idx in enumerate(test_indexes):
            d = min([D[idx, j] for j in train_indexes])
            min_dist[i] = d

        # 选择距离训练集最远的样本
        max_idx = np.argmax(min_dist)
        selected_idx = test_indexes[max_idx]

        # 将其添加到训练集
        train_indexes.append(selected_idx)
        test_indexes.remove(selected_idx)

    # 根据索引划分数据集
    X_train = X.iloc[train_indexes]
    X_test = X.iloc[test_indexes]
    y_train = y.iloc[train_indexes]
    y_test = y.iloc[test_indexes]

    print(f"SPXY划分完成，训练集大小: {len(X_train)}，测试集大小: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def load_processed_data(file_path):
    """加载预处理后的数据"""
    print(f"加载预处理后的数据: {file_path}")

    try:
        # 尝试不同的编码方式加载数据
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gbk')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')

        print(f"数据形状: {df.shape}")

        # 目标变量列
        target_cols = [col for col in df.columns if col.startswith('CD_')]

        # 特征列 (非目标变量列)
        feature_cols = [col for col in df.columns if col not in target_cols]

        # 识别分类变量和数值变量
        categorical_cols = []
        numeric_cols = []

        # 检测分类变量
        for col in feature_cols:
            # 检查是否是已知的分类列或编码列
            if col in ['品种类型', 'variety_binary', '原料大类'] or col.startswith('material_') or col.startswith(
                    'irr_type_'):
                categorical_cols.append(col)
            elif df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        print(f"特征数量: {len(feature_cols)}")
        print(f"数值特征: {len(numeric_cols)}")
        print(f"分类特征: {len(categorical_cols)}")
        print(f"目标变量: {target_cols}")

        # 数据概览
        print("\n数值列数据概览:")
        numeric_df = df[numeric_cols + target_cols]
        print(numeric_df.describe().T[['count', 'mean', 'min', 'max']])

        return df, feature_cols, target_cols

    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, [], []


def initialize_models():
    """初始化所有要测试的模型和它们的超参数网格

    返回9个模型:
    (a) LR - 线性回归
    (b) Bayesian Ridge - 贝叶斯岭回归
    (c) Ridge - 岭回归 (L2正则化)
    (d) XGBoost - 极端梯度提升
    (e) LightGBM - 轻量级梯度提升
    (f) RF - 随机森林
    (g) GBDT - 梯度提升决策树
    (h) ElasticNet - 弹性网络 (L1+L2正则化)
    (i) SVR - 支持向量回归
    """
    models = {
        'LR': {
            'model': LinearRegression(),
            'params': {}  # 线性回归没有需要调优的超参数
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
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],  # L2正则化强度
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }
        },

        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [2,3, 5, 7,10],
                'learning_rate': [0.01, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 1],  # L1正则化
                'reg_lambda': [0.1, 1,1.5,5, 10],  # L2正则化
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        },

        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'num_leaves': [15, 31, 50],  # 减小默认值，降低复杂度
                'max_depth': [3, 5, 7],  # 添加显式的深度限制
                'learning_rate': [0.01, 0.05, 0.1],
                'min_child_samples': [5, 10, 20],  # 确保每个叶节点有足够样本
                'min_split_gain': [0, 0.01, 0.1],  # 只使用非负值
                'reg_alpha': [0, 0.1, 1],  # L1正则化
                'reg_lambda': [0.1, 1, 10],  # 增加L2正则化选项
                'subsample': [0.7, 0.8, 1.0],  # 样本采样率
                'colsample_bytree': [0.7, 0.8, 1.0],  # 特征采样率
                'boosting_type': ['gbdt', 'dart']  # 尝试不同的提升类型
            }
        },

        'RF': {
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        },

        'GBDT': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },

        'ElasticNet': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0],  # 正则化强度
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95],  # L1正则化比例(0为纯L2，1为纯L1)
                'max_iter': [1000, 2000, 5000]
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
    """使用网格搜索找到模型的最佳参数"""
    print(f"\n为 {model_name} 模型执行参数网格搜索...")

    model = model_info['model']
    param_grid = model_info['params']

    if not param_grid:  # 如果没有参数需要调优
        print(f"{model_name} 没有需要调优的参数，使用默认配置")
        return model

    # 创建网格搜索对象
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,  # 使用所有可用的CPU
        verbose=0
    )

    # 适应训练数据
    try:
        grid_search.fit(X_train, y_train)

        # 打印最佳参数
        print(f"最佳参数:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"最佳R²得分: {grid_search.best_score_:.4f}")

        # 返回最佳模型
        return grid_search.best_estimator_

    except Exception as e:
        print(f"网格搜索失败: {str(e)}")
        print("使用默认参数继续")
        return model


def train_model_for_target(df, feature_cols, target_col, test_size=0.2, random_state=42, output_dir='.'):
    """对特定目标变量训练所有模型并评估"""
    print(f"\n对目标变量 {target_col} 进行模型训练和评估")

    try:
        # 提取特征和目标数据
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # 检查X是否包含非数值列并进行处理
        numeric_cols = []
        categorical_cols = []

        for col in X.columns:
            # 检查列类型并进行分类
            if X[col].dtype.kind in 'bifc':  # 布尔、整数、浮点数或复数
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
                print(f"发现分类变量: {col}")

        # 对于分类变量，进行独热编码
        if categorical_cols:
            print(f"对以下分类列进行处理: {categorical_cols}")
            for col in categorical_cols:
                # 检查是否已经是独热编码格式（以列名前缀判断）
                if not (col.startswith('material_') or col.startswith('irr_type_')):
                    # 创建独热编码
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    # 合并到X中
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                    print(f"已将 '{col}' 转换为独热编码，创建了 {dummies.shape[1]} 个新特征")

        # 更新特征列列表
        feature_cols = X.columns.tolist()
        print(f"处理后的特征列: {len(feature_cols)}")

        # 检查并删除X中的无效列(全NaN或全常数)
        for col in X.columns.copy():
            if X[col].isna().all():
                print(f"删除全NaN特征列: {col}")
                X = X.drop(columns=[col])
            elif X[col].nunique() <= 1:
                print(f"删除常数特征列: {col}, 唯一值: {X[col].unique()[0]}")
                X = X.drop(columns=[col])

        # 过滤掉有缺失值的样本
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        # 重新获取有效的特征列
        feature_cols = X.columns.tolist()

        # 检查样本量
        if len(X) < 10:
            print(f"警告: 样本量太小 ({len(X)}), 跳过对 {target_col} 的建模")
            return None, None, None

        print(f"有效样本数: {len(X)}")

        # 检查目标列数据范围
        print(f"目标变量 {target_col} 范围: [{y.min()}, {y.max()}]")

        # 保持目标变量的原始值，不进行变换
        # 注意: 对于预处理中使用了双边归一化的数据，不需要再次转换

        # 确保所有特征都不含缺失值
        # 对数值列用均值填充
        for col in X.select_dtypes(include=['number']).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].mean())
                print(f"用均值填充 '{col}' 的缺失值")

        # 处理非数值列（如果有）
        for col in X.select_dtypes(exclude=['number']).columns:
            if X[col].isna().any():
                mode_val = X[col].mode()[0]
                X[col] = X[col].fillna(mode_val)
                print(f"用众数填充 '{col}' 的缺失值: {mode_val}")

        # 安全处理无穷大值 - 只针对数值列
        for col in X.select_dtypes(include=['number']).columns:
            if np.any(np.isinf(X[col].values)):
                print(f"警告: '{col}' 中存在无穷大值，正在替换")
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(X[col].mean())

        # 使用SPXY方法进行数据分割
        X_train, X_test, y_train, y_test = spxy_train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

        # 标准化特征 - 只标准化数值列
        numeric_features = X_train.select_dtypes(include=['number']).columns

        if len(numeric_features) > 0:
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
            X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            print("警告: 没有数值特征可以标准化")

        # 初始化模型
        model_infos = initialize_models()

        # 存储结果
        results = []

        # 训练并评估每个模型
        for name, model_info in model_infos.items():
            try:
                print(f"\n训练模型: {name}")

                # 对于LightGBM模型的特殊处理
                if name == 'LightGBM':
                    # 使用较小的参数网格，减少复杂性
                    simplified_params = {
                        'n_estimators': [50, 100],
                        'num_leaves': [15, 31],
                        'max_depth': [3, 5],
                        'learning_rate': [0.05, 0.1],
                        'min_child_samples': [10, 20],
                        'min_split_gain': [0, 0.01],
                        'reg_alpha': [0, 0.1],
                        'reg_lambda': [1, 10],
                        'subsample': [0.8],
                        'colsample_bytree': [0.8],
                        'verbose': [-1]
                    }

                    # 替换参数网格
                    model_info['params'] = simplified_params

                    # 设置基本模型参数，避免警告
                    model_info['model'].set_params(verbose=-1, min_child_samples=10)

                # 选择适当的训练数据
                if name in ['LightGBM', 'XGBoost', 'RF', 'GBDT']:
                    X_train_model = X_train
                    X_test_model = X_test
                else:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled

                # 使用网格搜索找到最佳参数
                best_model = find_best_params(name, model_info, X_train_model, y_train, cv=5)
                cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)

                try:
                    cv_scores = cross_val_score(best_model, X_train_model, y_train, cv=cv, scoring='r2', error_score=-1)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    print(f"交叉验证 R² (平均 ± 标准差): {cv_mean:.4f} ± {cv_std:.4f}")
                except Exception as e:
                    print(f"交叉验证失败: {str(e)}")
                    cv_mean = -999
                    cv_std = 0

                # 在完整训练集上训练一次（最后的训练）
                if name == 'LightGBM':
                    try:
                        # 直接训练，不使用复杂的回调
                        best_model.fit(X_train_model, y_train, eval_set=[(X_test_model, y_test)], verbose=False)
                    except Exception as e:
                        print(f"LightGBM模型训练失败，使用简化模型: {str(e)}")
                        # 使用最简化的LightGBM模型作为后备
                        best_model = lgb.LGBMRegressor(
                            n_estimators=100, num_leaves=31, max_depth=5,
                            learning_rate=0.1, random_state=42, verbose=-1
                        )
                        best_model.fit(X_train_model, y_train)
                else:
                    best_model.fit(X_train_model, y_train)
                # 使用五重交叉验证并重复5次，按照要求进行
                cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)

                try:
                    cv_scores = cross_val_score(best_model, X_train_model, y_train,
                                                cv=cv, scoring='r2', error_score='raise')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()

                    print(f"交叉验证 R² (平均 ± 标准差): {cv_mean:.4f} ± {cv_std:.4f}")
                except Exception as e:
                    print(f"交叉验证失败: {str(e)}")
                    print("使用默认error_score=-1重试")
                    try:
                        cv_scores = cross_val_score(best_model, X_train_model, y_train,
                                                    cv=cv, scoring='r2', error_score=-1)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        print(f"重试成功，交叉验证 R²: {cv_mean:.4f} ± {cv_std:.4f}")
                    except Exception as e2:
                        print(f"交叉验证再次失败: {str(e2)}")
                        cv_mean = -999
                        cv_std = 0

                # 在完整训练集上训练
                best_model.fit(X_train_model, y_train)

                # 训练集和测试集预测
                y_train_pred = best_model.predict(X_train_model)
                y_test_pred = best_model.predict(X_test_model)

                # 评估指标 - 训练集
                r2_train = r2_score(y_train, y_train_pred)
                rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
                mae_train = mean_absolute_error(y_train, y_train_pred)

                # 评估指标 - 测试集
                r2_test = r2_score(y_test, y_test_pred)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae_test = mean_absolute_error(y_test, y_test_pred)

                print(f"训练集性能指标 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
                print(f"测试集性能指标 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

                # 如果是随机森林或其他树模型，获取特征重要性
                feature_importance = None
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
                    # 打印前5个最重要的特征
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("前5个最重要的特征:")
                    for feature, importance in top_features:
                        print(f"  {feature}: {importance:.4f}")

                # 存储训练结果
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
                    'Train_Indices': X_train.index,
                    'Test_Indices': X_test.index,
                    'Feature_Importance': feature_importance,
                    'Fitted_Model': best_model,
                    'Best_Params': best_model.get_params() if hasattr(best_model, 'get_params') else {}
                })

            except Exception as e:
                print(f"训练模型 {name} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        if not results:
            print(f"所有模型训练失败，跳过对 {target_col} 的处理")
            return None, None, None

        # 根据测试集R²排序结果
        results_df = pd.DataFrame(
            [{k: v for k, v in r.items() if
              k not in ['Train_Predictions', 'Test_Predictions', 'Train_Values', 'Test_Values',
                        'Train_Indices', 'Test_Indices', 'Fitted_Model', 'Feature_Importance', 'Best_Params']}
             for r in results])
        results_df = results_df.sort_values('R2_Test', ascending=False)

        # 创建更美观的性能表格
        print("\n模型性能排名 (按测试集 R² 排序):")
        performance_table = results_df[['Model', 'R2_Train', 'RMSE_Train', 'MAE_Train',
                                        'R2_Test', 'RMSE_Test', 'MAE_Test', 'CV_R2_Mean']]

        # 格式化数值列，保留4位小数
        for col in performance_table.columns:
            if col != 'Model':
                performance_table[col] = performance_table[col].map(lambda x: f"{x:.4f}")

        print(performance_table)

        # 打印每个模型的最佳参数
        print("\n各模型最佳参数:")
        for result in results:
            print(f"\n{result['Model']}:")
            params = result['Best_Params']
            for param, value in params.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    print(f"  {param}: {value}")

        # 获取最佳模型
        if len(results_df) > 0:
            best_model_idx = results_df['R2_Test'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'Model']
            best_result = next(r for r in results if r['Model'] == best_model_name)

            print(f"\n最佳模型: {best_model_name}, 测试集 R² = {best_result['R2_Test']:.4f}")

            # 为这个目标变量创建一个包含所有模型结果的大图
            plot_all_models_comparison(results, target_col, output_dir)

            # 如果最佳模型具有特征重要性，绘制特征重要性图
            if best_result['Feature_Importance'] is not None:
                plot_feature_importance(best_result['Feature_Importance'], target_col, best_model_name, output_dir)
        else:
            best_result = None

        # 将测试指标R²、RMSE和MAE添加到最佳结果中，以兼容主程序的指标提取
        if best_result is not None:
            best_result['R2'] = best_result['R2_Test']
            best_result['RMSE'] = best_result['RMSE_Test']
            best_result['MAE'] = best_result['MAE_Test']

        return results, results_df, best_result

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def plot_all_models_comparison(results, target_col, output_dir='.'):
    """为目标变量创建一个大图，包含所有9个模型的预测比较
    模型按顺序排列:
    (a) LR, (b) Bayesian Ridge, (c) Ridge, (d) XGBoost, (e) LightGBM,
    (f) RF, (g) GBDT, (h) ElasticNet, (i) SVR
    """
    try:
        if not results:
            print(f"没有可用于绘图的结果: {target_col}")
            return

        # 设置固定的行和列数 - 3行3列，9个模型
        n_rows = 3
        n_cols = 3
        model_order = ['LR', 'Bayesian Ridge', 'Ridge', 'XGBoost', 'LightGBM', 'RF', 'GBDT', 'ElasticNet', 'SVR']

        # 创建大图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16))
        axes = axes.flatten()

        # 使用特定的颜色方案
        train_color = 'red'
        test_color = 'blue'

        # 创建一个模型名称到结果的映射
        model_results = {result['Model']: result for result in results}

        # 获取最佳模型（按测试集R2）
        best_model = max(results, key=lambda x: x['R2_Test'])['Model']

        # 为每个模型位置创建子图
        for i, model_name in enumerate(model_order):
            ax = axes[i]

            # 检查模型是否在结果中
            if model_name in model_results:
                result = model_results[model_name]

                # 获取训练集和测试集数据
                y_train = result['Train_Values']
                y_train_pred = result['Train_Predictions']
                y_test = result['Test_Values']
                y_test_pred = result['Test_Predictions']

                # 获取性能指标
                r2_train = result['R2_Train']
                rmse_train = result['RMSE_Train']
                mae_train = result['MAE_Train']
                r2_test = result['R2_Test']
                rmse_test = result['RMSE_Test']
                mae_test = result['MAE_Test']

                # 绘制训练集数据点 (红色圆点)
                ax.scatter(y_train, y_train_pred, color=train_color, marker='o', alpha=0.7, s=30, label='Train')

                # 绘制测试集数据点 (蓝色三角形)
                ax.scatter(y_test, y_test_pred, color=test_color, marker='^', alpha=0.7, s=30, label='Test')

                # 找出数据范围
                all_y_values = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
                min_val = np.min(all_y_values)
                max_val = np.max(all_y_values)

                # 添加对角线 (y=x线)
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

                # 添加训练集和测试集性能指标文本框
                train_text = (f"R: Train\n"
                              f"R²: {r2_train:.3f}\n"
                              f"RMSE: {rmse_train:.3f}\n"
                              f"MAE: {mae_train:.3f}")
                ax.text(0.05, 0.95, train_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=train_color, alpha=0.1,
                                  edgecolor=train_color, linestyle='--'))

                test_text = (f"B: Test\n"
                             f"R²: {r2_test:.3f}\n"
                             f"RMSE: {rmse_test:.3f}\n"
                             f"MAE: {mae_test:.3f}")
                ax.text(0.70, 0.25, test_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=test_color, alpha=0.1,
                                  edgecolor=test_color, linestyle='--'))

                # 设置轴标签和标题 - 标记最佳模型
                model_label = chr(97 + i)  # a, b, c, ...
                is_best = " (Best)" if model_name == best_model else ""
                ax.set_title(f'({model_label}) {model_name}{is_best}\nR² = {r2_test:.3f}', fontsize=10)

                # 设置轴范围，确保包括所有数据点
                padding = (max_val - min_val) * 0.05  # 5%的边距
                ax.set_xlim(min_val - padding, max_val + padding)
                ax.set_ylim(min_val - padding, max_val + padding)

                # 添加网格线
                ax.grid(True, alpha=0.3)
            else:
                # 对于没有结果的模型位置
                model_label = chr(97 + i)  # a, b, c, ...
                ax.text(0.5, 0.5, f'({model_label}) {model_name}\n没有数据',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.axis('off')

        # 设置共享的x轴和y轴标签
        fig.text(0.5, 0.04, 'Observed Values', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=14)

        # 添加目标变量标题
        plt.suptitle(f'目标变量: {target_col} - 所有模型预测比较', fontsize=16, y=0.995)

        # 调整布局
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.97])  # 为顶部标题和侧边轴标签留出空间

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 保存图形
        save_path = os.path.join(output_dir, f'{target_col}_all_models_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"所有模型比较图已保存至 {save_path}")

        # 关闭图形，避免内存问题
        plt.close()

    except Exception as e:
        print(f"绘制所有模型比较图时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def plot_feature_importance(importance_dict, target_col, model_name, output_dir='.', top_n=10):
    """绘制特征重要性条形图"""
    try:
        # 按重要性排序并取前N个
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in top_features]
        importance = [f[1] for f in top_features]

        plt.figure(figsize=(12, 8))

        # 创建水平条形图
        bars = plt.barh(range(len(features)), importance, align='center')

        # 为条形着色
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / len(features)))

        # 设置y轴标签
        plt.yticks(range(len(features)), features)

        # 标签和标题
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance for {target_col}\nModel: {model_name}', fontsize=14)

        # 调整布局
        plt.tight_layout()

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 保存图形
        save_path = os.path.join(output_dir, f'{target_col}_{model_name}_feature_importance.png')
        plt.savefig(save_path, dpi=300)
        print(f"特征重要性图已保存至 {save_path}")

        # 关闭图形
        plt.close()

    except Exception as e:
        print(f"绘制特征重要性图时出错: {str(e)}")


def create_combined_plots(best_models, output_dir='.'):
    """创建所有目标变量最佳模型的组合图"""
    if not best_models:
        print("没有可用的最佳模型数据来创建组合图")
        return

    try:
        print("\n创建所有目标变量最佳模型的组合图...")

        # 计算需要的行数（每行3个子图）
        num_targets = len(best_models)
        num_rows = (num_targets + 2) // 3  # 向上取整

        fig, axes = plt.subplots(num_rows, 3, figsize=(18, 6 * num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes] if num_targets == 1 else axes

        for i, (target, result) in enumerate(best_models.items()):
            if result is not None:
                # 当前子图
                ax = axes[i]

                # 模型信息
                model_name = result['Model']
                r2_train = result['R2_Train']
                rmse_train = result['RMSE_Train']
                mae_train = result['MAE_Train']
                r2_test = result['R2_Test']
                rmse_test = result['RMSE_Test']
                mae_test = result['MAE_Test']

                # 获取训练和测试数据
                y_train = result['Train_Values']
                y_train_pred = result['Train_Predictions']
                y_test = result['Test_Values']
                y_test_pred = result['Test_Predictions']

                # 绘制散点图
                ax.scatter(y_train, y_train_pred, color='red', marker='o', alpha=0.7, s=30)
                ax.scatter(y_test, y_test_pred, color='blue', marker='^', alpha=0.7, s=30)

                # 找出数据范围
                all_y_values = np.concatenate([y_train, y_test, y_train_pred, y_test_pred])
                min_val = np.min(all_y_values)
                max_val = np.max(all_y_values)

                # 添加对角线
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

                # 添加训练集和测试集性能指标文本框
                train_text = (f"R: Train\n"
                              f"R²: {r2_train:.3f}\n"
                              f"RMSE: {rmse_train:.3f}\n"
                              f"MAE: {mae_train:.3f}")
                ax.text(0.05, 0.95, train_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.1,
                                  edgecolor='red', linestyle='--'))

                test_text = (f"B: Test\n"
                             f"R²: {r2_test:.3f}\n"
                             f"RMSE: {rmse_test:.3f}\n"
                             f"MAE: {mae_test:.3f}")
                ax.text(0.75, 0.25, test_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.1,
                                  edgecolor='blue', linestyle='--'))

                # 设置轴标签和标题
                ax.set_xlabel('Observed Values', fontsize=10)
                ax.set_ylabel('Predicted Values', fontsize=10)
                ax.set_title(f'{target}\nModel: {model_name}, R² = {r2_test:.3f}', fontsize=12)

                # 设置轴范围，确保包括所有数据点
                padding = (max_val - min_val) * 0.05  # 5%的边距
                ax.set_xlim(min_val - padding, max_val + padding)
                ax.set_ylim(min_val - padding, max_val + padding)

                # 添加网格线
                ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        # 调整布局
        plt.tight_layout()

        # 保存图形
        combined_path = os.path.join(output_dir, 'all_targets_best_models.png')
        plt.savefig(combined_path, dpi=300)
        print(f"组合图已保存至 {combined_path}")
        plt.close()

    except Exception as e:
        print(f"创建组合图时出错: {str(e)}")
        import traceback
        traceback.print_exc()


def train_all_targets(file_path, output_dir='.'):
    """对所有目标变量进行模型训练和评估

    为每个目标变量生成一张包含全部9个模型的大图，然后为所有目标变量的最佳模型创建一张组合图。
    这样可以大幅减少生成的图片数量，每个目标变量只生成一张图片，外加一张总览图。

    参数:
    file_path -- 预处理后的数据文件路径
    output_dir -- 模型结果输出目录

    返回:
    all_results -- 所有目标变量的所有模型结果字典
    best_models -- 所有目标变量的最佳模型结果字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df, feature_cols, target_cols = load_processed_data(file_path)

    if df is None or not target_cols:
        print("无法加载数据或未找到目标变量，训练终止")
        return {}, {}

    # 结果存储
    all_results = {}
    best_models = {}
    best_params = {}

    # 对每个目标变量进行训练
    for target_col in target_cols:
        print(f"\n{'=' * 50}")
        print(f"开始对 {target_col} 进行建模")
        print(f"{'=' * 50}")

        results, results_df, best_result = train_model_for_target(
            df, feature_cols, target_col, test_size=0.3, output_dir=output_dir
        )

        if results is not None:
            # 存储结果
            all_results[target_col] = results
            best_models[target_col] = best_result

            # 存储最佳模型参数
            if 'Best_Params' in best_result:
                best_params[target_col] = {
                    'model': best_result['Model'],
                    'params': best_result['Best_Params']
                }

            # 保存详细结果到CSV
            if results_df is not None:
                results_path = os.path.join(output_dir, f'{target_col}_model_results.csv')
                results_df.to_csv(results_path, index=False)
                print(f"模型结果已保存至 {results_path}")

    # 创建所有目标变量最佳模型的组合图（仅一张图）
    if best_models:
        create_combined_plots(best_models, output_dir)

    # 保存所有最佳模型的参数
    if best_params:
        params_df = pd.DataFrame([
            {
                'Target': target,
                'Best_Model': params['model'],
                **{f'param_{k}': str(v) for k, v in params['params'].items() if
                   isinstance(v, (int, float, str, bool)) or v is None}
            }
            for target, params in best_params.items()
        ])
        params_path = os.path.join(output_dir, 'best_model_parameters.csv')
        params_df.to_csv(params_path, index=False)
        print(f"\n最佳模型参数已保存至 {params_path}")

    # 打印统计结果
    print(f"\n{'=' * 50}")
    print(f"训练完成! 总共处理了 {len(target_cols)} 个目标变量。")
    print(f"生成了 {len(target_cols) + 1} 张图片:")
    print(f"- {len(target_cols)} 张各目标变量的9模型比较图")
    print(f"- 1 张所有目标变量最佳模型的组合图")
    print(f"{'=' * 50}")

    return all_results, best_models


# 主函数，当直接运行此脚本时执行
if __name__ == "__main__":
    # 设置参数
    import argparse

    parser = argparse.ArgumentParser(description='模型训练和评估')
    parser.add_argument('--data', type=str, required=True, help='预处理后的数据CSV文件路径')
    parser.add_argument('--output', type=str, default='model_results', help='输出目录 (默认: model_results)')

    args = parser.parse_args()

    # 运行训练
    all_results, best_models = train_all_targets(args.data, args.output)

    # 如果有最佳模型，打印每个目标变量的最佳模型性能
    if best_models:
        print("\n各目标变量最佳模型性能汇总:")
        print("-" * 80)
        print(f"{'目标变量':<15} {'最佳模型':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}")
        print("-" * 80)

        for target, result in best_models.items():
            if result:
                print(
                    f"{target:<15} {result['Model']:<15} {result['R2']:.4f}{'':>5} {result['RMSE']:.4f}{'':>5} {result['MAE']:.4f}{'':>5}")

        print("-" * 80)