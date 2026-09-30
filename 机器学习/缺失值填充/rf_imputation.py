import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings

warnings.filterwarnings('ignore')


def rf_imputation_with_optimization(df, target_col, numeric_cols, categorical_cols=None):
    """
    使用随机森林填充目标变量，并优化超参数

    参数:
    df: 数据框
    target_col: 需要填充的目标列
    numeric_cols: 数值特征列
    categorical_cols: 分类特征列，默认为None

    返回:
    填充后的数据框, 最优参数, 特征重要性
    """
    print(f"\n正在使用随机森林填充 {target_col}...")
    start_time = time.time()
    df_imputed = df.copy()

    # 检查目标列是否有缺失值
    missing_mask = df_imputed[target_col].isnull()
    missing_count = missing_mask.sum()

    if missing_count == 0:
        print("  没有缺失值，无需填充")
        return df_imputed, None, None

    print(f"  发现 {missing_count} 个缺失值")

    # 准备特征
    # 使用数值特征和经过编码的分类特征
    used_features = numeric_cols.copy()

    # 添加目标专用的影响编码
    # 检查是否已经存在目标专用影响编码
    impact_col = f'原料对{target_col}的影响'
    material_category_impact = f'原料大类对{target_col}的影响'

    if impact_col not in df_imputed.columns or material_category_impact not in df_imputed.columns:
        # 如果不存在，创建专用编码
        from data_preprocessing import encode_features_for_rf
        df_imputed, _ = encode_features_for_rf(df_imputed, target_col,
                                               categorical_cols if categorical_cols else [])

    # 在准备特征时
    # 明确排除原始分类列
    exclude_cols = ['原料', '品种', 'Irrigationmethod']
    used_features = [col for col in numeric_cols if col not in exclude_cols]

    # 明确添加编码列
    encoded_cols = [col for col in df_imputed.columns if
                    'encoded' in col or
                    '编码' in col or  # 这将包含'灌溉方式编码'
                    '对' in col and '影响' in col]
    used_features.extend(encoded_cols)

    print(f"  使用的特征数量: {len(used_features)}")
    print(f"  特征列表: {used_features}")

    # 划分训练集（无缺失值的样本）
    X_train = df_imputed.loc[~missing_mask, used_features]
    y_train = df_imputed.loc[~missing_mask, target_col]

    # 划分预测集（有缺失值的样本）
    X_missing = df_imputed.loc[missing_mask, used_features]

    # 检查样本量，确保有足够的训练数据
    if len(y_train) < 10:
        print("  警告: 训练样本数量过少，可能影响模型质量")
        if len(y_train) < 5:  # 样本太少，退化为均值填充
            print("  样本数量极少，退化为均值填充")
            mean_val = y_train.mean()
            df_imputed.loc[missing_mask, target_col] = mean_val

            stats = {
                'missing_count': missing_count,
                'method': 'mean_fallback',
                'fill_value': mean_val,
                'original_mean': mean_val,
                'original_std': y_train.std() if len(y_train) > 1 else 0,
                'original_min': y_train.min(),
                'original_max': y_train.max(),
                'filled_mean': df_imputed[target_col].mean(),
                'filled_std': df_imputed[target_col].std(),
                'filled_min': df_imputed[target_col].min(),
                'filled_max': df_imputed[target_col].max()
            }

            print(f"  {target_col}: 使用均值填充: {mean_val:.4f}")
            return df_imputed, {'method': 'mean_fallback'}, None

    # 创建训练集验证集划分（用于内部评估模型质量）
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # 定义参数网格
    param_dist = {
        'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [None, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 8, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # 根据缺失值数量和可用训练样本数量调整搜索强度
    n_iter = 30

    # 样本较少时降低复杂度
    if len(y_train) < 30:
        n_iter = 15
        param_dist['n_estimators'] = [50, 100, 150]
        param_dist['max_depth'] = [None, 5, 10]

    # 缺失值较多时增加搜索强度
    elif missing_count > 100:
        n_iter = 40

    print(f"  随机搜索将进行 {n_iter} 次迭代")

    # 创建随机森林模型
    rf = RandomForestRegressor(random_state=42)

    # 设置交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 创建随机搜索
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # 执行搜索
    print("  开始参数搜索...")
    search_start = time.time()
    random_search.fit(X_train, y_train)
    search_time = time.time() - search_start
    print(f"  参数搜索完成，用时 {search_time:.2f} 秒")

    # 获取最佳参数
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("\n  最优参数:")
    for param, value in best_params.items():
        print(f"    {param}: {value}")
    print(f"  最佳交叉验证MSE: {-best_score:.6f}")

    # 使用最佳参数训练最终模型
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train_fit, y_train_fit)

    # 在验证集上评估
    val_predictions = best_rf.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_mae = mean_absolute_error(y_val, val_predictions)
    print(f"  验证集MSE: {val_mse:.6f}")
    print(f"  验证集MAE: {val_mae:.6f}")

    # 记录特征重要性
    feature_importances = pd.DataFrame({
        'feature': used_features,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n  特征重要性 (前10个):")
    print(feature_importances.head(10))

    # 使用最佳参数训练完整模型（使用全部训练数据）
    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(X_train, y_train)

    # 预测缺失值
    predicted_values = final_model.predict(X_missing)

    # 填充缺失值
    df_imputed.loc[missing_mask, target_col] = predicted_values

    # 验证填充效果
    print("\n  填充统计:")
    print(f"    填充前均值: {df[target_col].mean():.4f}")
    print(f"    填充后均值: {df_imputed[target_col].mean():.4f}")
    print(f"    填充值范围: [{predicted_values.min():.4f}, {predicted_values.max():.4f}]")
    print(f"    填充值均值: {predicted_values.mean():.4f}")

    # 收集统计信息
    stats = {
        'missing_count': missing_count,
        'best_params': best_params,
        'cv_score': -best_score,
        'val_mse': val_mse,
        'val_mae': val_mae,
        'original_mean': df[target_col].mean(),
        'original_std': df[target_col].std(),
        'original_min': df[target_col].min(),
        'original_max': df[target_col].max(),
        'filled_mean': df_imputed[target_col].mean(),
        'filled_std': df_imputed[target_col].std(),
        'filled_min': df_imputed[target_col].min(),
        'filled_max': df_imputed[target_col].max(),
        'predictions_mean': predicted_values.mean(),
        'predictions_min': predicted_values.min(),
        'predictions_max': predicted_values.max()
    }

    # 可视化填充前后分布
    plt.figure(figsize=(15, 10))

    # 填充前后分布对比
    plt.subplot(2, 2, 1)
    sns.histplot(df[target_col].dropna(), kde=True, color='blue', label='原始数据')
    sns.histplot(df_imputed[target_col], kde=True, color='green', alpha=0.6, label='填充后数据')
    plt.title(f'{target_col} - 分布对比')
    plt.legend()

    # 原始值和预测值的分布对比
    plt.subplot(2, 2, 2)
    sns.histplot(df[target_col].dropna(), kde=True, color='blue', label='原始数据')
    sns.histplot(predicted_values, kde=True, color='red', alpha=0.6, label='预测值')
    plt.title(f'{target_col} - 原始值与预测值分布')
    plt.legend()

    # 特征重要性（前10个）
    plt.subplot(2, 2, 3)
    top_features = feature_importances.head(10)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('特征重要性（前10个）')

    # 验证集实际值与预测值对比
    plt.subplot(2, 2, 4)
    plt.scatter(y_val, val_predictions, alpha=0.6)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'验证集 - 实际值vs预测值 (MSE: {val_mse:.6f})')

    plt.tight_layout()
    plt.savefig(f'{target_col}_rf_imputation.png')
    print(f"  已保存填充结果图: {target_col}_rf_imputation.png")

    # 单独保存特征重要性图表
    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(15)  # 展示前15个特征
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'{target_col} - 特征重要性（前15个）')
    plt.tight_layout()
    plt.savefig(f'{target_col}_feature_importance.png')
    print(f"  已保存特征重要性图: {target_col}_feature_importance.png")

    # 计算总用时
    total_time = time.time() - start_time
    print(f"  随机森林填充完成，总用时: {total_time:.2f} 秒")

    return df_imputed, stats, feature_importances


def impute_numeric_features_rf(df, numeric_cols, categorical_cols=None):
    """
    使用简化的随机森林填充数值特征

    参数:
    df: 数据框
    numeric_cols: 数值特征列列表
    categorical_cols: 分类特征列列表

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始使用随机森林填充数值特征...")
    df_imputed = df.copy()
    all_stats = {}

    # 计算每列的缺失值数量
    missing_counts = {col: df[col].isnull().sum() for col in numeric_cols}

    # 按缺失值数量排序
    sorted_cols = sorted(missing_counts.items(), key=lambda x: x[1])

    # 按顺序填充
    for col, count in sorted_cols:
        if count > 0:
            print(f"\n填充特征: {col}，缺失值数量: {count}")

            # 对于数值特征，使用简化的随机森林模型（不进行大规模参数优化）
            # 确定哪些特征可以用于预测当前特征
            other_numeric = [c for c in numeric_cols if c != col and df_imputed[c].isnull().sum() == 0]

            # 确保有足够的特征可用
            if len(other_numeric) < 2:
                print(f"  可用特征不足，使用均值填充 {col}")
                mean_val = df_imputed[col].mean()
                df_imputed[col] = df_imputed[col].fillna(mean_val)

                all_stats[col] = {
                    'method': 'mean_fallback',
                    'missing_count': count,
                    'fill_value': mean_val
                }
                continue

            # 分离有缺失值和无缺失值的样本
            missing_mask = df_imputed[col].isnull()
            X_train = df_imputed.loc[~missing_mask, other_numeric]
            y_train = df_imputed.loc[~missing_mask, col]
            X_missing = df_imputed.loc[missing_mask, other_numeric]

            # 使用简单的随机森林模型（固定参数）
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )

            # 训练模型
            rf.fit(X_train, y_train)

            # 预测缺失值
            predicted_values = rf.predict(X_missing)

            # 填充缺失值
            df_imputed.loc[missing_mask, col] = predicted_values

            # 记录统计信息
            all_stats[col] = {
                'method': 'random_forest_simple',
                'missing_count': count,
                'original_mean': df[col].mean(),
                'filled_mean': df_imputed[col].mean(),
                'predictions_mean': predicted_values.mean(),
                'predictions_min': predicted_values.min(),
                'predictions_max': predicted_values.max()
            }

            print(f"  {col} 填充完成")

    return df_imputed, all_stats


def impute_target_variables_rf(df, target_cols, numeric_cols, categorical_cols=None):
    """
    按缺失值数量从小到大的顺序使用随机森林填充所有目标变量

    参数:
    df: 数据框
    target_cols: 目标变量列列表
    numeric_cols: 数值特征列列表
    categorical_cols: 分类特征列列表

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序使用随机森林填充目标变量...")
    df_imputed = df.copy()
    all_stats = {}
    all_importances = {}

    # 计算每列的缺失值数量
    missing_counts = {col: df[col].isnull().sum() for col in target_cols}

    # 按缺失值数量排序
    sorted_cols = sorted(missing_counts.items(), key=lambda x: x[1])

    # 按顺序填充
    for col, count in sorted_cols:
        if count > 0:
            print(f"\n填充目标变量: {col}，缺失值数量: {count}")
            df_imputed, stats, importances = rf_imputation_with_optimization(
                df_imputed, col, numeric_cols, categorical_cols
            )

            if stats is not None:  # 确保填充成功
                all_stats[col] = stats
                all_importances[col] = importances

    return df_imputed, all_stats, all_importances


def rf_imputation_pipeline(preprocessed_data_path, output_path='rf_imputed_data.csv'):
    """
    随机森林填充完整流程

    参数:
    preprocessed_data_path: 预处理后的数据文件路径
    output_path: 输出文件路径

    返回:
    填充后的数据框及相关统计信息
    """
    # 加载预处理后的数据
    df = pd.read_csv(preprocessed_data_path)

    # 识别数值特征和目标变量
    numeric_cols = ['pH', 'organicmatter', 'CEC', 'Cdcontent', '温度',
                    'pHsoil', 'Charcoal_application_ratio']

    target_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    categorical_cols = ['原料', '原料大类', '品种', '品种类型', 'Irrigationmethod']

    # 1. 首先填充数值特征
    df_features_imputed, feature_stats = impute_numeric_features_rf(df, numeric_cols, categorical_cols)

    # 2. 然后填充目标变量
    df_imputed, target_stats, feature_importances = impute_target_variables_rf(
        df_features_imputed, target_cols, numeric_cols, categorical_cols
    )

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\n随机森林填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'feature_importances': feature_importances
    }
    return df_imputed, all_stats

    if __name__ == "__main__":
        # 执行随机森林填充流程
        imputed_data, stats = rf_imputation_pipeline('preprocessed_data.csv')
        print("随机森林填充完成！")