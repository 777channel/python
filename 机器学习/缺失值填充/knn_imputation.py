import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import time
import warnings

warnings.filterwarnings('ignore')


def knn_imputation(df, target_col=None, n_neighbors=5, use_target_encoding=True):
    """
    使用KNN填充缺失值

    参数:
    df: 数据框
    target_col: 需要填充的目标列，默认为None (填充数值特征)
    n_neighbors: KNN的邻居数量
    use_target_encoding: 是否使用目标专用的影响编码

    返回:
    填充后的数据框及相关统计信息
    """
    print(f"\n正在使用KNN填充 {'数值特征' if target_col is None else target_col}...")
    start_time = time.time()
    df_imputed = df.copy()
    stats = {}

    # 确定要使用的特征
    # 选择数值列和编码后的分类特征
    encoded_features = [col for col in df.columns if ('编码' in col or '原料大类_' in col)]
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if
                    col not in encoded_features and
                    not col.startswith('CD_') and
                    '对' not in col]  # 排除目标影响编码

    # 如果是为特定目标变量填充，且需要使用目标专用影响编码
    if target_col is not None and use_target_encoding:
        # 为原料创建针对当前目标的影响编码
        impact_col = f'原料对{target_col}的影响'
        material_impact_col = f'原料大类对{target_col}的影响'

        # 检查是否已有这些编码
        if impact_col not in df_imputed.columns:
            from data_preprocessing import create_target_specific_encoding
            df_imputed, _ = create_target_specific_encoding(df_imputed, target_col, '原料')

        if material_impact_col not in df_imputed.columns:
            from data_preprocessing import create_target_specific_encoding
            df_imputed, _ = create_target_specific_encoding(df_imputed, target_col, '原料大类')

        # 添加这些影响编码到特征列表
        if impact_col in df_imputed.columns:
            encoded_features.append(impact_col)
        if material_impact_col in df_imputed.columns:
            encoded_features.append(material_impact_col)

    # 确定要填充的列
    if target_col is None:
        # 填充所有数值特征
        columns_to_fill = numeric_cols
    else:
        # 只填充指定的目标变量
        columns_to_fill = [target_col]

    # 准备KNN输入数据
    columns_for_knn = encoded_features + numeric_cols
    if target_col is not None:
        columns_for_knn.append(target_col)

    # 从KNN数据中移除没有需要填充的列
    knn_data = df_imputed[columns_for_knn].copy()

    # 检查是否有任何缺失值
    missing_counts = {col: knn_data[col].isnull().sum() for col in columns_to_fill}
    total_missing = sum(missing_counts.values())

    if total_missing > 0:
        print(f"  发现 {total_missing} 个缺失值需要填充")

        # 记录填充前的统计信息
        before_stats = {}
        for col in columns_to_fill:
            if missing_counts[col] > 0:
                before_stats[col] = {
                    'missing_count': missing_counts[col],
                    'mean': knn_data[col].mean(),
                    'std': knn_data[col].std(),
                    'min': knn_data[col].min(),
                    'max': knn_data[col].max()
                }
                print(f"  {col}: 缺失值数量 = {missing_counts[col]}")

        # 标准化数据
        scaler = StandardScaler()
        # 只标准化数值列，不处理编码列
        numeric_for_scale = [col for col in knn_data.columns if col not in encoded_features]
        knn_data[numeric_for_scale] = scaler.fit_transform(knn_data[numeric_for_scale].fillna(0))

        # 创建KNN填充器
        imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

        # 填充数据
        imputed_data = imputer.fit_transform(knn_data)
        imputed_df = pd.DataFrame(imputed_data, columns=knn_data.columns, index=knn_data.index)

        # 反向标准化
        imputed_df[numeric_for_scale] = scaler.inverse_transform(imputed_df[numeric_for_scale])

        # 将填充后的值复制回原始数据框
        for col in columns_to_fill:
            if missing_counts[col] > 0:
                # 获取原始缺失值的索引
                missing_mask = df_imputed[col].isnull()
                # 仅填充缺失位置
                df_imputed.loc[missing_mask, col] = imputed_df.loc[missing_mask, col]

                # 记录填充后的统计信息
                stats[col] = {
                    'missing_count': missing_counts[col],
                    'original_mean': before_stats[col]['mean'],
                    'original_std': before_stats[col]['std'],
                    'original_min': before_stats[col]['min'],
                    'original_max': before_stats[col]['max'],
                    'filled_mean': df_imputed[col].mean(),
                    'filled_std': df_imputed[col].std(),
                    'filled_min': df_imputed[col].min(),
                    'filled_max': df_imputed[col].max(),
                    'filled_values_mean': df_imputed.loc[missing_mask, col].mean(),
                    'filled_values_min': df_imputed.loc[missing_mask, col].min(),
                    'filled_values_max': df_imputed.loc[missing_mask, col].max()
                }

                print(f"  {col}: 已填充 {missing_counts[col]} 个值")
                print(f"    填充值范围: [{stats[col]['filled_values_min']:.4f}, {stats[col]['filled_values_max']:.4f}]")
                print(f"    填充值均值: {stats[col]['filled_values_mean']:.4f}")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"  KNN填充完成，用时: {execution_time:.2f} 秒")

    return df_imputed, stats


def impute_numeric_features_knn(df, numeric_cols, n_neighbors=5):
    print("\n开始按顺序使用KNN填充数值特征...")
    df_imputed = df.copy()
    all_stats = {}

    # 计算每列的缺失值数量
    missing_counts = {col: df[col].isnull().sum() for col in numeric_cols}

    # 按缺失值数量排序
    sorted_cols = sorted(missing_counts.items(), key=lambda x: x[1])

    # 打印col_stats的键
    print(f"准备填充的列: {[col for col, count in sorted_cols if count > 0]}")

    # 按顺序填充
    for col, count in sorted_cols:
        if count > 0:
            print(f"\n填充特征: {col}，缺失值数量: {count}")
            # 对于数值特征，不使用目标专用编码
            temp_df = df_imputed.copy()
            temp_columns = [col]  # 只填充当前列
            temp_df, col_stats = knn_imputation(temp_df, None, n_neighbors, False)

            # 将填充后的值复制回原始数据框
            df_imputed[col] = temp_df[col]

            # 添加安全检查
            if col in col_stats:
                all_stats[col] = col_stats[col]
            else:
                print(f"警告: 列 {col} 在结果中不存在，使用默认统计信息")
                # 创建基本统计信息
                filled_values = df_imputed.loc[df[col].isnull(), col]
                all_stats[col] = {
                    'missing_count': count,
                    'original_mean': df[col].dropna().mean(),
                    'filled_mean': df_imputed[col].mean(),
                    'filled_values_mean': filled_values.mean(),
                    'filled_values_min': filled_values.min(),
                    'filled_values_max': filled_values.max()
                }

    return df_imputed, all_stats

def impute_target_variables_knn(df, target_cols, n_neighbors=5):
    """
    按缺失值数量从小到大的顺序使用KNN填充所有目标变量

    参数:
    df: 数据框
    target_cols: 目标变量列列表
    n_neighbors: KNN的邻居数量

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序使用KNN填充目标变量...")
    df_imputed = df.copy()
    all_stats = {}

    # 计算每列的缺失值数量
    missing_counts = {col: df[col].isnull().sum() for col in target_cols}

    # 按缺失值数量排序
    sorted_cols = sorted(missing_counts.items(), key=lambda x: x[1])

    # 按顺序填充
    for col, count in sorted_cols:
        if count > 0:
            print(f"\n填充目标变量: {col}，缺失值数量: {count}")
            # 对于目标变量，使用目标专用编码
            df_imputed, col_stats = knn_imputation(df_imputed, col, n_neighbors, True)
            all_stats[col] = col_stats

    return df_imputed, all_stats


def knn_imputation_pipeline(preprocessed_data_path, output_path='knn_imputed_data.csv', n_neighbors=5):
    """
    KNN填充完整流程

    参数:
    preprocessed_data_path: 预处理后的数据文件路径
    output_path: 输出文件路径
    n_neighbors: KNN的邻居数量

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

    # 1. 首先填充数值特征
    df_features_imputed, feature_stats = impute_numeric_features_knn(df, numeric_cols, n_neighbors)

    # 2. 然后填充目标变量
    df_imputed, target_stats = impute_target_variables_knn(df_features_imputed, target_cols, n_neighbors)

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\nKNN填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'n_neighbors': n_neighbors
    }
    return df_imputed, all_stats


def optimize_knn_neighbors(df, target_col, n_values=[3, 5, 7, 9, 11]):
    """
    测试不同的邻居数量，找到最佳KNN参数

    参数:
    df: 数据框
    target_col: 目标变量列名
    n_values: 要测试的邻居数量列表

    返回:
    最佳的邻居数量
    """
    print(f"\n正在为 {target_col} 优化KNN邻居数量...")

    # 创建一个小的测试集（随机选择20%有值的样本，将其设为缺失值）
    df_test = df.copy()
    non_null_mask = df_test[target_col].notnull()
    non_null_indices = non_null_mask[non_null_mask].index

    if len(non_null_indices) < 5:
        print("  样本数量不足，无法进行优化")
        return 5  # 默认返回5

    # 随机选择20%作为测试集
    test_size = max(int(len(non_null_indices) * 0.2), 5)
    np.random.seed(42)
    test_indices = np.random.choice(non_null_indices, test_size, replace=False)

    # 保存原始值
    true_values = df_test.loc[test_indices, target_col].copy()

    # 将测试样本设为缺失值
    df_test.loc[test_indices, target_col] = np.nan

    results = {}
    best_n = None
    best_score = float('inf')

    for n in n_values:
        print(f"  测试 n_neighbors = {n}")
        # 使用KNN填充
        df_filled, _ = knn_imputation(df_test, target_col, n, True)

        # 计算均方误差
        predicted_values = df_filled.loc[test_indices, target_col]
        mse = ((true_values - predicted_values) ** 2).mean()
        mae = (true_values - predicted_values).abs().mean()

        results[n] = {
            'mse': mse,
            'mae': mae,
            'predicted_values': predicted_values,
            'true_values': true_values
        }

        print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")

        # 更新最佳值
        if mse < best_score:
            best_score = mse
            best_n = n

    print(f"\n最佳邻居数量: {best_n} (MSE: {results[best_n]['mse']:.6f}, MAE: {results[best_n]['mae']:.6f})")

    # 可视化不同邻居数量的效果
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    mse_values = [results[n]['mse'] for n in n_values]
    plt.plot(n_values, mse_values, 'o-')
    plt.xlabel('邻居数量')
    plt.ylabel('均方误差 (MSE)')
    plt.title('KNN邻居数量vs.MSE')

    plt.subplot(1, 2, 2)
    mae_values = [results[n]['mae'] for n in n_values]
    plt.plot(n_values, mae_values, 'o-')
    plt.xlabel('邻居数量')
    plt.ylabel('平均绝对误差 (MAE)')
    plt.title('KNN邻居数量vs.MAE')

    plt.tight_layout()
    return best_n, results


def knn_imputation_with_optimization(preprocessed_data_path, output_path='knn_imputed_data_optimized.csv'):
    """
    使用优化后的参数进行KNN填充

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

    # 1. 首先填充数值特征（使用默认邻居数量）
    df_features_imputed, feature_stats = impute_numeric_features_knn(df, numeric_cols, n_neighbors=5)

    # 2. 对每个目标变量优化邻居数量并填充
    df_imputed = df_features_imputed.copy()
    target_stats = {}
    best_neighbors = {}

    for col in target_cols:
        missing_count = df_imputed[col].isnull().sum()
        if missing_count > 0:
            print(f"\n处理目标变量: {col}，缺失值数量: {missing_count}")

            # 优化邻居数量
            best_n, opt_results = optimize_knn_neighbors(df_imputed, col)
            best_neighbors[col] = best_n

            # 使用最佳邻居数量填充
            print(f"\n使用最佳邻居数量 {best_n} 填充 {col}")
            df_imputed, col_stats = knn_imputation(df_imputed, col, best_n, True)
            target_stats[col] = col_stats

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\n优化KNN填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'best_neighbors': best_neighbors
    }
    return df_imputed, all_stats


if __name__ == "__main__":
    # 执行带优化的KNN填充流程
    imputed_data, stats = knn_imputation_with_optimization('preprocessed_data.csv')
    print("KNN填充完成！")