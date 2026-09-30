import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
import time
import warnings

warnings.filterwarnings('ignore')


def interpolation_imputation(df, target_col=None, method='linear', use_grouping=True):
    """
    使用插值法填充缺失值，考虑分类特征分组

    参数:
    df: 数据框 (使用prepare_data_for_interpolation函数处理)
    target_col: 需要填充的目标列，默认为None (填充数值特征)
    method: 插值方法，默认为'linear'
    use_grouping: 是否使用分组插值策略

    返回:
    填充后的数据框及相关统计信息
    """
    print(f"\n正在使用插值填充 {'数值特征' if target_col is None else target_col}...")
    start_time = time.time()
    df_imputed = df.copy()
    stats = {}

    # 确定要填充的列
    if target_col is None:
        # 填充所有数值特征
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # 排除目标变量和编码列
        columns_to_fill = [col for col in numeric_cols if
                           not col.startswith('CD_') and
                           not (col.endswith('编码') or col.endswith('的影响'))]
    else:
        # 只填充指定的目标变量
        columns_to_fill = [target_col]

    # 如果是目标变量且需要分组，确保有分组编码
    if target_col is not None and use_grouping:
        if 'interpolation_group' not in df_imputed.columns:
            # 创建分组
            from data_preprocessing import prepare_data_for_interpolation
            df_imputed, _ = prepare_data_for_interpolation(df_imputed, target_col,
                                                           ['原料', '原料大类', '品种类型', 'Irrigationmethod'])

    # 遍历每个需要填充的列
    for col in columns_to_fill:
        missing_count = df_imputed[col].isnull().sum()

        if missing_count > 0:
            print(f"  {col}: 发现 {missing_count} 个缺失值")

            # 记录填充前统计信息
            before_stats = {
                'missing_count': missing_count,
                'mean': df_imputed[col].mean(),
                'std': df_imputed[col].std(),
                'min': df_imputed[col].min(),
                'max': df_imputed[col].max()
            }

            # 使用分组插值策略（对于目标变量）
            if target_col is not None and use_grouping and 'interpolation_group' in df_imputed.columns:
                # 统计分组情况
                groups = df_imputed['interpolation_group'].unique()
                print(f"  共有 {len(groups)} 个分组")

                # 记录分组填充情况
                group_stats = {}
                still_missing = missing_count

                # 分组插值（第一轮：使用完整分组）
                for group in groups:
                    # 获取组内数据
                    group_data = df_imputed[df_imputed['interpolation_group'] == group]
                    group_missing = group_data[col].isnull().sum()

                    if group_missing > 0 and len(group_data) > group_missing:
                        # 只有组内同时有缺失值和非缺失值时才能进行插值
                        try:
                            # 创建组内索引
                            df_imputed.loc[df_imputed['interpolation_group'] == group, col] = (
                                df_imputed.loc[df_imputed['interpolation_group'] == group, col]
                                .interpolate(method=method)
                                .fillna(method='ffill')
                                .fillna(method='bfill')
                            )

                            # 更新统计
                            filled_in_group = group_missing - df_imputed[df_imputed['interpolation_group'] == group][
                                col].isnull().sum()
                            still_missing -= filled_in_group

                            group_stats[group] = {
                                'original_missing': group_missing,
                                'filled': filled_in_group,
                                'still_missing': group_missing - filled_in_group
                            }

                            print(f"    分组 '{group}': 填充了 {filled_in_group}/{group_missing} 个值")
                        except:
                            print(f"    分组 '{group}' 插值失败，可能是数据不足")

                # 如果仍有缺失值，尝试简化分组
                if still_missing > 0:
                    print(f"  仍有 {still_missing} 个缺失值，尝试简化分组策略")

                    # 创建一个简化的分组列，只使用原料大类和品种类型
                    if '原料大类' in df_imputed.columns and '品种类型' in df_imputed.columns:
                        df_imputed['simple_group'] = df_imputed['原料大类'] + '_' + df_imputed['品种类型']

                        # 获取简化分组列表
                        simple_groups = df_imputed['simple_group'].unique()
                        print(f"  简化后有 {len(simple_groups)} 个分组")

                        # 对每个简化分组进行插值
                        for group in simple_groups:
                            # 获取组内还有缺失值的数据
                            group_data = df_imputed[(df_imputed['simple_group'] == group) &
                                                    (df_imputed[col].isnull())]
                            group_missing = len(group_data)

                            if group_missing > 0:
                                try:
                                    # 创建组内索引
                                    df_imputed.loc[df_imputed['simple_group'] == group, col] = (
                                        df_imputed.loc[df_imputed['simple_group'] == group, col]
                                        .interpolate(method=method)
                                        .fillna(method='ffill')
                                        .fillna(method='bfill')
                                    )

                                    # 更新统计
                                    filled_in_group = group_missing - df_imputed[(df_imputed['simple_group'] == group) &
                                                                                 (df_imputed[col].isnull())].shape[0]
                                    still_missing -= filled_in_group

                                    print(f"    简化分组 '{group}': 填充了 {filled_in_group}/{group_missing} 个值")
                                except:
                                    print(f"    简化分组 '{group}' 插值失败")

                    # 如果还有缺失值，直接全局插值
                    if still_missing > 0:
                        print(f"  仍有 {still_missing} 个缺失值，进行全局插值")

                        # 全局插值
                        df_imputed[col] = df_imputed[col].interpolate(method=method)

                        # 处理可能的头尾缺失值
                        df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')

                        final_missing = df_imputed[col].isnull().sum()
                        if final_missing > 0:
                            # 如果仍然有缺失值，使用均值填充
                            print(f"  全局插值后仍有 {final_missing} 个缺失值，使用均值填充")
                            mean_val = df_imputed[col].mean()
                            df_imputed[col] = df_imputed[col].fillna(mean_val)
            else:
                # 直接全局插值（不分组）
                try:
                    # 尝试索引插值
                    df_imputed[col] = df_imputed[col].interpolate(method=method)
                except:
                    # 退化为线性插值
                    print(f"    方法 '{method}' 失败，退化为线性插值")
                    df_imputed[col] = df_imputed[col].interpolate(method='linear')

                # 处理可能的头尾缺失值
                df_imputed[col] = df_imputed[col].fillna(method='ffill').fillna(method='bfill')

                # 如果仍然有缺失值（可能是全部都是缺失值的情况），使用均值填充
                still_missing = df_imputed[col].isnull().sum()
                if still_missing > 0:
                    print(f"    插值后仍有 {still_missing} 个缺失值，使用均值填充")
                    mean_val = df[col].dropna().mean()
                    if np.isnan(mean_val):  # 如果所有值都是缺失的
                        mean_val = 5
                    df_imputed[col] = df_imputed[col].fillna(mean_val)

            # 记录填充后统计信息
            stats[col] = {
                'missing_count': missing_count,
                'method': method,
                'use_grouping': use_grouping,
                'original_mean': before_stats['mean'],
                'original_std': before_stats['std'],
                'original_min': before_stats['min'],
                'original_max': before_stats['max'],
                'filled_mean': df_imputed[col].mean(),
                'filled_std': df_imputed[col].std(),
                'filled_min': df_imputed[col].min(),
                'filled_max': df_imputed[col].max()
            }

            # 检查填充结果
            final_missing = df_imputed[col].isnull().sum()
            if final_missing == 0:
                print(f"    {col} 填充完成！所有 {missing_count} 个缺失值已填充")
            else:
                print(f"    警告: {col} 仍有 {final_missing} 个缺失值！")

            # 可视化填充结果
            if col in columns_to_fill:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                sns.histplot(df[col].dropna(), kde=True, color='blue')
                plt.title(f'{col} - 填充前')

                plt.subplot(1, 2, 2)
                sns.histplot(df_imputed[col], kde=True, color='green')
                plt.title(f'{col} - 填充后')

                plt.tight_layout()
                plt.savefig(f'{col}_interpolation.png')
                print(f"  已保存填充结果图: {col}_interpolation.png")
        else:
            print(f"  {col}: 没有缺失值，无需填充")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"  插值填充完成，用时: {execution_time:.2f} 秒")

    return df_imputed, stats


def impute_numeric_features_interpolation(df, numeric_cols, method='linear'):
    """
    按缺失值数量从小到大的顺序使用插值填充所有数值特征

    参数:
    df: 数据框
    numeric_cols: 数值特征列列表
    method: 插值方法，默认为'linear'

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序使用插值填充数值特征...")
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
            # 对于数值特征，不使用分组策略
            df_temp = df_imputed.copy()
            df_temp[col] = None  # 临时列，避免修改原始数据框
            df_filled, col_stats = interpolation_imputation(df_temp, col, method, use_grouping=False)

            # 将填充结果复制回原始数据框
            df_imputed[col] = df_filled[col]
            all_stats[col] = col_stats[col]

    return df_imputed, all_stats


def impute_target_variables_interpolation(df, target_cols, method='linear'):
    """
    按缺失值数量从小到大的顺序使用插值填充所有目标变量

    参数:
    df: 数据框
    target_cols: 目标变量列列表
    method: 插值方法，默认为'linear'

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序使用插值填充目标变量...")
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
            # 对于目标变量，使用分组策略
            df_imputed, col_stats = interpolation_imputation(df_imputed, col, method, use_grouping=True)
            all_stats[col] = col_stats

    return df_imputed, all_stats


def interpolation_pipeline(preprocessed_data_path, output_path='interpolation_imputed_data.csv', method='linear'):
    """
    插值填充完整流程

    参数:
    preprocessed_data_path: 预处理后的数据文件路径
    output_path: 输出文件路径
    method: 插值方法，默认为'linear'

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
    df_features_imputed, feature_stats = impute_numeric_features_interpolation(df, numeric_cols, method)

    # 2. 然后填充目标变量
    df_imputed, target_stats = impute_target_variables_interpolation(df_features_imputed, target_cols, method)

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\n插值填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'method': method
    }
    return df_imputed, all_stats


def compare_interpolation_methods(df, target_col, methods=['linear', 'cubic', 'nearest', 'spline']):
    """
    比较不同插值方法的效果

    参数:
    df: 数据框
    target_col: 目标变量列名
    methods: 要比较的插值方法列表

    返回:
    最佳的插值方法
    """
    print(f"\n正在为 {target_col} 比较不同插值方法...")

    # 创建一个小的测试集（随机选择20%有值的样本，将其设为缺失值）
    df_test = df.copy()
    non_null_mask = df_test[target_col].notnull()
    non_null_indices = non_null_mask[non_null_mask].index

    if len(non_null_indices) < 5:
        print("  样本数量不足，无法进行比较")
        return 'linear'  # 默认返回linear

    # 随机选择20%作为测试集
    test_size = max(int(len(non_null_indices) * 0.2), 5)
    np.random.seed(42)
    test_indices = np.random.choice(non_null_indices, test_size, replace=False)

    # 保存原始值
    true_values = df_test.loc[test_indices, target_col].copy()

    # 将测试样本设为缺失值
    df_test.loc[test_indices, target_col] = np.nan

    results = {}
    best_method = None
    best_score = float('inf')

    for method in methods:
        try:
            print(f"  测试插值方法: {method}")
            # 使用插值填充
            df_filled, _ = interpolation_imputation(df_test, target_col, method, use_grouping=True)

            # 计算误差
            predicted_values = df_filled.loc[test_indices, target_col]
            mse = ((true_values - predicted_values) ** 2).mean()
            mae = (true_values - predicted_values).abs().mean()

            results[method] = {
                'mse': mse,
                'mae': mae,
                'predicted_values': predicted_values,
                'true_values': true_values
            }

            print(f"    MSE: {mse:.6f}, MAE: {mae:.6f}")

            # 更新最佳方法
            if mse < best_score:
                best_score = mse
                best_method = method
        except Exception as e:
            print(f"  方法 {method} 出错: {str(e)}")

    if best_method:
        print(
            f"\n最佳插值方法: {best_method} (MSE: {results[best_method]['mse']:.6f}, MAE: {results[best_method]['mae']:.6f})")
    else:
        print("\n所有方法都失败，默认使用线性插值")
        best_method = 'linear'

    # 可视化不同方法的效果
    if len(results) > 0:
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        mse_values = [results[m]['mse'] for m in methods if m in results]
        methods_available = [m for m in methods if m in results]
        plt.bar(methods_available, mse_values)
        plt.xlabel('插值方法')
        plt.ylabel('均方误差 (MSE)')
        plt.title('不同插值方法的MSE对比')

        plt.subplot(1, 2, 2)
        mae_values = [results[m]['mae'] for m in methods if m in results]
        plt.bar(methods_available, mae_values)
        plt.xlabel('插值方法')
        plt.ylabel('平均绝对误差 (MAE)')
        plt.title('不同插值方法的MAE对比')

        plt.tight_layout()
    return best_method, results


def interpolation_pipeline_with_optimization(preprocessed_data_path,
                                             output_path='interpolation_imputed_data_optimized.csv'):
    """
    使用优化后的参数进行插值填充

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

    # 要测试的插值方法
    methods = ['linear', 'cubic', 'nearest', 'slinear']

    # 1. 首先填充数值特征（使用线性插值）
    df_features_imputed, feature_stats = impute_numeric_features_interpolation(df, numeric_cols, 'linear')

    # 2. 对每个目标变量选择最佳插值方法并填充
    df_imputed = df_features_imputed.copy()
    target_stats = {}
    best_methods = {}

    for col in target_cols:
        missing_count = df_imputed[col].isnull().sum()
        if missing_count > 0:
            print(f"\n处理目标变量: {col}，缺失值数量: {missing_count}")

            # 比较不同插值方法
            best_method, comparison_results = compare_interpolation_methods(df_imputed, col, methods)
            best_methods[col] = best_method

            # 使用最佳插值方法填充
            print(f"\n使用最佳插值方法 {best_method} 填充 {col}")
            df_imputed, col_stats = interpolation_imputation(df_imputed, col, best_method, use_grouping=True)
            target_stats[col] = col_stats

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\n优化插值填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'best_methods': best_methods
    }
    return df_imputed, all_stats


if __name__ == "__main__":
    # 执行带优化的插值填充流程
    imputed_data, stats = interpolation_pipeline_with_optimization('preprocessed_data.csv')
    print("插值填充完成！")