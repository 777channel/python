import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def mean_imputation(df, numeric_cols, target_col=None):
    """
    使用均值填充数值列

    参数:
    df: 数据框
    numeric_cols: 需要填充的数值特征列
    target_col: 需要填充的目标列，默认为None

    返回:
    填充后的数据框及相关统计信息
    """
    print(f"\n正在使用均值填充 {'数值特征' if target_col is None else target_col}...")
    df_imputed = df.copy()
    stats = {}

    # 填充数值特征
    if target_col is None:
        for col in numeric_cols:
            if df_imputed[col].isnull().sum() > 0:
                # 统计填充前信息
                missing_count = df[col].isnull().sum()
                mean_val = df_imputed[col].mean()
                std_val = df_imputed[col].std()

                # 执行填充
                df_imputed[col] = df_imputed[col].fillna(mean_val)

                # 记录统计信息
                stats[col] = {
                    'missing_count': missing_count,
                    'fill_value': mean_val,
                    'original_mean': mean_val,
                    'original_std': std_val,
                    'filled_mean': df_imputed[col].mean(),
                    'filled_std': df_imputed[col].std()
                }

                print(f"  {col}: 填充了 {missing_count} 个缺失值，均值 = {mean_val:.4f}")

    # 填充目标变量
    else:
        if df_imputed[target_col].isnull().sum() > 0:
            # 统计填充前信息
            missing_count = df[target_col].isnull().sum()
            mean_val = df_imputed[target_col].mean()
            std_val = df_imputed[target_col].std()
            min_val = df_imputed[target_col].min()
            max_val = df_imputed[target_col].max()

            # 执行填充
            df_imputed[target_col] = df_imputed[target_col].fillna(mean_val)

            # 记录统计信息
            stats = {
                'missing_count': missing_count,
                'fill_value': mean_val,
                'original_mean': mean_val,
                'original_std': std_val,
                'original_min': min_val,
                'original_max': max_val,
                'filled_mean': df_imputed[target_col].mean(),
                'filled_std': df_imputed[target_col].std(),
                'filled_min': df_imputed[target_col].min(),
                'filled_max': df_imputed[target_col].max()
            }

            print(f"  {target_col}: 填充了 {missing_count} 个缺失值，均值 = {mean_val:.4f}")

            # 可视化填充结果
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(df[target_col].dropna(), kde=True, color='blue')
            plt.title(f'{target_col} - 填充前')
            plt.axvline(mean_val, color='red', linestyle='--')
            plt.annotate(f'均值 = {mean_val:.4f}',
                         xy=(mean_val, 0),
                         xytext=(mean_val + 0.05, 5),
                         arrowprops=dict(facecolor='black', shrink=0.05))

            plt.subplot(1, 2, 2)
            sns.histplot(df_imputed[target_col], kde=True, color='green')
            plt.title(f'{target_col} - 填充后')
            plt.axvline(mean_val, color='red', linestyle='--')

            plt.tight_layout()
    return df_imputed, stats


def impute_numeric_features(df, numeric_cols):
    """
    按缺失值数量从小到大的顺序填充所有数值特征

    参数:
    df: 数据框
    numeric_cols: 数值特征列列表

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序填充数值特征...")
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
            df_imputed, col_stats = mean_imputation(df_imputed, [col])
            all_stats[col] = col_stats[col]

    return df_imputed, all_stats


def impute_target_variables(df, target_cols):
    """
    按缺失值数量从小到大的顺序填充所有目标变量

    参数:
    df: 数据框
    target_cols: 目标变量列列表

    返回:
    填充后的数据框及相关统计信息
    """
    print("\n开始按顺序填充目标变量...")
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
            df_imputed, col_stats = mean_imputation(df_imputed, None, col)
            all_stats[col] = col_stats

    return df_imputed, all_stats


def mean_imputation_pipeline(preprocessed_data_path, output_path='mean_imputed_data.csv'):
    """
    均值填充完整流程

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

    # 1. 首先填充数值特征
    df_features_imputed, feature_stats = impute_numeric_features(df, numeric_cols)

    # 2. 然后填充目标变量
    df_imputed, target_stats = impute_target_variables(df_features_imputed, target_cols)

    # 保存结果
    df_imputed.to_csv(output_path, index=False)
    print(f"\n均值填充完成，结果已保存到 {output_path}")

    # 汇总统计信息
    all_stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats
    }

    # 生成总结报告
    report = "均值填充方法统计报告\n"
    report += "=" * 50 + "\n\n"

    report += "数值特征填充统计:\n"
    for col, stats in feature_stats.items():
        report += f"* {col}:\n"
        report += f"  - 缺失值数量: {stats['missing_count']}\n"
        report += f"  - 填充值: {stats['fill_value']:.4f}\n"
        report += f"  - 原始均值: {stats['original_mean']:.4f}\n"
        report += f"  - 填充后均值: {stats['filled_mean']:.4f}\n"
        report += f"  - 原始标准差: {stats['original_std']:.4f}\n"
        report += f"  - 填充后标准差: {stats['filled_std']:.4f}\n\n"

    report += "目标变量填充统计:\n"
    for col, stats in target_stats.items():
        report += f"* {col}:\n"
        report += f"  - 缺失值数量: {stats['missing_count']}\n"
        report += f"  - 填充值: {stats['fill_value']:.4f}\n"
        report += f"  - 原始均值: {stats['original_mean']:.4f}\n"
        report += f"  - 填充后均值: {stats['filled_mean']:.4f}\n"
        report += f"  - 原始范围: [{stats['original_min']:.4f}, {stats['original_max']:.4f}]\n"
        report += f"  - 填充后范围: [{stats['filled_min']:.4f}, {stats['filled_max']:.4f}]\n\n"

    # 保存报告
    with open('mean_imputation_report.txt', 'w') as f:
        f.write(report)

    print("统计报告已保存到 mean_imputation_report.txt")

    return df_imputed, all_stats


if __name__ == "__main__":
    # 执行均值填充流程
    imputed_data, stats = mean_imputation_pipeline('preprocessed_data.csv')
    print("均值填充完成！")