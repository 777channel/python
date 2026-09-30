import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def load_data(file_path):
    """加载数据并进行初步检查"""
    # 尝试不同的编码方式加载数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')

    print(f"数据加载成功，形状: {df.shape}")

    # 检查并输出缺失值情况
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    print("\n原始数据缺失值统计:")
    missing_stats = pd.DataFrame({
        '缺失值数量': missing_values,
        '缺失百分比': missing_percentage.round(2)
    })
    print(missing_stats[missing_stats['缺失值数量'] > 0].sort_values(by='缺失值数量', ascending=False))

    return df


def preprocess_data_without_imputation(df):
    """对数据进行预处理和特征工程，但不填充缺失值"""
    df_processed = df.copy()

    # 处理品种列 - 严格按照归一化代码进行二元编码
    if '品种' in df_processed.columns:
        print("\n处理品种列...")
        # 检查品种列的唯一值
        unique_varieties = df_processed['品种'].unique()
        print(f"品种列唯一值: {unique_varieties}")

        # 对于非缺失值进行二元编码: 低累积品种=0, 普通品种=1
        df_processed['variety_binary'] = df_processed['品种'].apply(
            lambda x: 0 if pd.notna(x) and '低累积' in str(x) else (1 if pd.notna(x) else np.nan)
        )

        # 查看编码结果
        variety_counts = df_processed['variety_binary'].value_counts(dropna=False)
        print("\n品种二元编码统计:")
        print(variety_counts)

        # 删除原始品种列
        df_processed = df_processed.drop('品种', axis=1)
        print("品种列已编码为variety_binary")

    # 处理原料列 - 直接移除，不参与相关性分析
    if 'ingredient' in df_processed.columns:
        print("\n移除原料列(多类别变量)，不参与相关性分析")
        df_processed = df_processed.drop('ingredient', axis=1)

    # 处理灌溉方式列 - 按照归一化代码进行编码
    if 'Irrigationmethod' in df_processed.columns:
        print("\n处理灌溉方式列...")
        # 检查灌溉方式列的唯一值
        unique_methods = df_processed['Irrigationmethod'].dropna().unique()
        print(f"灌溉方式列唯一值: {unique_methods}")

        # 获取唯一的灌溉方式（排除NaN）
        irrigation_methods = sorted([m for m in unique_methods if not pd.isna(m) and m != ''])

        # 创建映射字典
        method_mapping = {method: i for i, method in enumerate(irrigation_methods)}
        method_mapping['nan'] = -1  # 处理缺失值
        print(f"灌溉方式编码映射: {method_mapping}")

        # 使用单一编码而非多个二元编码
        df_processed['irrigation_type'] = df_processed['Irrigationmethod'].apply(
            lambda x: method_mapping.get(x, -1) if pd.notna(x) and x != '' else -1
        )

        # 查看编码结果
        method_counts = df_processed['irrigation_type'].value_counts(dropna=False)
        print("\n灌溉方式编码分布:")
        print(method_counts)

        # 删除原始列
        df_processed = df_processed.drop('Irrigationmethod', axis=1)
        print("灌溉方式列已编码为irrigation_type")

    # 重命名目标变量以简化
    rename_dict = {
        'CD_soil_change': 'CD_soil',
        'CD_brown_rice_change': 'CD_rice',
        'CD_stem_change': 'CD_stem',
        'CD_leaf_change': 'CD_leaf',
        'CD_shell_change': 'CD_shell',
        'CD_root_change': 'CD_root'
    }

    # 如果列存在，则重命名
    for old_name, new_name in rename_dict.items():
        if old_name in df_processed.columns:
            df_processed = df_processed.rename(columns={old_name: new_name})

    # 确保数值列的类型正确
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            print(f"尝试将列 '{col}' 转换为数值类型")
            # 尝试转换为数值类型，但保留缺失值
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # 检查每列的非缺失值数量和比例
    non_missing_counts = df_processed.count()
    non_missing_percentage = (non_missing_counts / len(df_processed)) * 100

    print("\n每列的非缺失值统计:")
    non_missing_stats = pd.DataFrame({
        '非缺失值数量': non_missing_counts,
        '非缺失值百分比': non_missing_percentage.round(2)
    })
    print(non_missing_stats)

    # 检查是否存在全部缺失或几乎全部缺失的列
    almost_empty_threshold = 5  # 定义"几乎全部缺失"的阈值
    almost_empty_cols = non_missing_counts[non_missing_counts < almost_empty_threshold].index.tolist()

    if almost_empty_cols:
        print(f"\n警告: 以下列几乎全部为缺失值(非缺失值<{almost_empty_threshold}): {almost_empty_cols}")
        print("这些列将在分析时可能导致问题，建议注意")

    return df_processed


def calculate_pairwise_correlation(df, method='pearson', min_samples=5):
    """计算成对相关系数（忽略缺失对）"""
    columns = df.columns
    n = len(columns)
    corr_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)

    # 记录样本不足的对
    insufficient_pairs = []

    for i in range(n):
        for j in range(n):
            # 获取两列
            col_i = columns[i]
            col_j = columns[j]

            # 如果是同一列，相关系数为1
            if col_i == col_j:
                corr_matrix.loc[col_i, col_j] = 1.0
                continue

            # 只保留两列都非缺失的行
            valid_data = df[[col_i, col_j]].dropna()

            # 如果有足够的有效数据点，计算相关系数
            if len(valid_data) >= min_samples:
                try:
                    corr = valid_data[col_i].corr(valid_data[col_j], method=method)
                    corr_matrix.loc[col_i, col_j] = corr
                except Exception as e:
                    print(f"计算 {col_i} 和 {col_j} 的相关系数时出错: {e}")
                    corr_matrix.loc[col_i, col_j] = np.nan
            else:
                insufficient_pairs.append((col_i, col_j, len(valid_data)))

    if insufficient_pairs:
        print("\n以下特征对因样本数不足无法计算相关系数:")
        for col_i, col_j, count in insufficient_pairs:
            if col_i in df.columns and col_j in df.columns:
                print(f"  {col_i} - {col_j}: {count} 个样本 (最小要求: {min_samples})")

    return corr_matrix


def count_valid_pairs(df, feature_cols, target_cols):
    """计算每对特征-目标变量的有效样本数"""
    # 验证输入数据
    if not feature_cols or not target_cols:
        print("警告: 没有特征或目标变量")
        return pd.DataFrame([[0]], index=['无特征'], columns=['无目标'])

    # 创建有效样本数矩阵
    valid_counts = pd.DataFrame(0, index=feature_cols, columns=target_cols)
    has_valid_pair = False

    for feature in feature_cols:
        for target in target_cols:
            # 检查列是否在数据框中
            if feature in df.columns and target in df.columns:
                # 计算两列都非缺失的行数
                valid_count = df[[feature, target]].dropna().shape[0]
                valid_counts.loc[feature, target] = valid_count

                if valid_count > 0:
                    has_valid_pair = True

    # 验证结果
    if not has_valid_pair:
        print("警告: 没有任何特征-目标对有有效样本")
        # 确保返回可用于绘图的数据框
        return pd.DataFrame([[0]], index=['无有效特征'], columns=['无有效目标'])

    # 检查是否有无效的索引或列
    if valid_counts.isnull().values.any():
        print("警告: 有效样本数矩阵包含NaN值，将替换为0")
        valid_counts = valid_counts.fillna(0)

    return valid_counts


def plot_valid_counts_heatmap(valid_counts, output_file=None):
    """绘制有效样本数热图"""
    if valid_counts.empty or valid_counts.size == 0:
        print("错误: 无法绘制空的有效样本数矩阵")
        return False

    # 确保没有NaN值
    valid_counts_clean = valid_counts.fillna(0).astype(int)

    # 检查是否有数据可绘制
    if valid_counts_clean.shape[0] <= 1 and valid_counts_clean.shape[1] <= 1 and valid_counts_clean.iloc[0, 0] == 0:
        print("错误: 没有有效数据可绘制热图")
        return False

    try:
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(valid_counts_clean, annot=True, fmt="d", cmap="YlGnBu")
        plt.title("每对特征-目标变量的有效样本数")
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"有效样本数热图已保存到: {output_file}")

        plt.show()
        return True
    except Exception as e:
        print(f"绘制有效样本数热图时出错: {e}")
        # 尝试使用替代方法
        try:
            print("尝试使用替代方法绘制热图...")
            plt.figure(figsize=(12, 10))
            plt.matshow(valid_counts_clean, cmap="YlGnBu")
            plt.title("每对特征-目标变量的有效样本数")
            plt.colorbar()

            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"有效样本数热图已保存到: {output_file}")

            plt.show()
            return True
        except Exception as e2:
            print(f"替代方法也失败了: {e2}")
            print("将有效样本数导出到CSV文件...")
            valid_counts_clean.to_csv("valid_samples_counts.csv")
            print("有效样本数已保存到valid_samples_counts.csv")
            return False


def plot_correlation_heatmap(correlation_matrix, feature_cols, target_cols, output_file=None, method='pearson',
                             annot_fmt=".2f"):
    """绘制相关系数热图"""
    if correlation_matrix.empty or correlation_matrix.shape[0] <= 1 or correlation_matrix.shape[1] <= 1:
        print("错误: 相关矩阵为空或维度太小，无法绘制热图")
        return correlation_matrix

    # 确认所有列都存在于相关系数矩阵中
    available_features = [f for f in feature_cols if f in correlation_matrix.index]
    available_targets = [t for t in target_cols if t in correlation_matrix.columns]

    if not available_features or not available_targets:
        print("错误: 没有足够的特征或目标变量存在于相关系数矩阵中")
        return correlation_matrix

    # 创建简化的列名映射
    simplified_names = {}
    for col in available_targets:
        if col.startswith('CD_'):
            # 使用下标表示
            suffix = col.split('_')[1]
            simplified_names[col] = f"CD$_{{{suffix}}}$"

    # 重新排序相关系数矩阵的列和行
    ordered_columns = available_features + available_targets
    subset_matrix = correlation_matrix.loc[ordered_columns, ordered_columns]

    # 替换列名为简化版本
    display_matrix = subset_matrix.rename(columns=simplified_names, index=simplified_names)

    try:
        # 设置图表大小和样式
        plt.figure(figsize=(14, 12))
        sns.set(font="SimHei")  # 设置中文字体

        # 创建热图
        mask = np.zeros_like(subset_matrix, dtype=bool)
        # 如果只想显示下三角矩阵，可以取消下面一行的注释
        # mask[np.triu_indices_from(mask)] = True

        # 绘制热图
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
        heatmap = sns.heatmap(display_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                              square=True, linewidths=.5, annot=True, fmt=annot_fmt, annot_kws={"size": 8})

        # 设置标题和标签
        plt.title(f"特征与目标变量的{method}相关系数热图\n(忽略缺失值对，不含多类别变量)", fontsize=16, pad=20)
        plt.xticks(rotation=90, ha='right', fontsize=10)
        plt.yticks(fontsize=10)

        # 调整布局
        plt.tight_layout()

        # 保存图片
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"热图已保存到: {output_file}")

        # 显示图表
        plt.show()
    except Exception as e:
        print(f"绘制相关系数热图时出错: {e}")
        print("将相关系数矩阵导出到CSV文件...")
        subset_matrix.to_csv("correlation_matrix.csv")
        print("相关系数矩阵已保存到correlation_matrix.csv")

    return subset_matrix


def analyze_correlations(correlation_matrix, feature_cols, target_cols):
    """分析特征与目标变量之间的相关性"""
    # 确保所有列都在相关系数矩阵中
    available_features = [f for f in feature_cols if f in correlation_matrix.index]
    available_targets = [t for t in target_cols if t in correlation_matrix.columns]

    if not available_features or not available_targets:
        print("错误: 没有足够的特征或目标变量存在于相关系数矩阵中")
        return None

    # 提取特征与目标变量之间的相关系数
    feature_target_corr = correlation_matrix.loc[available_features, available_targets]

    # 计算每个特征的平均绝对相关系数（忽略NaN）
    avg_abs_corr = feature_target_corr.abs().mean(axis=1, skipna=True).sort_values(ascending=False)

    print("\n按平均绝对相关系数排序的特征重要性:")
    for feature, score in avg_abs_corr.items():
        # 计算特征与多少个目标变量有相关系数
        valid_targets = feature_target_corr.loc[feature].dropna().count()
        print(f"{feature}: {score:.4f} (有效目标数: {valid_targets}/{len(available_targets)})")

    print("\n每个目标变量的最相关特征:")
    for target in available_targets:
        target_corrs = feature_target_corr[target].dropna().abs().sort_values(ascending=False)

        if not target_corrs.empty:
            print(f"\n{target} (有效特征数: {len(target_corrs)}/{len(available_features)}):")
            for feature, corr in target_corrs.head(3).items():
                sign = "正相关" if feature_target_corr.loc[feature, target] > 0 else "负相关"
                print(f"  {feature}: {feature_target_corr.loc[feature, target]:.4f} ({sign})")
        else:
            print(f"\n{target}: 没有足够的非缺失值对计算相关系数")

    return feature_target_corr


def main():
    # 文件路径
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv"  # 替换为您的实际文件路径
    output_file_counts = "有效样本数热图_无多类别变量.png"  # 有效样本数热图输出路径
    output_file_corr = "镉变化相关系数热图_无多类别变量.png"  # 相关系数热图输出路径

    # 相关系数计算方法：'pearson'（默认，线性关系）, 'spearman'（单调关系，对异常值不敏感）, 'kendall'（序关系）
    correlation_method = 'pearson'

    # 最小样本数要求，少于此数量的对将不计算相关系数
    min_samples = 5

    try:
        # 加载数据
        print("正在加载数据...")
        df = load_data(file_path)

        # 查看原始数据的列
        print("\n原始数据列:")
        print(df.columns.tolist())

        # 数据预处理和特征工程（不填充缺失值，移除多类别变量）
        print("\n正在进行预处理和特征工程（不填充缺失值，移除多类别变量）...")
        df_processed = preprocess_data_without_imputation(df)

        # 查看处理后的列
        print("\n处理后的数据列:")
        print(df_processed.columns.tolist())

        # 查看处理后的数据类型
        print("\n处理后的数据类型:")
        print(df_processed.dtypes)

        # 区分特征列和目标列
        feature_cols = [col for col in df_processed.columns if not col.startswith('CD_')]
        target_cols = [col for col in df_processed.columns if col.startswith('CD_')]

        print(f"\n特征列 ({len(feature_cols)}):")
        for feature in feature_cols:
            non_missing = df_processed[feature].count()
            print(f"  {feature}: {non_missing} 个非缺失值 ({non_missing / len(df_processed) * 100:.1f}%)")

        print(f"\n目标列 ({len(target_cols)}):")
        for target in target_cols:
            non_missing = df_processed[target].count()
            print(f"  {target}: {non_missing} 个非缺失值 ({non_missing / len(df_processed) * 100:.1f}%)")

        # 检查是否有全部缺失的列，如果有，则移除
        empty_cols = []
        for col in feature_cols + target_cols:
            if df_processed[col].count() == 0:
                empty_cols.append(col)

        if empty_cols:
            print(f"\n警告: 以下列全部为缺失值，将被移除: {empty_cols}")
            df_processed = df_processed.drop(columns=empty_cols)
            # 更新特征和目标列
            feature_cols = [col for col in df_processed.columns if not col.startswith('CD_')]
            target_cols = [col for col in df_processed.columns if col.startswith('CD_')]
            print(f"更新后的特征列数量: {len(feature_cols)}")
            print(f"更新后的目标列数量: {len(target_cols)}")

        # 计算每对特征-目标变量的有效样本数
        print("\n计算每对特征-目标变量的有效样本数...")
        valid_counts = count_valid_pairs(df_processed, feature_cols, target_cols)

        # 绘制有效样本数热图
        print("\n绘制有效样本数热图...")
        plot_valid_counts_heatmap(valid_counts, output_file_counts)

        # 计算成对相关系数（忽略缺失对）
        print(f"\n计算成对相关系数（使用{correlation_method}方法，忽略缺失对，最小样本数={min_samples}）...")
        correlation_matrix = calculate_pairwise_correlation(df_processed, method=correlation_method,
                                                            min_samples=min_samples)

        # 绘制相关系数热图
        print("\n绘制相关系数热图...")
        correlation_matrix = plot_correlation_heatmap(
            correlation_matrix, feature_cols, target_cols, output_file_corr, method=correlation_method
        )

        # 分析相关性
        print("\n分析特征与目标变量之间的相关性...")
        feature_target_corr = analyze_correlations(correlation_matrix, feature_cols, target_cols)

        print("\n分析完成!")
        return df_processed, correlation_matrix, feature_target_corr, valid_counts

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    df_processed, correlation_matrix, feature_target_corr, valid_counts = main()