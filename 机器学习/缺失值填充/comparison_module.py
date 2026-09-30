import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import os
import warnings

warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman，字号为22
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22


def load_all_imputed_data():
    """加载所有填充结果"""
    print("正在加载所有填充结果...")

    imputed_data = {}
    file_mapping = {
        'mean': 'mean_imputed_data.csv',
        'knn': 'knn_imputed_data_optimized.csv',
        'rf': 'rf_imputed_data.csv',
        'interpolation': 'interpolation_imputed_data_optimized.csv'
    }

    # 加载各个方法的填充结果
    for method, filename in file_mapping.items():
        if os.path.exists(filename):
            imputed_data[method] = pd.read_csv(filename)
            print(f"已加载{method}填充结果")
        elif method == 'knn' and os.path.exists('knn_imputed_data.csv'):
            imputed_data[method] = pd.read_csv('knn_imputed_data.csv')
            print(f"已加载{method}填充结果（非优化版）")
        elif method == 'interpolation' and os.path.exists('interpolation_imputed_data.csv'):
            imputed_data[method] = pd.read_csv('interpolation_imputed_data.csv')
            print(f"已加载{method}填充结果（非优化版）")

    # 加载原始数据
    if os.path.exists('preprocessed_data.csv'):
        imputed_data['original'] = pd.read_csv('preprocessed_data.csv')
        print("已加载原始数据")

    return imputed_data


def create_distribution_comparison(data_dict, column, output_dir='comparison_plots'):
    """
    创建单个列在四种方法下的分布对比图（与示例格式一致）

    参数:
    data_dict: 包含不同填充方法结果的字典
    column: 要比较的列名
    output_dir: 输出目录
    """
    print(f"\n创建{column}的分布对比图...")

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 方法名称和标题映射
    method_names = {
        'mean': 'Mean Imputation',
        'knn': 'KNN Imputation',
        'rf': 'Random Forest Imputation',
        'interpolation': 'Interpolate Imputation'
    }

    method_labels = {
        'mean': 'Mean Filled',
        'knn': 'KNN Filled',
        'rf': 'Random Forest Filled',
        'interpolation': 'Interpolate Filled'
    }

    # 获取四种方法列表
    methods = [m for m in data_dict.keys() if m != 'original']
    if len(methods) < 4:
        print(f"警告: 只找到 {len(methods)} 种填充方法，少于需要的4种")
        return

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    # 获取所有数据的范围来统一x轴
    all_values = []
    for method in methods:
        if column in data_dict[method].columns:
            all_values.extend(data_dict[method][column].dropna().tolist())
    if 'original' in data_dict and column in data_dict['original'].columns:
        all_values.extend(data_dict['original'][column].dropna().tolist())

    # 计算统一的x轴范围
    min_val = min(all_values)
    max_val = max(all_values)
    x_range = [min_val - 0.2 * (max_val - min_val), max_val + 0.2 * (max_val - min_val)]

    # 为每种方法创建子图
    for i, method in enumerate(methods[:4]):
        ax = axes[i]

        # 绘制原始数据分布
        if 'original' in data_dict and column in data_dict['original'].columns:
            sns.histplot(data_dict['original'][column].dropna(), kde=True, color='skyblue',
                         label='Original', alpha=0.7, ax=ax)

        # 绘制填充后的数据分布
        sns.histplot(data_dict[method][column], kde=True, color='gold',
                     label=method_labels[method], alpha=0.7, ax=ax)

        # 设置图表属性
        ax.set_title(method_names[method], fontsize=22)
        ax.set_xlabel('Value', fontsize=22)
        ax.set_ylabel('Frequency', fontsize=22)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(fontsize=22)

        # 统一x轴范围
        ax.set_xlim(x_range)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{column}_distribution_comparison.pdf', dpi=300, bbox_inches='tight')
    print(f"已保存{column}的分布对比图: {output_dir}/{column}_distribution_comparison.pdf")
    plt.close()


def create_all_comparisons(data_dict, output_dir='comparison_plots'):
    """
    为特征列和目标列分别创建分布对比图

    参数:
    data_dict: 包含不同填充方法结果的字典
    output_dir: 输出目录
    """
    # 识别特征列和输出列
    feature_cols = ['pH', 'organicmatter', 'CEC', 'Cdcontent', '温度',
                    'pHsoil', 'Charcoal_application_ratio']

    output_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    # 1. 为一个样例特征列创建分布对比图
    example_feature = None
    for col in feature_cols:
        if col in data_dict['original'].columns and data_dict['original'][col].isnull().sum() > 0:
            example_feature = col
            break

    if example_feature is None:
        example_feature = feature_cols[0]  # 默认使用第一个特征列

    create_distribution_comparison(data_dict, example_feature, output_dir)

    # 2. 为一个样例输出列创建分布对比图
    example_output = output_cols[0]  # 使用第一个输出列
    create_distribution_comparison(data_dict, example_output, output_dir)


def export_rf_results(data_dict, output_path=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\rf_imputed_results(备用）.csv'):
    """
    导出随机森林填充结果为CSV文件

    参数:
    data_dict: 包含不同填充方法结果的字典
    output_path: 输出文件路径
    """
    print("\n导出随机森林填充结果...")

    # 检查是否有随机森林结果
    if 'rf' not in data_dict:
        print("错误: 未找到随机森林填充结果")
        return

    # 导出随机森林结果
    rf_data = data_dict['rf']
    rf_data.to_csv(output_path, index=False)
    print(f"随机森林填充结果已保存到 {output_path}")


def main():
    """主函数"""
    print("开始创建比较图表和导出结果...")

    # 加载所有填充结果
    imputed_data = load_all_imputed_data()

    if len(imputed_data) < 2:
        print("错误: 未找到足够的填充结果进行比较")
        return

    # 创建分布对比图
    create_all_comparisons(imputed_data)

    # 导出随机森林填充结果
    export_rf_results(imputed_data)

    print("比较图表创建和结果导出完成！")


if __name__ == "__main__":
    main()