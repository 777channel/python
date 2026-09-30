import os
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')


def run_full_pipeline(file_path=r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv"):
    """
    运行完整的填充流程

    参数:
    file_path: 输入数据文件路径
    """
    start_time = time.time()
    print("=" * 60)
    print("启动完整缺失值填充流程")
    print("=" * 60)

    # 1. 数据预处理
    print("\n第一步: 数据预处理")
    print("-" * 60)

    from data_preprocessing import main as preprocess_main
    preprocess_result = preprocess_main()

    # 2. 均值填充
    print("\n第二步: 均值填充")
    print("-" * 60)

    from mean_imputation import mean_imputation_pipeline
    mean_result, mean_stats = mean_imputation_pipeline('preprocessed_data.csv')

    # 3. KNN填充（带优化）
    print("\n第三步: KNN填充（带优化）")
    print("-" * 60)

    from knn_imputation import knn_imputation_with_optimization
    knn_result, knn_stats = knn_imputation_with_optimization('preprocessed_data.csv')

    # 4. 随机森林填充
    print("\n第四步: 随机森林填充")
    print("-" * 60)

    from rf_imputation import rf_imputation_pipeline
    rf_result, rf_stats = rf_imputation_pipeline('preprocessed_data.csv')

    # 5. 插值填充（带优化）
    print("\n第五步: 插值填充（带优化）")
    print("-" * 60)

    from interpolation_imputation import interpolation_pipeline_with_optimization
    interp_result, interp_stats = interpolation_pipeline_with_optimization('preprocessed_data.csv')

    # 6. 比较和评估
    print("\n第六步: 比较和评估填充结果")
    print("-" * 60)

    from comparison_module import main as comparison_main
    comparison_main()

    # 计算总耗时
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print(f"完整流程完成！总用时: {total_time:.2f} 秒")
    print("=" * 60)

    return {
        'preprocessed': preprocess_result,
        'mean': mean_result,
        'knn': knn_result,
        'rf': rf_result,
        'interpolation': interp_result
    }


def run_only_preprocessing(file_path=r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv"):
    """
    仅运行预处理步骤

    参数:
    file_path: 输入数据文件路径
    """
    from data_preprocessing import main as preprocess_main
    print("运行数据预处理...")
    preprocess_result = preprocess_main()
    print("数据预处理完成！")
    return preprocess_result


def run_specific_imputation(method, input_path='preprocessed_data.csv'):
    """
    运行特定的填充方法

    参数:
    method: 填充方法 ('mean', 'knn', 'rf', 'interpolation')
    input_path: 输入数据文件路径

    返回:
    填充结果
    """
    print(f"运行{method}填充方法...")

    if method == 'mean':
        from mean_imputation import mean_imputation_pipeline
        result, stats = mean_imputation_pipeline(input_path)
    elif method == 'knn':
        from knn_imputation import knn_imputation_with_optimization
        result, stats = knn_imputation_with_optimization(input_path)
    elif method == 'rf':
        from rf_imputation import rf_imputation_pipeline
        result, stats = rf_imputation_pipeline(input_path)
    elif method == 'interpolation':
        from interpolation_imputation import interpolation_pipeline_with_optimization
        result, stats = interpolation_pipeline_with_optimization(input_path)
    else:
        raise ValueError(f"不支持的填充方法: {method}")

    print(f"{method}填充完成！")
    return result, stats


def run_comparison_only():
    """
    仅运行比较和评估模块
    """
    from comparison_module import main as comparison_main
    print("运行比较和评估模块...")
    comparison_main()
    print("比较和评估完成！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='缺失值填充流程')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'mean', 'knn', 'rf', 'interpolation', 'compare'],
                        help='运行模式 (默认: full)')
    parser.add_argument('--input', type=str, default=r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv",
                        help='输入数据文件路径 (默认: finall_data.csv)')

    args = parser.parse_args()

    if args.mode == 'full':
        run_full_pipeline(args.input)
    elif args.mode == 'preprocess':
        run_only_preprocessing(args.input)
    elif args.mode in ['mean', 'knn', 'rf', 'interpolation']:
        run_specific_imputation(args.mode, 'preprocessed_data.csv')
    elif args.mode == 'compare':
        run_comparison_only()
    else:
        print(f"错误: 不支持的模式 {args.mode}")