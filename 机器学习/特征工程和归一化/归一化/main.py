import os
import time
import argparse
import pandas as pd
import numpy as np
import traceback
from data_normalization import full_preprocessing_pipeline
from model_training import train_all_targets


def main(data_path, force_preprocessing=False, output_dir='results'):
    """
    主函数 - 执行完整的数据处理和模型训练流程

    参数:
    data_path: 原始数据CSV文件路径
    force_preprocessing: 是否强制重新执行预处理(即使已存在预处理后的文件)
    output_dir: 输出目录
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置处理后数据文件路径
        processed_data_path = os.path.join(output_dir, 'fully_processed_data.csv')

        # 记录开始时间
        start_time = time.time()

        print(f"{'=' * 70}")
        print(f"开始数据分析和建模流程")
        print(f"{'=' * 70}")
        print(f"数据文件: {data_path}")
        print(f"输出目录: {output_dir}")

        # 步骤1: 数据预处理
        if force_preprocessing or not os.path.exists(processed_data_path):
            print("\n执行数据预处理...")

            # 检查输入文件是否存在
            if not os.path.exists(data_path):
                print(f"错误: 找不到输入文件 {data_path}")
                return

            try:
                processed_df, target_cols = full_preprocessing_pipeline(data_path)

                # 检查预处理结果
                print("\n预处理完成。检查结果...")
                print(f"处理后数据形状: {processed_df.shape}")
                print(f"目标变量列: {target_cols}")

                # 保存处理后的数据
                processed_df.to_csv(processed_data_path, index=False)
                print(f"预处理完成，处理后的数据已保存到 {processed_data_path}")
            except Exception as e:
                print(f"预处理过程中出错: {str(e)}")
                print("详细错误信息:")
                traceback.print_exc()
                return
        else:
            print(f"\n找到已处理的数据文件 {processed_data_path}，跳过预处理步骤")
            try:
                # 加载已处理的数据
                processed_df = pd.read_csv(processed_data_path)
                target_cols = [col for col in processed_df.columns if col.startswith('CD_')]
                print(f"已加载处理后的数据，形状: {processed_df.shape}")
                print(f"目标变量列: {target_cols}")
            except Exception as e:
                print(f"读取处理后的数据文件时出错: {str(e)}")
                print("执行完整预处理...")
                try:
                    processed_df, target_cols = full_preprocessing_pipeline(data_path)
                    processed_df.to_csv(processed_data_path, index=False)
                except Exception as e2:
                    print(f"预处理过程中出错: {str(e2)}")
                    print("详细错误信息:")
                    traceback.print_exc()
                    return

        # 步骤2: 模型训练和评估
        print(f"\n{'=' * 70}")
        print(f"开始模型训练和评估")
        print(f"{'=' * 70}")

        try:
            all_results, best_models = train_all_targets(processed_data_path, output_dir)

            # 汇总结果
            if best_models:
                summary_data = []
                for target, result in best_models.items():
                    if result is not None:
                        summary_data.append({
                            'Target': target,
                            'Best_Model': result['Model'],
                            'R2': result['R2'],
                            'RMSE': result['RMSE'],
                            'MAE': result['MAE'],
                            'CV_R2': result['CV_R2_Mean']
                        })

                if summary_data:
                    summary = pd.DataFrame(summary_data)
                    # 按R²降序排序
                    summary = summary.sort_values('R2', ascending=False)

                    # 打印汇总结果
                    print(f"\n{'=' * 70}")
                    print("模型性能汇总 (按R²排序):")
                    print(f"{'=' * 70}")
                    print(summary)

                    # 保存汇总结果
                    summary_path = os.path.join(output_dir, 'modeling_summary.csv')
                    summary.to_csv(summary_path, index=False)
                    print(f"\n结果汇总已保存至 {summary_path}")
                else:
                    print("\n没有成功的模型训练结果可汇总")
            else:
                print("\n模型训练未能生成有效结果")
        except Exception as e:
            print(f"\n模型训练过程中出错: {str(e)}")
            print("详细错误信息:")
            traceback.print_exc()

        # 计算总运行时间
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)

        print(f"\n总运行时间: {int(hours)}时 {int(minutes)}分 {int(seconds)}秒")
        print(f"\n完整流程已完成! 所有结果文件已保存在 {output_dir} 目录。")

    except Exception as e:
        print(f"运行过程中发生未捕获的错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='数据预处理和模型训练流程')
    parser.add_argument('--data', type=str, default=r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\data_finall version.csv',
                        help='原始数据CSV文件路径 (默认: data_finall version.csv)')
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--force-preprocess', action='store_true',
                        help='强制重新执行预处理 (即使已存在预处理后的文件)')

    args = parser.parse_args()

    main(args.data, args.force_preprocess, args.output)