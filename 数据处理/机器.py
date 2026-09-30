import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_explore import AgriDataProcessor


def main():
    # 设置日志级别和格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_processing.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    try:
        # 初始化处理器
        processor = AgriDataProcessor(validation_threshold=0.75)  # 设置填充质量阈值为75%

        # 加载数据
        data_path = Path(r"C:\Users\AMDYE\OneDrive\桌面\data_pre_sheet1.csv")
        if not processor.load_data(data_path):
            logger.error("数据加载失败，程序退出")
            return

        # 数据处理前的可视化
        visualize_missing_data(processor.data, "处理前的缺失值分布")

        # 填充缺失值
        filled_data = processor.fill_missing_values_with_ml(n_neighbors=5)
        if filled_data is None:
            logger.error("缺失值填充失败，程序退出")
            return

        # 评估填充质量
        quality_metrics = processor.evaluate_fill_quality()

        # 输出评估结果
        print_evaluation_results(quality_metrics)

        # 可视化填充质量
        visualize_quality_metrics(quality_metrics)

        # 保存处理后的数据
        output_path = data_path.parent / f"{data_path.stem}_processed{data_path.suffix}"
        processor.save_processed_data(output_path)

        logger.info("数据处理完成！")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")


def visualize_missing_data(data: pd.DataFrame, title: str):
    """可视化数据缺失情况"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png')
    plt.close()


def print_evaluation_results(metrics: dict):
    """打印评估结果"""
    print("\n=== 填充质量评估结果 ===")
    print("-" * 40)
    for column, results in metrics.items():
        print(f"\n列: {column}")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    print("-" * 40)


def visualize_quality_metrics(metrics: dict):
    """可视化填充质量指标"""
    # 准备数据
    columns = []
    scores = []
    metric_types = []

    for column, results in metrics.items():
        for metric, value in results.items():
            columns.append(column)
            scores.append(value)
            metric_types.append(metric)

    # 创建数据框
    df = pd.DataFrame({
        '列名': columns,
        '分数': scores,
        '指标类型': metric_types
    })

    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    sns.barplot(x='列名', y='分数', hue='指标类型', data=df)
    plt.title('填充质量评估')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('quality_metrics.png')
    plt.close()


if __name__ == "__main__":
    main()
