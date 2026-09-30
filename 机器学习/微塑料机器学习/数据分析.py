import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def analyze_data_statistics(file_path, columns_to_analyze):
    """
    加载CSV文件，使用KNN填充缺失值，并输出指定格式的统计表格。
    """
    try:
        # 1. 加载数据
        df = pd.read_csv(file_path)
        
        # 2. 清理列名：保持与原有逻辑一致
        df.columns = df.columns.str.replace('.', '-', regex=False)
        
        # 检查指定的列是否存在
        existing_cols = [col for col in columns_to_analyze if col in df.columns]
        missing_cols = list(set(columns_to_analyze) - set(existing_cols))
        
        if missing_cols:
            print(f"警告: 以下列在文件中未找到: {missing_cols}")
        
        if not existing_cols:
            print("错误: 没有找到任何待分析的列。")
            return

        # 3. KNN 填充缺失值
        # 仅对需要分析的列进行提取和填充，避免不相关列的影响
        data_to_impute = df[existing_cols]
        
        print(f"正在使用 KNN 填充缺失值 (n_neighbors=5)...")
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(data_to_impute)
        
        # 将填充后的数据转回 DataFrame
        df_filled = pd.DataFrame(imputed_data, columns=existing_cols)

        # 4. 数据统计
        stats_list = []
        for col in existing_cols:
            series = df_filled[col]
            stats_list.append({
                '变量': col,
                '计数': int(series.count()),
                '最小值': series.min(),
                '最大值': series.max(),
                '均值': series.mean(),
                '中位数': series.median(),
                '标准差': series.std()
            })
        
        # 5. 构建结果表格
        results_df = pd.DataFrame(stats_list)
        
        # 设置数值显示精度（保留 2 到 4 位小数，根据您的图片习惯调整）
        pd.options.display.float_format = '{:.2f}'.format

        print("\n" + "="*80)
        print("数据统计分析报告 (KNN填充后)")
        print("="*80)
        
        # 打印表格
        # 使用 to_string 对齐列名，或者直接 print(results_df)
        print(results_df.to_string(index=False, justify='center'))
        print("="*80)

        # 可选：保存到本地
        # results_df.to_csv("统计结果.csv", index=False, encoding='utf-8-sig')

    except FileNotFoundError:
        print(f"错误: 无法找到文件 '{file_path}'。")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

# --- 用户配置 ---
FILE_PATH = "C:/Users/AMDYE/OneDrive/桌面/微塑料机器学习/因果推断.csv"

# 您定义的列名列表
COLUMNS = [
    'pH-S', 'OM', 'CEC', 'Cd-C', 'BET', 'T', 
    'pH-B', 'BR', 'Size', 'MPR', 'ΔpH', 'ΔCd'
]

# --- 运行分析 ---
if __name__ == "__main__":
    analyze_data_statistics(FILE_PATH, COLUMNS)