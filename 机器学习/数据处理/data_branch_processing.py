import pandas as pd

def split_dataframe_by_delimiter(df, split_columns, base_column='施炭量', delimiter='/'):
    """
    根据分隔符对指定列进行分行处理，使用基准列作为参考

    参数:
    df (pandas.DataFrame): 输入的数据表
    split_columns (list): 需要进行分行处理的列名列表
    base_column (str): 作为基准的列名（默认为'Charcoal_application_ratio'）
    delimiter (str): 数据中使用的分隔符（默认为'/'）

    返回:
    pandas.DataFrame: 处理后的新数据表
    """

    # 创建新行列表用于存储处理后的数据
    new_rows = []

    # 遍历原始数据表中的每一行
    for idx, row in df.iterrows():
        # 获取基准列的值并分割
        base_values = row[base_column].split(delimiter) if pd.notna(row[base_column]) else ['']

        # 创建字典存储每列分割后的值
        split_values = {}
        for col in split_columns:
            if pd.notna(row[col]) and row[col] != '':
                values = row[col].split(delimiter)
                # 如果值的数量少于基准列，用空字符串补充
                if len(values) < len(base_values):
                    values.extend([''] * (len(base_values) - len(values)))
                split_values[col] = values[:len(base_values)]  # 如果超过基准列长度则截断
            else:
                split_values[col] = [''] * len(base_values)

        # 根据分割后的值创建新行
        for i in range(len(base_values)):
            new_row = row.copy()
            # 更新需要分行的列的对应值
            for col in split_columns:
                new_row[col] = split_values[col][i]
            new_rows.append(new_row)

    # 从新行创建新的DataFrame并返回
    return pd.DataFrame(new_rows)


# 读取CSV文件
def read_csv_file(file_path):
    """
    读取CSV文件

    参数:
    file_path (str): CSV文件路径

    返回:
    pandas.DataFrame: 读取的数据表
    """
    try:
        # 使用 'utf-8-sig' 编码和 'ignore' 错误处理来读取文件，避免 BOM 或其他编码问题
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功读取文件：{file_path}")
        return df
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None


# 保存处理后的数据到CSV文件
def save_to_csv(df, output_file_path):
    """
    将处理后的数据保存到CSV文件

    参数:
    df (pandas.DataFrame): 要保存的数据表
    output_file_path (str): 输出文件路径
    """
    try:
        # 使用 'utf-8-sig' 编码保存文件，确保处理好 BOM 标记
        df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"数据已成功保存到：{output_file_path}")
    except Exception as e:
        print(f"保存文件时发生错误：{str(e)}")


# 主程序运行示例
if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\人工老化数据处理.csv"
    output_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据.csv"

    # 需要分行的列名列表
    columns_to_split = ['施炭量','土pH变化','土壤镉变化','可溶性有机碳变化']

    # 读取数据
    input_df = read_csv_file(input_file)

    if input_df is not None:
        # 处理数据
        result_df = split_dataframe_by_delimiter(input_df, columns_to_split)

        # 保存结果
        save_to_csv(result_df, output_file)

        # 输出处理结果统计
        print(f"处理前行数：{len(input_df)}")
        print(f"处理后行数：{len(result_df)}")
