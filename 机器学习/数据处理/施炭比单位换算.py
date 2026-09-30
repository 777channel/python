import pandas as pd
import re

def process_shitanbi(value):
    """
    处理‘施炭比’列：
    - 百分比数据直接复制。
    - t/ha单位的数据，所有用/分隔的数字均除以1.95。
    - 其他数据保持不变。

    参数:
    value (str): 单元格内容

    返回:
    str: 处理后的结果
    """
    try:
        # 如果为空值，直接返回 None
        if pd.isna(value):
            return None

        # 如果是百分比数据，直接返回
        if "%" in str(value):
            return value

        # 如果是 t/ha 数据
        if "t/ha" in str(value):
            # 提取用 / 分隔的数字部分
            numbers = re.findall(r"\d+\.?\d*", str(value))  # 提取所有数字
            # 将每个数字除以 1950，并保留6位小数，换算成百分比
            processed_numbers = [f"{str(round(float(num) / 1950*100, 2))}%" for num in numbers]
            # 返回处理后的字符串，用 / 分隔，并保留 t/ha 单位
            return "/".join(processed_numbers) + ""
        return value
    except Exception as e:
        # 如果出现异常，返回原始值
        return value

# 读取数据
file_path = 'C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data_updated111.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# 如果存在‘施炭比’列，创建新的‘施炭比_处理后’列
if '施炭比' in df.columns:
    df['施炭比_处理后'] = df['施炭比'].apply(process_shitanbi)

# 保存处理后的数据到新的 CSV 文件
output_path = 'C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data_updated1111.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("处理完成！结果已保存到:", output_path)
