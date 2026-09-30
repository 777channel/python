import pandas as pd
import re

def annotate_rice_variety(input_file, rice_variety_file, output_file):
    """
    根据水稻品种的高累积和低累积分类，对第一个文件的品种列进行标注，并保留其他列不变。

    参数:
    input_file (str): 输入的主文件路径，包含“品种”列。
    rice_variety_file (str): 包含水稻品种分类的文件路径（高累积和低累积品种）。
    output_file (str): 输出文件路径。

    返回:
    None
    """
    # 读取主文件和水稻品种文件
    main_df = pd.read_csv(input_file, encoding='utf-8')
    rice_variety_df = pd.read_csv(rice_variety_file, encoding='utf-8')

    # 将水稻品种文件分为高累积和低累积两组
    high_accumulation = set(rice_variety_df['高累积水稻品种'].dropna().str.strip())  # 去除空值并去掉多余空格
    low_accumulation = set(rice_variety_df['低累积水稻品种'].dropna().str.strip())

    # 定义一个函数，根据品种名称添加标注
    def annotate_variety(variety):
        if variety in low_accumulation:
            return f"{variety} (低累积)"
        elif variety:
            return f"{variety} (普通品种)"
        else:
            return variety  # 如果没有匹配到，直接返回原值

    # 对主文件的“品种”列进行处理，同时其他列保持不变
    if '品种' in main_df.columns:
        main_df['品种'] = main_df['品种'].apply(annotate_variety)
    else:
        raise ValueError("主文件中未找到“品种”列，请检查文件格式。")

    # 保存处理后的文件，其他列不变
    main_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"处理完成，结果已保存到：{output_file}")

# 示例调用
input_file = 'C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/finall_data(rice).csv'  # 主文件路径
rice_variety_file = 'C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/Rice varieties.csv'  # 水稻品种文件路径
output_file = 'C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/finall_data2.csv'  # 输出文件路径

annotate_rice_variety(input_file, rice_variety_file, output_file)
