import pandas as pd
import re

# 读取数据
data = pd.read_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data_updated（正常版）.csv', encoding='utf-8')

# 初始化新的列用于存储不同部分的数据
fields = ['brown_rice', 'stem', 'leaf', 'shell', 'root']
for field in fields:
    data[f'CD_{field}'] = None


# 定义一个函数来提取最新的年份/季度数据
def extract_latest_data(text, field_name):
    if pd.isna(text):
        return None

    # 将中文field_name映射到对应的字段名
    field_map = {
        'brown_rice': '糙米',
        'stem': '茎',
        'leaf': '叶',
        'shell': '壳',
        'root': '根'
    }

    chinese_field = field_map[field_name]

    # 匹配模式：查找字段及其后面的内容，直到下一个字段或结束
    pattern = f"{chinese_field}[:：]?(.*?)(?=糙米|茎|叶|壳|根|$)"
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # 获取该字段的最后一个匹配项
    field_text = matches[-1].strip()

    # 提取所有年份/季度的数据
    year_season_pattern = "第[一二三四五六七八九十]年|第[一二三四五六七八九十]季"
    year_season_matches = re.findall(year_season_pattern, field_text)

    if year_season_matches:
        # 如果找到年份/季度，提取最后一组数据
        latest_period = year_season_matches[-1]
        # 获取最后一个年份/季度之后的数据
        final_data_pattern = f"{latest_period}[^第]*"
        final_data_match = re.search(final_data_pattern, field_text)
        if final_data_match:
            return final_data_match.group().strip()

    # 如果没有年份/季度信息，返回整个字段内容
    return field_text.strip()


# 处理每一行数据
for i in range(len(data)):
    text = data.loc[i, 'CD_rice']
    if not pd.isna(text):
        # 对每个字段进行处理
        for field in fields:
            data.loc[i, f'CD_{field}'] = extract_latest_data(text, field)

# 保存更新后的数据到新的 CSV 文件
data.to_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data_updated11.csv', encoding='utf-8-sig', index=False)
print("提取完成并保存到 'data_updated11.csv'")