import pandas as pd
import re

data = pd.read_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data.csv', encoding='utf-8')

data['CD_brown_rice'] = None  # 糙米列
data['CD_stem'] = None        # 茎列
data['CD_leaf'] = None        # 叶列
data['CD_shell'] = None       # 壳列
data['CD_root'] = None        # 根列

for i in range(len(data)):
    text = data.loc[i, 'CD_rice']  # 提取 CD_rice 列中的值

    if pd.isna(text):
        continue

    if "糙米" in text:
        pattern = r"(糙米[:：]?.*?)(?=茎|叶|壳|根|$)"
        matches = re.findall(pattern, text)
        if matches:
            data.loc[i, 'CD_brown_rice'] = "; ".join(matches)  # 以分号分隔提取的内容


    if "茎" in text:
        pattern = r"(茎[:：]?.*?)(?=糙米|叶|壳|根|$)"
        matches = re.findall(pattern, text)
        if matches:
            data.loc[i, 'CD_stem'] = "; ".join(matches)  # 以分号分隔提取的内容

    if "叶" in text:
        pattern = r"(叶[:：]?.*?)(?=糙米|茎|壳|根|$)"
        matches = re.findall(pattern, text)
        if matches:
            data.loc[i, 'CD_leaf'] = "; ".join(matches)  # 以分号分隔提取的内容

    if "壳" in text:
        pattern = r"(壳[:：]?.*?)(?=糙米|茎|叶|根|$)"
        matches = re.findall(pattern, text)
        if matches:
            data.loc[i, 'CD_shell'] = "; ".join(matches)  # 以分号分隔提取的内容

    if "根" in text:
        pattern = r"(根[:：]?.*?)(?=糙米|茎|叶|壳|$)"
        matches = re.findall(pattern, text)
        if matches:
            data.loc[i, 'CD_root'] = "; ".join(matches)  # 以分号分隔提取的内容

data.to_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/data_updated.csv', encoding='utf-8-sig', index=False)
print("提取完成并保存到 'data_updated.csv'")
