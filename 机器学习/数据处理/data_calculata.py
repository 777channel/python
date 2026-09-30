import pandas as pd
import re
import pandas as pd
# 读取数据，确保编码格式正确
data = pd.read_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/annotated_output.csv', encoding='utf-8')

# 初始化一个新列，用于存储计算后的变化百分比结果
data['CD_brown_rice_change'] = None
data['CD_stem_change'] = None
data['CD_leaf_change'] = None
data['CD_shell_change'] = None
data['CD_root_change'] = None


# 定义一个函数，计算每个数据相对于第一个数据的变化百分比
def calculate_changes(text):
    if pd.isna(text):  # 如果为空值，直接返回 None
        return None

    try:
        # 提取数值部分，忽略非数值部分
        pattern = r'([\d.]+(?:/[\d.]+)*)'
        match = re.search(pattern, text)
        if not match:
            return None

        # 分割提取的数值部分
        numbers = [float(x) for x in match.group(1).split('/')]
        if len(numbers) < 2:  # 如果只有一个值或无值，返回 None
            return None

        # 第一个值
        first_value = numbers[0]

        # 计算变化百分比
        changes = [(x - first_value) / first_value * 100 for x in numbers[1:]]

        # 格式化结果，第一个值保留原始值，其余显示百分比变化
        result = f"{first_value:.2f}/" + '/'.join(f"{change:.2f}%" for change in changes)
        return result
    except Exception as e:
        # 如果出现异常，返回 None
        return None


# 对多列进行处理，计算变化百分比
data['CD_brown_rice_change'] = data['CD_brown'].apply(calculate_changes)
data['CD_stem_change'] = data['CD_stem'].apply(calculate_changes)
data['CD_leaf_change'] = data['CD_leaf'].apply(calculate_changes)
data['CD_shell_change'] = data['CD_shell'].apply(calculate_changes)
data['CD_root_change'] = data['CD_root'].apply(calculate_changes)
# 保存处理后的数据到新的 CSV 文件
data.to_csv('C:/Users/AMDYE/OneDrive/桌面/机器学习/数据/annotated_output1.csv', encoding='utf-8-sig', index=False)

# 打印处理完成的信息
print("处理完成并保存到 'annotated_output1.csv'")
