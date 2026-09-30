import pandas as pd

def process_repeated_data_strict(input_csv_file_path, target_column, output_csv_file_path):
    """
    读取CSV文件，严格处理指定列中的重复数据块，将分割数据按行展开并精确对应。
    假设重复行数严格对应分割数据个数。

    参数:
        input_csv_file_path (str): 输入CSV文件的路径。
        target_column (str): 需要处理的列的名称 (例如, '土pH变化')。
        output_csv_file_path (str): 写入处理结果的输出CSV文件路径。
    """
    try:
        df = pd.read_csv(input_csv_file_path, encoding='utf-8-sig')
        print(f"成功使用 UTF-8 with BOM (utf-8-sig) 编码读取文件 '{input_csv_file_path}'。")
    except FileNotFoundError:
        print(f"错误: 未找到输入文件 '{input_csv_file_path}'。请检查文件路径和名称。")
        return
    except UnicodeDecodeError:
        print(f"错误: 无法使用 'utf-8-sig' 编码读取文件 '{input_csv_file_path}'。")
        print("请检查文件编码或尝试其他编码。")
        return
    except Exception as e:
        print(f"读取CSV文件时发生未知错误: {e}")
        return

    # 处理列名中的 BOM
    df.columns = [col.lstrip('\ufeff') for col in df.columns]

    if target_column not in df.columns:
        print(f"错误: CSV文件中未找到列 '{target_column}'。请检查列名是否正确。")
        print(f"文件中存在的列有: {df.columns.tolist()}")
        return

    original_column_data = df[target_column].copy()
    processed_column_data = [None] * len(df) # 预分配列表存储结果

    i = 0
    while i < len(df):
        current_value_str = str(original_column_data.iloc[i]).strip()

        if pd.isna(original_column_data.iloc[i]) or current_value_str == '':
            processed_column_data[i] = '' # 空值处理
            i += 1
            continue

        # 查找重复块的结束位置
        j = i
        while j < len(df) and str(original_column_data.iloc[j]).strip() == current_value_str:
            j += 1
        
        repeat_block_size = j - i # 重复行的数量

        try:
            # 分割当前行的值，并过滤掉空字符串
            split_values = [v.strip() for v in current_value_str.split('/') if v.strip()]
            split_values_count = len(split_values)

            # --- 严格对应逻辑和不匹配警告 ---
            if repeat_block_size != split_values_count:
                print(f"警告: 从行 {i + 1} 开始的重复块数据不完全对应。")
                print(f"  重复行数 (N): {repeat_block_size}，分割数据个数 (M): {split_values_count}。")
                print(f"  原始值: '{current_value_str}'。")
                print("  将根据重复行数进行截取或填充，请检查数据源的规范性。")
                # 对于不匹配的情况，我们仍然需要处理，这里选择：
                # 如果 N > M，超出部分留空或重复最后一个（按您的澄清，这种不该出现，但为了健壮性保留处理）
                # 如果 M > N，多余部分舍弃（按您的澄清，这种不该出现，但为了健壮性保留处理）

            # 遍历重复块中的每一行，进行数据分配
            for k in range(repeat_block_size):
                if k < split_values_count:
                    # 分配对应索引的数据
                    processed_column_data[i + k] = split_values[k]
                else:
                    # 这种情况理论上不应该发生 (N > M)，如果发生，根据实际业务需求处理
                    # 这里选择留空，或者您可以选择重复最后一个有效数据：
                    # processed_column_data[i + k] = split_values[-1] if split_values_count > 0 else ''
                    processed_column_data[i + k] = '' # 保持空白
                    print(f"  行 {i + k + 1} 超出分割数据范围，已设置为空白。")

        except Exception as e:
            print(f"警告: 处理行 {i + 1} 开始的重复块时发生错误: {e}。原始值: '{current_value_str}'。")
            # 遇到错误，可以将整个重复块标记为错误
            for k_err in range(repeat_block_size):
                 processed_column_data[i + k_err] = f"ERROR_Processing({current_value_str})"

        i = j # 跳转到下一个不重复的块开始的位置

    # 将处理后的数据赋值回 DataFrame 的目标列
    df[target_column] = processed_column_data

    # 保存到新的CSV文件
    try:
        df.to_csv(output_csv_file_path, index=False, encoding='utf-8')
        print(f"数据处理成功！处理后的数据已写入 '{output_csv_file_path}'。")
        print(f"请确保您的查看器（如 Excel）也以 UTF-8 编码打开此文件。")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

# --- 如何使用代码 ---
input_csv_file = r'C:\Users\AMDYE\OneDrive\python\naturliche Aging.csv'           # 您的输入CSV文件名
target_column = '土壤镉变化'              # 需要处理的列的名称
output_csv_file = 'processed_strict_expanded_data1.csv' # 处理后数据将保存到的新CSV文件名

process_repeated_data_strict(input_csv_file, target_column, output_csv_file)