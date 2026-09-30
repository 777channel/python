import pandas as pd

def process_and_save_new_csv(input_csv_file_path, column_name_to_process, output_csv_file_path, new_column_name=None):
    """
    从输入CSV文件读取数据，处理指定列，并将包含新列的结果保存到新的CSV文件。
    会依次尝试 UTF-8 和 GBK 编码进行读取。

    参数:
        input_csv_file_path (str): 输入CSV文件的路径。
        column_name_to_process (str): 需要处理的列的名称 (例如, '土pH变化')。
        output_csv_file_path (str): 写入处理结果的输出CSV文件路径。
        new_column_name (str, optional): 新列的名称。如果未指定，默认为 'Processed_' + 原始列名。
    """
    df = None # 初始化 DataFrame 为 None

    # --- 尝试多种编码读取 CSV ---
    possible_encodings = ['utf-8', 'gbk', 'big5', 'latin1']
    read_successfully = False
    
    for encoding in possible_encodings:
        try:
            print(f"尝试使用 {encoding} 编码读取文件 '{input_csv_file_path}'...")
            df = pd.read_csv(input_csv_file_path, encoding=encoding)
            read_successfully = True
            print(f"成功使用 {encoding} 编码读取文件。")
            break # 成功读取，跳出循环
        except FileNotFoundError:
            print(f"错误: 未找到输入文件 '{input_csv_file_path}'。请检查文件路径和名称。")
            return
        except UnicodeDecodeError:
            print(f"使用 {encoding} 编码读取失败。")
            continue # 尝试下一个编码
        except Exception as e:
            print(f"读取CSV文件时发生未知错误 (使用 {encoding} 编码): {e}")
            return
    
    if not read_successfully:
        print("\n错误: 尝试了所有常见编码 (utf-8, gbk, big5, latin1) 都无法正确读取文件。")
        print("请手动检查CSV文件的实际编码，并在代码中指定正确的编码。")
        return # 无法读取文件，终止函数执行

    # --- 文件读取成功后，继续处理 ---

    if column_name_to_process not in df.columns:
        print(f"错误: 输入CSV文件中未找到列 '{column_name_to_process}'。请检查列名是否正确。")
        print(f"文件中存在的列有: {df.columns.tolist()}")
        return

    # 确定新列的名称
    if new_column_name is None:
        new_column_name = f'Processed_{column_name_to_process}'
    
    processed_results = []

    for index, cell_value in df[column_name_to_process].items():
        if pd.isna(cell_value) or str(cell_value).strip() == '': # 处理空单元格或空白字符串
            processed_results.append('') # 对于空值，在输出中也保留为空
            continue

        try:
            # 将单元格值转换为字符串，确保 split 函数正常工作
            ph_values_str = str(cell_value).strip().split('/')
            
            # 过滤掉空的字符串，以防出现 "1.0//2.0" 这样的情况
            ph_values_str = [val for val in ph_values_str if val] 
            
            if not ph_values_str: # 如果过滤后没有有效值，则跳过
                processed_results.append('')
                continue

            ph_values = [float(val) for val in ph_values_str]

            first_value = ph_values[0]
            # # 计算所有值与第一个值的差值
            # processed_line_values = [val - first_value for val in ph_values]
            new_calculated_values = []
            if first_value == 0: # 避免除以零错误
                # 如果第一个值为0，根据业务逻辑处理，这里选择将该行标记为错误或特殊值
                print(f"警告: 行 {index + 1} 的第一个数据为零，无法进行除法运算。原始值: '{cell_value}'。")
                processed_results.append(f"ERROR: DivideByZero({cell_value})")
                continue # 跳过当前行，处理下一行

            for val in ph_values:
                calculated_value = (first_value - val) / first_value
                new_calculated_values.append(calculated_value)
            
            # 将处理后的值重新格式化为用斜杠分隔的字符串，保留两位小数
            # formatted_line = '/'.join([f"{val:.2f}" for val in processed_line_values])
            formatted_line = '/'.join([f"{val:.3f}" for val in new_calculated_values]) # 注意这里要用新的列表

            processed_results.append(formatted_line)
            # # 将处理后的值重新格式化为用斜杠分隔的字符串，保留两位小数
            # formatted_line = '/'.join([f"{val:.2f}" for val in processed_line_values])
            # processed_results.append(formatted_line)

        except ValueError as e:
            print(f"警告: 无法处理行 {index + 1} (原始值: '{cell_value}')。错误: {e}。将此单元格标记为错误。")
            processed_results.append(f"ERROR: {cell_value}") # 将错误标记在输出中
        except Exception as e:
            print(f"警告: 处理行 {index + 1} 时发生未知错误 (原始值: '{cell_value}')。错误: {e}。将此单元格标记为错误。")
            processed_results.append(f"UNKNOWN_ERROR: {cell_value}")

    # 将新的处理后的数据列添加到 DataFrame
    df[new_column_name] = processed_results

    # 将 DataFrame 保存到新的CSV文件，明确指定 encoding='utf-8'
    try:
        df.to_csv(output_csv_file_path, index=False, encoding='utf-8')
        print(f"数据处理成功！新列 '{new_column_name}' 已添加到 '{output_csv_file_path}'。")
        print(f"请使用支持 UTF-8 编码的软件打开 '{output_csv_file_path}'。")
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

# --- 如何使用代码 ---

# 请替换为您的实际文件名和列名
input_csv_file = r"C:\Users\AMDYE\OneDrive\python\naturliche Aging.csv" # 您的输入CSV文件名
column_to_process = '土壤镉变化'          # 您需要处理的列的准确名称
output_csv_file = 'processed_data1.csv'   # 处理后数据将保存到的新CSV文件名
new_calculated_column_name = 'cd_Difference' # (可选) 您可以指定新列的名称，如果不指定则为 'Processed_土pH变化'

# 调用函数进行处理
process_and_save_new_csv(input_csv_file, column_to_process, output_csv_file, new_column_name=new_calculated_column_name)