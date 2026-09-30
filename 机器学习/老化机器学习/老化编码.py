import pandas as pd
import numpy as np # 仍然需要 numpy 用于 pd.to_numeric 的 NaN
import os
import re # 用于处理老化方式文本，即使在这里不直接用，但考虑其在特征工程中的相关性
import traceback # 导入 traceback 模块

import warnings
warnings.filterwarnings('ignore') # 忽略警告信息

# --- 数据加载函数 ---
def load_and_process_data(file_path):
    """加载数据，对所有列进行数值转换和初步检查。"""
    try:
        # 尝试多种编码读取，确保兼容性
        read_successfully = False
        encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"数据从 '{file_path}' (使用 {enc} 编码) 加载成功。")
                read_successfully = True
                break
            except UnicodeDecodeError:
                continue # 尝试下一个编码
            except Exception as e:
                print(f"尝试使用 {enc} 编码读取文件 '{file_path}' 时发生错误: {e}")
                return None
        
        if not read_successfully:
            print(f"错误: 尝试所有常用编码后仍无法读取文件 '{file_path}'。请检查文件编码或损坏。")
            return None

    except Exception as e:
        print(f"加载数据时发生意外错误: {e}")
        return None
    
    # --- 根据用户要求：仅移除 BOM，不进行列名大小写转换和空格去除 ---
    df.columns = [col.lstrip('\ufeff') for col in df.columns] # 移除BOM
    print("\n已移除列名开头的BOM字符。列名的大小写和空格将保持不变，请确保与代码中的列名精确匹配。")

    # --- 将所有列转换为数值型，无法转换的变为NaN ---
    print("\n尝试将所有列转换为数值型...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")
            
    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    print(f"\n缺失值统计 (所有列):")
    missing_info = df.isnull().sum()
    missing_info_present = missing_info[missing_info > 0]
    if len(missing_info_present) > 0:
        print(missing_info_present)
    else:
        print("没有发现缺失值。")
    
    return df

# --- 老化方式组合编号函数 ---
def assign_aging_combination_id_sequential(df, 
                                           col_dry_wet_days='DW-D', 
                                           col_freeze_thaw_days='FT-D', 
                                           col_cth_30_40_days='C-30%-40%-D', 
                                           col_cth_100_days='C-100%-D',
                                           col_aging_method_text='有效老化', # <--- FIX: 确保这里有这个参数
                                           new_id_col='AgingCombinationID'): # 新的组合ID列名
    """
    根据四种老化方式的组合，为每行数据分配一个唯一的递增序列编号。
    假设每行数据有且仅两种老化方式同时存在（两两组合）或仅一种。
    此函数假定输入的DataFrame列名已保持原始大小写。
    """

    # 检查所有必需的列是否存在（使用传入的原始大小写列名）
    required_cols = [col_dry_wet_days, col_freeze_thaw_days, col_cth_30_40_days, col_cth_100_days]
    for col_name in required_cols: 
        if col_name not in df.columns:
            raise ValueError(f"DataFrame中缺少必需的列: '{col_name}'。请检查列名是否与CSV文件中的实际列名完全匹配（包括大小写和空格）。")

    # 定义所有可能的老化方式名称的内部简化名称（这些是用于COMBINATION_ID_MAP的键）
    DRY_WET_NAME = 'dry_wet'
    FREEZE_THAW_NAME = 'freeze_thaw'
    CTH_30_40_NAME = 'cth_30_40'
    CTH_100_NAME = 'cth_100'

    # 将传入的列名映射到内部的简化名称，便于COMBINATION_ID_MAP使用
    ACTIVE_COL_MAP = {
        col_dry_wet_days: DRY_WET_NAME,
        col_freeze_thaw_days: FREEZE_THAW_NAME,
        col_cth_30_40_days: CTH_30_40_NAME,
        col_cth_100_days: CTH_100_NAME,
    }

    # 定义所有可能的组合及其对应的递增编号
    # 单一方式优先
    COMBINATION_ID_MAP = {
        # 单一方式
        (DRY_WET_NAME,): 1,
        (FREEZE_THAW_NAME,): 2,
        (CTH_30_40_NAME,): 3,
        (CTH_100_NAME,): 4,
        
        # 两两组合 (按字母顺序排列内部元素，确保键的唯一性)
        tuple(sorted([DRY_WET_NAME, FREEZE_THAW_NAME])): 5,
        tuple(sorted([DRY_WET_NAME, CTH_30_40_NAME])): 6,
        tuple(sorted([DRY_WET_NAME, CTH_100_NAME])): 7,
        tuple(sorted([FREEZE_THAW_NAME, CTH_30_40_NAME])): 8,
        tuple(sorted([FREEZE_THAW_NAME, CTH_100_NAME])): 9,
        tuple(sorted([CTH_30_40_NAME, CTH_100_NAME])): 10,
    }
    
    df[new_id_col] = 0 # 初始化新列

    # 检查 col_aging_method_text 是否存在 (用于警告信息)
    _col_aging_method_text_exists = col_aging_method_text in df.columns

    for index, row in df.iterrows():
        active_aging_simplified_names = []
        
        # 检查哪些老化方式有非零天数，即为活跃的老化方式
        for col_df_name, simplified_name in ACTIVE_COL_MAP.items():
            # 确保列存在且不是NaN，并且值大于0
            if col_df_name in row.index and pd.notna(row[col_df_name]) and row[col_df_name] > 0:
                active_aging_simplified_names.append(simplified_name)
        
        # 将活跃方式列表排序并转换为元组，作为字典的键
        combination_key = tuple(sorted(active_aging_simplified_names))

        # 查找对应的组合编号
        if combination_key in COMBINATION_ID_MAP:
            df.loc[index, new_id_col] = COMBINATION_ID_MAP[combination_key]
        else:
            if not active_aging_simplified_names:
                df.loc[index, new_id_col] = 0 # 0 表示没有明确老化方式（可以与对照/未老化关联）
            else:
                df.loc[index, new_id_col] = -1 # -1 表示不符合预期的组合模式（非单阶段/两两组合）
                warning_msg = f"警告: 行 {index} 的老化方式组合 '{combination_key}' 不符合预定义模式或未定义。已分配编号 -1。"
                if _col_aging_method_text_exists:
                    warning_msg += f" 原始文本: '{row[col_aging_method_text]}'"
                print(warning_msg)
                
    return df

# --- 主程序执行部分 ---
if __name__ == "__main__":
    # !!! 请根据您的实际情况修改以下路径和列名 !!!
    
    # 输入文件路径
    input_csv_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"
    
    # 输出文件路径 (可以与输入文件在同一目录，但文件名不同)
    output_directory = os.path.dirname(input_csv_file)
    output_csv_file = os.path.join(output_directory, "naturliche Aging_组合编号_vFinal.csv")

    # 您的四个天数列的精确名称 (请确保与CSV文件中的列名大小写和空格完全匹配)
    col_dw_days_name = 'DW-D'
    col_ft_days_name = 'FT-D'
    col_cth_30_40_days_name = 'C-30%-40%-D'
    col_cth_100_days_name = 'C-100%-D'
    
    # 原始老化方式文本的列名（可选，用于验证输出）
    col_aging_method_text_name = '有效老化' 

    print("--- 开始老化方式组合编号处理 ---")
    
    df_original = load_and_process_data(input_csv_file)

    if df_original is not None:
        # 调用函数分配组合编号
        df_processed_with_id = assign_aging_combination_id_sequential(
            df_original.copy(), # 传入副本以避免直接修改原始DataFrame
            col_dry_wet_days=col_dw_days_name, 
            col_freeze_thaw_days=col_ft_days_name, 
            col_cth_30_40_days=col_cth_30_40_days_name, 
            col_cth_100_days=col_cth_100_days_name,
            col_aging_method_text=col_aging_method_text_name, # 传入原始老化方式文本列名
            new_id_col='AgingCombinationID' # 新生成的编号列名
        )

        # 导出新的CSV文件
        try:
            # 保存时使用 utf-8-sig 编码，确保 Excel 等软件能正确识别中文
            df_processed_with_id.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
            print(f"\n处理后的数据已成功保存到: '{output_csv_file}'")
            
            # 打印部分结果进行验证
            print("\n部分结果验证:")
            display_cols = []
            if col_aging_method_text_name in df_processed_with_id.columns:
                display_cols.append(col_aging_method_text_name)
            display_cols.extend([col_dw_days_name, col_ft_days_name, col_cth_30_40_days_name, col_cth_100_days_name])
            display_cols.append('AgingCombinationID')
            
            # 筛选实际存在的列进行显示，防止因用户未提供的列而报错
            display_cols_actual = [col_name for col_name in display_cols if col_name in df_processed_with_id.columns]
            
            print(df_processed_with_id[display_cols_actual].head(10))

        except Exception as e:
            print(f"保存CSV文件时发生错误: {e}")
            traceback.print_exc()

    else:
        print("数据加载失败，无法执行编号处理。")