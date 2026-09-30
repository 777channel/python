import pandas as pd
import re
import os

# --- 常量定义 ---
ROOM_TEMP_IMPUTE = 20.0
DEFAULT_PARAM_NA = -1.0 # 默认的缺失参数值

# --- 全局常量，用于控制恒温恒湿的最大阶段数 ---
# 根据您的说明：'+' 号最多一个。
# 这意味着最多只有两个老化阶段。
# 如果两个阶段都是恒温恒湿，那么最多就是2个恒温恒湿阶段。
MAX_CTH_PHASES = 2 # <-- 确认此值，因为'+'最多一个，所以最多两段恒温恒湿

def parse_single_aging_treatment_string_v5(aging_text_raw): # 更新函数版本号
    """
    解析单条老化处理字符串，识别不同老化方式及其参数，
    并为多阶段恒温恒湿创建独立的列。
    此版本假定字符串中 '+' 号最多出现一次。
    """
    # 基础的默认解析结果，包含所有可能的列
    parsed_info = {
        '老化方式_含对照或其他': 0,
        '老化方式_含干湿': 0,
        '老化方式_含冻融': 0,
        '干湿_天数': 0,
        '冻融_量': 0,
    }
    # 为每个可能的恒温恒湿阶段预设列，并初始化为默认值
    for i in range(MAX_CTH_PHASES):
        parsed_info[f'恒温恒湿_天数_Phase{i+1}'] = 0
        parsed_info[f'参数_恒温恒湿_温度_Phase{i+1}'] = DEFAULT_PARAM_NA
        parsed_info[f'参数_恒温恒湿_水分_Phase{i+1}'] = DEFAULT_PARAM_NA
    parsed_info['老化方式_含恒温恒湿'] = 0 # 总体标志

    if pd.isna(aging_text_raw) or not str(aging_text_raw).strip():
        parsed_info['老化方式_含对照或其他'] = 1
        return pd.Series(parsed_info)
    
    aging_text = str(aging_text_raw).strip()

    is_control_or_other_flag = False
    # CK / 未老化 / None / 无 / 不处理 等同于对照或其他
    if "CK" in aging_text.upper() or "未老化" in aging_text or \
       aging_text.upper() == 'NONE' or aging_text == '无' or aging_text == '不处理':
        parsed_info['老化方式_含对照或其他'] = 1
        is_control_or_other_flag = True

    # ---- 识别并处理老化处理字符串：按 '+' 分割成最多两个阶段 ----
    aging_stages = [s.strip() for s in aging_text.split('+') if s.strip()]

    cth_current_phase_idx = 0 # 用于跟踪当前正在处理的恒温恒湿阶段的索引 (0-based)
    overall_has_cth = False # 标记是否包含任何恒温恒湿阶段

    # 遍历每个解析出的阶段字符串（最多两个阶段）
    for stage_text in aging_stages:
        has_cth_in_stage = "恒温恒湿" in stage_text
        has_wd_in_stage = "干湿" in stage_text
        has_ft_in_stage = "冻融" in stage_text

        # 恒温恒湿阶段处理
        if has_cth_in_stage:
            overall_has_cth = True # 标记总体包含恒温恒湿

            # 由于 MAX_CTH_PHASES 已经设置为 2，且 '+' 最多一个，所以这里不会越界
            current_phase_num = cth_current_phase_idx + 1 # 从1开始计数 Phase1, Phase2...
            
            # 提取恒温恒湿相关参数
            days_match_cth = re.search(r'(\d+)\s*(?:天|d|D)', stage_text)
            moisture_match = re.search(r'水分\s*(\d+)\s*%', stage_text) or \
                             re.search(r'(\d+)\s*%\s*水分', stage_text)
            temp_match_numeric = re.search(r'(\d+\.?\d*)\s*(?:C|c|度)', stage_text) or \
                                 re.search(r'温度\s*(\d+\.?\d*)', stage_text)

            if days_match_cth:
                parsed_info[f'恒温恒湿_天数_Phase{current_phase_num}'] = int(days_match_cth.group(1))
            if moisture_match:
                parsed_info[f'参数_恒温恒湿_水分_Phase{current_phase_num}'] = float(moisture_match.group(1))
            
            temp_val_found_numerically = False
            if temp_match_numeric:
                parsed_info[f'参数_恒温恒湿_温度_Phase{current_phase_num}'] = float(temp_match_numeric.group(1))
                temp_val_found_numerically = True
            
            if not temp_val_found_numerically and ("室温" in stage_text or "RT" in stage_text.upper()):
                parsed_info[f'参数_恒温恒湿_温度_Phase{current_phase_num}'] = ROOM_TEMP_IMPUTE
            
            cth_current_phase_idx += 1 # 恒温恒湿阶段计数增加

        # 干湿循环处理 (假设一个字符串中只有一个干湿循环或只关注第一个，且DW/FT不分阶段)
        if has_wd_in_stage and parsed_info['老化方式_含干湿'] == 0: 
            parsed_info['老化方式_含干湿'] = 1
            wd_patterns = [
                r'(\d+)\s*(?:天|d|D|次)\s*干湿', r'干湿\s*(\d+)\s*(?:天|d|D|次)',
                r'(\d+)\s*干湿', r'干湿\s*(\d+)'
            ]
            for pat in wd_patterns:
                wd_match = re.search(pat, stage_text)
                if wd_match:
                    parsed_info['干湿_天数'] = int(wd_match.group(1))
                    break
            if parsed_info['干湿_天数'] == 0:
                 general_days_match = re.search(r'(\d+)\s*(?:天|d|D)', stage_text)
                 if general_days_match: parsed_info['干湿_天数'] = int(general_days_match.group(1))


        # 冻融循环处理 (假设一个字符串中只有一个冻融循环或只关注第一个，且DW/FT不分阶段)
        if has_ft_in_stage and parsed_info['老化方式_含冻融'] == 0: 
            parsed_info['老化方式_含冻融'] = 1
            ft_patterns = [
                r'(\d+)\s*(?:天|d|D|次)\s*冻融', r'冻融\s*(\d+)\s*(?:天|d|D|次)',
                r'(\d+)\s*冻融', r'冻融\s*(\d+)'
            ]
            for pat in ft_patterns:
                ft_match = re.search(pat, stage_text)
                if ft_match:
                    parsed_info['冻融_量'] = int(ft_match.group(1))
                    break
            if parsed_info['冻融_量'] == 0:
                general_days_match = re.search(r'(\d+)\s*(?:天|d|D|次)', stage_text)
                if general_days_match: parsed_info['冻融_量'] = int(general_days_match.group(1))

    # 更新总体恒温恒湿标志 (只有当确实解析到恒温恒湿阶段且不是对照时才设置)
    if overall_has_cth and not is_control_or_other_flag:
        parsed_info['老化方式_含恒温恒湿'] = 1
    
    # 最终检查，如果没有任何特定的老化方式且不是对照，则标记为对照或其他
    if not is_control_or_other_flag and \
       parsed_info['老化方式_含恒温恒湿'] == 0 and \
       parsed_info['老化方式_含干湿'] == 0 and \
       parsed_info['老化方式_含冻融'] == 0:
        parsed_info['老化方式_含对照或其他'] = 1

    return pd.Series(parsed_info)

def engineer_aging_features_from_df_v5(df, aging_column_name='有效老化'): # 更新函数版本号
    """
    对DataFrame进行特征工程，解析老化方式列并生成新的特征列。
    此版本假定字符串中 '+' 号最多出现一次。
    """
    if aging_column_name not in df.columns:
        print(f"错误：列 '{aging_column_name}' 在DataFrame中未找到。")
        return df

    df_processed = df.copy()
    
    # 构建所有可能的新列名 (包括恒温恒湿的阶段性列)
    new_col_names_ordered = [
        '老化方式_含对照或其他', '老化方式_含干湿', '老化方式_含冻融',
        '干湿_天数', '冻融_量' # 这些是非阶段性的
    ]
    for i in range(MAX_CTH_PHASES):
        new_col_names_ordered.extend([
            f'恒温恒湿_天数_Phase{i+1}',
            f'参数_恒温恒湿_温度_Phase{i+1}',
            f'参数_恒温恒湿_水分_Phase{i+1}'
        ])
    new_col_names_ordered.append('老化方式_含恒温恒湿') # 总体恒温恒湿标志放在最后

    parsed_features_list = []
    # 调用新的 v5 解析函数
    for text_entry in df_processed[aging_column_name]:
        parsed_series = parse_single_aging_treatment_string_v5(text_entry)
        parsed_features_list.append(parsed_series)
    
    parsed_features_df = pd.DataFrame(parsed_features_list, index=df_processed.index)
    
    # 将解析出的特征合并到原始DataFrame
    for col in new_col_names_ordered:
        if col not in parsed_features_df.columns:
            # 确保即使解析函数没有返回的列也能被正确初始化
            if '天数' in col or '量' in col:
                df_processed[col] = 0
            elif '参数_' in col:
                df_processed[col] = DEFAULT_PARAM_NA
            else: # 老化方式_含X这种布尔标志
                df_processed[col] = 0 
        else:
            df_processed[col] = parsed_features_df[col]
            
    return df_processed

# --- 主程序执行部分 ---
if __name__ == '__main__':
    # !!! 根据您的要求更新文件路径和列名 !!!
    input_file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（删除ck）.csv"
    aging_column_to_process = '有效老化' # !!! 需要处理的列名 !!!
    
    output_directory = os.path.dirname(input_file_path)
    output_file_name = "naturliche Aging（删除ck）_特征工程_阶段展开_v2.csv" # 建议修改输出文件名
    output_file_path = os.path.join(output_directory, output_file_name)

    try:
        # 尝试多种编码读取，确保兼容性
        read_successfully = False
        encodings_to_try = ['gbk', 'gb18030', 'utf-8-sig', 'utf-8', 'latin1'] 
        
        df_original = None 
        for enc in encodings_to_try:
            try:
                df_original = pd.read_csv(input_file_path, encoding=enc)
                print(f"成功使用 {enc} 编码读取文件。")
                read_successfully = True
                break
            except UnicodeDecodeError:
                print(f"使用 {enc} 编码读取失败，尝试下一个。")
            except FileNotFoundError:
                print(f"错误: 文件 '{input_file_path}' 未找到。请确保文件路径正确。")
                exit() 
            except Exception as e:
                print(f"读取CSV文件时发生未知错误 ({enc}): {e}")
                exit()
        
        if not read_successfully:
            print("错误: 尝试所有常用编码后仍无法读取文件。请检查文件编码或损坏。")
            exit()
            
        # 处理原始DataFrame的列名，去除BOM
        df_original.columns = [col.lstrip('\ufeff') for col in df_original.columns]

        print("--- 原始数据预览 ---")
        print(f"原始数据形状: {df_original.shape}")
        print(f"原始 '{aging_column_to_process}' 列示例:")
        if aging_column_to_process in df_original.columns:
            print(df_original[aging_column_to_process].head(10))
        else:
            print(f"错误: 原始DataFrame中未找到列 '{aging_column_to_process}'。")
            exit()


        # 调用特征工程函数，传入正确的列名
        df_featured = engineer_aging_features_from_df_v5(df_original, aging_column_name=aging_column_to_process)

        print("\n--- 处理后的数据预览 ---")
        print(f"处理后数据形状: {df_featured.shape}")
        
        # 动态生成要显示的列名
        cols_to_show_for_verification = [aging_column_to_process] + [
            '老化方式_含对照或其他', '老化方式_含干湿', '老化方式_含冻融', '干湿_天数', '冻融_量'
        ]
        for i in range(MAX_CTH_PHASES):
            cols_to_show_for_verification.extend([
                f'恒温恒湿_天数_Phase{i+1}',
                f'参数_恒温恒湿_温度_Phase{i+1}',
                f'参数_恒温恒湿_水分_Phase{i+1}'
            ])
        cols_to_show_for_verification.append('老化方式_含恒温恒湿')
        
        # 筛选出实际存在于 df_featured 中的列
        cols_to_show_for_verification_actual = [col for col in cols_to_show_for_verification if col in df_featured.columns]

        print("\n用于验证的部分列的前10行:")
        print(df_featured[cols_to_show_for_verification_actual].head(10))
        
        # 验证包含“恒温恒湿”或“+”的复杂条目是否正确提取
        print(f"\n验证包含'恒温恒湿'或'+'的复杂条目 (来自 '{aging_column_to_process}' 列，前10行):")
        complex_entries = df_featured[
            (df_featured[aging_column_to_process].astype(str).str.contains("恒温恒湿", na=False)) | 
            (df_featured[aging_column_to_process].astype(str).str.contains(r"\+", na=False, regex=True)) 
        ]
        if not complex_entries.empty:
            print(complex_entries[cols_to_show_for_verification_actual].head(10)) 
        else:
            print(f"未找到包含'恒温恒湿'或'+'的复杂条目在 '{aging_column_to_process}' 列中进行验证。")

        print("\n处理后DataFrame的所有列名 (部分):")
        all_cols = df_featured.columns.tolist()
        if len(all_cols) > 30: 
            print(all_cols[:15] + ['...'] + all_cols[-15:])
        else:
            print(all_cols)
        
        # 确保原始老化处理列被保留
        if aging_column_to_process in df_featured.columns:
            print(f"\n原始 '{aging_column_to_process}' 列已保留.")
        else:
            print(f"\n错误: 原始 '{aging_column_to_process}' 列丢失了!")

        # 确保保存时使用 utf-8-sig
        df_featured.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n处理后的数据已成功保存到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_file_path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")