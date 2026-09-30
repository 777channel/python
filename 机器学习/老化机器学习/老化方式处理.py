import pandas as pd
import re

# --- 常量定义 ---
ROOM_TEMP_IMPUTE = 20.0
DEFAULT_PARAM_NA = -1.0

def parse_single_aging_treatment_string_v3(aging_text_raw):
    parsed_info = {
        '老化方式_含对照或其他': 0,
        '老化方式_含干湿': 0,
        '老化方式_含冻融': 0,
        '老化方式_含恒温恒湿': 0,
        '干湿_天数': 0,
        '冻融_量': 0,
        '恒温恒湿_天数': 0,
        '参数_恒温恒湿_温度': DEFAULT_PARAM_NA,
        '参数_恒温恒湿_水分': DEFAULT_PARAM_NA,
    }

    if pd.isna(aging_text_raw):
        parsed_info['老化方式_含对照或其他'] = 1
        return pd.Series(parsed_info)
    
    aging_text = str(aging_text_raw).strip()
    if not aging_text:
        parsed_info['老化方式_含对照或其他'] = 1
        return pd.Series(parsed_info)

    is_control_or_other_flag = False # Flag to track if CK/未老化 was primary reason
    if "CK" in aging_text.upper() or "未老化" in aging_text:
        parsed_info['老化方式_含对照或其他'] = 1
        is_control_or_other_flag = True # Mark that this is a control/unaged sample

    has_chh = "恒温恒湿" in aging_text
    has_wd = "干湿" in aging_text
    has_ft = "冻融" in aging_text

    days_chh = 0
    days_wd = 0
    days_ft_val = 0

    if has_chh:
        if not is_control_or_other_flag: parsed_info['老化方式_含恒温恒湿'] = 1 # Don't override if it was CK

        days_match_chh = re.search(r'(\d+)\s*(?:天|d|D)', aging_text)
        if days_match_chh:
            parsed_info['恒温恒湿_天数'] = int(days_match_chh.group(1))
        
        # 修正后的水分提取: % 是必需的
        moisture_match = re.search(r'水分\s*(\d+)\s*%', aging_text) or \
                         re.search(r'(\d+)\s*%\s*水分', aging_text)
        if moisture_match:
            val = moisture_match.group(1) # 直接取第一个捕获组
            if val: parsed_info['参数_恒温恒湿_水分'] = float(val)
        
        # 修正后的温度提取
        temp_val_found_numerically = False
        temp_match_numeric = re.search(r'(\d+\.?\d*)\s*(?:C|c|度)', aging_text) or \
                             re.search(r'温度\s*(\d+\.?\d*)', aging_text)
        if temp_match_numeric:
            val = temp_match_numeric.group(1) # 直接取第一个捕获组
            if val:
                parsed_info['参数_恒温恒湿_温度'] = float(val)
                temp_val_found_numerically = True
        
        if not temp_val_found_numerically and "室温" in aging_text:
            parsed_info['参数_恒温恒湿_温度'] = ROOM_TEMP_IMPUTE
        
    if has_wd:
        if not is_control_or_other_flag: parsed_info['老化方式_含干湿'] = 1
        wd_patterns = [
            r'(\d+)\s*(?:天|d|D|次)\s*干湿', r'干湿\s*(\d+)\s*(?:天|d|D|次)',
            r'(\d+)\s*干湿', r'干湿\s*(\d+)'
        ]
        for pat in wd_patterns:
            wd_match = re.search(pat, aging_text)
            if wd_match:
                parsed_info['干湿_天数'] = int(wd_match.group(1))
                break
        if parsed_info['干湿_天数'] == 0 and not has_chh and not has_ft:
             general_days_match = re.search(r'(\d+)\s*(?:天|d|D)', aging_text)
             if general_days_match: parsed_info['干湿_天数'] = int(general_days_match.group(1))

    if has_ft:
        if not is_control_or_other_flag: parsed_info['老化方式_含冻融'] = 1
        ft_patterns = [
            r'(\d+)\s*(?:天|d|D|次)\s*冻融', r'冻融\s*(\d+)\s*(?:天|d|D|次)',
            r'(\d+)\s*冻融', r'冻融\s*(\d+)'
        ]
        for pat in ft_patterns:
            ft_match = re.search(pat, aging_text)
            if ft_match:
                parsed_info['冻融_量'] = int(ft_match.group(1))
                break
        if parsed_info['冻融_量'] == 0 and not has_chh and not has_wd :
             general_days_match = re.search(r'(\d+)\s*(?:天|d|D|次)', aging_text)
             if general_days_match: parsed_info['冻融_量'] = int(general_days_match.group(1))

    if not is_control_or_other_flag and \
       parsed_info['老化方式_含恒温恒湿'] == 0 and \
       parsed_info['老化方式_含干湿'] == 0 and \
       parsed_info['老化方式_含冻融'] == 0:
        parsed_info['老化方式_含对照或其他'] = 1
        # For "other" types that are not CK/unaged, try to get general days if specific days are 0
        # This means day columns specific to WD, FT, CHH would remain 0 if this "other" type is not them.
        # If an "other" type had days, e.g. "自然风干30天", and you want those days captured,
        # you would need an '其他_天数' column or a general '总天数' if not captured by specific types.
        # For now, specific day columns are only populated if their type flag is set.

    return pd.Series(parsed_info)

def engineer_aging_features_from_df_v3(df, aging_column_name='老化处理'):
    if aging_column_name not in df.columns:
        print(f"错误：列 '{aging_column_name}' 在DataFrame中未找到。")
        return df

    df_processed = df.copy()
    
    new_col_names_ordered = [
        '老化方式_含对照或其他', '老化方式_含干湿', '老化方式_含冻融', '老化方式_含恒温恒湿',
        '干湿_天数', '冻融_量', '恒温恒湿_天数',
        '参数_恒温恒湿_温度', '参数_恒温恒湿_水分'
    ]
    
    parsed_features_list = []
    for text_entry in df_processed[aging_column_name]:
        parsed_series = parse_single_aging_treatment_string_v3(text_entry)
        parsed_features_list.append(parsed_series)
    
    parsed_features_df = pd.DataFrame(parsed_features_list, index=df_processed.index)
    
    for col in new_col_names_ordered:
        if col not in parsed_features_df.columns: 
            df_processed[col] = 0 if '天数' in col or '量' in col else (DEFAULT_PARAM_NA if '参数_' in col else 0)
        else:
            df_processed[col] = parsed_features_df[col]
            
    return df_processed

# --- 主程序执行部分 ---
if __name__ == '__main__':
    input_file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据.csv"
    import os
    output_directory = os.path.dirname(input_file_path)
    output_file_name = "分列数据_特征工程后_v4.csv" # Updated version
    output_file_path = os.path.join(output_directory, output_file_name)

    try:
        try:
            df_original = pd.read_csv(input_file_path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                df_original = pd.read_csv(input_file_path, encoding='gb18030')
            except UnicodeDecodeError:
                df_original = pd.read_csv(input_file_path, encoding='utf-8')
        
        print("--- 原始数据 ---")
        print(f"原始数据形状: {df_original.shape}")
        # print(df_original[['老化处理']].head(20))

        df_featured = engineer_aging_features_from_df_v3(df_original, aging_column_name='老化处理')

        print("\n--- 处理后的数据 ---")
        print(f"处理后数据形状: {df_featured.shape}")
        
        created_feature_cols = [
            '老化方式_含对照或其他', '老化方式_含干湿', '老化方式_含冻融', '老化方式_含恒温恒湿',
            '干湿_天数', '冻融_量', '恒温恒湿_天数',
            '参数_恒温恒湿_温度', '参数_恒温恒湿_水分'
        ]
        
        cols_to_show_for_verification = ['老化处理'] + [col for col in created_feature_cols if col in df_featured.columns]
        
        print("\n用于验证的部分列的前10行:")
        print(df_featured[cols_to_show_for_verification].head(10))
        
        # 验证包含“水分”的条目是否正确提取
        print("\n验证包含'水分'的条目:")
        水分相关的条目 = df_featured[df_featured['老化处理'].str.contains("水分", na=False)]
        if not 水分相关的条目.empty:
            print(水分相关的条目[cols_to_show_for_verification].head())
        else:
            print("未找到包含'水分'的条目进行验证。")


        print("\n处理后DataFrame的所有列名 (部分):")
        all_cols = df_featured.columns.tolist()
        if len(all_cols) > 20:
            print(all_cols[:10] + ['...'] + all_cols[-10:])
        else:
            print(all_cols)
        
        if '老化处理' in df_featured.columns:
            print("\n原始 '老化处理' 列已保留.")
        else:
            print("\n错误: 原始 '老化处理' 列丢失了!")

        df_featured.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n处理后的数据已成功保存到: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 '{input_file_path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")