import pandas as pd
import numpy as np
import ast
import os
from datetime import datetime

def safe_eval_list(list_str):
    """安全地将字符串转换为列表"""
    if pd.isna(list_str) or list_str == '':
        return []
    
    try:
        # 尝试使用ast.literal_eval安全解析
        if isinstance(list_str, str):
            # 移除可能的多余空格和换行符
            cleaned_str = list_str.strip()
            if cleaned_str.startswith('[') and cleaned_str.endswith(']'):
                return ast.literal_eval(cleaned_str)
            else:
                # 如果不是标准列表格式，尝试按逗号分割
                return [float(x.strip()) for x in cleaned_str.split(',') if x.strip()]
        elif isinstance(list_str, list):
            return list_str
        else:
            return []
    except:
        print(f"警告: 无法解析列表字符串: {list_str}")
        return []

def check_cell_value(cell_value):
    """检查单元格值是否为有效的非空值"""
    try:
        # 处理各种可能的空值情况
        if cell_value is None:
            return False
        if pd.isna(cell_value):
            return False
        if isinstance(cell_value, str) and cell_value.strip() == '':
            return False
        return True
    except:
        return False

def extract_decay_rates_from_csv(input_file_path, output_file_path=None):
    """
    从CSV文件中提取衰减率数据并保存为新的CSV文件
    
    Parameters:
    input_file_path (str): 输入CSV文件路径
    output_file_path (str): 输出CSV文件路径，如果为None则自动生成
    """
    
    print("="*60)
    print("衰减率数据提取工具")
    print("="*60)
    
    # 读取原始数据
    try:
        df = pd.read_csv(input_file_path, encoding='utf-8-sig')
        print(f"成功读取文件: {input_file_path}")
        print(f"数据形状: {df.shape}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return
    
    # 检查必要的列
    required_columns = ['Region', 'Scenario']
    decay_rate_columns = [
        'Cd_Natural_Decay_Rates', 
        'Cd_Artificial_Decay_Rates',
        'pH_Natural_Decay_Rates', 
        'pH_Artificial_Decay_Rates'
    ]
    
    missing_required = [col for col in required_columns if col not in df.columns]
    missing_decay = [col for col in decay_rate_columns if col not in df.columns]
    
    if missing_required:
        print(f"错误: 缺少必要列: {missing_required}")
        return
    
    if missing_decay:
        print(f"警告: 缺少衰减率列: {missing_decay}")
    
    available_decay_columns = [col for col in decay_rate_columns if col in df.columns]
    print(f"找到的衰减率列: {available_decay_columns}")
    
    # 过滤出包含衰减率数据的行（通常是VALIDATION_SUMMARY行）
    summary_rows = df[df['Year'] == 'VALIDATION_SUMMARY'].copy()
    
    if summary_rows.empty:
        print("警告: 未找到VALIDATION_SUMMARY行，尝试查找包含衰减率数据的其他行...")
        # 查找任何包含衰减率数据的行
        has_decay_data = pd.Series([False] * len(df))
        for col in available_decay_columns:
            has_decay_data = has_decay_data | df[col].notna()
        summary_rows = df[has_decay_data].copy()
    
    if summary_rows.empty:
        print("错误: 未找到任何包含衰减率数据的行")
        return
    
    print(f"找到 {len(summary_rows)} 行包含衰减率数据")
    
    # 提取和处理衰减率数据
    extracted_data = []
    
    for idx, row in summary_rows.iterrows():
        base_info = {
            'Region': row['Region'],
            'Scenario': row['Scenario'],
            'Combination_ID': f"{row['Region']}_{row['Scenario'].replace('=', '').replace(',', '_').replace(' ', '')}"
        }
        
        # 处理每个衰减率列
        for decay_col in available_decay_columns:
            if decay_col in row:
                cell_value = row[decay_col]
                # 使用新的检查函数
                if check_cell_value(cell_value):
                    # 将字符串列表转换为实际列表
                    decay_list = safe_eval_list(cell_value)
                    
                    if decay_list:  # 如果列表不为空
                        base_info[decay_col] = decay_list
                        base_info[f'{decay_col}_Length'] = len(decay_list)
                        base_info[f'{decay_col}_Mean'] = round(np.mean(decay_list), 6)
                        base_info[f'{decay_col}_Std'] = round(np.std(decay_list), 6)
                        base_info[f'{decay_col}_Min'] = round(min(decay_list), 6)
                        base_info[f'{decay_col}_Max'] = round(max(decay_list), 6)
                    else:
                        base_info[decay_col] = []
                        base_info[f'{decay_col}_Length'] = 0
                        base_info[f'{decay_col}_Mean'] = np.nan
                        base_info[f'{decay_col}_Std'] = np.nan
                        base_info[f'{decay_col}_Min'] = np.nan
                        base_info[f'{decay_col}_Max'] = np.nan
                else:
                    base_info[decay_col] = []
                    base_info[f'{decay_col}_Length'] = 0
                    base_info[f'{decay_col}_Mean'] = np.nan
                    base_info[f'{decay_col}_Std'] = np.nan
                    base_info[f'{decay_col}_Min'] = np.nan
                    base_info[f'{decay_col}_Max'] = np.nan
            else:
                base_info[decay_col] = []
                base_info[f'{decay_col}_Length'] = 0
                base_info[f'{decay_col}_Mean'] = np.nan
                base_info[f'{decay_col}_Std'] = np.nan
                base_info[f'{decay_col}_Min'] = np.nan
                base_info[f'{decay_col}_Max'] = np.nan
        
        extracted_data.append(base_info)
    
    # 创建DataFrame
    result_df = pd.DataFrame(extracted_data)
    
    # 生成输出文件名
    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"decay_rates_extracted_{timestamp}.csv"
    
    # 保存结果
    try:
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n衰减率数据已成功提取并保存至: {output_file_path}")
        print(f"输出数据形状: {result_df.shape}")
    except Exception as e:
        print(f"保存文件失败: {e}")
        return
    
    # 显示数据摘要
    print("\n" + "="*60)
    print("数据摘要")
    print("="*60)
    
    print(f"\n提取的组合数量: {len(result_df)}")
    print(f"\n区域分布:")
    print(result_df['Region'].value_counts().to_string())
    
    print(f"\n衰减率数据统计:")
    for decay_col in available_decay_columns:
        length_col = f'{decay_col}_Length'
        if length_col in result_df.columns:
            valid_count = (result_df[length_col] > 0).sum()
            total_count = len(result_df)
            print(f"  {decay_col}: {valid_count}/{total_count} 个组合有数据")
            
            if valid_count > 0:
                mean_col = f'{decay_col}_Mean'
                if mean_col in result_df.columns:
                    valid_means = result_df[result_df[length_col] > 0][mean_col]
                    print(f"    平均衰减率范围: {valid_means.min():.4f} 到 {valid_means.max():.4f}")
    
    # 显示前几行数据作为示例
    print(f"\n前3行数据示例:")
    print("-"*60)
    display_columns = ['Region', 'Scenario', 'Combination_ID'] + [col for col in available_decay_columns]
    display_df = result_df[display_columns].head(3)
    for idx, row in display_df.iterrows():
        print(f"\n第 {idx+1} 行:")
        print(f"  区域: {row['Region']}")
        print(f"  情景: {row['Scenario']}")
        print(f"  组合ID: {row['Combination_ID']}")
        for decay_col in available_decay_columns:
            if decay_col in row and check_cell_value(row[decay_col]):
                decay_data = row[decay_col] if isinstance(row[decay_col], list) else safe_eval_list(str(row[decay_col]))
                if decay_data:
                    print(f"  {decay_col}: {[round(x, 4) for x in decay_data]}")
    
    return result_df

def create_detailed_decay_analysis(input_file_path, output_file_path=None):
    """
    创建详细的衰减率分析，包括年度展开数据
    """
    
    print("="*60)
    print("详细衰减率分析工具")
    print("="*60)
    
    # 首先提取基础数据
    basic_df = extract_decay_rates_from_csv(input_file_path, None)
    if basic_df is None:
        return
    
    # 创建详细分析数据
    detailed_data = []
    
    decay_rate_columns = [
        'Cd_Natural_Decay_Rates', 
        'Cd_Artificial_Decay_Rates',
        'pH_Natural_Decay_Rates', 
        'pH_Artificial_Decay_Rates'
    ]
    
    for idx, row in basic_df.iterrows():
        base_info = {
            'Region': row['Region'],
            'Scenario': row['Scenario'],
            'Combination_ID': row['Combination_ID']
        }
        
        # 获取所有衰减率数据的最大长度
        max_length = 0
        decay_data = {}
        
        for decay_col in decay_rate_columns:
            if decay_col in row:
                cell_value = row[decay_col]
                # 使用新的检查函数
                if check_cell_value(cell_value):
                    decay_list = cell_value if isinstance(cell_value, list) else safe_eval_list(str(cell_value))
                    decay_data[decay_col] = decay_list
                    max_length = max(max_length, len(decay_list))
                else:
                    decay_data[decay_col] = []
            else:
                decay_data[decay_col] = []
        
        # 按年度展开数据
        if max_length > 0:
            for year_idx in range(max_length):
                year_data = base_info.copy()
                year_data['Year_Index'] = year_idx + 1  # 年份从1开始
                year_data['Year_Transition'] = f"Year_{year_idx+1}_to_Year_{year_idx+2}"
                
                for decay_col in decay_rate_columns:
                    if year_idx < len(decay_data[decay_col]):
                        year_data[decay_col] = decay_data[decay_col][year_idx]
                    else:
                        year_data[decay_col] = np.nan
                
                detailed_data.append(year_data)
        else:
            # 如果没有衰减率数据，也保留基础信息
            no_data = base_info.copy()
            no_data['Year_Index'] = np.nan
            no_data['Year_Transition'] = 'No_Data'
            for decay_col in decay_rate_columns:
                no_data[decay_col] = np.nan
            detailed_data.append(no_data)
    
    # 创建详细分析DataFrame
    detailed_df = pd.DataFrame(detailed_data)
    
    # 生成输出文件名
    if output_file_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"decay_rates_detailed_analysis_{timestamp}.csv"
    
    # 保存详细分析结果
    try:
        detailed_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n详细衰减率分析已保存至: {output_file_path}")
        print(f"详细数据形状: {detailed_df.shape}")
    except Exception as e:
        print(f"保存详细分析文件失败: {e}")
        return
    
    return detailed_df

# 主程序
if __name__ == "__main__":
    # 配置文件路径
    input_csv_file = "dual_target_bayesian_optimization_with_decay_validation.csv"
    
    print("衰减率数据提取和分析工具")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_csv_file):
        print(f"错误: 找不到输入文件 {input_csv_file}")
        print("请确保文件路径正确")
        exit()
    
    print("请选择操作:")
    print("1. 基础衰减率提取")
    print("2. 详细衰减率分析（按年度展开）")
    print("3. 两者都执行")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n执行基础衰减率提取...")
        basic_result = extract_decay_rates_from_csv(
            input_file_path=input_csv_file,
            output_file_path="decay_rates_basic_extract.csv"
        )
    
    if choice in ['2', '3']:
        print("\n执行详细衰减率分析...")
        detailed_result = create_detailed_decay_analysis(
            input_file_path=input_csv_file,
            output_file_path="decay_rates_detailed_analysis.csv"
        )
    
    if choice not in ['1', '2', '3']:
        print("无效选择，退出程序")
    else:
        print("\n" + "="*60)
        print("处理完成!")
        print("="*60)