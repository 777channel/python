import pandas as pd
import io
import numpy as np # 虽然此代码可能不直接用numpy，但保留它以防万一
import os # 用于检查文件路径

# --- 配置 ---
# 使用原始字符串 (r"...") 处理 Windows 路径中的反斜杠
csv_file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\生物炭结果\3% result.csv"
analysis_threshold = 3 # 只详细分析出现次数 >= 3 次的原料

# --- 数据加载与预处理 ---
try:
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"错误：找不到文件 '{csv_file_path}'")

    # 读取 CSV 文件
    df = pd.read_csv(csv_file_path, skiprows=1, header=0, encoding='utf-8')
    print(f"成功从 '{csv_file_path}' 加载数据。")

    # 清理列名（去除首尾空格）
    df.columns = df.columns.str.strip()
    print("数据列名已清理。")
    # print("清理后的列名:", list(df.columns)) # 取消注释以查看列名

    # 检查必需的列是否存在
    required_cols = ['土壤pH', '初始镉浓度(mg/kg)', '生物炭原料类型', '生物炭热解温度(°C)', '灌溉方式']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV 文件缺少以下必需的列: {missing_cols}")
    print("必需列检查通过。")

    # 定义温度区间函数
    def categorize_temp(temp):
        if pd.isna(temp): # 处理可能的 NaN 值
            return 'Unknown'
        if temp <= 400:
            return 'Low (≤400°C)'
        elif temp <= 550:
            return 'Mid (401-550°C)'
        else: # > 550
            return 'High (>550°C)'

    # 应用温度区间分类
    # 先确保温度列是数值类型，强制转换无效值为 NaN
    df['生物炭热解温度(°C)'] = pd.to_numeric(df['生物炭热解温度(°C)'], errors='coerce')
    df['温度区间'] = df['生物炭热解温度(°C)'].apply(categorize_temp)
    print("温度区间分类已应用。")


    # 分离数据
    df_awd = df[df['灌溉方式'] == '干湿交替'].copy()
    df_cf = df[df['灌溉方式'] == '持续淹水'].copy()
    print(f"干湿交替数据: {len(df_awd)} 条")
    print(f"持续淹水数据: {len(df_cf)} 条")


    # --- 定义辅助函数生成分析文本 ---
    def generate_detailed_analysis_text(df_subset, condition_col_name, condition_val):
        """
        为特定条件下的数据子集生成包含量化数据的分析文本。
        """
        if df_subset.empty:
            return f"在 {condition_col_name}={condition_val} 条件下无数据记录。\n"

        total_n = len(df_subset)
        # 使用 .iloc[0] 安全地获取第一个值，避免多值警告
        condition_str = f"{condition_col_name}={df_subset[condition_col_name].iloc[0] if total_n > 0 else condition_val}"
        text = f"{condition_str} (总 N={total_n}):\n"

        # 统计原料类型频率
        feedstock_counts = df_subset['生物炭原料类型'].value_counts()

        # 分析主要原料
        frequent_feedstocks = feedstock_counts[feedstock_counts >= analysis_threshold]
        if frequent_feedstocks.empty:
            all_materials = ", ".join([f"{idx}(N={val})" for idx, val in feedstock_counts.items()])
            text += f"  - 无单一原料研究次数达到 {analysis_threshold} 次。涉及的原料类型包括: {all_materials}。\n" if all_materials else "  - 未记录原料类型。\n"
        else:
            text += "  - 主要研究的原料类型及温度分布：\n"
            for material, count in frequent_feedstocks.items():
                df_material = df_subset[df_subset['生物炭原料类型'] == material]
                # 统计每个温度区间的数量
                temp_dist = df_material['温度区间'].value_counts().reindex(['Low (≤400°C)', 'Mid (401-550°C)', 'High (>550°C)'], fill_value=0) # 保证顺序和包含所有区间
                dist_str = ", ".join([f"{idx}: {num}" for idx, num in temp_dist.items()])
                min_temp = df_material['生物炭热解温度(°C)'].min()
                max_temp = df_material['生物炭热解温度(°C)'].max()
                median_temp = df_material['生物炭热解温度(°C)'].median() # 中位数更能代表中心趋势
                text += f"    - {material} (N={count}): 温度范围 {min_temp:.0f}-{max_temp:.0f}°C (中位数 {median_temp:.0f}°C)。温度区间分布: {dist_str}。\n"

        # 分析整体温度趋势
        if total_n > 0:
            temp_dist_overall = df_subset['温度区间'].value_counts().reindex(['Low (≤400°C)', 'Mid (401-550°C)', 'High (>550°C)'], fill_value=0)
            dist_str_overall = "; ".join([f"{idx}: {num} ({num/total_n:.1%})" for idx, num in temp_dist_overall.items()])
            text += f"  - 总体温度分布: {dist_str_overall}。\n"

        return text + "\n"

    # --- 执行分析并打印结果 ---

    print("\n" + "="*10 + " 分析一：不同土壤 pH 条件与灌溉方式下的生物炭研究热点 " + "="*10)

    print("\n--- 1. 干湿交替 (Alternate Wetting and Drying, AWD) ---")
    ph_values_awd = sorted(df_awd['土壤pH'].unique())
    for ph in ph_values_awd:
        print(generate_detailed_analysis_text(df_awd[df_awd['土壤pH'] == ph], "土壤pH", ph))

    print("\n--- 2. 持续淹水 (Continuous Flooding, CF) ---")
    ph_values_cf = sorted(df_cf['土壤pH'].unique())
    for ph in ph_values_cf:
        print(generate_detailed_analysis_text(df_cf[df_cf['土壤pH'] == ph], "土壤pH", ph))

    print("\n" + "="*10 + " 分析二：不同初始镉浓度与灌溉方式下的生物炭研究热点 " + "="*10)

    print("\n--- 1. 干湿交替 (Alternate Wetting and Drying, AWD) ---")
    cd_values_awd = sorted(df_awd['初始镉浓度(mg/kg)'].unique())
    for cd in cd_values_awd:
        print(generate_detailed_analysis_text(df_awd[df_awd['初始镉浓度(mg/kg)'] == cd], "初始镉浓度(mg/kg)", cd))

    print("\n--- 2. 持续淹水 (Continuous Flooding, CF) ---")
    cd_values_cf = sorted(df_cf['初始镉浓度(mg/kg)'].unique())
    for cd in cd_values_cf:
        print(generate_detailed_analysis_text(df_cf[df_cf['初始镉浓度(mg/kg)'] == cd], "初始镉浓度(mg/kg)", cd))

    print("\n分析完成。请复制上面的统计结果。")

# --- 异常处理 ---
except FileNotFoundError as e:
    print(e)
except KeyError as e:
    print(f"\n错误：CSV 文件中缺少必需的列名，或列名不正确: {e}")
    print("请检查 CSV 文件的标题行（应为第二行）。")
except Exception as e:
    print(f"\n执行脚本时发生未知错误: {e}")