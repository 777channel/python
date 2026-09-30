import pandas as pd
import numpy as np

def preprocess_aging_data(input_csv_path, output_csv_path):
    """
    对老化处理数据进行预处理，创建单一处理变量 T 和有效的条件特征。

    参数:
    input_csv_path (str): 输入的 CSV 文件路径。
    output_csv_path (str): 处理后数据保存的 CSV 文件路径。
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"原始数据已加载，共 {df.shape[0]} 行, {df.shape[1]} 列。")
        print("原始数据前5行：")
        print(df.head())
        print("\n原始数据列名：")
        print(df.columns.tolist())
        print("-" * 30)

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_csv_path}")
        return
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return

    # --- 步骤一：确认或创建基础处理的二元指示列 ---
    # 假设您的数据中已经有 DW, FT 列代表是否应用了干湿循环和冻融循环 (0或1)
    # 您需要指定代表“恒温恒湿”处理的列名，这里我们用 'C_applied' 作为占位符
    # 如果这些列不存在，您需要先根据您的原始数据创建它们。

    dw_applied_col = 'DW'        # 请替换为实际的“干湿循环是否应用”的列名 (0/1)
    ft_applied_col = 'FT'        # 请替换为实际的“冻融循环是否应用”的列名 (0/1)
    c_applied_col = 'C'  # 请替换为实际的“恒温恒湿是否应用”的列名 (0/1)

    # 检查这些基础处理指示列是否存在
    required_binary_treatment_cols = [dw_applied_col, ft_applied_col, c_applied_col]
    for col in required_binary_treatment_cols:
        if col not in df.columns:
            print(f"错误：列 '{col}' 在数据中未找到。请确保该列存在或提供正确的列名。")
            # 如果C_applied列不存在，您可以根据其他逻辑创建它，例如：
            # df[c_applied_col] = ((df[dw_applied_col] == 0) & (df[ft_applied_col] == 0)).astype(int) # 示例逻辑
            # 此处为了代码能继续，我们假设如果列不存在则填充0，但这通常需要您根据实际情况处理
            # df[col] = 0 # 这是一个危险的假设，请务必确认！
            print(f"警告：列 '{col}' 不存在，这可能会导致 'T_aging_protocol' 创建不正确。请检查您的列名或数据。")
            # 为了演示，如果列不存在，我们暂时创建一个全0的列，但这需要您根据实际情况调整
            if col not in df.columns :
                 df[col] = 0
                 print(f"为演示目的，已创建全0的列 '{col}'。请您务必替换为真实数据或逻辑。")


    # --- 步骤二：创建单一的组合处理变量 T ---
    print("\n正在创建组合处理变量 'T_aging_protocol'...")
    conditions = [
        (df[dw_applied_col] == 0) & (df[ft_applied_col] == 0) & (df[c_applied_col] == 0), # T=0 (控制组)
        (df[dw_applied_col] == 1) & (df[ft_applied_col] == 0) & (df[c_applied_col] == 0), # T=1 (仅 DW)
        (df[dw_applied_col] == 0) & (df[ft_applied_col] == 1) & (df[c_applied_col] == 0), # T=2 (仅 FT)
        (df[dw_applied_col] == 0) & (df[ft_applied_col] == 0) & (df[c_applied_col] == 1), # T=3 (仅 C)
        (df[dw_applied_col] == 1) & (df[ft_applied_col] == 1) & (df[c_applied_col] == 0), # T=4 (DW + FT)
        (df[dw_applied_col] == 1) & (df[ft_applied_col] == 0) & (df[c_applied_col] == 1), # T=5 (DW + C)
        (df[dw_applied_col] == 0) & (df[ft_applied_col] == 1) & (df[c_applied_col] == 1), # T=6 (FT + C)
        (df[dw_applied_col] == 1) & (df[ft_applied_col] == 1) & (df[c_applied_col] == 1)  # T=7 (DW + FT + C)
    ]
    choices = [0, 1, 2, 3, 4, 5, 6, 7]
    df['T_aging_protocol'] = np.select(conditions, choices, default=np.nan)

    if df['T_aging_protocol'].isnull().any():
        print("警告：'T_aging_protocol' 列中存在 NaN 值。这意味着有些行的处理组合不符合预设条件。")
        print("请检查您的 'DW_applied', 'FT_applied', 'C_applied' 列的组合是否覆盖了所有情况，")
        print("或者 default 值是否应该设置为某个特定的控制组编码。")
    print("'T_aging_protocol' 列已创建。各处理组别的样本数量：")
    print(df['T_aging_protocol'].value_counts().sort_index())


    # --- 步骤三：处理条件性特征 ---
    print("\n正在创建有效的条件性特征 (如 *_eff 列)...")
    # DW 相关
    dw_duration_col = 'DW-D'  # 请替换为实际的“干湿循环天数”列名
    if dw_duration_col in df.columns and dw_applied_col in df.columns:
        df['DW_D_eff'] = df[dw_applied_col] * df[dw_duration_col]
        print(f"列 'DW_D_eff' 已创建。")
    else:
        print(f"警告：列 '{dw_duration_col}' 或 '{dw_applied_col}' 不存在，无法创建 'DW_D_eff'。")

    # FT 相关
    ft_duration_col = 'FT-D'  # 请替换为实际的“冻融循环天数”列名
    if ft_duration_col in df.columns and ft_applied_col in df.columns:
        df['FT_D_eff'] = df[ft_applied_col] * df[ft_duration_col]
        print(f"列 'FT_D_eff' 已创建。")
    else:
        print(f"警告：列 '{ft_duration_col}' 或 '{ft_applied_col}' 不存在，无法创建 'FT_D_eff'。")

    # C (恒温恒湿) 相关特征
    c_days_col = 'C-D'          # 请替换为实际的“恒温恒湿天数”列名
    c_temp_col = 'C-T'          # 请替换为实际的“恒温恒湿温度”列名
    c_moisture_col = 'C-W'      # 请替换为实际的“恒温恒湿水分”列名
    # 您可以添加更多 C 相关的条件特征

    if c_days_col in df.columns and c_applied_col in df.columns:
        df['C_days_eff'] = df[c_applied_col] * df[c_days_col]
        print(f"列 'C_days_eff' 已创建。")
    else:
        print(f"警告：列 '{c_days_col}' 或 '{c_applied_col}' 不存在，无法创建 'C_days_eff'。")

    if c_temp_col in df.columns and c_applied_col in df.columns:
        df['C_temp_eff'] = df[c_applied_col] * df[c_temp_col]
        print(f"列 'C_temp_eff' 已创建。")
    else:
        print(f"警告：列 '{c_temp_col}' 或 '{c_applied_col}' 不存在，无法创建 'C_temp_eff'。")

    if c_moisture_col in df.columns and c_applied_col in df.columns:
        df['C_moisture_eff'] = df[c_applied_col] * df[c_moisture_col]
        print(f"列 'C_moisture_eff' 已创建。")
    else:
        print(f"警告：列 '{c_moisture_col}' 或 '{c_applied_col}' 不存在，无法创建 'C_moisture_eff'。")

    # --- 步骤四：保存处理后的数据 ---
    try:
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # utf-8-sig 确保中文正确显示
        print(f"\n处理后的数据已成功保存到: {output_csv_path}")
        print("新数据的前5行：")
        print(df.head())
    except Exception as e:
        print(f"保存文件时出错: {e}")

# --- 使用示例 ---
# 请将下面的路径替换为您的实际文件路径
input_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv" # 这是您上传的文件名
output_file = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\因果推断.csv'

# 在调用函数前，请确保您已经根据实际情况修改了函数内部的列名占位符
print("重要提示：请确保您已在 preprocess_aging_data 函数内部，")
print("将 dw_applied_col, ft_applied_col, c_applied_col, dw_duration_col, ft_duration_col, c_days_col 等")
print("占位符列名替换为您的 CSV 文件中实际的列名。\n")

# 调用函数进行处理
preprocess_aging_data(input_file, output_file)