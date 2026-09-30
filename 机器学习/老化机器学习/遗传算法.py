import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicTransformer
import os
import sys

def create_optimal_feature(input_path, output_path, target_column):
    """
    使用遗传算法对指定特征进行组合，生成最优的新特征。
    """
    # 1. 加载数据
    try:
        df = pd.read_csv(input_path)
        print(f"成功从 '{input_path}' 加载数据，共 {df.shape[0]} 行, {df.shape[1]} 列。")
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_path}'")
        sys.exit(1)

    # 2. 定义需要被组合的特征
    features_to_combine = ['C-D', 'C-T', 'C-W']
    
    for col in features_to_combine + [target_column]:
        if col not in df.columns:
            print(f"错误: 数据文件中缺少必需的列 '{col}'。")
            sys.exit(1)

    X_for_gp = df[features_to_combine]
    y_for_gp = df[target_column]

    # 3. 定义并初始化遗传编程转换器
    print("\n初始化遗传编程模型...")
    gp_transformer = SymbolicTransformer(
        n_components=1,
        population_size=1000,
        generations=20,
        stopping_criteria=0.01,
        function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'max', 'min'),
        metric='pearson',
        feature_names=features_to_combine,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    # 4. 运行进化过程
    print("\n开始运行遗传编程来寻找最佳组合公式 (这可能需要几分钟甚至更长时间)...")
    gp_transformer.fit(X_for_gp, y_for_gp)

    # 5. 从 "名人堂" 中提取并打印出找到的最佳公式
    # gp_transformer.programs_ 是一个列表的列表, [0]代表我们想要的第一个组件
    # 第一个组件的[0]代表该组件名人堂中的第一个（即最佳）公式
    best_program_object = gp_transformer._programs[0][0]
    # --- 增强的打印输出 ---
    print("\n" + "="*25)
    print("  遗传算法进化完成！")
    print("="*25)
    print(f"\n找到的最佳公式 (Best Formula Found):\n")
    # 直接打印这个 program 对象，会自动调用它的 __str__ 方法显示为可读的公式
    print(f"  {best_program_object}\n")
    print("="*55 + "\n")
    # --- 结束 ---

    # 6. 应用公式，生成新的特征列
    new_feature_values = gp_transformer.transform(X_for_gp)

    # 7. 构建新的DataFrame并保存
    df_output = df.copy()
    df_output['C'] = new_feature_values.flatten()
    print(f"提醒：已创建新特征 'C'，如果原数据中存在同名列，它已被覆盖。")
    df_output.drop(columns=features_to_combine, inplace=True)
    print(f"已移除了原始三列: {features_to_combine}")
    
    df_output.to_csv(output_path, index=False)
    print(f"成功将包含新特征的数据保存至: '{output_path}'")


# --- ##  主程序执行区域  ## ---
if __name__ == "__main__":
    
    # --- 步骤1：在这里配置您的文件和参数 ---
    
    # 输入文件路径 (请确保路径正确)
    # 注意：Windows路径中的反斜杠`\`最好在前面加上`r`，防止转义字符问题
    input_file = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"

    # 您希望遗传算法针对哪个目标进行优化？
    # 可选项: 'pH-S-CH' 或 'Cd-CH'
    target_to_optimize = 'Cd-CH'

    # 自动生成输出文件名
    # 例如：..._v4.csv -> ..._v4_GA_optimized.csv
    directory, filename = os.path.split(input_file)
    base, ext = os.path.splitext(filename)
    output_file = os.path.join(directory, f"{base}_GA_optimized.csv")

    # --- 步骤2：运行主函数 ---
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"优化目标: {target_to_optimize}")
    
    # 调用主函数执行所有操作
    create_optimal_feature(input_file, output_file, target_to_optimize)