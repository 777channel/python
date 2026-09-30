import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 定义从中文原始原料到英文大类的映射函数
def map_feedstock_to_english_category(feedstock_name_cn):
    if pd.isna(feedstock_name_cn):
        return "Unknown_Type" # 处理原始数据中的NaN
    
    # 映射规则 (根据您的分类方案)
    name_str = str(feedstock_name_cn).strip()
# 0 - 水果类
# 1 - 壳类
# 2 - 粪便类
# 3 - 混合类
# 4 - 秸秆类
# 5 - 木质类
    if name_str in ["水稻秸秆", "小麦秸秆", "玉米秸秆", "油菜秸秆", "花生秸秆", "向日葵秸秆", "秸秆", "稻草"]:
        return "Straw_Type"# 2
    elif name_str in ["稻壳", "花生壳",'椰子壳','山胡桃果壳']:
        return "Husk_Shell_Type" # 0
    elif name_str in ["木屑", "竹", "玉米芯","匍匐茎",'荔枝木','棕榈丝','木屑']:
        return "Woody_Type"# 3
    elif name_str in ["牛粪", "鸡粪"]:
        return "Manure_Type"
    elif name_str == "废苹果":
        return "Fruit_Waste_Type"
    elif name_str == "玉米秸秆，污泥混合": # 请注意原始数据中逗号是中文还是英文
        return "Mixed_Type"
    else:
        
        return "Other_Plant_Type" # 1


# --- 主程序 ---
if __name__ == '__main__':
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（删除ck）_特征工程_阶段展开_v2.csv" # 请确保这个文件在您的Python脚本的工作目录下，或者提供完整路径
    output_file_processed_feedstock =r'C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（训练版本）.csv' # 定义输出文件名

    try:
        # 尝试使用常见的中文编码加载CSV
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='gb18030')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='utf-8')
        
        print("--- 原始数据 ---")
        print(f"原始数据形状: {df.shape}")
        print("原始 '原料' 列的前5行:")
        print(df[['原料']].head())

        # 1. 创建副本进行操作，以确保原始df对象不变（如果函数外部还需要原始df）
        #    如果直接用 df = ... 的方式赋值，原始df会被修改。
        #    为了清晰，我们创建一个新变量df_processed
        df_processed = df.copy()

        # 2. 应用映射规则，创建新的英文大类列
        df_processed['Feedstock_Category'] = df_processed['原料'].apply(map_feedstock_to_english_category)
        
        print("\n--- 映射后的英文大类 ('Feedstock_Category'列) ---")
        print(df_processed[['原料', 'Feedstock_Category']].head(10))
        print("\n英文大类的唯一值及其数量:")
        print(df_processed['Feedstock_Category'].value_counts())

        # 3. 对新的英文大类进行标签编码 (生成1列)
        label_encoder = LabelEncoder()
        df_processed['Feedstock_Cat_LabelEncoded'] = label_encoder.fit_transform(df_processed['Feedstock_Category'])
        
        print("\n--- 标签编码后的结果 ('Feedstock_Cat_LabelEncoded'列) ---")
        print(df_processed[['Feedstock_Category', 'Feedstock_Cat_LabelEncoded']].head(10))
        le_classes = label_encoder.classes_
        le_mapping = dict(zip(le_classes, label_encoder.transform(le_classes)))
        print("标签编码的映射关系:", le_mapping)

        # 4. 对新的英文大类进行独热编码 (生成N列)
        # 为了确保独热编码列的稳定性和顺序，明确指定类别
        expected_categories = sorted(list(df_processed['Feedstock_Category'].unique()))
        
        df_ohe_feedstock = pd.get_dummies(df_processed['Feedstock_Category'], 
                                          prefix='FeedstockCat', 
                                          columns=expected_categories, # 指定列以确保顺序和完整性
                                          dummy_na=False) 
        
        # 将独热编码的列合并回主DataFrame
        df_processed = pd.concat([df_processed, df_ohe_feedstock], axis=1)
        
        print("\n--- 独热编码后的结果 (部分列展示) ---")
        ohe_cols_to_show = ['Feedstock_Category'] + [col for col in df_processed.columns if col.startswith('FeedstockCat_')]
        print(df_processed[ohe_cols_to_show].head(10))
        
        print("\n--- 最终DataFrame的列名 (部分) ---")
        all_cols = df_processed.columns.tolist()
        print(f"处理后总列数: {len(all_cols)}")
        if len(all_cols) > 25: # 调整显示数量，确保能看到新增的列
            print(all_cols[:5] + ['...'] + [col for col in all_cols if 'Feedstock_' in col or col.startswith('FeedstockCat_')])
        else:
            print(all_cols)
        
        # 检查原始列是否都保留 (与初始加载的df的列进行比较)
        original_cols_in_df = df.columns.tolist() # 获取初始加载时的列
        preserved_check = all(col in df_processed.columns for col in original_cols_in_df)
        print(f"\n所有原始列是否都已保留在处理后的DataFrame中: {preserved_check}")
        if not preserved_check:
            missing_cols = [col for col in original_cols_in_df if col not in df_processed.columns]
            print(f"在处理后的DataFrame中未能找到的原始列: {missing_cols}")


        # 保存包含所有原始列以及新生成的分类和编码列的DataFrame
        df_processed.to_csv(output_file_processed_feedstock, index=False, encoding='utf-8-sig')
        print(f"\n包含所有原始列、分类和编码步骤的数据已保存到: {output_file_processed_feedstock}")
        print("\n提示：下一步，您可以根据您的模型需求，从此文件中选择性地删除不需要的中间列（如 '原料', 'Feedstock_Category'）或某种编码方式的列。")
        print("例如，如果选择独热编码，则保留 'FeedstockCat_*' 列；如果选择标签编码，则保留 'Feedstock_Cat_LabelEncoded' 列。")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请确保文件路径正确。")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")