import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import warnings

warnings.filterwarnings('ignore')


def bilateral_minmax(x):
    """双边Min-Max归一化，保持零点不变"""
    pos_max = np.max(np.maximum(x, 0))
    neg_min = np.min(np.minimum(x, 0))

    result = np.zeros_like(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0

    if pos_max > 0:
        result[pos_mask] = x[pos_mask] / pos_max

    if neg_min < 0:
        result[neg_mask] = x[neg_mask] / abs(neg_min)

    return result


def load_and_preprocess_data(file_path):
    """加载数据并进行基础预处理"""
    print("正在加载和预处理数据...")

    # 尝试不同的编码方式加载数据
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1')

    print(f"数据加载成功，形状: {df.shape}")
    print("\n列信息:")
    print(df.columns.tolist())

    # 删除不需要的列
    if '水稻种植方式' in df.columns:
        df = df.drop('水稻种植方式', axis=1)
        print("\n已删除水稻种植方式列")

    if '地点' in df.columns:
        df = df.drop('地点', axis=1)
        print("\n已删除地点列")

    # 处理CD_shell_change列中的空白字符串
    if 'CD_shell_change' in df.columns:
        df['CD_shell_change'] = df['CD_shell_change'].replace(r'^\s*$', np.nan, regex=True)
        df['CD_shell_change'] = pd.to_numeric(df['CD_shell_change'], errors='coerce')

    # 确保所有数值列的类型正确
    numeric_cols = ['pH', 'organicmatter', 'CEC', 'Cdcontent', 'temperature',
                    'pHsoil', 'Charcoal_application_ratio']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确定分类列和目标列
    categorical_cols = []
    if '品种' in df.columns:
        categorical_cols.append('品种')
    if 'ingredient' in df.columns:
        categorical_cols.append('ingredient')
    if 'Irrigationmethod' in df.columns:
        categorical_cols.append('Irrigationmethod')

    target_cols = [col for col in df.columns if col.startswith('CD_')]

    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])

    print("\n数据基础预处理完成")
    return df, numeric_cols, categorical_cols, target_cols


def normalize_features(df, numeric_cols):
    """对数值特征应用不同的归一化方法"""
    print("\n对数值特征进行归一化...")
    df_normalized = df.copy()
    # 新增: 异常值检测与处理
    for col in numeric_cols:
        if col in df_normalized.columns and df_normalized[col].dtype != 'object':
            # 计算IQR
            Q1 = df_normalized[col].quantile(0.25)
            Q3 = df_normalized[col].quantile(0.75)
            IQR = Q3 - Q1

            # 定义异常值界限
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 检测异常值
            outliers = df_normalized[(df_normalized[col] < lower_bound) | (df_normalized[col] > upper_bound)][col]
            if len(outliers) > 0:
                print(f"检测到 {col} 中有 {len(outliers)} 个异常值")

                # 将异常值替换为边界值而不是删除
                df_normalized.loc[df_normalized[col] < lower_bound, col] = lower_bound
                df_normalized.loc[df_normalized[col] > upper_bound, col] = upper_bound
                print(f"已将 {col} 中的异常值替换为IQR边界值")

    # 检查数据类型，确保是数值类型
    for col in numeric_cols:
        if col in df_normalized.columns and df_normalized[col].dtype == 'object':
            print(f"警告: 将'{col}'从对象类型转换为数值类型")
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')

    # 1. Min-Max归一化 [0,1] - 适用于pH, pH_soil, temperature, Charcoal_application_ratio
    minmax_features = [col for col in ['pH', 'pHsoil', 'temperature', 'Charcoal_application_ratio']
                       if col in numeric_cols and col in df.columns]

    if minmax_features:
        # 确保没有缺失值
        for col in minmax_features:
            if df_normalized[col].isna().any():
                df_normalized[col] = df_normalized[col].fillna(df_normalized[col].mean())

        minmax_scaler = MinMaxScaler()
        df_normalized[minmax_features] = minmax_scaler.fit_transform(df_normalized[minmax_features])
        print(f"Min-Max归一化已应用于: {minmax_features}")

    # 2. Z-score标准化 - 适用于organicmatter, CEC
    std_features = [col for col in ['organicmatter', 'CEC']
                    if col in numeric_cols and col in df.columns]

    if std_features:
        # 确保没有缺失值
        for col in std_features:
            if df_normalized[col].isna().any():
                df_normalized[col] = df_normalized[col].fillna(df_normalized[col].mean())

        std_scaler = StandardScaler()
        df_normalized[std_features] = std_scaler.fit_transform(df_normalized[std_features])
        print(f"Z-score标准化已应用于: {std_features}")

    # 3. 对数变换+标准化 - 适用于Cdcontent
    if 'Cdcontent' in numeric_cols and 'Cdcontent' in df.columns:
        # 确保没有缺失值
        if df_normalized['Cdcontent'].isna().any():
            df_normalized['Cdcontent'] = df_normalized['Cdcontent'].fillna(df_normalized['Cdcontent'].mean())

        # 如果有0或负值，使用log1p而非log
        if (df_normalized['Cdcontent'] <= 0).any():
            min_cd = df_normalized['Cdcontent'].min()
            shift = 0 if min_cd >= 0 else abs(min_cd) + 1
            df_normalized['log_Cdcontent'] = np.log1p(df_normalized['Cdcontent'] + shift)
        else:
            df_normalized['log_Cdcontent'] = np.log(df_normalized['Cdcontent'])

        # 标准化
        log_cd = df_normalized['log_Cdcontent'].values.reshape(-1, 1)
        log_cd_std = StandardScaler().fit_transform(log_cd)
        df_normalized['log_Cdcontent'] = log_cd_std

        # 删除原始列
        df_normalized = df_normalized.drop('Cdcontent', axis=1)
        print("对数变换+标准化已应用于: Cdcontent")

    print("数值特征归一化完成")
    return df_normalized


def classify_varieties(df):
    """分类品种并进行二元编码 - 处理带括号的品种格式"""
    print("\n对品种进行分类和编码...")
    df_encoded = df.copy()

    if '品种' not in df_encoded.columns:
        print("警告: 数据中没有'品种'列")
        return df_encoded

    # 打印品种的一些示例值
    print("品种列示例值:")
    sample_varieties = df_encoded['品种'].sample(min(5, len(df_encoded))).tolist()
    for var in sample_varieties:
        print(f"  - {var}")

    # 检查品种字段的数据类型
    print(f"品种列数据类型: {df_encoded['品种'].dtype}")

    # 确保品种是字符串类型
    df_encoded['品种'] = df_encoded['品种'].astype(str)

    # 创建品种类型列 - 使用简单的包含逻辑
    df_encoded['品种类型'] = df_encoded['品种'].apply(
        lambda x: '低累积品种' if '低累积' in x else '普通品种'
    )

    # 创建二元编码列 - 直接使用整数赋值
    df_encoded['variety_binary'] = 0  # 先将所有值设为0
    df_encoded.loc[df_encoded['品种类型'] == '普通品种', 'variety_binary'] = 1

    # 确保variety_binary是浮点数类型
    df_encoded['variety_binary'] = df_encoded['variety_binary'].astype(float)

    # 统计品种类型分布
    variety_counts = df_encoded['品种类型'].value_counts()
    print("\n品种类型统计:")
    print(variety_counts)

    # 检查二元编码结果
    binary_counts = df_encoded['variety_binary'].value_counts()
    print("\n品种二元编码统计:")
    print(binary_counts)

    # 可以选择删除原始品种列和中间类别列
    df_encoded = df_encoded.drop(['品种', '品种类型'], axis=1)

    return df_encoded


def classify_materials(df):
    """将原料分类为单一特征进行编码，使用更详细的分类"""
    print("\n对原料进行分类和单一编码...")
    df_encoded = df.copy()

    if 'ingredient' not in df_encoded.columns:
        print("警告: 数据中没有'ingredient'列")
        return df_encoded

    # 打印原料的一些示例值
    print("原料列示例值:")
    sample_materials = df_encoded['ingredient'].sample(min(5, len(df_encoded))).tolist()
    for mat in sample_materials:
        print(f"  - {mat}")

    # 确保原料是字符串类型
    df_encoded['ingredient'] = df_encoded['ingredient'].astype(str)

    # 更详细的原料分类映射
    detailed_categories = {
        'rice_straw': ['水稻秸秆', '高硅水稻秸秆', '低硅水稻秸秆', '稻草'],  # 水稻秸秆类
        'wheat_straw': ['小麦秸秆'],  # 小麦秸秆
        'corn_straw': ['玉米秸秆', '玉米芯'],  # 玉米秸秆类
        'other_crop_straw': ['花生秸秆', '油菜秸秆', '豆秸', '大蒜秸秆'],  # 其他作物秸秆
        'nutshell': ['山胡桃果壳', '花生壳', '椰壳', '椰子壳', '稻壳'],  # 坚果壳类
        'wood': ['木屑', '硬木', '竹'],  # 木材类
        'animal_derived': ['蛋壳'],  # 动物源材料
        'waste': ['污泥', '酒糟']  # 废弃物
    }

    def detailed_category(material_string):
        for category, materials in detailed_categories.items():
            for material in materials:
                if material in material_string:
                    return category
        return 'other_mixed'  # 默认分类

    # 应用分类
    df_encoded['material_type'] = df_encoded['ingredient'].apply(detailed_category)

    # 统计分类结果
    material_counts = df_encoded['material_type'].value_counts()
    print("\n原料分类统计:")
    print(material_counts)

    # 创建数值编码
    unique_categories = df_encoded['material_type'].unique()
    material_mapping = {category: i for i, category in enumerate(sorted(unique_categories))}

    # 应用编码
    df_encoded['material_type_code'] = df_encoded['material_type'].map(material_mapping)
    print(f"已创建原料类型编码列: material_type_code")
    print(f"原料类型编码映射: {material_mapping}")

    # 删除原始列和中间分类列
    df_encoded = df_encoded.drop(['ingredient', 'material_type'], axis=1)

    return df_encoded


def encode_irrigation_method(df):
    """对灌溉方式进行单一编码而非多个二元编码"""
    print("\n对灌溉方式进行编码...")
    df_encoded = df.copy()

    if 'Irrigationmethod' not in df_encoded.columns:
        print("警告: 数据中没有'Irrigationmethod'列")
        return df_encoded

    # 打印灌溉方式的一些示例值
    print("灌溉方式列示例值:")
    sample_methods = df_encoded['Irrigationmethod'].sample(min(5, len(df_encoded))).tolist()
    for method in sample_methods:
        print(f"  - {method}")

    # 确保灌溉方式是字符串类型
    df_encoded['Irrigationmethod'] = df_encoded['Irrigationmethod'].astype(str)

    # 获取唯一的灌溉方式并编码
    irrigation_methods = [m for m in df_encoded['Irrigationmethod'].unique() if m != 'nan']

    # 创建映射字典
    method_mapping = {method: i for i, method in enumerate(irrigation_methods)}
    method_mapping['nan'] = -1  # 处理缺失值

    # 应用映射创建单一特征
    df_encoded['irrigation_type'] = df_encoded['Irrigationmethod'].map(method_mapping)

    # 确保为数值类型并填充缺失值
    df_encoded['irrigation_type'] = pd.to_numeric(df_encoded['irrigation_type'], errors='coerce').fillna(-1)

    print(f"已创建灌溉方式单一编码列: irrigation_type")
    print(f"灌溉方式编码映射: {method_mapping}")

    # 删除原始列
    df_encoded = df_encoded.drop('Irrigationmethod', axis=1)

    return df_encoded


def normalize_targets(df, target_cols):
    """对目标变量应用Z-score标准化，不再使用双边Min-Max归一化或对数变换"""
    from sklearn.preprocessing import StandardScaler
    print("\n对目标变量进行Z-score标准化...")
    df_normalized = df.copy()

    for col in target_cols:
        if col in df.columns:
            # 检查是否全是缺失值
            if df[col].isna().all():
                print(f"跳过 {col}: 全是缺失值")
                continue

            # 处理缺失值
            if df[col].isna().any():
                print(f"警告: {col}中有缺失值，将以0填充")
                df_normalized[col] = df[col].fillna(0)

            # 获取有效数据
            valid_data = df_normalized[col].values.reshape(-1, 1)

            try:
                # 应用Z-score标准化
                scaler = StandardScaler()
                df_normalized[col] = scaler.fit_transform(valid_data).flatten()
                print(f"对 {col} 应用Z-score标准化")
                
                # 保存标准化器（可选）
                # 如果需要保存标准化器，可以在此添加代码
                
            except Exception as e:
                print(f"标准化 {col} 时出错: {str(e)}")

            # 打印统计信息
            non_zero = df_normalized[col][df_normalized[col] != 0]
            pos_values = df_normalized[col][df_normalized[col] > 0]
            neg_values = df_normalized[col][df_normalized[col] < 0]

            print(f"{col} 标准化后统计:")
            print(f"  非零值: {len(non_zero)}, 正值: {len(pos_values)}, 负值: {len(neg_values)}")
            print(f"  范围: [{df_normalized[col].min()}, {df_normalized[col].max()}]")

    print("目标变量标准化完成")
    return df_normalized


def full_preprocessing_pipeline(file_path):
    """完整的数据预处理流程"""
    print("开始完整的数据预处理流程...")

    # 1. 加载和基础预处理
    df, numeric_cols, categorical_cols, target_cols = load_and_preprocess_data(file_path)

    # 2. 特征编码（优先进行分类特征处理）
    # 品种分类和编码
    if '品种' in df.columns:
        df = classify_varieties(df)

    # 原料分类和编码 - 使用简化版本
    if 'ingredient' in df.columns:
        df = classify_materials(df)

    # 灌溉方式编码
    if 'Irrigationmethod' in df.columns:
        df = encode_irrigation_method(df)

    # 3. 数值特征归一化 (已增强异常值处理)
    df = normalize_features(df, numeric_cols)

    # 4. 目标变量归一化 (已优化变换方法)
    df = normalize_targets(df, target_cols)

    # 5. 统计优化后的特征数量
    feature_cols = [col for col in df.columns if not col.startswith('CD_')]
    print(f"\n优化后的特征数量: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")

    print("\n优化后的数据预处理流程已完成!")
    return df, target_cols


# 如果直接运行此脚本
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\data_finall version.csv"
    processed_df, target_cols = full_preprocessing_pipeline(file_path)

    # 保存处理后的数据
    processed_df.to_csv('fully_processed_data(3).csv', index=False)
    print("\n处理后的数据已保存到 'fully_processed_data(3).csv'")