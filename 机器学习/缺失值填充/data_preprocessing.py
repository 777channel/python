import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(file_path):
    """
    加载数据并进行预处理
    """
    print("正在加载和预处理数据...")

    # 加载数据
    df = pd.read_csv(file_path)

    # 查看基本信息
    print(f"数据集大小: {df.shape}")
    print("\n列信息:")
    print(df.columns.tolist())

    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values[missing_values > 0])

    # 处理CD_shell_change列中的空白字符串
    if 'CD_shell_change' in df.columns:
        # 将空白字符串转换为NaN
        df['CD_shell_change'] = df['CD_shell_change'].replace(r'^\s*$', np.nan, regex=True)
        # 转换为数值类型
        df['CD_shell_change'] = pd.to_numeric(df['CD_shell_change'], errors='coerce')

    # 删除地点列（如果存在）
    if '地点' in df.columns:
        df = df.drop('地点', axis=1)
        print("\n已删除地点列")

    # 确保所有数值列的类型正确
    numeric_cols = ['pH', 'organicmatter', 'CEC', 'Cdcontent', 'temperature',
                    'pH_soil', 'Charcoal_application_ratio']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 确定分类列和目标列
    categorical_cols = ['原料', '品种', 'Irrigationmethod']
    target_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    print("\n数据预处理完成")

    return df, numeric_cols, categorical_cols, target_cols


def classify_varieties(df):
    """
    根据品种名称中的标记识别低累积品种和普通品种
    """
    print("\n正在对品种进行分类...")

    # 创建品种类型列
    df['品种类型'] = df['品种'].apply(
        lambda x: '低累积品种' if '低累积' in str(x) else '普通品种'
    )

    # 统计各类品种数量
    variety_counts = df['品种类型'].value_counts()
    print("\n品种类型统计:")
    print(variety_counts)

    # 查看低累积品种列表
    low_accumulation = df[df['品种类型'] == '低累积品种']['品种'].unique()
    print(f"\n低累积品种列表 ({len(low_accumulation)}个):")
    for v in low_accumulation:
        print(f"  - {v}")

    return df


def classify_materials(df):
    """
    对原料进行分类
    """
    print("\n正在对原料进行分类...")

    # 定义原料分类映射
    material_categories = {
        # 谷物秸秆类
        '谷物秸秆类': ['小麦秸秆', '水稻秸秆', '玉米秸秆', '高硅水稻秸秆', '低硅水稻秸秆', '稻草'],

        # 豆类/油料作物秸秆类
        '豆类油料秸秆类': ['花生秸秆', '油菜秸秆', '豆秸', '大蒜秸秆'],

        # 果壳/硬壳类
        '果壳硬壳类': ['稻壳', '山胡桃果壳', '花生壳', '椰壳', '椰子壳'],

        # 木质/竹类
        '木质竹类': ['竹', '木屑', '硬木'],

        # 其他植物副产品
        '其他植物副产品': ['玉米芯'],

        # 非植物来源/特殊类
        '非植物来源': ['蛋壳', '污泥', '酒糟'],

        # 混合物
        '混合物': ['玉米秸秆、花生壳、稻壳混合']
    }

    # 创建反向映射
    material_to_category = {}
    for category, materials in material_categories.items():
        for material in materials:
            material_to_category[material] = category

    # 添加分类列
    df['原料大类'] = df['原料'].map(material_to_category)

    # 检查是否有未分类的原料
    unclassified = df[df['原料大类'].isna()]['原料'].unique()
    if len(unclassified) > 0:
        print(f"警告: 发现未分类的原料: {unclassified}")
        # 将未分类的原料归为"其他"类别
        df['原料大类'] = df['原料大类'].fillna('其他')

    print("\n原料分类完成")
    print(df['原料大类'].value_counts())

    return df, material_categories


def classify_irrigation_method(df):
    """
    识别和分类灌溉方式
    """
    print("\n正在对灌溉方式进行分类...")

    # 查看唯一的灌溉方式
    irrigation_methods = df['Irrigationmethod'].unique()
    print(f"发现的灌溉方式: {irrigation_methods}")

    # 简化为两类编码（如果有多于两种，可以根据实际情况调整）
    if len(irrigation_methods) == 2:
        # 如果刚好有两种，直接编码为0和1
        le = LabelEncoder()
        df['灌溉方式编码'] = le.fit_transform(df['Irrigationmethod'])

        # 输出编码映射
        mapping = {method: code for method, code in zip(le.classes_, range(len(le.classes_)))}
        print("灌溉方式编码映射:")
        for method, code in mapping.items():
            print(f"  {method} -> {code}")
    else:
        # 如果超过两种，可能需要更复杂的处理
        print("注意: 发现超过两种灌溉方式，可能需要调整编码策略")
        # 这里仍然进行编码，但提醒用户
        le = LabelEncoder()
        df['灌溉方式编码'] = le.fit_transform(df['Irrigationmethod'])

    return df


def create_target_specific_encoding(df, target_col, categorical_col='原料'):
    """
    为目标变量创建专用的影响编码
    """
    # 计算每种原料对目标变量的平均影响
    impact_means = df.groupby(categorical_col)[target_col].mean().to_dict()

    # 填充缺失值
    # 使用所有非缺失值的平均值作为默认值
    default_impact = df[target_col].mean()

    # 创建影响映射函数
    def get_impact_level(material):
        impact = impact_means.get(material, default_impact)
        if pd.isna(impact):
            return np.nan
        elif impact < -0.5:
            return 1  # 强负影响
        elif impact < -0.2:
            return 2  # 中负影响
        elif impact < 0:
            return 3  # 弱负影响
        else:
            return 4  # 中性/正面影响

    # 应用影响级别编码
    encoding_col_name = f'{categorical_col}对{target_col}的影响'
    df[encoding_col_name] = df[categorical_col].apply(get_impact_level)

    # 打印编码统计
    print(f"\n{encoding_col_name} 编码统计:")
    print(df[encoding_col_name].value_counts())

    # 返回原料对该目标的影响分组
    impact_groups = {}
    for material in df[categorical_col].unique():
        if material in impact_means:
            impact = impact_means[material]
            if pd.isna(impact):
                continue
            level = get_impact_level(material)
            if level not in impact_groups:
                impact_groups[level] = []
            impact_groups[level].append((material, impact))

    # 打印每个影响级别的原料
    print("\n各影响级别的原料:")
    impact_level_names = {
        1: "强负影响 (<-0.5)",
        2: "中负影响 (-0.5~-0.2)",
        3: "弱负影响 (-0.2~0)",
        4: "中性/正面影响 (>=0)"
    }

    for level, materials in impact_groups.items():
        print(f"{impact_level_names[level]}:")
        for mat, impact in sorted(materials, key=lambda x: x[1]):
            print(f"  {mat}: {impact:.2f}")

    return df, impact_groups


def encode_features_for_knn(df, categorical_cols):
    """
    为KNN填充创建混合编码（独热编码+目标影响编码）
    """
    print("\n为KNN填充创建编码...")
    df_encoded = df.copy()

    # 为分类特征创建独热编码
    for col in categorical_cols:
        if col == '原料大类':  # 原料大类独热编码
            # 使用pandas的get_dummies函数进行独热编码
            dummies = pd.get_dummies(df_encoded[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            # 保留原始列
        elif col in ['品种类型', '灌溉方式编码']:  # 二元特征直接编码
            # 品种类型已经是文本，转换为数值编码
            if col == '品种类型':
                df_encoded['品种类型编码'] = df_encoded['品种类型'].map({'低累积品种': 0, '普通品种': 1})
            # 灌溉方式编码已经是数值，无需额外处理

    # 注意：目标影响编码将在每个目标变量的填充过程中单独添加

    print("\nKNN编码完成，编码后的特征:")
    encoded_features = [col for col in df_encoded.columns if ('编码' in col or '原料大类_' in col)]
    print(encoded_features)

    return df_encoded


def encode_features_for_rf(df, target_col, categorical_cols):
    """
    为随机森林填充创建目标专用的影响编码
    """
    print(f"\n为随机森林填充创建 {target_col} 的专用编码...")
    df_encoded = df.copy()

    # 为原料创建目标专用的影响编码
    df_encoded, impact_groups = create_target_specific_encoding(df_encoded, target_col, '原料')

    # 为原料大类也创建目标专用的影响编码
    df_encoded, _ = create_target_specific_encoding(df_encoded, target_col, '原料大类')

    # 为品种类型使用简单的二元编码
    if '品种类型' in df_encoded.columns:
        df_encoded['品种类型编码'] = df_encoded['品种类型'].map({'低累积品种': 0, '普通品种': 1})

    # 灌溉方式已经编码，无需额外处理

    print("\n随机森林编码完成")

    return df_encoded, impact_groups


def prepare_data_for_interpolation(df, target_col, categorical_cols):
    """
    为插值填充准备数据，使用两阶段编码进行分组
    """
    print(f"\n为插值填充准备 {target_col} 的数据...")
    df_grouped = df.copy()

    # 对原料进行两阶段编码
    # 第一阶段：原料大类
    # 第二阶段：根据对目标的影响
    df_grouped, impact_groups = create_target_specific_encoding(df_grouped, target_col, '原料')

    # 为分组创建组合键
    # 使用原料大类和对目标的影响级别组合
    df_grouped['interpolation_group'] = df_grouped['原料大类'] + '_' + df_grouped[f'原料对{target_col}的影响'].astype(
        str)

    # 添加品种类型到分组中
    if '品种类型' in df_grouped.columns:
        df_grouped['interpolation_group'] += '_' + df_grouped['品种类型']

    # 添加灌溉方式到分组中
    if '灌溉方式编码' in df_grouped.columns:
        df_grouped['interpolation_group'] += '_' + df_grouped['灌溉方式编码'].astype(str)

    # 查看分组情况
    group_counts = df_grouped['interpolation_group'].value_counts()
    print(f"\n插值分组统计 (前10组):")
    print(group_counts.head(10))
    print(f"总共 {len(group_counts)} 个分组")

    # 检查组内样本分布
    small_groups = group_counts[group_counts < 3]
    if not small_groups.empty:
        print(f"\n警告: 有 {len(small_groups)} 个分组的样本量少于3，可能影响插值效果")
        print(f"样本量最少的分组: {small_groups.index[-1]} (样本数: {small_groups.iloc[-1]})")

    return df_grouped, group_counts


def main():
    # 设置文件路径
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv"

    # 加载并预处理数据
    df, numeric_cols, categorical_cols, target_cols = load_and_preprocess_data(file_path)

    # 数据分析
    missing_targets = pd.DataFrame({
        'column': target_cols,
        'missing_count': [df[col].isnull().sum() for col in target_cols]
    }).sort_values('missing_count')

    print("\n目标变量缺失值情况:")
    print(missing_targets)

    # 对原料进行分类
    df, material_categories = classify_materials(df)

    # 对品种进行分类（根据名称中是否包含"低累积"）
    df = classify_varieties(df)

    # 对灌溉方式进行分类
    df = classify_irrigation_method(df)

    # 准备各种填充方法的数据
    # 1. KNN填充的编码
    df_knn = encode_features_for_knn(df, categorical_cols + ['原料大类', '品种类型'])

    # 2. 随机森林填充的编码 (为第一个目标变量示例)
    first_target = target_cols[0]
    df_rf, impact_groups_rf = encode_features_for_rf(df, first_target, categorical_cols + ['原料大类', '品种类型'])

    # 3. 插值填充的分组 (为第一个目标变量示例)
    df_interp, group_counts = prepare_data_for_interpolation(df, first_target, categorical_cols + ['品种类型'])

    # 保存预处理后的数据
    df.to_csv('preprocessed_data.csv', index=False)
    df_knn.to_csv('knn_encoded_data.csv', index=False)
    df_rf.to_csv('rf_encoded_data.csv', index=False)
    df_interp.to_csv('interpolation_grouped_data.csv', index=False)

    print("\n预处理完成，数据已保存")

    # 返回处理后的数据和信息
    return {
        'original_data': df,
        'knn_encoded': df_knn,
        'rf_encoded': df_rf,
        'interpolation_grouped': df_interp,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'target_cols': target_cols,
        'missing_targets': missing_targets,
        'material_categories': material_categories,
        'impact_groups': impact_groups_rf
    }


if __name__ == "__main__":
    result = main()