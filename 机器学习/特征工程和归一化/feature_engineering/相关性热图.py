#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征工程脚本：对数据进行特征编码、工程和分析

使用方法:
    python feature_engineering.py <CSV文件路径> <输出目录>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 26  # 设置全局字体大小为24号
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data(csv_path, encoding='utf-8'):
    """加载数据并进行基本预处理"""
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"数据加载成功，形状: {df.shape}")

        # 处理CD_shell_change列中的空白字符串（如果存在）
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
        numeric_cols = ['pH', 'organicmatter', 'CEC', 'Cdcontent', '温度',
                        'pHsoil', 'Charcoal_application_ratio']

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 识别分类特征和数值特征
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # 识别目标变量
        target_cols = [col for col in numeric_features if col.startswith('CD_')]
        feature_cols = [col for col in numeric_features if col not in target_cols]

        # 分析缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print("\n缺失值统计:")
            print(missing_values[missing_values > 0])

        print(f"\n分类特征 ({len(categorical_features)}): {categorical_features}")
        print(f"数值特征 ({len(feature_cols)}): {feature_cols}")
        print(f"目标变量 ({len(target_cols)}): {target_cols}")

        return df, categorical_features, numeric_features, target_cols
    except Exception as e:
        print(f"数据读取失败: {e}")
        return None, [], [], []


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

    # 创建品种类型的二元编码 - 修正映射键值
    df['variety_binary'] = df['品种类型'].map({'低累积品种': 0, '普通品种': 1})

    # 检查编码结果
    print("\n品种二元编码结果:")
    print(df['variety_binary'].value_counts())

    # 检查是否有缺失值
    missing = df['variety_binary'].isna().sum()
    if missing > 0:
        print(f"警告: variety_binary中有{missing}个缺失值")
        # 检查哪些品种类型值未被映射
        unmapped_types = df[df['variety_binary'].isna()]['品种类型'].unique()
        print(f"未映射的品种类型: {unmapped_types}")
        # 填充缺失值
        df['variety_binary'] = df['variety_binary'].fillna(1)  # 默认当作普通品种
        print("已将缺失值填充为1（普通品种）")

    return df
    
    
def classify_materials(df, material_col='ingredient'):
    """对原料进行分类"""
    print("\n正在对原料进行分类...")

    if material_col not in df.columns:
        print(f"警告: 找不到原料列 '{material_col}'")
        return df, {}

    # 定义原料分类映射
    material_categories = {
        # 谷物秸秆类
        '谷物秸秆类': ['小麦秸秆', '水稻秸秆', '玉米秸秆', '高硅水稻秸秆', '低硅水稻秸秆', '稻草',
                       '水稻', '小麦', '玉米', '谷物', '稻草', '麦秆', '玉米秸秆'],

        # 豆类/油料作物秸秆类
        '豆类油料秸秆类': ['花生秸秆', '油菜秸秆', '豆秸', '大蒜秸秆',
                           '大豆', '油菜', '花生', '芝麻', '豆类', '油料作物'],

        # 果壳/硬壳类
        '果壳硬壳类': ['稻壳', '山胡桃果壳', '花生壳', '椰壳', '椰子壳',
                       '果壳', '核桃壳', '栗壳', '坚果壳', '杏壳', '桃核', '椰壳'],

        # 木质/竹类
        '木质竹类': ['竹', '木屑', '硬木',
                     '木材', '竹子', '树皮', '木屑', '锯末', '竹屑', '木质材料'],

        # 其他植物副产品
        '其他植物副产品': ['玉米芯', '棉花', '甘蔗渣', '茶渣', '咖啡渣', '甜菜', '甘蔗', '香蕉皮'],

        # 非植物来源/特殊类
        '非植物来源': ['蛋壳', '污泥', '酒糟',
                       '动物骨', '贝壳', '羽毛', '污泥', '垃圾', '城市固废', '塑料'],

        # 混合物
        '混合物': ['玉米秸秆、花生壳、稻壳混合',
                   '混合秸秆', '混合物', '复合材料', '多种材料']
    }

    # 创建反向映射
    material_to_category = {}
    for category, materials in material_categories.items():
        for material in materials:
            material_to_category[material] = category

    # 添加分类列
    df['原料大类'] = df[material_col].map(material_to_category)

    # 检查是否有未分类的原料
    unclassified = df[df['原料大类'].isna()][material_col].unique()
    if len(unclassified) > 0:
        print(f"警告: 发现未分类的原料: {unclassified}")
        # 将未分类的原料归为"其他"类别
        df['原料大类'] = df['原料大类'].fillna('其他植物副产品')

    # 创建独热编码 - 每个类别是一个特征
    material_dummies = pd.get_dummies(df['原料大类'], prefix='material')
    df = pd.concat([df, material_dummies], axis=1)

    print("\n原料分类完成")
    print(df['原料大类'].value_counts())

    return df, material_categories


def classify_irrigation_method(df, irrigation_col='Irrigationmethod'):
    """识别和分类灌溉方式 - 确保每种灌溉方式是一个特征"""
    print("\n正在对灌溉方式进行分类...")

    if irrigation_col not in df.columns:
        print(f"警告: 找不到灌溉方式列 '{irrigation_col}'")
        return df, {}

    # 创建一个副本，避免修改原始列
    df['irrigation_method'] = df[irrigation_col].copy()

    # 查看唯一的灌溉方式
    irrigation_methods = df['irrigation_method'].unique()
    print(f"发现的灌溉方式: {irrigation_methods}")

    # 创建灌溉方式的编码 - 使用简单的映射而非LabelEncoder
    irrigation_mapping = {}
    for i, method in enumerate(sorted(df['irrigation_method'].dropna().unique())):
        irrigation_mapping[method] = i

    # 应用映射，并确保处理任何缺失值
    df['irrigation_code'] = df['irrigation_method'].map(irrigation_mapping)
    df['irrigation_code'] = df['irrigation_code'].fillna(-1).astype(int)  # 确保是整数

    # 输出编码映射
    print("灌溉方式编码映射:")
    for method, code in irrigation_mapping.items():
        print(f"  {method} -> {code}")

    # 创建独热编码 - 每种灌溉方式是一个特征
    # 首先获取所有唯一的灌溉方式
    unique_methods = sorted(irrigation_mapping.keys())

    # 为每种灌溉方式创建一个二进制列
    for i, method in enumerate(unique_methods):
        col_name = f'irr_type_{i}'  # 使用数字索引而不是方法名称
        df[col_name] = (df['irrigation_method'] == method).astype(int)
        print(f"创建灌溉方式编码列: {col_name} (对应 '{method}')")

    # 添加一个数据字典来追踪灌溉方式的编码
    irrigation_dict = {f'irr_type_{i}': method for i, method in enumerate(unique_methods)}

    # 删除原始的irrigation_method列，避免后续处理中的字符串问题
    df = df.drop('irrigation_method', axis=1)

    print("\n灌溉方式分类完成")
    return df, irrigation_dict


def classify_planting_method(df, method_col='水稻种植方式'):
    """对水稻种植方式进行分类"""
    print("\n正在对水稻种植方式进行分类...")

    if method_col not in df.columns:
        print(f"警告: 找不到水稻种植方式列 '{method_col}'")
        return df

    # 创建种植方式类型列（大田或盆栽）
    df['planting_type'] = df[method_col].apply(
        lambda x: '大田' if '大田' in str(x) else '盆栽' if '盆栽' in str(x) else '其他'
    )

    # 创建种植季节类型列
    def classify_season(method):
        method = str(method).lower()
        if '一季' in method or '单季' in method:
            return '一季'
        elif '两季' in method or '双季' in method:
            return '两季'
        elif '三季' in method:
            return '三季'
        else:
            return '未知'

    df['planting_season'] = df[method_col].apply(classify_season)

    # 创建种植方式的组合分类
    df['planting_combination'] = df['planting_type'] + '_' + df['planting_season']

    # 统计各类种植方式数量
    method_counts = df['planting_combination'].value_counts()
    print("\n种植方式组合统计:")
    print(method_counts)

    # 创建种植方式的独热编码
    planting_dummies = pd.get_dummies(df['planting_combination'], prefix='planting')
    df = pd.concat([df, planting_dummies], axis=1)

    # 创建大田/盆栽的二元编码
    df['field_pot_binary'] = df['planting_type'].map({'大田': 1, '盆栽': 0, '其他': -1})

    # 创建季节数的编码
    season_mapping = {'一季': 1, '两季': 2, '三季': 3, '未知': 0}
    df['season_count'] = df['planting_season'].map(season_mapping)

    print("\n种植方式分类完成")
    return df


def classify_location(df, location_col='地点'):
    """对种植地点进行分类"""
    print("\n正在对种植地点进行分类...")

    if location_col not in df.columns:
        print(f"警告: 找不到地点列 '{location_col}'")
        return df

    # 提取省份信息
    def extract_province(location):
        location = str(location)
        # 常见省份列表
        provinces = ['北京', '天津', '上海', '重庆', '河北', '山西', '辽宁', '吉林',
                     '黑龙江', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南',
                     '湖北', '湖南', '广东', '海南', '四川', '贵州', '云南', '陕西',
                     '甘肃', '青海', '台湾', '内蒙古', '广西', '西藏', '宁夏', '新疆']

        for province in provinces:
            if province in location:
                return province

        # 处理特殊情况
        if '京' in location:
            return '北京'
        elif '沪' in location:
            return '上海'
        # 可以添加更多特殊情况的处理

        return '其他'

    df['province'] = df[location_col].apply(extract_province)

    # 统计各省份数量
    province_counts = df['province'].value_counts()
    print("\n省份统计:")
    print(province_counts)

    # 将出现次数少于5次的省份归为"其他"类别
    min_count = 5
    rare_provinces = province_counts[province_counts < min_count].index
    df['province'] = df['province'].apply(lambda x: '其他' if x in rare_provinces else x)

    # 重新统计整合后的省份
    print(f"\n整合后的省份统计 (将少于{min_count}次的归为'其他'):")
    print(df['province'].value_counts())

    # 创建省份的独热编码
    province_dummies = pd.get_dummies(df['province'], prefix='province')
    df = pd.concat([df, province_dummies], axis=1)

    # 创建区域分组（可选）
    region_mapping = {
        '北京': '北方', '天津': '北方', '河北': '北方', '山西': '北方', '内蒙古': '北方',
        '辽宁': '东北', '吉林': '东北', '黑龙江': '东北',
        '上海': '东部', '江苏': '东部', '浙江': '东部', '安徽': '东部', '福建': '东部',
        '江西': '东部', '山东': '东部',
        '河南': '中部', '湖北': '中部', '湖南': '中部',
        '广东': '南方', '广西': '南方', '海南': '南方',
        '重庆': '西南', '四川': '西南', '贵州': '西南', '云南': '西南', '西藏': '西南',
        '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北',
        '台湾': '东部', '其他': '其他'
    }

    df['region'] = df['province'].map(region_mapping)

    # 创建区域的独热编码
    region_dummies = pd.get_dummies(df['region'], prefix='region')
    df = pd.concat([df, region_dummies], axis=1)

    print("\n地点分类完成")
    return df

def create_target_specific_encoding(df, target_col, categorical_col='ingredient'):
    """为目标变量创建专用的影响编码"""
    print(f"\n为 {target_col} 创建 {categorical_col} 的影响编码...")

    if categorical_col not in df.columns or target_col not in df.columns:
        print(f"警告: 找不到所需列 '{categorical_col}' 或 '{target_col}'")
        return df, {}

    # 计算每种原料对目标变量的平均影响
    impact_means = df.groupby(categorical_col)[target_col].mean().to_dict()

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
    encoding_col_name = f'{categorical_col}_impact_{target_col}'
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


def create_optimized_heatmap(df, output_dir='./', method='pearson'):
    """
    创建优化的相关性热图:
    1. 保持图片中的颜色方案
    2. 只在下三角和主对角线显示数值
    3. 将0.00和-0.00显示为0
    4. 优先显示所有单个特征，然后再显示交互特征
    5. 最多展示25个特征
    6. 改进文本在深色背景上的可见性
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap

    # 检查并修复输出目录
    if not output_dir:
        output_dir = './'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 修复variety_binary列
    if 'variety_binary' in df.columns:
        # 检查数据类型和可能的问题
        print(f"variety_binary的数据类型: {df['variety_binary'].dtype}")
        print(f"variety_binary的缺失值数量: {df['variety_binary'].isna().sum()}")

        # 转换为数值并填充缺失值
        df['variety_binary'] = pd.to_numeric(df['variety_binary'], errors='coerce')
        if df['variety_binary'].isna().any():
            df['variety_binary'] = df['variety_binary'].fillna(1)  # 默认为普通品种
            print("已填充variety_binary的缺失值")
    elif '品种类型' in df.columns:
        # 如果不存在variety_binary但存在品种类型，创建正确的映射
        df['variety_binary'] = df['品种类型'].map({'低累积品种': 0, '普通品种': 1})
        print("从品种类型创建了variety_binary列")

    # 识别所有特征列
    all_columns = df.columns.tolist()
    target_cols = [col for col in all_columns if col.startswith('CD_')]

    # 识别数值型特征
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in target_cols]

    # 分类特征和基本特征
    basic_features = [
        'pH', 'organicmatter', 'CEC', 'Cdcontent', 'pHsoil', 'temperature',
        'Charcoal_application_ratio', 'variety_binary', 'field_pot_binary',
        'irrigation_code', 'season_count'
    ]

    # 编码特征 (material_, province_, region_)
    encoded_features = [col for col in feature_cols if
                        col.startswith('material_') or
                        col.startswith('province_') or
                        col.startswith('region_') or
                        col.startswith('irr_type_') or
                        col.startswith('planting_')]

    # 交互特征 (改为空列表，因为我们不再创建交互特征)
    interaction_features = []

    # 确保所有特征分组的完整性
    other_features = [col for col in feature_cols if
                      col not in basic_features and
                      col not in encoded_features and
                      col not in interaction_features]

    if other_features:
        print(f"其他未分类特征: {other_features}")
        # 添加到适当的分组中
        for col in other_features:
            basic_features.append(col)

    # 打印特征分组统计
    print(f"基本特征 ({len(basic_features)}个)")
    print(f"编码特征 ({len(encoded_features)}个)")

    # 按优先级排序特征: 先单个特征，后交互特征
    priority_features = []

    # 添加基本特征
    for feat in basic_features:
        if feat in feature_cols:
            priority_features.append(feat)

    # 添加编码特征中最重要的几个
    # 为了控制总数，可能需要限制编码特征的数量
    important_encoded = []
    if encoded_features:
        # 计算每个编码特征与目标的相关性
        if target_cols:
            encoded_importance = {}
            for col in encoded_features:
                if col in df.columns:
                    # 计算与所有目标的平均绝对相关性
                    corrs = []
                    for target in target_cols:
                        if target in df.columns:
                            corr = df[col].corr(df[target], method=method)
                            if not pd.isna(corr):
                                corrs.append(abs(corr))
                    if corrs:
                        encoded_importance[col] = sum(corrs) / len(corrs)

            # 按重要性排序
            important_encoded = sorted(encoded_importance.items(),
                                       key=lambda x: x[1], reverse=True)
            important_encoded = [x[0] for x in important_encoded[:5]]  # 最多5个编码特征

        # 如果无法计算重要性，则选择前几个
        if not important_encoded:
            important_encoded = encoded_features[:5]

    # 添加重要的编码特征
    priority_features.extend(important_encoded)
    priority_features.extend(target_cols)
    # 移除不需要的特征
    if 'ingredient_impact_CD_soil_change' in priority_features:
        priority_features.remove('ingredient_impact_CD_soil_change')
    # 确保不超过25个特征
    if len(priority_features) > 25:
        priority_features = priority_features[:25]
        print(f"限制为前25个特征")

    # 计算相关性矩阵
    # 首先确保所有特征都是数值型的
    df_numeric = df[priority_features].copy()
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

    # 填充缺失值
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # 计算相关性
    corr_matrix = df_numeric.corr(method=method)

    # 创建特征简称映射
    abbr_mapping = {
        'pH': 'pH-B',
        'organicmatter': 'OM',
        'CEC': 'CEC',
        'Cdcontent': 'Cd-C',
        'temperature': 'T',
        'pHsoil': 'pH-S',
        'Charcoal_application_ratio': 'BR',
        'variety_binary': 'RS',
        'irrigation_code': 'I',
        'field_pot_binary': 'Field',
        'season_count': 'Season',
        'CD_soil_change': 'Cd-S',
        'CD_brown_rice_change': 'Cd-R',
        'CD_stem_change': 'Cd-ST',
        'CD_leaf_change': 'Cd-L',
        'CD_shell_change': 'Cd-SH',
        'CD_root_change': 'Cd-RO'
    }

    # 为material_类特征创建简称
    material_prefixes = {
        'material_谷物秸秆类': 'M_Grain',
        'material_豆类油料秸秆类': 'M_Bean',
        'material_果壳硬壳类': 'M_Shell',
        'material_木质竹类': 'M_Wood',
        'material_其他植物副产品': 'M_Plant',
        'material_非植物来源': 'M_NonP',
        'material_混合物': 'M_Mix'
    }

    # 合并所有映射
    label_mapping = {}
    for col in corr_matrix.columns:
        # 检查是否是material_开头的列
        if any(col.startswith(prefix) for prefix in material_prefixes.keys()):
            for prefix, abbr in material_prefixes.items():
                if col.startswith(prefix):
                    label_mapping[col] = abbr
                    break
        # 否则使用通用简称映射
        elif col in abbr_mapping:
            label_mapping[col] = abbr_mapping[col]
        # 如果没有预定义简称，保持原样
        else:
            label_mapping[col] = col

    # 创建使用简短标签的DataFrame
    corr_matrix_relabeled = corr_matrix.copy()
    corr_matrix_relabeled.index = [label_mapping.get(col, col) for col in corr_matrix.index]
    corr_matrix_relabeled.columns = [label_mapping.get(col, col) for col in corr_matrix.columns]

    # 创建自定义颜色映射 - 使用图片中的颜色方案但调整透明度以提高文本可见性
    # 蓝色-白色-红色，深色为正负1.0的极限值
    colors = ['#043872', '#5099CC', '#F4EAE8', '#FC704F', '#6E0210']
    custom_cmap = LinearSegmentedColormap.from_list('custom_RdBu', colors, N=100)

    # 创建热图
    plt.figure(figsize=(18, 16))

    # 创建注释矩阵 - 自定义格式化数字
    annot_matrix = np.array(corr_matrix_relabeled.round(2), dtype=str)

    # 将0.00和-0.00格式化为0
    annot_matrix[annot_matrix == '0.0'] = '0'
    annot_matrix[annot_matrix == '-0.0'] = '0'
    annot_matrix[annot_matrix == '0.00'] = '0'
    annot_matrix[annot_matrix == '-0.00'] = '0'

    # 去除小数点后末尾的0
    for i in range(len(annot_matrix)):
        for j in range(len(annot_matrix)):
            if annot_matrix[i, j] not in ['0', '-0']:
                if annot_matrix[i, j].endswith('.0'):
                    annot_matrix[i, j] = annot_matrix[i, j][:-2]  # 移除.0

    # 将上三角区域设为空白（只保留颜色）
    mask = np.zeros_like(corr_matrix_relabeled, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True 
    # 创建用于显示的注释矩阵
    display_annot_matrix = annot_matrix.copy()
    display_annot_matrix[mask] = ''  # 将上三角区域设为空字符串

    # 绘制热图 - 使用稍微透明的背景色提高文本可见性
    ax = sns.heatmap(
        corr_matrix_relabeled,
        annot=display_annot_matrix,  # 使用自定义的注释矩阵
        fmt='',  # 使用空字符串，因为我们已经格式化了数值
        cmap=custom_cmap,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 20, "weight": "bold"}  
    )

    # 获取文本对象并设置颜色和轮廓
    texts = ax.texts
    for i, text in enumerate(texts):
        if text.get_text():  # 只处理非空文本
            row, col = divmod(i, len(corr_matrix_relabeled))
            if row == col:  # 对角线元素
                text.set_color('black')  # 设置对角线上的文本为黑色
                text.set_weight('bold')  # 对角线文本加粗
            else:
                # 根据相关性值的大小调整文本
                try:
                    value = float(corr_matrix_relabeled.iloc[row, col])
                    abs_value = abs(value)

                    # 为高度相关的单元格设置文本颜色和背景处理
                    text.set_weight('bold')
                    text.set_color('black')
                except:
                    pass

    # 设置标题和标签样式
    plt.title('Feature Correlation Matrix (PCC)', fontsize=24, pad=20, fontweight='bold')

    # 设置轴标签
    plt.xticks(fontsize=20, rotation=90, ha='center',fontweight='bold')
    plt.yticks(fontsize=20,fontweight='bold')

    # 调整布局
    plt.tight_layout()

    # 加强边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

    # 保存图像
    output_path = os.path.join(output_dir, '相关性热力图.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"优化的相关性热图已保存至: {output_path}")
    plt.close()
    return corr_matrix, priority_features, label_mapping
# 主函数
if __name__ == "__main__":
    # 使用命令行参数获取CSV文件路径
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\data_finall version.csv"  # 默认文件

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = ''  # 默认输出目录

    # 添加以下代码来启动分析流程
    print("=" * 50)
    print("开始特征工程和相关性分析")
    print("=" * 50)

    # 步骤1: 加载和分析数据
    print("\n===== 步骤1: 加载和分析数据 =====")
    df, categorical_features, numeric_features, target_cols = load_data(csv_path)
    if df is None:
        print("数据加载失败，程序终止")
        sys.exit(1)

    # 步骤2: 对分类特征进行处理
    print("\n===== 步骤2: 分类特征处理 =====")

    # 水稻品种分类处理
    df = classify_varieties(df)

    # 水稻种植方式分类处理
    df = classify_planting_method(df, method_col='水稻种植方式')

    # 地点分类处理
    df = classify_location(df, location_col='地点')

    # 原料分类处理 - 确保每个材料类别是一个特征
    df, material_categories = classify_materials(df, material_col='ingredient')

    # 灌溉方式处理 - 确保每种灌溉方式是一个特征
    df, irrigation_dict = classify_irrigation_method(df, irrigation_col='Irrigationmethod')

    # 步骤3: 创建目标变量特定的编码
    print("\n===== 步骤3: 创建目标特定编码 =====")
    # 为CD_soil_change创建专用编码（示例）
    if 'CD_soil_change' in target_cols:
        df, impact_groups = create_target_specific_encoding(df, 'CD_soil_change', 'ingredient')


    # 步骤4: 创建相关性热图
    print("\n===== 步骤4: 创建特征相关性热图 =====")
    print("特征工程和分析完成！")
    print("=" * 50)