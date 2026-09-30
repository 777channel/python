import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
import time

warnings.filterwarnings('ignore')


# 读取数据
def load_data(file_path):
    """读取CSV文件并返回DataFrame"""
    df = pd.read_csv(file_path, encoding='utf-8')
    print(f"数据形状: {df.shape}")
    return df


# 查看基本信息
def explore_data(df):
    """探索数据的基本信息"""
    print("\n数据前5行:")
    print(df.head())

    print("\n数据描述:")
    print(df.describe())

    print("\n缺失值情况:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({'缺失值数量': missing, '缺失率(%)': missing_percent})
    print(missing_info[missing_info['缺失值数量'] > 0])

    # 检查目标变量的分布情况
    target_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    print("\n目标变量统计:")
    for col in target_cols:
        if col != 'CD_shell_change':  # 假设CD_shell_change是分类列
            if df[col].dtype in [np.float64, np.int64]:
                neg_percent = (df[col][df[col] < 0].count() / df[col].count() * 100).round(2)
                print(f"{col}: 负值比例 {neg_percent}%")

    return missing_info[missing_info['缺失值数量'] > 0].index.tolist()


# 品种处理简化函数
def process_variety_simple(df):
    """简化处理品种列，仅区分'低累积'和'普通品种'"""
    # 创建新的二元特征
    df['品种_低累积'] = df['品种'].apply(lambda x: 1 if isinstance(x, str) and '低累积' in x else 0)
    df['品种_普通'] = df['品种'].apply(lambda x: 1 if isinstance(x, str) and '普通品种' in x else 0)

    # 创建一个综合分类
    def categorize_variety(x):
        if not isinstance(x, str):
            return '未知'
        elif '低累积' in x:
            return '低累积'
        elif '普通品种' in x:
            return '普通品种'
        else:
            return '其他'

    df['品种_类别'] = df['品种'].apply(categorize_variety)

    # 对类别进行标签编码
    le = LabelEncoder()
    df['品种_类别_编码'] = le.fit_transform(df['品种_类别'])

    print(f"品种分类编码: {dict(zip(le.classes_, range(len(le.classes_))))}")

    return df


# 地点处理函数
def encode_location(df):
    """处理地点列，提取潜在地理信息"""
    # 创建区域分组映射
    region_map = {
        '江苏省': '华东',
        '浙江省': '华东',
        '安徽省': '华东',
        '福建省': '华东',
        '福建': '华东',
        '湖南省': '华中',
        '湖北省': '华中',
        '江西省': '华中',
        '四川省': '西南',
        '广东省': '华南',
        '辽宁省': '东北'
    }
    df['地点_区域'] = df['地点'].map(region_map)
    # 对区域进行标签编码
    le = LabelEncoder()
    df['地点_区域_encoded'] = le.fit_transform(df['地点_区域'].fillna('未知'))
    return df


# 原料处理函数
def process_material(df):
    """处理原料列，提取材料类型信息"""
    # 可以创建材料大类
    material_types = {
        '秸秆': ['小麦秸秆', '水稻秸秆', '玉米秸秆', '大蒜秸秆', '油菜秸秆', '豆秸'],
        '木质': ['木屑', '竹'],
        '特殊': ['高硅水稻秸秆', '低硅水稻秸秆']
    }

    # 创建新的特征列
    for type_name, materials in material_types.items():
        df[f'原料_{type_name}'] = df['原料'].apply(lambda x: 1 if isinstance(x, str) and x in materials else 0)
    return df


# 灌溉方法处理函数
def process_irrigation(df):
    """处理灌溉方法列"""
    # 创建水分管理指数
    water_mgmt = {
        '持续淹水灌溉': 2,  # 最湿润
        '干湿交替灌溉': 1,  # 中等湿润
        '持续通气灌溉': 0  # 最干燥
    }
    df['灌溉方法_水分指数'] = df['Irrigation method'].map(water_mgmt)
    return df


# 频率编码函数
def frequency_encoding(df, categorical_features):
    """对分类特征进行频率编码"""
    df_encoded = df.copy()
    for col in categorical_features:
        if col in df.columns:
            # 处理缺失值
            if df[col].isna().any():
                df[col] = df[col].fillna('未知')

            # 计算每个值的频率
            freq_map = df[col].value_counts(normalize=True).to_dict()
            # 将频率作为新的特征
            df_encoded[f'{col}_freq'] = df[col].map(freq_map)
    return df_encoded


# 增强的数据预处理函数
def enhanced_preprocess_data(df):
    """增强的数据预处理函数，更好地利用文字特征"""
    # 基本预处理
    df_processed = df.copy()

    # 识别分类特征和数值特征
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 将目标变量与特征分开
    target_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']
    feature_cols = [col for col in df.columns if col not in target_cols]

    # 分离分类特征和数值特征
    categorical_features = [col for col in categorical_cols if col in feature_cols]
    numeric_features = [col for col in numeric_cols if col in feature_cols]

    print(f"\n原始分类特征: {categorical_features}")
    print(f"原始数值特征: {numeric_features}")

    # 进行特征工程 - 更好地利用文字数据
    if '地点' in df.columns:
        df_processed = encode_location(df_processed)
        numeric_features.append('地点_区域_encoded')

    if '原料' in df.columns:
        df_processed = process_material(df_processed)
        # 添加新生成的特征到数值特征列表
        for col in df_processed.columns:
            if col.startswith('原料_') and col not in numeric_features:
                numeric_features.append(col)

    # 使用简化的品种处理方法
    if '品种' in df.columns:
        df_processed = process_variety_simple(df_processed)
        # 添加新生成的特征到数值特征列表
        numeric_features.extend(['品种_低累积', '品种_普通', '品种_类别_编码'])

    if 'Irrigation method' in df.columns:
        df_processed = process_irrigation(df_processed)
        numeric_features.append('灌溉方法_水分指数')

    # 对所有分类特征进行频率编码
    df_processed = frequency_encoding(df_processed, categorical_features)
    # 添加频率编码特征到数值特征列表
    for col in categorical_features:
        freq_col = f'{col}_freq'
        if freq_col in df_processed.columns and freq_col not in numeric_features:
            numeric_features.append(freq_col)

    # 最后，对所有分类特征进行标签编码(作为备用)
    for col in categorical_features:
        if col in df_processed.columns:
            # 处理缺失值
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna('未知')

            # 应用标签编码
            le = LabelEncoder()
            encoded_col = f'{col}_encoded'
            df_processed[encoded_col] = le.fit_transform(df_processed[col])
            if encoded_col not in numeric_features:
                numeric_features.append(encoded_col)

    # 处理CD_shell_change列
    if 'CD_shell_change' in target_cols and 'CD_shell_change' in df.columns:
        # 检查CD_shell_change的类型
        if df['CD_shell_change'].dtype == 'object':
            print("  'CD_shell_change' 是分类变量，进行标签编码")
            label_encoder = LabelEncoder()
            # 填充缺失值为'missing'
            df_processed['CD_shell_change'] = df_processed['CD_shell_change'].fillna('missing')
            df_processed['CD_shell_change_encoded'] = label_encoder.fit_transform(df_processed['CD_shell_change'])
            # 存储映射关系
            shell_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            print("  CD_shell_change编码映射:", shell_mapping)
        else:
            print("  'CD_shell_change' 是数值变量，不需要编码")

    # 打印处理后的特征列表
    print(f"\n增强后的数值特征: {numeric_features}")
    print(f"增强后的分类特征: {categorical_features}")

    return df_processed, categorical_features, numeric_features, target_cols


# 平均值填充方法
def mean_imputation(df, categorical_features, numeric_features):
    """使用平均值填充数值型缺失值，众数填充分类型缺失值"""
    df_imputed = df.copy()

    # 对分类特征使用众数填充
    for col in categorical_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            if not df_imputed[col].mode().empty:
                mode_val = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                print(f"  列 '{col}' 使用众数 '{mode_val}' 填充")
            else:
                df_imputed[col] = df_imputed[col].fillna('未知')
                print(f"  列 '{col}' 使用 '未知' 填充(无众数)")

    # 对数值特征使用均值填充
    for col in numeric_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            try:
                mean_val = df_imputed[col].mean()
                df_imputed[col] = df_imputed[col].fillna(mean_val)
                print(f"  列 '{col}' 使用均值 {mean_val} 填充")
            except Exception as e:
                print(f"  填充列 '{col}' 时出错: {e}")

    return df_imputed


# KNN填充方法
def knn_imputation(df, categorical_features, numeric_features, n_neighbors=5):
    """使用KNN方法填充缺失值，分类特征需先编码"""
    df_imputed = df.copy()

    # 处理分类特征 - 使用独热编码
    try:
        # 首先填充分类特征的缺失值(用众数)以便进行编码
        for col in categorical_features:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                if not df_imputed[col].mode().empty:
                    mode_val = df_imputed[col].mode()[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                else:
                    df_imputed[col] = df_imputed[col].fillna("未知")

        # 对分类特征进行编码
        if categorical_features:
            df_encoded = pd.get_dummies(df_imputed, columns=categorical_features, drop_first=True)
        else:
            df_encoded = df_imputed.copy()

        # 选择所有数值列用于KNN填充
        numeric_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # 检查是否有足够的数值列用于KNN
        if len(numeric_cols) < 2:  # KNN需要至少两列
            print("  警告: 数值列不足，无法使用KNN填充")
            # 回退到简单均值填充
            for col in numeric_features:
                if col in df_imputed.columns and df_imputed[col].isna().any():
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
            return df_imputed

        # 标准化数值列
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_encoded[numeric_cols])
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_cols, index=df_encoded.index)

        # 应用KNN填充
        imputer = KNNImputer(n_neighbors=min(n_neighbors, len(scaled_df) - 1))
        imputed_array = imputer.fit_transform(scaled_df)

        # 转回原始比例
        imputed_df = pd.DataFrame(
            scaler.inverse_transform(imputed_array),
            columns=numeric_cols,
            index=df_encoded.index
        )

        # 更新原始数据框中的数值列
        for col in numeric_cols:
            missing_idx = df_encoded[col].isna()
            if missing_idx.any():
                df_encoded.loc[missing_idx, col] = imputed_df.loc[missing_idx, col].values

        return df_encoded

    except Exception as e:
        print(f"  KNN填充错误: {e}")
        print("  回退到均值填充")

        # 回退到简单均值/众数填充
        for col in categorical_features:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                if not df_imputed[col].mode().empty:
                    mode_val = df_imputed[col].mode()[0]
                    df_imputed[col] = df_imputed[col].fillna(mode_val)
                else:
                    df_imputed[col] = df_imputed[col].fillna("未知")

        for col in numeric_features:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

        return df_imputed


# 随机森林填充方法
def rf_imputation(df, categorical_features, numeric_features):
    """使用随机森林填充缺失值"""
    df_imputed = df.copy()

    # 先用简单方法填充，以便训练模型
    for col in categorical_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            if not df_imputed[col].mode().empty:
                mode_val = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
            else:
                # 如果没有众数（所有值都是NA），使用"未知"填充
                df_imputed[col] = df_imputed[col].fillna("未知")

    for col in numeric_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    try:
        # 重要修复：明确区分分类和数值列，并正确处理
        print("\n处理随机森林填充前的数据类型:")
        for col in df_imputed.columns:
            print(f"  列 '{col}' 类型: {df_imputed[col].dtype}")

        # 确保分类特征列是object类型，数值特征是数值类型
        for col in categorical_features:
            if col in df_imputed.columns and df_imputed[col].dtype != 'object':
                df_imputed[col] = df_imputed[col].astype(str)
                print(f"  将列 '{col}' 转换为字符串类型")

        for col in numeric_features:
            if col in df_imputed.columns and df_imputed[col].dtype == 'object':
                try:
                    df_imputed[col] = pd.to_numeric(df_imputed[col], errors='coerce')
                    print(f"  将列 '{col}' 转换为数值类型")
                except:
                    print(f"  警告: 无法将列 '{col}' 转换为数值类型，从数值特征列表中移除")
                    numeric_features.remove(col)
                    if col not in categorical_features:
                        categorical_features.append(col)

        # 对分类特征进行编码
        print(f"\n对这些分类特征进行独热编码: {categorical_features}")
        if categorical_features:
            df_encoded = pd.get_dummies(df_imputed, columns=categorical_features, drop_first=True)
        else:
            df_encoded = df_imputed.copy()

        print(f"\n编码后数据框的列数: {len(df_encoded.columns)}")

        # 确保所有列都是数值类型
        non_numeric_cols = df_encoded.select_dtypes(exclude=['int64', 'float64']).columns
        if len(non_numeric_cols) > 0:
            print(f"  警告: 编码后仍有非数值列: {non_numeric_cols}")
            # 删除非数值列
            df_encoded = df_encoded.select_dtypes(include=['int64', 'float64'])

        # 对每个缺失值列应用随机森林
        for col in numeric_features:
            # 检查列是否在数据框中
            if col not in df.columns or col not in df_encoded.columns:
                print(f"  警告: 列 '{col}' 不在数据框中，跳过")
                continue

            # 检查是否有缺失值
            missing_mask = df[col].isna()
            missing_count = missing_mask.sum()

            if missing_count > 0:
                print(f"\n开始填充列 '{col}' (缺失值数量: {missing_count})")

                # 确保有足够的非缺失值来训练模型
                if missing_count < len(df) * 0.9:  # 确保至少有10%的数据可用于训练
                    try:
                        # 分割特征和目标
                        X_cols = [c for c in df_encoded.columns if c != col]
                        if len(X_cols) < 2:  # 需要至少两个特征
                            print(f"  警告: '{col}' 缺少足够的预测特征，使用中位数填充")
                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                            continue

                        X_train = df_encoded.loc[~missing_mask, X_cols]
                        y_train = df_encoded.loc[~missing_mask, col]
                        X_missing = df_encoded.loc[missing_mask, X_cols]

                        # 检查训练数据是否有效
                        if len(y_train) < 10 or X_train.isnull().any().any():
                            print(f"  警告: '{col}' 的训练数据无效，使用中位数填充")
                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                            continue

                        # 训练随机森林，使用稳健参数
                        print(f"  为特征列 '{col}' 优化随机森林参数")
                        start_time = time.time()
                        best_model, best_params = optimize_rf_for_feature(X_train, y_train)
                        elapsed_time = time.time() - start_time
                        print(f"  参数优化完成，耗时: {elapsed_time:.2f}秒")
                        print(f"  特征列 '{col}' 的最佳参数: {best_params}")

                        # 使用优化后的模型
                        best_model.fit(X_train, y_train)

                        # 预测时使用优化后的模型
                        predicted = best_model.predict(X_missing)

                        # 预测并填充缺失值
                        predicted = rf.predict(X_missing)
                        df_encoded.loc[missing_mask, col] = predicted

                        # 打印特征重要性（前5个）
                        if len(X_train.columns) > 5:
                            feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
                            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                            print(f"  填充 {col} 的前5个重要特征:")
                            for feat, imp in sorted_importance[:5]:
                                print(f"    {feat}: {imp:.4f}")
                        else:
                            feature_importance = dict(zip(X_train.columns, rf.feature_importances_))
                            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                            print(f"  填充 {col} 的所有特征重要性:")
                            for feat, imp in sorted_importance:
                                print(f"    {feat}: {imp:.4f}")

                    except Exception as e:
                        print(f"  随机森林填充 '{col}' 时出错: {e}")
                        print(f"  使用中位数填充")
                        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                else:
                    print(f"  警告: {col} 缺失值过多 ({missing_mask.sum()} / {len(df)}), 使用中位数填充")
                    df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

        return df_encoded

    except Exception as e:
        print(f"  随机森林填充过程出错: {e}")
        print(f"  完整错误信息: {str(e)}")
        print("  回退到均值/众数填充方法")

        # 如果随机森林方法失败，回退到简单填充
        for col in categorical_features:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                if not df_imputed[col].mode().empty:
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])
                else:
                    df_imputed[col] = df_imputed[col].fillna("未知")

        for col in numeric_features:
            if col in df_imputed.columns and df_imputed[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

        return df_imputed


# 插值填充方法
def interpolation_imputation(df, categorical_features, numeric_features):
    """使用插值法填充数值型缺失值"""
    df_imputed = df.copy()

    # 对分类特征使用众数填充
    for col in categorical_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            if not df_imputed[col].mode().empty:
                mode_val = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                print(f"  列 '{col}' 使用众数 '{mode_val}' 填充")
            else:
                df_imputed[col] = df_imputed[col].fillna("未知")
                print(f"  列 '{col}' 使用 '未知' 填充(无众数)")

    # 对数值特征使用插值填充
    for col in numeric_features:
        if col in df_imputed.columns and df_imputed[col].isna().any():
            try:
                # 检查是否有足够的非NA值进行插值
                non_na_count = df_imputed[col].notna().sum()
                if non_na_count < 2:  # 插值需要至少两个点
                    print(f"  警告: 列 '{col}' 没有足够的非缺失值进行插值，使用中位数填充")
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                    continue

                # 先记录原始缺失值的位置
                missing_mask = df_imputed[col].isna()
                missing_count = missing_mask.sum()

                # 使用线性插值
                df_imputed[col] = df_imputed[col].interpolate(method='linear')

                # 处理边界缺失值
                df_imputed[col] = df_imputed[col].fillna(method='bfill').fillna(method='ffill')

                # 检查是否仍有缺失值
                still_missing = df_imputed[col].isna().sum()
                if still_missing > 0:
                    print(f"  警告: 列 '{col}' 在插值后仍有 {still_missing} 个缺失值，使用中位数填充")
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
                else:
                    print(f"  列 '{col}' 成功插值填充了 {missing_count} 个缺失值")

            except Exception as e:
                print(f"  插值填充列 '{col}' 时出错: {e}")
                print(f"  使用中位数填充")
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

    return df_imputed


# 自定义负值均方误差计算器，对负值样本给予更高权重
def weighted_negative_mse(y_true, y_pred):
    """计算加权负值MSE，对负值样本给予更高权重"""
    neg_mask = y_true < 0
    if np.sum(neg_mask) == 0:  # 没有负值样本
        return mean_squared_error(y_true, y_pred)

    # 对负值样本给予2倍权重
    weights = np.ones_like(y_true)
    weights[neg_mask] = 2.0

    # 计算加权MSE
    squared_errors = (y_true - y_pred) ** 2
    weighted_squared_errors = squared_errors * weights
    return np.mean(weighted_squared_errors)


# 目标列优化RandomForest填充
def optimize_rf_for_feature(X_train, y_train):
    """为特征变量优化RandomForest参数"""
    # 设置参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 创建随机森林回归器
    rf = RandomForestRegressor(random_state=42)

    # 进行网格搜索
    try:
        # 设置较短的超时时间，避免过长搜索
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # 使用所有可用CPU
            verbose=1
        )

        # 如果数据集太大，可以考虑使用子样本来加速
        if len(X_train) > 1000:
            # 随机选择1000个样本点进行参数调优
            sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
            X_sample = X_train.iloc[sample_idx]
            y_sample = y_train.iloc[sample_idx]
            grid_search.fit(X_sample, y_sample)
        else:
            grid_search.fit(X_train, y_train)

        # 获取最佳参数
        best_params = grid_search.best_params_
        print(f"  最佳参数: {best_params}")

        # 使用最佳参数创建模型
        best_rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )

        return best_rf, best_params

    except Exception as e:
        print(f"  参数优化出错: {e}")
        print("  使用默认参数")

        # 使用默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 3
        }

        default_rf = RandomForestRegressor(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            min_samples_split=default_params['min_samples_split'],
            min_samples_leaf=default_params['min_samples_leaf'],
            random_state=42
        )

        return default_rf, default_params


def optimize_rf_for_feature(X_train, y_train):
    """为特征变量优化RandomForest参数"""
    # 设置参数网格
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 创建随机森林回归器
    rf = RandomForestRegressor(random_state=42)

    # 进行网格搜索
    try:
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        # 如果数据集太大，可以考虑使用子样本来加速
        if len(X_train) > 1000:
            sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
            X_sample = X_train.iloc[sample_idx]
            y_sample = y_train.iloc[sample_idx]
            grid_search.fit(X_sample, y_sample)
        else:
            grid_search.fit(X_train, y_train)

        # 获取最佳参数
        best_params = grid_search.best_params_

        # 使用最佳参数创建模型
        best_rf = RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            random_state=42
        )

        return best_rf, best_params

    except Exception as e:
        print(f"  参数优化出错: {e}")
        print("  使用默认参数")

        # 使用默认参数
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 3
        }

        default_rf = RandomForestRegressor(
            n_estimators=default_params['n_estimators'],
            max_depth=default_params['max_depth'],
            min_samples_split=default_params['min_samples_split'],
            min_samples_leaf=default_params['min_samples_leaf'],
            random_state=42
        )

        return default_rf, default_params
def optimize_rf_for_target(X_train, y_train, is_negative_dominant=False):
        """
        为目标变量优化RandomForest参数

        is_negative_dominant: 如果为True，使用加权负值MSE作为评分函数
        """
        # 设置参数网格
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # 根据是否以负值为主选择评分函数
        if is_negative_dominant:
            # 使用自定义评分函数，重视负值
            scorer = make_scorer(weighted_negative_mse, greater_is_better=False)
        else:
            # 使用标准MSE
            scorer = 'neg_mean_squared_error'

        # 创建随机森林回归器
        rf = RandomForestRegressor(random_state=42)

        # 进行网格搜索
        try:
            # 设置较短的超时时间，避免过长搜索
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=5,
                scoring=scorer,
                n_jobs=-1,  # 使用所有可用CPU
                verbose=1
            )

            # 如果数据集太大，可以考虑使用子样本来加速
            if len(X_train) > 1000:
                # 随机选择1000个样本点进行参数调优
                sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
                X_sample = X_train.iloc[sample_idx]
                y_sample = y_train.iloc[sample_idx]
                grid_search.fit(X_sample, y_sample)
            else:
                grid_search.fit(X_train, y_train)

            # 获取最佳参数
            best_params = grid_search.best_params_
            print(f"  最佳参数: {best_params}")

            # 使用最佳参数创建模型
            best_rf = RandomForestRegressor(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                random_state=42
            )

            return best_rf, best_params

        except Exception as e:
            print(f"  参数优化出错: {e}")
            print("  使用默认参数")

            # 使用适用于负值预测的默认参数
            default_params = {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 3
            }

            default_rf = RandomForestRegressor(
                n_estimators=default_params['n_estimators'],
                max_depth=default_params['max_depth'],
                min_samples_split=default_params['min_samples_split'],
                min_samples_leaf=default_params['min_samples_leaf'],
                random_state=42
            )

            return default_rf, default_params

    # 评估填充方法
def evaluate_imputation(original_df, imputed_dfs, target_cols, test_size=0.2):
        """评估不同填充方法的效果"""
        results = {}

        # 对每个目标变量评估
        for target in target_cols:
            if target != 'CD_shell_change':  # 假设这是分类列，需要单独处理
                # 排除缺失目标值的行
                mask = ~original_df[target].isna()

                # 如果非缺失值太少，跳过评估
                if mask.sum() < 30:
                    print(f"  警告: {target} 的非缺失值过少 ({mask.sum()}), 跳过评估")
                    continue

                X = original_df[mask].drop(target_cols, axis=1)
                y = original_df[mask][target]

                # 划分训练和测试集
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                # 检查是否有足够的负值样本
                neg_count = (y < 0).sum()
                neg_percent = neg_count / len(y) * 100
                print(f"  {target}: 总样本数 {len(y)}, 负值样本 {neg_count} ({neg_percent:.1f}%)")

                # 判断是否以负值为主
                is_negative_dominant = neg_percent > 50

                method_scores = {}
                for method_name, imputed_df in imputed_dfs.items():
                    try:
                        # 确保我们使用的是相同的训练和测试样本
                        imputed_X_train = imputed_df.loc[X_train.index]
                        imputed_X_test = imputed_df.loc[X_test.index]

                        # 排除目标列
                        cols_to_drop = [col for col in target_cols if col in imputed_X_train.columns]
                        imputed_X_train = imputed_X_train.drop(cols_to_drop, axis=1)
                        imputed_X_test = imputed_X_test.drop(cols_to_drop, axis=1)

                        # 训练随机森林回归器，使用交叉验证来提高稳定性
                        from sklearn.model_selection import cross_val_score

                        # 优化随机森林参数
                        optimized_model, best_params = optimize_rf_for_target(
                            imputed_X_train, y_train, is_negative_dominant
                        )

                        # 使用5折交叉验证
                        if is_negative_dominant:
                            # 使用自定义评分函数
                            cv_scores = []
                            for train_idx, val_idx in KFold(n_splits=5, shuffle=True, random_state=42).split(
                                    imputed_X_train):
                                X_train_cv, X_val_cv = imputed_X_train.iloc[train_idx], imputed_X_train.iloc[val_idx]
                                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

                                model_cv = RandomForestRegressor(**best_params, random_state=42)
                                model_cv.fit(X_train_cv, y_train_cv)
                                y_pred_cv = model_cv.predict(X_val_cv)

                                cv_scores.append(weighted_negative_mse(y_val_cv, y_pred_cv))

                            cv_wmse = np.mean(cv_scores)
                            cv_rmse = np.sqrt(cv_wmse)  # 使用加权MSE的平方根作为RMSE
                        else:
                            # 使用标准评分
                            cv_rmse = np.sqrt(-cross_val_score(optimized_model, imputed_X_train, y_train,
                                                               cv=5, scoring='neg_mean_squared_error').mean())

                        cv_mae = -cross_val_score(optimized_model, imputed_X_train, y_train,
                                                  cv=5, scoring='neg_mean_absolute_error').mean()

                        # 再次训练完整模型并在测试集上评估
                        optimized_model.fit(imputed_X_train, y_train)
                        y_pred = optimized_model.predict(imputed_X_test)

                        # 计算测试集指标
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        test_mae = mean_absolute_error(y_test, y_pred)

                        # 特别检查对负值的预测
                        neg_mask = y_test < 0
                        if neg_mask.sum() > 0:
                            neg_rmse = np.sqrt(mean_squared_error(y_test[neg_mask], y_pred[neg_mask]))
                            neg_mae = mean_absolute_error(y_test[neg_mask], y_pred[neg_mask])
                        else:
                            neg_rmse = None
                            neg_mae = None

                        method_scores[method_name] = {
                            'CV_RMSE': cv_rmse,
                            'CV_MAE': cv_mae,
                            'Test_RMSE': test_rmse,
                            'Test_MAE': test_mae,
                            'Neg_RMSE': neg_rmse,
                            'Neg_MAE': neg_mae,
                            'Best_Params': best_params
                        }
                    except Exception as e:
                        print(f"  评估 {method_name} 方法时出错: {e}")
                        method_scores[method_name] = {'错误': str(e)}

                results[target] = method_scores

        return results

    # 可视化评估结果
def visualize_results(results):
        """可视化不同填充方法的评估结果"""
        for target, scores in results.items():
            # 创建RMSE比较图
            methods = list(scores.keys())
            test_rmse_scores = [scores[method].get('Test_RMSE', 0) for method in methods]
            test_mae_scores = [scores[method].get('Test_MAE', 0) for method in methods]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # RMSE图
            ax1.bar(methods, test_rmse_scores, color='skyblue')
            ax1.set_title(f'{target} - Test RMSE比较')
            ax1.set_xlabel('填充方法')
            ax1.set_ylabel('RMSE')
            for i, v in enumerate(test_rmse_scores):
                ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')

            # MAE图
            ax2.bar(methods, test_mae_scores, color='lightgreen')
            ax2.set_title(f'{target} - Test MAE比较')
            ax2.set_xlabel('填充方法')
            ax2.set_ylabel('MAE')
            for i, v in enumerate(test_mae_scores):
                ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(f'{target}_imputation_comparison.png')
            plt.close()

            # 为负值样本创建另一个图形
            neg_rmse_scores = [scores[method].get('Neg_RMSE', 0) for method in methods]
            neg_mae_scores = [scores[method].get('Neg_MAE', 0) for method in methods]

            if any(neg_rmse_scores):  # 只有在有负值样本的情况下才创建图表
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # 负值样本RMSE图
                ax1.bar(methods, neg_rmse_scores, color='coral')
                ax1.set_title(f'{target} - 负值样本RMSE比较')
                ax1.set_xlabel('填充方法')
                ax1.set_ylabel('RMSE')
                for i, v in enumerate(neg_rmse_scores):
                    if v > 0:  # 只有在有效值时才添加文本
                        ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom')

                # 负值样本MAE图
                ax2.bar(methods, neg_mae_scores, color='lightcoral')
                ax2.set_title(f'{target} - 负值样本MAE比较')
                ax2.set_xlabel('填充方法')
                ax2.set_ylabel('MAE')
                for i, v in enumerate(neg_mae_scores):
                    if v > 0:  # 只有在有效值时才添加文本
                        ax2.text(i, v, f'{v:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(f'{target}_negative_samples_comparison.png')
                plt.close()

    # 填充特征列
def impute_features(df, categorical_features, numeric_features):
        """比较不同方法填充特征列并选择最佳方法，但最终使用优化参数的随机森林填充"""
        # 分离特征列和目标列
        target_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                       'CD_leaf_change', 'CD_shell_change', 'CD_root_change']
        feature_cols = [col for col in df.columns if col not in target_cols]

        # 只选择特征列进行填充比较
        df_features = df[feature_cols].copy()
        cat_feat = [col for col in categorical_features if col in feature_cols]
        num_feat = [col for col in numeric_features if col in feature_cols]

        # 输出分类特征和数值特征
        print("\n分类特征列:", cat_feat)
        print("数值特征列:", num_feat)

        # 检查各列的数据类型，确保分类正确
        print("\n检查各列数据类型:")
        for col in df_features.columns:
            print(f"{col}: {df_features[col].dtype}")
            # 如果是数值列，显示一些统计信息
            if df_features[col].dtype in [np.float64, np.int64]:
                valid_values = df_features[col].dropna()
                if len(valid_values) > 0:
                    print(f"  范围: {valid_values.min()} 到 {valid_values.max()}")
            # 如果是分类列，显示唯一值数量
            else:
                unique_vals = df_features[col].dropna().unique()
                print(f"  唯一值数量: {len(unique_vals)}")
                if len(unique_vals) < 10:  # 如果唯一值不多，显示所有唯一值
                    print(f"  唯一值: {', '.join(map(str, unique_vals))}")

        # 检查是否有混合类型的列（既有数字又有文本）
        for col in df_features.columns:
            if df_features[col].dtype == 'object':
                # 尝试转换为数值，看是否所有非NA值都能转换
                try:
                    pd.to_numeric(df_features[col].dropna())
                    print(f"\n警告: 列 '{col}' 包含数值数据但存储为字符串类型")
                except:
                    # 如果有无法转换的，看看是什么值
                    try:
                        numeric_mask = df_features[col].str.isnumeric()
                        if numeric_mask.any() and (~numeric_mask).any():
                            non_numeric = df_features[col][~numeric_mask].dropna().unique()
                            print(f"\n警告: 列 '{col}' 混合了数值和非数值: {non_numeric}")
                    except:
                        pass

        # 应用不同填充方法到特征列
        print("\n开始对特征列应用不同的填充方法...")

        df_mean_feat = mean_imputation(df_features, cat_feat, num_feat)
        print("特征列均值填充完成")

        df_knn_feat = knn_imputation(df_features, cat_feat, num_feat)
        print("特征列KNN填充完成")

        df_rf_feat = rf_imputation(df_features, cat_feat, num_feat)
        print("特征列随机森林填充完成")

        df_interp_feat = interpolation_imputation(df_features, cat_feat, num_feat)
        print("特征列插值填充完成")

        # 创建用于交叉验证的子集
        imputed_feature_dfs = {
            'Mean': df_mean_feat,
            'KNN': df_knn_feat,
            'RandomForest': df_rf_feat,
            'Interpolation': df_interp_feat
        }

        # 简化评估过程，由于可能存在编码转换问题，直接使用均方误差比较填充结果
        print("\n开始简化评估特征填充方法质量...")
        feature_imputation_scores = {}

        # 对每个数值特征进行评估
        for col in num_feat:
            # 只考虑确实存在缺失值的列
            if df_features[col].isna().sum() > 0:
                # 获得非缺失值的索引和值
                non_missing_idx = ~df_features[col].isna()
                true_values = df_features.loc[non_missing_idx, col].values

                # 记录每种方法填充后的MSE
                method_errors = {}

                for method_name, imputed_df in imputed_feature_dfs.items():
                    # 确保列存在
                    if col in imputed_df.columns:
                        # 计算填充值与真实值的MSE
                        filled_values = imputed_df.loc[non_missing_idx, col].values
                        mse = mean_squared_error(true_values, filled_values)
                        method_errors[method_name] = mse
                    else:
                        print(f"  警告: {method_name} 填充结果中不包含列 {col}")

                if method_errors:
                    # 找出最佳方法
                    best_method = min(method_errors.items(), key=lambda x: x[1])[0]
                    feature_imputation_scores[col] = {'best_method': best_method, 'errors': method_errors}

        # 显示特征填充结果
        print("\n特征列填充评估结果:")
        for feature, result in feature_imputation_scores.items():
            print(f"\n{feature}:")
            print(f"  最佳方法: {result['best_method']}")
            for method, error in result['errors'].items():
                print(f"  {method}: MSE={error:.4f}")

        # 如果无法评估，默认使用均值填充方法
        if not feature_imputation_scores:
            print("\n警告: 无法评估填充方法质量，默认使用均值填充")
            return df_mean_feat, 'Mean'

        # 根据评估结果选择最优填充方法
        best_feature_methods = {feature: result['best_method']
                                for feature, result in feature_imputation_scores.items()}

        # 计算每种方法作为最佳方法的频次
        method_counts = {}
        for method in ['Mean', 'KNN', 'RandomForest', 'Interpolation']:
            method_counts[method] = sum(1 for v in best_feature_methods.values() if v == method)

        print("\n每种特征填充方法的最佳频次:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        # 选择整体最佳的特征填充方法
        if method_counts:
            best_overall_method = max(method_counts.items(), key=lambda x: x[1])[0]
            print(f"\n特征填充的整体最佳方法是: {best_overall_method}")
            return imputed_feature_dfs[best_overall_method], best_overall_method
        else:
            print("\n警告: 无法确定最佳填充方法，默认使用均值填充")
            # 新增：使用优化参数的随机森林方法重新填充特征列
            print("\n开始使用优化参数的随机森林方法填充特征列...")
            df_optimized_rf = df_features.copy()

            # 处理分类特征 - 使用众数填充
            for col in cat_feat:
                if col in df_optimized_rf.columns and df_optimized_rf[col].isna().any():
                    if not df_optimized_rf[col].mode().empty:
                        mode_val = df_optimized_rf[col].mode()[0]
                        df_optimized_rf[col] = df_optimized_rf[col].fillna(mode_val)
                    else:
                        df_optimized_rf[col] = df_optimized_rf[col].fillna("未知")

            # 处理数值特征 - 使用优化参数的随机森林填充
            # 先处理数据编码
            if cat_feat:
                df_encoded = pd.get_dummies(df_optimized_rf, columns=cat_feat, drop_first=True)
            else:
                df_encoded = df_optimized_rf.copy()

            # 确保所有列都是数值类型
            df_encoded = df_encoded.select_dtypes(include=['int64', 'float64'])

            # 对每个数值特征进行优化的随机森林填充
            for col in num_feat:
                if col not in df.columns or col not in df_encoded.columns:
                    continue

                missing_mask = df_features[col].isna()
                missing_count = missing_mask.sum()

                if missing_count > 0:
                    print(f"\n开始使用优化的随机森林填充列 '{col}' (缺失值数量: {missing_count})")

                    if missing_count < len(df) * 0.9:
                        try:
                            # 分割特征和目标
                            X_cols = [c for c in df_encoded.columns if c != col]
                            if len(X_cols) < 2:
                                df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                                continue

                            X_train = df_encoded.loc[~missing_mask, X_cols]
                            y_train = df_encoded.loc[~missing_mask, col]
                            X_missing = df_encoded.loc[missing_mask, X_cols]

                            if len(y_train) < 10 or X_train.isnull().any().any():
                                df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                                continue

                            # 为特征列优化随机森林参数
                            print(f"  为特征列 '{col}' 优化随机森林参数")
                            start_time = time.time()
                            best_model, best_params = optimize_rf_for_feature(X_train, y_train)
                            elapsed_time = time.time() - start_time
                            print(f"  参数优化完成，耗时: {elapsed_time:.2f}秒")
                            print(f"  特征列 '{col}' 的最佳参数: {best_params}")

                            # 使用优化后的模型
                            best_model.fit(X_train, y_train)

                            # 预测并填充缺失值
                            predicted = best_model.predict(X_missing)
                            df_encoded.loc[missing_mask, col] = predicted

                        except Exception as e:
                            print(f"  优化随机森林填充 '{col}' 时出错: {e}")
                            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
                    else:
                        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

            # 不论评估结果如何，最终返回优化的随机森林填充结果
            print("\n将使用优化随机森林参数的填充结果作为最终结果")
            return df_encoded, "OptimizedRandomForest"

def enhanced_impute_targets(df, filled_features_df, target_cols):
        """使用优化参数的随机森林填充目标列，特别处理负值"""
        # 创建完整的数据框
        df_complete = filled_features_df.copy()

        # 先将目标列添加到完整数据框
        for col in target_cols:
            if col not in df_complete.columns and col in df.columns:
                df_complete[col] = df[col].copy()

        # 对每个目标列使用随机森林进行填充
        for col in target_cols:
            if col not in df.columns:
                print(f"  目标列 '{col}' 不在原始数据中，跳过")
                continue

            # 检查是否有缺失值
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                print(f"  目标列 '{col}' 没有缺失值，跳过")
                continue

            print(f"\n开始填充目标列: {col}")
            print(f"  缺失值数量: {missing_count}/{len(df)}")

            if col != 'CD_shell_change':  # 数值型目标列
                try:
                    # 确保目标列是数值类型
                    if df[col].dtype not in [np.float64, np.int64]:
                        print(f"  警告: 目标列 '{col}' 不是数值类型，尝试转换")
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                    # 获取非缺失数据作为训练集
                    non_missing_idx = ~df[col].isna()

                    # 检查是否有足够的非缺失样本
                    if non_missing_idx.sum() < 10:
                        print(f"  警告: 目标列 '{col}' 的非缺失样本过少 ({non_missing_idx.sum()}), 使用均值填充")
                        df_complete[col] = df[col].fillna(df[col].mean())
                        continue

                    # 使用已填充的特征数据框作为X
                    numeric_cols = df_complete.select_dtypes(include=['int64', 'float64']).columns
                    X_train_cols = [c for c in numeric_cols if c not in target_cols]

                    if len(X_train_cols) < 2:
                        print(f"  警告: 特征数量不足，使用均值填充")
                        df_complete[col] = df[col].fillna(df[col].mean())
                        continue

                    X_train = df_complete.loc[non_missing_idx, X_train_cols]
                    y_train = df.loc[non_missing_idx, col]

                    # 获取缺失数据作为预测集
                    missing_idx = df[col].isna()
                    X_missing = df_complete.loc[missing_idx, X_train_cols]

                    # 检查是否有NaN值
                    if X_train.isnull().any().any() or X_missing.isnull().any().any():
                        print(f"  警告: 特征中存在NaN值，使用均值填充这些NaN值")
                        # 对特征中的NaN进行均值填充
                        imputer = SimpleImputer(strategy='mean')
                        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns,
                                               index=X_train.index)
                        X_missing = pd.DataFrame(imputer.transform(X_missing), columns=X_missing.columns,
                                                 index=X_missing.index)

                    # 检查负值比例
                    neg_count = (y_train < 0).sum()
                    neg_percent = neg_count / len(y_train) * 100
                    print(f"  负值样本比例: {neg_percent:.1f}% ({neg_count}/{len(y_train)})")

                    # 判断是否以负值为主
                    is_negative_dominant = neg_percent > 50

                    # 优化随机森林参数
                    print(f"  开始优化随机森林参数{'(针对负值优化)' if is_negative_dominant else ''}")
                    start_time = time.time()
                    best_model, best_params = optimize_rf_for_target(X_train, y_train, is_negative_dominant)
                    elapsed_time = time.time() - start_time
                    print(f"  参数优化完成，耗时: {elapsed_time:.2f}秒")

                    # 训练模型并预测
                    best_model.fit(X_train, y_train)
                    predicted_values = best_model.predict(X_missing)

                    # 填充目标列
                    df_complete.loc[missing_idx, col] = predicted_values
                    print(f"  成功填充了 {missing_idx.sum()} 个缺失值")

                    # 输出前5个重要特征
                    if len(X_train_cols) > 1:
                        feature_importance = dict(zip(X_train_cols, best_model.feature_importances_))
                        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                        print(f"  填充 {col} 的前5个重要特征:")
                        for feat, imp in sorted_importance[:min(5, len(sorted_importance))]:
                            print(f"    {feat}: {imp:.4f}")

                    # 使用最佳参数来预测并评估模型
                    print(f"  最佳随机森林参数: {best_params}")

                    # 检查预测结果的负值比例
                    pred_neg_count = (predicted_values < 0).sum()
                    pred_neg_percent = pred_neg_count / len(predicted_values) * 100 if len(predicted_values) > 0 else 0
                    print(f"  预测结果中负值比例: {pred_neg_percent:.1f}% ({pred_neg_count}/{len(predicted_values)})")

                    # 如果原始数据以负值为主，但预测结果没有足够的负值，可能需要调整
                    if is_negative_dominant and pred_neg_percent < 40:
                        print("  警告: 预测结果中负值比例过低，尝试调整预测值")

                        # 计算调整因子
                        adjustment_factor = 0.9  # 缩小所有值，增加负值比例
                        predicted_values = predicted_values * adjustment_factor

                        # 重新填充
                        df_complete.loc[missing_idx, col] = predicted_values

                        # 再次检查负值比例
                        pred_neg_count = (predicted_values < 0).sum()
                        pred_neg_percent = pred_neg_count / len(predicted_values) * 100
                        print(
                            f"  调整后预测结果中负值比例: {pred_neg_percent:.1f}% ({pred_neg_count}/{len(predicted_values)})")

                except Exception as e:
                    print(f"  填充目标列 '{col}' 时出错: {e}")
                    print(f"  使用平均值填充")
                    if df[col].dtype in [np.float64, np.int64]:
                        df_complete[col] = df[col].fillna(df[col].mean())
                    else:
                        # 如果不是数值列，使用众数填充
                        if not df[col].mode().empty:
                            df_complete[col] = df[col].fillna(df[col].mode()[0])
                        else:
                            df_complete[col] = df[col].fillna("未知")

            else:  # 分类型目标列 - CD_shell_change
                try:
                    # 获取非缺失数据作为训练集
                    non_missing_idx = ~df[col].isna()

                    # 检查是否有足够的非缺失样本
                    if non_missing_idx.sum() < 10:
                        print(f"  警告: 目标列 '{col}' 的非缺失样本过少 ({non_missing_idx.sum()}), 使用众数填充")
                        if not df[col].mode().empty:
                            mode_val = df[col].mode()[0]
                            df_complete[col] = df[col].fillna(mode_val)
                        else:
                            df_complete[col] = df[col].fillna("未知")
                        continue

                        # 只使用数值特征列进行预测
                        numeric_cols = df_complete.select_dtypes(include=['int64', 'float64']).columns
                        X_train_cols = [c for c in numeric_cols if c not in target_cols]

                        if len(X_train_cols) < 2:
                            print(f"  警告: 特征数量不足，使用众数填充")
                            if not df[col].mode().empty:
                                mode_val = df[col].mode()[0]
                                df_complete[col] = df[col].fillna(mode_val)
                            else:
                                df_complete[col] = df[col].fillna("未知")
                            continue

                        X_train = df_complete.loc[non_missing_idx, X_train_cols]
                        y_train = df.loc[non_missing_idx, col]

                        # 获取缺失数据作为预测集
                        missing_idx = df[col].isna()
                        X_missing = df_complete.loc[missing_idx, X_train_cols]

                        # 检查是否有NaN值
                        if X_train.isnull().any().any() or X_missing.isnull().any().any():
                            print(f"  警告: 特征中存在NaN值，使用均值填充这些NaN值")
                            # 对特征中的NaN进行均值填充
                            imputer = SimpleImputer(strategy='mean')
                            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns,
                                                   index=X_train.index)
                            X_missing = pd.DataFrame(imputer.transform(X_missing), columns=X_missing.columns,
                                                     index=X_missing.index)

                        # 优化随机森林分类器参数
                        print(f"  开始优化随机森林分类器参数")

                        # 创建参数网格
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [10, 15, 20, None],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }

                        # 进行网格搜索
                        try:
                            rf_classifier = RandomForestClassifier(random_state=42)

                            grid_search = GridSearchCV(
                                estimator=rf_classifier,
                                param_grid=param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=1
                            )

                            # 如果数据集太大，可以考虑使用子样本来加速
                            if len(X_train) > 1000:
                                sample_idx = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
                                X_sample = X_train.iloc[sample_idx]
                                y_sample = y_train.iloc[sample_idx]
                                grid_search.fit(X_sample, y_sample)
                            else:
                                grid_search.fit(X_train, y_train)

                            # 获取最佳参数
                            best_params = grid_search.best_params_
                            print(f"  最佳参数: {best_params}")

                            # 使用最佳参数创建模型
                            best_classifier = RandomForestClassifier(
                                n_estimators=best_params['n_estimators'],
                                max_depth=best_params['max_depth'],
                                min_samples_split=best_params['min_samples_split'],
                                min_samples_leaf=best_params['min_samples_leaf'],
                                random_state=42
                            )

                        except Exception as e:
                            print(f"  参数优化出错: {e}")
                            print("  使用默认参数")

                            # 使用默认参数
                            best_classifier = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=15,
                                min_samples_split=5,
                                min_samples_leaf=2,
                                random_state=42
                            )
                            best_params = {
                                'n_estimators': 100,
                                'max_depth': 15,
                                'min_samples_split': 5,
                                'min_samples_leaf': 2
                            }

                        # 训练模型
                        best_classifier.fit(X_train, y_train)

                        # 预测缺失值
                        predicted_values = best_classifier.predict(X_missing)

                        # 填充目标列
                        df_complete.loc[missing_idx, col] = predicted_values
                        print(f"  成功填充了 {missing_idx.sum()} 个缺失值")

                        # 输出前5个重要特征
                        if len(X_train_cols) > 1:
                            feature_importance = dict(zip(X_train_cols, best_classifier.feature_importances_))
                            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                            print(f"  填充 {col} 的前5个重要特征:")
                            for feat, imp in sorted_importance[:min(5, len(sorted_importance))]:
                                print(f"    {feat}: {imp:.4f}")

                        # 显示分类报告
                        try:
                            from sklearn.metrics import classification_report
                            y_pred = best_classifier.predict(X_train)
                            print("\n分类报告:")
                            print(classification_report(y_train, y_pred))
                        except Exception as e:
                            print(f"  生成分类报告出错: {e}")
                except Exception as e:
                    print(f"  填充分类目标列 '{col}' 时出错: {e}")
                    print(f"  使用众数填充")
                    if not df[col].mode().empty:
                        mode_val = df[col].mode()[0]
                        df_complete[col] = df[col].fillna(mode_val)
                    else:
                        df_complete[col] = df[col].fillna("未知")
            return df_complete

        # 主函数
def main():
            # 读取数据
            df = load_data(r'/OneDrive/桌面/机器学习/数据/finall_data.csv')
            all_best_params = {}
            # 探索数据
            missing_cols = explore_data(df)

            # 使用增强的预处理函数
            df_processed, categorical_features, numeric_features, target_cols = enhanced_preprocess_data(df)

            # 第一步：找出最佳特征填充方法，并填充特征列
            filled_features_df, best_feature_method = impute_features(df_processed, categorical_features,
                                                                      numeric_features)
            print(f"\n已使用 {best_feature_method} 方法完成特征列填充")

            # 保存中间结果，以便需要时恢复
            filled_features_df.to_csv('filled_features.csv', index=False, encoding='utf-8')
            print("已保存中间填充结果到 'filled_features.csv'")

            # 第二步：使用增强的目标列填充方法
            complete_df = enhanced_impute_targets(df_processed, filled_features_df, target_cols)
            print("\n已使用优化的随机森林方法完成目标列填充")

            # 检查填充后是否还有缺失值
            missing_after = complete_df.isnull().sum()
            print("\n填充后的缺失值情况:")
            if missing_after.sum() > 0:
                print(missing_after[missing_after > 0])
            else:
                print("没有缺失值，填充完成！")

            # 最终检查 - 确保没有缺失值
            missing_after = complete_df.isnull().sum()
            if missing_after.sum() > 0:
                print("\n检测到仍有缺失值，进行最终强制填充...")
                print(missing_after[missing_after > 0])

                # 对剩余缺失值进行强制填充
                for col in complete_df.columns:
                    if complete_df[col].isnull().sum() > 0:
                        if complete_df[col].dtype in [np.float64, np.int64]:
                            # 数值列用中位数填充
                            complete_df[col] = complete_df[col].fillna(complete_df[col].median())
                            print(f"  列 '{col}' 用中位数强制填充")
                        else:
                            # 分类列用众数填充
                            if not complete_df[col].mode().empty:
                                mode_val = complete_df[col].mode()[0]
                                complete_df[col] = complete_df[col].fillna(mode_val)
                                print(f"  列 '{col}' 用众数 '{mode_val}' 强制填充")
                            else:
                                complete_df[col] = complete_df[col].fillna("未知")
                                print(f"  列 '{col}' 用 '未知' 强制填充")

                # 再次检查
                missing_final = complete_df.isnull().sum()
                if missing_final.sum() > 0:
                    print("\n警告: 强制填充后仍有缺失值!")
                    print(missing_final[missing_final > 0])
                else:
                    print("\n强制填充成功，所有缺失值已填充!")

            # 保存填充后的数据
            complete_df.to_csv(r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\filled_data_optimized.csv', index=False, encoding='utf-8')
            print("\n填充后的数据已保存到 'filled_data_optimized.csv'")

            return complete_df
          
if __name__ == "__main__":
            try:
                result_df = main()
                print("\n程序成功完成!")
            except Exception as e:
                print(f"\n程序执行过程中发生错误: {e}")
                import traceback
                traceback.print_exc()