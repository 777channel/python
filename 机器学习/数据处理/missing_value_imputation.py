import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns


def mean_imputation(data, columns_to_impute):
    """改进的均值填充，考虑正负值分布"""
    data_imputed = data.copy()

    # 输出列，通常有高负值偏向
    output_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    for column in columns_to_impute:
        # 获取缺失值掩码
        mask = data[column].isnull()
        if not mask.any():
            continue

        print(f"\n处理列: {column}")

        # 检查是否为输出列
        is_output_col = column in output_cols

        # 特殊处理高度偏向负值的列
        if is_output_col:
            valid_values = data[column].dropna()
            neg_ratio = (valid_values < 0).mean()
            pos_ratio = (valid_values > 0).mean()
            zero_ratio = (valid_values == 0).mean()
            print(f"列 {column} 的分布: 负值={neg_ratio:.2%}, 正值={pos_ratio:.2%}, 零值={zero_ratio:.2%}")

            # 根据原始分布填充值
            missing_idx = np.where(mask)[0]
            values_to_fill = np.zeros(len(missing_idx))

            # 计算不同值区间的均值
            neg_mean = valid_values[valid_values < 0].mean() if (valid_values < 0).any() else -0.1
            pos_mean = valid_values[valid_values > 0].mean() if (valid_values > 0).any() else 0.1
            zero_mean = 0

            # 按原始分布比例分配缺失值
            if neg_ratio > 0:
                n_neg = int(neg_ratio * len(missing_idx))
                values_to_fill[:n_neg] = neg_mean

            if pos_ratio > 0:
                n_pos = int(pos_ratio * len(missing_idx))
                start_idx = int(neg_ratio * len(missing_idx))
                end_idx = min(start_idx + n_pos, len(missing_idx))
                values_to_fill[start_idx:end_idx] = pos_mean

            if zero_ratio > 0 and (neg_ratio + pos_ratio) < 1:
                start_idx = int((neg_ratio + pos_ratio) * len(missing_idx))
                values_to_fill[start_idx:] = zero_mean

            # 随机打乱，避免填充顺序偏差
            np.random.shuffle(values_to_fill)
            data_imputed.loc[mask, column] = values_to_fill
        else:
            # 分别处理正值和负值
            mask_positive = data[column] >= 0
            mask_negative = data[column] < 0

            # 对正值使用正值的均值填充
            positive_mean = data.loc[mask_positive, column].mean()
            if pd.notna(positive_mean):
                data_imputed.loc[mask_positive & data[column].isna(), column] = positive_mean

            # 对负值使用负值的均值填充
            negative_mean = data.loc[mask_negative, column].mean()
            if pd.notna(negative_mean):
                data_imputed.loc[mask_negative & data[column].isna(), column] = negative_mean

            # 如果还有未填充的值，使用总体均值
            if data_imputed[column].isna().any():
                total_mean = data[column].mean()
                if pd.notna(total_mean):
                    data_imputed.loc[data_imputed[column].isna(), column] = total_mean

    return data_imputed


def knn_imputation(data, columns_to_impute):
    """改进的KNN填充，严格按类别填充，只使用同类特征"""
    data_imputed = data.copy()

    # 定义特征类别 - 使用正确的列名
    biochar_features = ['pH', 'material_type_code', 'temperature']  # 生物炭特性
    soil_features = ['CEC', 'pHsoil', 'organicmatter', 'log_Cdcontent']  # 土壤特性
    rice_features = ['variety_binary', 'irrigation_type']  # 水稻特性和实验条件

    # 定义输出列
    output_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    # 将数据标准化，保持正负值的相对关系
    scaler_data = data[columns_to_impute].copy()
    for column in columns_to_impute:
        if not scaler_data[column].isna().all():
            mean_val = scaler_data[column].mean()
            std_val = scaler_data[column].std()
            if std_val != 0:
                scaler_data.loc[:, column] = (scaler_data[column] - mean_val) / std_val

    # 基本KNN参数
    base_n_neighbors = 5

    for column in columns_to_impute:
        mask = data[column].isnull()
        if not mask.any():
            continue

        print(f"\n处理列: {column}")

        # 确定当前列属于哪个类别，并严格只选择对应类别内的其他特征进行填充
        if column in biochar_features:
            features_to_use = [f for f in biochar_features if f != column and f in data.columns]
            print(f"使用生物炭特性填充: {features_to_use}")
        elif column in soil_features:
            features_to_use = [f for f in soil_features if f != column and f in data.columns]
            print(f"使用土壤特性填充: {features_to_use}")
        elif column in rice_features:
            features_to_use = [f for f in rice_features if f != column and f in data.columns]
            print(f"使用水稻特性填充: {features_to_use}")
        elif column in output_cols:
            # 对于输出列，使用所有特征列（但不使用其他输出列）
            features_to_use = biochar_features + soil_features + rice_features
            features_to_use = [f for f in features_to_use if f in data.columns]
            print(f"使用所有特征列填充输出列: {features_to_use[:5]}...")
        else:
            # 如果不属于定义的类别，使用所有非输出列特征
            features_to_use = [f for f in columns_to_impute
                               if f != column and f not in output_cols]
            print(f"使用所有非输出特征填充: {features_to_use[:5]}...")

        # 检查是否有足够的特征用于填充
        if len(features_to_use) < 2:
            print(f"警告：类别内特征不足以填充 {column}，使用均值/中位数填充")
            if pd.api.types.is_numeric_dtype(data[column]):
                data_imputed.loc[mask, column] = data[column].median()
            else:
                mode_val = data[column].mode().iloc[0] if not data[column].mode().empty else None
                data_imputed.loc[mask, column] = mode_val
            continue

        # 检查是否为输出列
        is_output_col = column in output_cols

        if is_output_col:
            # 分析正负值分布
            valid_values = data[column].dropna()
            neg_ratio = (valid_values < 0).mean()
            pos_ratio = (valid_values > 0).mean()
            print(f"列 {column} 的分布: 负值={neg_ratio:.2%}, 正值={pos_ratio:.2%}")

            # 对于高度偏向负值的列，使用特殊处理
            if neg_ratio > 0.7:
                # 1. 先分类预测正负，再预测具体值
                # 准备数据
                known_data = scaler_data[~mask]
                missing_data = scaler_data[mask]

                # 特征矩阵 - 只使用指定特征
                feature_cols = [f for f in features_to_use if f in known_data.columns]
                if not feature_cols:
                    print(f"警告：没有找到可用特征填充 {column}，使用备选填充方法")
                    data_imputed.loc[mask, column] = valid_values[valid_values < 0].mean()
                    continue

                X_train = known_data[feature_cols]
                y_sign = (known_data[column] >= 0).astype(int)  # 是否为正值
                X_missing = missing_data[feature_cols]

                # 填充特征矩阵中的缺失值
                temp_imputer = SimpleImputer(strategy='mean')
                X_train = pd.DataFrame(
                    temp_imputer.fit_transform(X_train),
                    index=X_train.index,
                    columns=X_train.columns
                )
                X_missing = pd.DataFrame(
                    temp_imputer.transform(X_missing),
                    index=X_missing.index,
                    columns=X_missing.columns
                )

                # 训练KNN分类器预测符号
                from sklearn.neighbors import KNeighborsClassifier
                sign_imputer = KNeighborsClassifier(n_neighbors=min(base_n_neighbors, len(y_sign)))
                sign_imputer.fit(X_train, y_sign)
                sign_pred = sign_imputer.predict(X_missing)

                # 对于正负值分别训练KNN回归模型
                neg_mask = known_data[column] < 0
                pos_mask = known_data[column] >= 0

                # 预测具体值
                predicted_values = np.zeros(len(X_missing))

                # 对负值使用负值模型
                if neg_mask.sum() > base_n_neighbors:
                    X_train_neg = X_train.loc[neg_mask]
                    y_train_neg = known_data.loc[neg_mask, column].values

                    neg_imputer = KNeighborsRegressor(n_neighbors=min(base_n_neighbors, len(y_train_neg)))
                    neg_imputer.fit(X_train_neg, y_train_neg)

                    # 找出预测为负值的样本
                    neg_pred_mask = sign_pred == 0
                    if any(neg_pred_mask):
                        predicted_values[neg_pred_mask] = neg_imputer.predict(X_missing.iloc[neg_pred_mask])

                # 对正值使用正值模型
                if pos_mask.sum() > base_n_neighbors:
                    X_train_pos = X_train.loc[pos_mask]
                    y_train_pos = known_data.loc[pos_mask, column].values

                    pos_imputer = KNeighborsRegressor(n_neighbors=min(base_n_neighbors, len(y_train_pos)))
                    pos_imputer.fit(X_train_pos, y_train_pos)

                    # 找出预测为正值的样本
                    pos_pred_mask = sign_pred == 1
                    if any(pos_pred_mask):
                        predicted_values[pos_pred_mask] = pos_imputer.predict(X_missing.iloc[pos_pred_mask])

                # 如果某些样本没有被预测（由于数据不足），使用整体模型
                unpredicted_mask = predicted_values == 0
                if any(unpredicted_mask):
                    y_train = known_data[column].values
                    overall_imputer = KNeighborsRegressor(n_neighbors=min(base_n_neighbors, len(y_train)))
                    overall_imputer.fit(X_train, y_train)
                    predicted_values[unpredicted_mask] = overall_imputer.predict(X_missing.iloc[unpredicted_mask])

                # 反标准化
                if not data[column].isna().all():
                    mean_val = data[column].mean()
                    std_val = data[column].std()
                    if std_val != 0:
                        predicted_values = (predicted_values * std_val) + mean_val

                # 后处理：确保负值比例符合原始分布
                pred_neg_ratio = (predicted_values < 0).mean()

                if pred_neg_ratio < neg_ratio * 0.9:  # 允许10%误差
                    # 计算需要调整的数量
                    n_to_adjust = int((neg_ratio - pred_neg_ratio) * len(predicted_values))

                    # 找出最小的正值
                    pos_indices = np.where(predicted_values >= 0)[0]
                    if len(pos_indices) > 0:
                        # 按值排序
                        sorted_indices = pos_indices[np.argsort(predicted_values[pos_indices])]
                        to_adjust = sorted_indices[:min(n_to_adjust, len(sorted_indices))]

                        # 计算负值平均值
                        neg_mean = data.loc[data[column] < 0, column].mean()

                        # 调整为负值
                        for idx in to_adjust:
                            predicted_values[idx] = neg_mean * 0.8  # 略高于平均负值

                data_imputed.loc[mask, column] = predicted_values
            else:
                # 非高度偏向负值的列使用类别内标准KNN
                standard_category_knn_imputation(data, data_imputed, scaler_data, column, mask, base_n_neighbors,
                                                 features_to_use)
        else:
            # 非输出列使用类别内标准KNN
            standard_category_knn_imputation(data, data_imputed, scaler_data, column, mask, base_n_neighbors,
                                             features_to_use)

    return data_imputed


def standard_category_knn_imputation(data, data_imputed, scaler_data, column, mask, n_neighbors, features_to_use):
    """标准KNN填充方法，但仅使用指定的类别内特征"""
    if mask.any():
        # 检查是否有足够的特征
        if not features_to_use:
            print(f"警告：没有可用特征填充 {column}，使用均值填充")
            if pd.api.types.is_numeric_dtype(data[column]):
                data_imputed.loc[mask, column] = data[column].mean()
            else:
                mode_val = data[column].mode().iloc[0] if not data[column].mode().empty else None
                data_imputed.loc[mask, column] = mode_val
            return

        known_data = scaler_data[~mask]
        missing_data = scaler_data[mask]

        # 准备特征矩阵，只使用指定的类别内特征
        feature_cols = [f for f in features_to_use if f in known_data.columns]
        X_train = known_data[feature_cols]
        y_train = known_data[column]
        X_missing = missing_data[feature_cols]

        # 填充特征矩阵中的缺失值
        temp_imputer = SimpleImputer(strategy='mean')
        X_train = pd.DataFrame(
            temp_imputer.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_missing = pd.DataFrame(
            temp_imputer.transform(X_missing),
            index=X_missing.index,
            columns=X_missing.columns
        )

        # 训练和预测
        if len(y_train) > 0 and len(feature_cols) > 0:
            imputer = KNeighborsRegressor(n_neighbors=min(n_neighbors, len(y_train)))
            imputer.fit(X_train, y_train)
            predicted_values = imputer.predict(X_missing)

            # 反标准化
            if not data[column].isna().all():
                mean_val = data[column].mean()
                std_val = data[column].std()
                if std_val != 0:
                    predicted_values = (predicted_values * std_val) + mean_val

            data_imputed.loc[mask, column] = predicted_values
        else:
            print(f"警告：无法为 {column} 训练KNN模型，使用均值填充")
            if pd.api.types.is_numeric_dtype(data[column]):
                data_imputed.loc[mask, column] = data[column].mean()
            else:
                mode_val = data[column].mode().iloc[0] if not data[column].mode().empty else None
                data_imputed.loc[mask, column] = mode_val


def correlation_based_imputation(data, columns_to_impute):
    """基于相关性的填充策略：先填充特征，再用特征填充输出"""
    data_imputed = data.copy()

    # 定义输出列
    output_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    # 区分特征列和输出列
    feature_columns = [col for col in columns_to_impute if col not in output_cols]
    output_columns = [col for col in columns_to_impute if col in output_cols]

    print(f"特征列: {feature_columns}")
    print(f"输出列: {output_columns}")

    # 计算特征列之间的相关系数矩阵
    feature_corr_matrix = data[feature_columns].corr(method='pearson')

    # 第一步：填充特征列
    print("\n===== 第一步：基于特征间相关性填充特征列 =====")
    for column in feature_columns:
        mask = data[column].isnull()
        if not mask.any():
            print(f"列 {column} 没有缺失值，跳过")
            continue

        print(f"\n处理特征列: {column}")

        # 为每个列选择相关性最强的特征
        correlation_threshold = 0.2  # 相关系数阈值
        top_n = 5  # 最多使用前5个相关性最强的特征

        # 获取与当前列相关性最强的特征列（不包括输出列）
        if column in feature_corr_matrix.index:
            # 获取相关性的绝对值，并按降序排序
            corr_series = feature_corr_matrix[column].drop(column)  # 排除自己
            abs_corr = corr_series.abs().sort_values(ascending=False)

            # 选择相关性超过阈值的前N个特征
            strong_correlations = abs_corr[abs_corr > correlation_threshold].head(top_n)
            features_to_use = strong_correlations.index.tolist()

            # 打印选择的特征及其相关系数
            print(f"根据相关性选择的特征:")
            for feat in features_to_use:
                print(f"  - {feat}: {corr_series[feat]:.4f}")
        else:
            # 如果没有相关性数据，使用所有其他特征列
            features_to_use = [f for f in feature_columns if f != column]
            print(f"无法获取相关性数据，使用所有其他特征")

        # 检查是否有足够的特征
        if len(features_to_use) < 2:
            print(f"警告：相关特征不足以填充 {column}，添加更多特征")
            # 添加更多特征直到达到最小数量
            additional_features = [f for f in feature_columns
                                   if f != column and f not in features_to_use]
            features_to_use.extend(additional_features[:max(0, 2 - len(features_to_use))])

        # 处理特征中可能存在的缺失值
        features_to_use_cleaned = []
        for feature in features_to_use:
            if feature in data.columns:
                if data[feature].isnull().sum() / len(data) < 0.3:  # 只使用缺失率低于30%的特征
                    features_to_use_cleaned.append(feature)
                    # 对特征中的缺失值进行简单填充
                    if data[feature].isnull().any():
                        if pd.api.types.is_numeric_dtype(data[feature]):
                            data.loc[:, feature] = data[feature].fillna(data[feature].median())
                        else:
                            data.loc[:, feature] = data[feature].fillna(
                                data[feature].mode().iloc[0] if not data[feature].mode().empty else "Unknown")

        features_to_use = features_to_use_cleaned
        print(f"清洗后使用的特征: {features_to_use}")

        # 如果没有可用的特征，使用简单均值填充
        if not features_to_use:
            print(f"警告：没有可用特征填充 {column}，使用均值填充")
            if pd.api.types.is_numeric_dtype(data[column]):
                data_imputed.loc[mask, column] = data[column].mean()
            else:
                most_common = data[column].mode().iloc[0] if not data[column].mode().empty else None
                data_imputed.loc[mask, column] = most_common
            continue

        # 使用线性回归填充特征列
        try:
            X_train = data.loc[~mask, features_to_use]
            y_train = data.loc[~mask, column]

            # 处理分类特征
            from sklearn.preprocessing import OneHotEncoder
            categorical_features = [f for f in features_to_use if not pd.api.types.is_numeric_dtype(data[f])]

            if categorical_features:
                # 创建只包含数值特征的DataFrame
                X_numeric = X_train.select_dtypes(include=['number'])

                # 使用OneHotEncoder处理分类特征
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_cats = encoder.fit_transform(X_train[categorical_features])
                encoded_df = pd.DataFrame(encoded_cats, index=X_train.index,
                                          columns=[f"{col}_{i}" for col in categorical_features
                                                   for i in range(encoded_cats.shape[1])])

                # 合并数值特征和编码后的分类特征
                X_train_processed = pd.concat([X_numeric, encoded_df], axis=1)

                # 对测试数据做相同处理
                X_test = data.loc[mask, features_to_use]
                X_test_numeric = X_test.select_dtypes(include=['number'])
                encoded_test = encoder.transform(X_test[categorical_features])
                encoded_test_df = pd.DataFrame(encoded_test, index=X_test.index,
                                               columns=encoded_df.columns)
                X_test_processed = pd.concat([X_test_numeric, encoded_test_df], axis=1)
            else:
                X_train_processed = X_train
                X_test_processed = data.loc[mask, features_to_use]

            # 如果特征和样本数量足够，使用线性回归
            if len(X_train_processed.columns) > 0 and len(X_train_processed) > 5:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train_processed, y_train)
                predicted_values = model.predict(X_test_processed)

                # 查看系数了解特征重要性
                if len(model.coef_) > 0:
                    coef_series = pd.Series(model.coef_, index=X_train_processed.columns)
                    print("\n线性回归特征系数(前5个):")
                    print(coef_series.sort_values(ascending=False).head())

                # 确保预测值在合理范围内
                if column == 'organicmatter':
                    predicted_values = np.clip(predicted_values, data[column].min(), data[column].max())

                data_imputed.loc[mask, column] = predicted_values
                print(f"成功使用线性回归填充 {column}")
            else:
                # 如果特征或样本不足，使用均值填充
                print(f"特征或样本不足，使用均值填充 {column}")
                data_imputed.loc[mask, column] = data[column].mean()
        except Exception as e:
            print(f"线性回归填充失败: {str(e)}")
            # 回退到简单均值填充
            data_imputed.loc[mask, column] = data[column].mean()

    # 更新数据，确保第二步使用已填充的特征
    for col in feature_columns:
        data.loc[:, col] = data_imputed[col]

    # 第二步：使用所有填充后的特征列来填充输出列
    print("\n===== 第二步：使用所有特征列填充输出列 =====")
    for column in output_columns:
        mask = data[column].isnull()
        if not mask.any():
            print(f"列 {column} 没有缺失值，跳过")
            continue

        print(f"\n处理输出列: {column}")

        # 分析正负值分布
        valid_values = data[column].dropna()
        neg_ratio = (valid_values < 0).mean()
        pos_ratio = (valid_values > 0).mean()
        print(f"列 {column} 的分布: 负值={neg_ratio:.2%}, 正值={pos_ratio:.2%}")

        # 使用所有已填充的特征列
        features_to_use = feature_columns
        print(f"使用所有特征列预测: {features_to_use[:5]}...")

        # 使用线性回归填充输出列
        try:
            X_train = data.loc[~mask, features_to_use]
            y_train = data.loc[~mask, column]

            # 处理分类特征
            from sklearn.preprocessing import OneHotEncoder
            categorical_features = [f for f in features_to_use if not pd.api.types.is_numeric_dtype(data[f])]

            if categorical_features:
                # 处理分类特征逻辑（与特征列相同）
                X_numeric = X_train.select_dtypes(include=['number'])
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoded_cats = encoder.fit_transform(X_train[categorical_features])
                encoded_df = pd.DataFrame(encoded_cats, index=X_train.index,
                                          columns=[f"{col}_{i}" for col in categorical_features
                                                   for i in range(encoded_cats.shape[1])])
                X_train_processed = pd.concat([X_numeric, encoded_df], axis=1)

                X_test = data.loc[mask, features_to_use]
                X_test_numeric = X_test.select_dtypes(include=['number'])
                encoded_test = encoder.transform(X_test[categorical_features])
                encoded_test_df = pd.DataFrame(encoded_test, index=X_test.index,
                                               columns=encoded_df.columns)
                X_test_processed = pd.concat([X_test_numeric, encoded_test_df], axis=1)
            else:
                X_train_processed = X_train
                X_test_processed = data.loc[mask, features_to_use]

            # 针对高度偏向负值的列使用特殊处理
            high_neg_bias = neg_ratio > 0.7
            if high_neg_bias:
                print(f"检测到高度偏向负值的列: {column}，使用分别建模处理")

                # 分离正负值样本
                neg_mask = y_train < 0
                pos_mask = y_train >= 0

                # 负值模型
                neg_model = None
                if neg_mask.sum() > 5:
                    from sklearn.linear_model import LinearRegression
                    neg_model = LinearRegression()
                    neg_model.fit(X_train_processed[neg_mask], y_train[neg_mask])

                # 正值模型
                pos_model = None
                if pos_mask.sum() > 5:
                    from sklearn.linear_model import LinearRegression
                    pos_model = LinearRegression()
                    pos_model.fit(X_train_processed[pos_mask], y_train[pos_mask])

                # 预测正负性
                from sklearn.linear_model import LogisticRegression
                sign_model = LogisticRegression(class_weight='balanced')
                sign_target = (y_train >= 0).astype(int)
                sign_model.fit(X_train_processed, sign_target)

                # 预测符号
                pred_signs = sign_model.predict(X_test_processed)

                # 根据符号使用对应模型预测值
                predicted_values = np.zeros(len(X_test_processed))

                for i, is_positive in enumerate(pred_signs):
                    if is_positive == 1 and pos_model is not None:
                        predicted_values[i] = pos_model.predict([X_test_processed.iloc[i]])[0]
                    elif is_positive == 0 and neg_model is not None:
                        predicted_values[i] = neg_model.predict([X_test_processed.iloc[i]])[0]
                    else:
                        # 回退策略
                        if is_positive == 1:
                            predicted_values[i] = y_train[y_train >= 0].mean() if pos_mask.any() else 0.1
                        else:
                            predicted_values[i] = y_train[y_train < 0].mean() if neg_mask.any() else -0.1

                # 检查预测的负值比例
                current_neg_ratio = (predicted_values < 0).mean()
                print(f"预测值的负值比例: {current_neg_ratio:.2%}, 目标负值比例: {neg_ratio:.2%}")

                # 调整负值比例
                if current_neg_ratio < neg_ratio * 0.9:  # 允许10%误差
                    n_to_adjust = int((neg_ratio - current_neg_ratio) * len(predicted_values))
                    print(f"调整 {n_to_adjust} 个值以匹配负值比例")

                    # 找出最接近0的正值
                    pos_indices = np.where(predicted_values >= 0)[0]
                    if len(pos_indices) > 0:
                        # 按值排序
                        sorted_indices = pos_indices[np.argsort(predicted_values[pos_indices])]
                        to_adjust = sorted_indices[:min(n_to_adjust, len(sorted_indices))]

                        # 计算负值平均值
                        neg_mean = y_train[y_train < 0].mean()

                        # 调整为负值
                        for idx in to_adjust:
                            predicted_values[idx] = neg_mean * 0.8  # 稍微高于平均负值
            else:
                # 对于不太偏向负值的列，使用单一线性模型
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train_processed, y_train)
                predicted_values = model.predict(X_test_processed)

                # 查看系数了解特征重要性
                if len(model.coef_) > 0:
                    coef_series = pd.Series(model.coef_, index=X_train_processed.columns)
                    print("\n线性回归特征系数(前5个):")
                    print(coef_series.sort_values(ascending=False).head())

            data_imputed.loc[mask, column] = predicted_values
            print(f"成功使用线性回归填充 {column}")

        except Exception as e:
            print(f"线性回归填充输出列失败: {str(e)}")
            # 使用考虑正负值分布的均值填充作为备选
            valid_values = data[column].dropna()
            neg_values = valid_values[valid_values < 0]
            pos_values = valid_values[valid_values >= 0]

            if neg_ratio > 0.7:  # 高度偏向负值
                # 大部分使用负值均值填充
                neg_mean = neg_values.mean()
                data_imputed.loc[mask, column] = neg_mean

                # 少量使用正值填充以匹配分布
                if pos_ratio > 0:
                    pos_mean = pos_values.mean() if not pos_values.empty else 0.1
                    n_pos = int(pos_ratio * mask.sum())
                    if n_pos > 0:
                        random_indices = np.random.choice(
                            np.where(mask)[0],
                            size=n_pos,
                            replace=False
                        )
                        if len(random_indices) > 0:
                            data_imputed.loc[random_indices, column] = pos_mean
            else:
                # 使用总体均值
                data_imputed.loc[mask, column] = data[column].mean()

            print(f"已使用备选方法填充 {mask.sum()} 个值")

    # 最终检查，确保所有值都已填充
    for column in columns_to_impute:
        if data_imputed[column].isnull().any():
            missing_count = data_imputed[column].isnull().sum()
            print(f"警告：最终检查发现列 {column} 仍有 {missing_count} 个未填充的值，使用全局均值填充")

            if pd.api.types.is_numeric_dtype(data_imputed[column]):
                data_imputed[column].fillna(data_imputed[column].mean(), inplace=True)
            else:
                mode_val = data_imputed[column].mode().iloc[0] if not data_imputed[column].mode().empty else None
                data_imputed[column].fillna(mode_val, inplace=True)

    return data_imputed

def interpolation_imputation(data, columns_to_impute):
    """改进的插值填充，考虑数据的连续性和负值偏向"""
    data_imputed = data.copy()

    # 输出列，通常有高负值偏向
    output_cols = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change',
                   'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    for column in columns_to_impute:
        # 获取缺失值掩码
        mask = data[column].isnull()
        if not mask.any():
            continue

        print(f"\n处理列: {column}")

        # 检查是否为输出列
        is_output_col = column in output_cols

        if is_output_col:
            # 分析正负值分布
            valid_values = data[column].dropna()
            neg_ratio = (valid_values < 0).mean()
            pos_ratio = (valid_values > 0).mean()
            print(f"列 {column} 的分布: 负值={neg_ratio:.2%}, 正值={pos_ratio:.2%}")

            # 对于高度偏向负值的列，使用特殊处理
            if neg_ratio > 0.7:
                # 特殊方法：先创建符号列，再分别插值
                # 创建符号列
                sign_column = pd.Series(index=data.index)
                sign_column[~mask] = np.where(data.loc[~mask, column] < 0, -1, 1)

                # 对于缺失值，按原始分布随机分配符号
                missing_indices = data.index[mask]
                n_neg = int(neg_ratio * len(missing_indices))
                n_pos = len(missing_indices) - n_neg

                # 分配符号
                random_signs = np.concatenate([
                    np.full(n_neg, -1),
                    np.full(n_pos, 1)
                ])
                np.random.shuffle(random_signs)
                sign_column[missing_indices] = random_signs

                # 分类进行插值
                neg_mask = sign_column == -1
                pos_mask = sign_column == 1

                # 分别为正负值创建临时数据
                temp_neg = data[column].copy()
                temp_pos = data[column].copy()

                # 分别插值
                if neg_mask.sum() > 0:
                    for method in ['linear', 'cubic', 'nearest']:
                        try:
                            temp_neg[neg_mask] = data[column].reindex(data.index[neg_mask]).interpolate(
                                method=method, limit_direction='both'
                            )
                            break
                        except:
                            continue

                    # 如果插值后仍有缺失，使用均值填充
                    neg_mean = data.loc[data[column] < 0, column].mean()
                    temp_neg[neg_mask & temp_neg.isna()] = neg_mean

                if pos_mask.sum() > 0:
                    for method in ['linear', 'cubic', 'nearest']:
                        try:
                            temp_pos[pos_mask] = data[column].reindex(data.index[pos_mask]).interpolate(
                                method=method, limit_direction='both'
                            )
                            break
                        except:
                            continue

                    # 如果插值后仍有缺失，使用均值填充
                    pos_mean = data.loc[data[column] >= 0, column].mean()
                    temp_pos[pos_mask & temp_pos.isna()] = pos_mean

                # 合并结果
                temp_combined = pd.Series(index=data.index)
                temp_combined[neg_mask] = temp_neg[neg_mask]
                temp_combined[pos_mask] = temp_pos[pos_mask]

                # 填充结果
                data_imputed.loc[mask, column] = temp_combined[mask]

                # 检查并确保负值比例一致
                filled_values = data_imputed.loc[mask, column]
                filled_neg_ratio = (filled_values < 0).mean()

                # 如果填充后负值比例不符合预期，调整部分值
                if filled_neg_ratio < neg_ratio * 0.9:  # 允许10%误差
                    n_to_adjust = int((neg_ratio - filled_neg_ratio) * len(filled_values))

                    # 找出最小的正值
                    pos_indices = mask & (data_imputed[column] >= 0)
                    pos_indices = data_imputed.loc[pos_indices, column].sort_values().index[:n_to_adjust]

                    # 重新填充为负值
                    if len(pos_indices) > 0:
                        data_imputed.loc[pos_indices, column] = neg_mean * 0.8  # 稍微高于平均负值
            else:
                # 对于不那么偏向负值的列，使用常规插值，但仍然保持正负比例
                # 使用多种插值方法
                methods = ['linear', 'cubic', 'spline', 'quadratic', 'nearest']

                # 尝试不同的插值方法
                for method in methods:
                    try:
                        temp_filled = data[column].interpolate(method=method, limit_direction='both')
                        # 如果减少了缺失值，使用这个方法
                        if temp_filled.isna().sum() < data[column].isna().sum():
                            data_imputed.loc[:, column] = temp_filled

                            # 检查并调整正负值比例
                            filled_values = data_imputed.loc[mask, column]
                            filled_neg_ratio = (filled_values < 0).mean()

                            # 如果比例不符合，进行简单调整
                            if abs(filled_neg_ratio - neg_ratio) > 0.1:  # 10%容差
                                # 根据需要调整部分值
                                if filled_neg_ratio < neg_ratio:
                                    # 需要增加负值
                                    pos_indices = mask & (data_imputed[column] >= 0)
                                    n_to_adjust = int((neg_ratio - filled_neg_ratio) * mask.sum())

                                    if n_to_adjust > 0 and pos_indices.sum() > 0:
                                        # 选择最小的正值调整为负值
                                        idx_to_change = data_imputed.loc[pos_indices, column].sort_values().index[
                                                        :n_to_adjust]
                                        data_imputed.loc[idx_to_change, column] = data_imputed.loc[
                                                                                      idx_to_change, column] * -1
                                else:
                                    # 需要增加正值
                                    neg_indices = mask & (data_imputed[column] < 0)
                                    n_to_adjust = int((filled_neg_ratio - neg_ratio) * mask.sum())

                                    if n_to_adjust > 0 and neg_indices.sum() > 0:
                                        # 选择最接近0的负值调整为正值
                                        idx_to_change = data_imputed.loc[neg_indices, column].sort_values(
                                            ascending=False).index[:n_to_adjust]
                                        data_imputed.loc[idx_to_change, column] = data_imputed.loc[
                                                                                      idx_to_change, column] * -1

                            break
                    except Exception as e:
                        print(f"方法 {method} 失败: {str(e)}")
                        continue

                # 如果插值后仍有缺失值，使用前向和后向填充
                if data_imputed[column].isna().any():
                    temp = data_imputed[column].fillna(method='ffill').fillna(method='bfill')
                    data_imputed.loc[data_imputed[column].isna(), column] = temp[data_imputed[column].isna()]
        else:
            # 非输出列使用标准插值方法
            # 先尝试不同的插值方法
            methods = ['linear', 'cubic', 'spline', 'quadratic', 'nearest']
            for method in methods:
                try:
                    # 尝试使用当前方法进行插值
                    interpolated = data[column].interpolate(method=method, limit_direction='both')
                    if interpolated.isna().sum() < data[column].isna().sum():
                        data_imputed.loc[:, column] = interpolated
                        break
                except Exception as e:
                    print(f"方法 {method} 失败: {str(e)}")
                    continue

            # 如果插值后仍有缺失值，使用前向和后向填充
            if data_imputed[column].isna().any():
                temp = data_imputed[column].fillna(method='ffill').fillna(method='bfill')
                data_imputed.loc[data_imputed[column].isna(), column] = temp[data_imputed[column].isna()]

                # 如果仍有缺失，使用均值填充
                if data_imputed[column].isna().any():
                    data_imputed.loc[data_imputed[column].isna(), column] = data_imputed[column].mean()

    return data_imputed


def create_enhanced_features(data, target_col, numeric_columns, is_output_col=False, neg_ratio=0):
    """创建增强的特征矩阵，针对高度偏向负值的目标列添加特殊特征"""
    features = data.copy()

    if target_col in numeric_columns:
        # 预先计算所有要添加的特征
        new_features = {}

        # 基本统计特征
        windows = [3, 5]
        stats_funcs = {
            'mean': lambda x: x.mean(),
            'std': lambda x: x.std(),
            'median': lambda x: x.median(),
            'q25': lambda x: x.quantile(0.25),
            'q75': lambda x: x.quantile(0.75)
        }

        for col in numeric_columns:
            if col != target_col:
                # 滚动窗口统计特征
                for window in windows:
                    for stat_name, stat_func in stats_funcs.items():
                        feat_name = f'{col}_{stat_name}_w{window}'
                        new_features[feat_name] = features[col].rolling(
                            window=window,
                            center=True,
                            min_periods=1
                        ).apply(stat_func)

                # 比率特征
                if not features[target_col].isnull().all() and not features[target_col].eq(0).all():
                    non_zero_mask = features[target_col] != 0
                    ratio_feat = pd.Series(index=features.index)
                    ratio_feat.loc[non_zero_mask] = features.loc[non_zero_mask, col] / features.loc[
                        non_zero_mask, target_col]
                    new_features[f'ratio_{col}_{target_col}'] = ratio_feat

        # 特征交互
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i + 1:]:
                if col1 != target_col and col2 != target_col:
                    # 差值特征
                    new_features[f'diff_{col1}_{col2}'] = features[col1] - features[col2]

                    # 乘积特征
                    new_features[f'prod_{col1}_{col2}'] = features[col1] * features[col2]

                    # 比率特征 (安全处理除以零)
                    non_zero_mask = features[col2] != 0
                    if non_zero_mask.any():
                        ratio_feat = pd.Series(index=features.index)
                        ratio_feat.loc[non_zero_mask] = features.loc[non_zero_mask, col1] / features.loc[
                            non_zero_mask, col2]
                        new_features[f'ratio_{col1}_{col2}'] = ratio_feat

        # 对高度偏向负值的输出列添加特殊特征
        if is_output_col and neg_ratio > 0.7:
            # 全局负值比例特征
            new_features['global_neg_ratio'] = np.ones(len(features)) * neg_ratio

            # 为每个特征添加与目标负值相关的条件概率特征
            for col in numeric_columns:
                if col != target_col:
                    try:
                        # 仅使用非缺失值
                        valid_mask = ~features[col].isnull() & ~features[target_col].isnull()
                        if valid_mask.sum() > 20:  # 确保有足够的样本
                            # 尝试使用分位数分组计算条件概率
                            col_series = features.loc[valid_mask, col]
                            target_series = features.loc[valid_mask, target_col]

                            # 创建特征分组
                            try:
                                # 使用分位数分组
                                q_labels = ['q1', 'q2', 'q3', 'q4']
                                col_groups = pd.qcut(col_series, 4, labels=q_labels, duplicates='drop')

                                # 计算每组内负值比例
                                group_neg_ratios = {}
                                for group in col_groups.unique():
                                    group_targets = target_series[col_groups == group]
                                    group_neg_ratios[group] = (group_targets < 0).mean()

                                # 将分组信息添加为新特征
                                all_groups = pd.qcut(features[col], 4, labels=q_labels, duplicates='drop')
                                new_features[f'{col}_neg_prob'] = all_groups.map(group_neg_ratios).fillna(neg_ratio)
                            except Exception as e:
                                # 如果分组失败，添加一个简单的是否大于中位数特征
                                col_median = col_series.median()
                                is_above_median = (features[col] > col_median).astype(float)
                                above_neg_ratio = (target_series[col_series > col_median] < 0).mean()
                                below_neg_ratio = (target_series[col_series <= col_median] < 0).mean()

                                # 创建简单的二分特征
                                new_features[f'{col}_above_median_neg_prob'] = is_above_median * above_neg_ratio + (
                                            1 - is_above_median) * below_neg_ratio
                    except Exception as e:
                        print(f"为特征 {col} 创建负值概率特征时出错: {str(e)}")

        # 添加所有新特征
        features = pd.concat([features, pd.DataFrame(new_features)], axis=1)

    return features


def evaluate_imputation(original_data, imputed_data, column, mask):
    """评估填充质量，使用多个指标，针对负值偏向添加特殊评估"""
    metrics = {}
    valid_mask = ~mask

    if valid_mask.any():
        true_values = original_data.loc[valid_mask, column]
        pred_values = imputed_data.loc[valid_mask, column]

        # 常规评估指标
        metrics['mse'] = mean_squared_error(true_values, pred_values)
        metrics['mae'] = mean_absolute_error(true_values, pred_values)
        metrics['r2'] = r2_score(true_values, pred_values)

        # KS统计量评估分布相似度
        ks_stat, _ = ks_2samp(true_values, pred_values)
        metrics['ks_stat'] = ks_stat

        # 针对高度偏向负值的列添加特殊评估
        if 'CD_' in column:
            # 计算正负值比例
            true_neg_ratio = (true_values < 0).mean()
            pred_neg_ratio = (pred_values < 0).mean()

            # 添加负值比例保持度量
            metrics['true_neg_ratio'] = true_neg_ratio
            metrics['pred_neg_ratio'] = pred_neg_ratio
            metrics['neg_ratio_diff'] = abs(true_neg_ratio - pred_neg_ratio)

            # 分别评估负值和正值的预测精度
            if true_neg_ratio > 0 and true_neg_ratio < 1:
                neg_mask = true_values < 0
                pos_mask = true_values >= 0

                if neg_mask.any():
                    metrics['neg_mse'] = mean_squared_error(true_values[neg_mask], pred_values[neg_mask])
                    metrics['neg_mae'] = mean_absolute_error(true_values[neg_mask], pred_values[neg_mask])

                if pos_mask.any():
                    metrics['pos_mse'] = mean_squared_error(true_values[pos_mask], pred_values[pos_mask])
                    metrics['pos_mae'] = mean_absolute_error(true_values[pos_mask], pred_values[pos_mask])

    return metrics


def optimize_rf_params(X_train, y_train, X_val, y_val):
    """优化随机森林参数"""
    param_grid = {
        'max_depth': [None, 5, 10],
        'n_estimators': [100, 200],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2', None]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_params_
def compare_methods(data, columns_to_impute):
    """比较不同填充方法的性能"""
    methods = {
        '均值填充': mean_imputation,
        'KNN填充': knn_imputation,
        '线性回归填充': correlation_based_imputation,  # 替换随机森林填充
        '插值填充': interpolation_imputation
    }

    results = {}
    imputed_data_dict = {}  # 存储每种方法的填充结果

    for method_name, method_func in methods.items():
        print(f"\n正在评估: {method_name}")
        try:
            imputed_data = method_func(data.copy(), columns_to_impute)
            imputed_data_dict[method_name] = imputed_data  # 保存填充后的数据

            method_metrics = {}
            for column in columns_to_impute:
                mask = data[column].isnull()
                metrics = evaluate_imputation(data, imputed_data, column, mask)
                method_metrics[column] = metrics

            results[method_name] = method_metrics
        except Exception as e:
            print(f"{method_name} 评估失败: {str(e)}")
            continue

    # 保存比较结果
    try:
        comparison_df = pd.DataFrame()
        for method, metrics in results.items():
            for column, values in metrics.items():
                for metric_name, value in values.items():
                    comparison_df.loc[f"{column}_{metric_name}", method] = value

        comparison_df.to_csv('method_comparison.csv')
        print("\n方法比较结果已保存到 method_comparison.csv")
    except Exception as e:
        print(f"保存比较结果时出错: {str(e)}")

    return results, imputed_data_dict


def plot_simplified_comparison(original_data, imputed_data_dict, columns_to_impute):
    """
    简化版本的比较函数，只展示总体对比和一个代表性的列
    """
    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # 使用默认样式

    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22  # 设置基础字号为22

    # 转换方法名称为英文
    method_mapping = {
        '均值填充': 'Mean Imputation',
        'KNN填充': 'KNN Imputation',
        '线性回归填充': 'Linear Regression Imputation',  # 改为线性回归
        '插值填充': 'Interpolate Imputation'
    }

    # 创建一个画布，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 第一个子图：总体对比
    # 准备合并所有列的数据，用于总体对比
    method_data = {}

    for method_name, imputed_data in imputed_data_dict.items():
        original_all_cols = pd.Series()
        imputed_all_cols = pd.Series()

        for column in columns_to_impute:
            mask = ~original_data[column].isnull()
            # 合并非缺失的原始数据
            original_all_cols = pd.concat([original_all_cols, original_data.loc[mask, column]])
            # 合并填充后的数据（对应的位置）
            imputed_all_cols = pd.concat([imputed_all_cols, imputed_data.loc[mask, column]])

        method_data[method_name] = {
            'original': original_all_cols,
            'imputed': imputed_all_cols
        }

    # 绘制总体对比直方图（第一个子图）
    for method_name, data in method_data.items():
        kde_plot = sns.kdeplot(
            data['original'],
            ax=ax1,
            label=f"Original",
            color='black',
            linestyle='-',
            linewidth=2
        )
        break  # 只需要绘制一次原始数据

    # 为每种方法绘制填充后的KDE曲线
    colors = ['blue', 'red', 'green', 'purple']
    line_styles = ['-', '--', '-.', ':']

    for i, (method_name, data) in enumerate(method_data.items()):
        english_method_name = method_mapping.get(method_name, method_name)
        kde_plot = sns.kdeplot(
            data['imputed'],
            ax=ax1,
            label=f"{english_method_name}",
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2
        )

    ax1.set_title('Overall Distribution Comparison', fontsize=24, fontfamily='Times New Roman')
    ax1.set_xlabel('Value', fontsize=22, fontfamily='Times New Roman')
    ax1.set_ylabel('Density', fontsize=22, fontfamily='Times New Roman')
    ax1.legend(fontsize=18, prop={'family': 'Times New Roman'})
    ax1.grid(True, alpha=0.3)

    # 第二个子图：代表性列的对比
    # 选择一个代表性的列，优先选择CD_开头的列
    representative_col = next((col for col in columns_to_impute if col.startswith('CD_')), columns_to_impute[0])

    print(f"选择的代表性列: {representative_col}")

    # 绘制原始数据和每种方法填充后的数据
    mask = ~original_data[representative_col].isnull()
    kde_plot = sns.kdeplot(
        original_data.loc[mask, representative_col],
        ax=ax2,
        label=f"Original",
        color='black',
        linestyle='-',
        linewidth=2
    )

    for i, (method_name, imputed_data) in enumerate(imputed_data_dict.items()):
        english_method_name = method_mapping.get(method_name, method_name)
        kde_plot = sns.kdeplot(
            imputed_data[representative_col],
            ax=ax2,
            label=f"{english_method_name}",
            color=colors[i % len(colors)],
            linestyle=line_styles[i % len(line_styles)],
            linewidth=2
        )

    ax2.set_title(f'Column: {representative_col}', fontsize=24, fontfamily='Times New Roman')
    ax2.set_xlabel('Value', fontsize=22, fontfamily='Times New Roman')
    ax2.set_ylabel('Density', fontsize=22, fontfamily='Times New Roman')
    ax2.legend(fontsize=18, prop={'family': 'Times New Roman'})
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.savefig('simplified_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_all_methods_plot(original_data, imputed_data_dict, columns_to_impute):
    """比较所有填充方法并生成对比图，使用Times New Roman字体，字号22，所有列合并到一个图中"""
    # 设置全局字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22

    # 方法名称映射
    method_mapping = {
        '均值填充': 'Mean Imputation',
        'KNN填充': 'KNN Imputation',
        '线性回归填充': 'Linear Regression Imputation',
        '插值填充': 'Interpolate Imputation'
    }

    # 创建整体画布
    plt.figure(figsize=(20, 15))

    # 遍历每种方法
    for i, (method_name, imputed_data) in enumerate(imputed_data_dict.items(), 1):
        # 创建子图
        ax = plt.subplot(2, 2, i)

        # 合并所有列数据
        original_all_cols = pd.Series(dtype=float)
        imputed_all_cols = pd.Series(dtype=float)

        for column in columns_to_impute:
            mask = ~original_data[column].isnull()
            original_all_cols = pd.concat([original_all_cols, original_data.loc[mask, column]])
            imputed_all_cols = pd.concat([imputed_all_cols, imputed_data.loc[mask, column]])

        # 绘制直方图
        ax.hist(original_all_cols, bins=30, alpha=0.5, label='Original', color='skyblue')
        ax.hist(imputed_all_cols, bins=30, alpha=0.5, label='Filled', color='orange')

        # 设置标题，字体加粗
        english_method_name = method_mapping.get(method_name, method_name)
        ax.set_title(f'{english_method_name}', fontsize=28, fontweight='bold', fontfamily='Times New Roman')

        # 移除x轴和y轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 图例字体加粗
        ax.legend(prop={'family': 'Times New Roman', 'size': 22, 'weight': 'bold'})

        # 设置坐标轴刻度字体加粗
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontsize(22)
            label.set_fontfamily('Times New Roman')

        # 坐标轴线加粗
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # 刻度线加粗
        ax.tick_params(axis='both', which='both', width=2, length=6)

    plt.tight_layout()
    plt.savefig('imputation_comparison_all_columns.pdf', dpi=600, bbox_inches='tight')


def compare_neg_ratio_methods_plot(original_data, imputed_data_dict, columns_to_impute):
    """比较所有填充方法的负值比例保持能力"""
    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22  # 设置基础字号

    # 转换方法名称为英文
    method_mapping = {
        '均值填充': 'Mean Imputation',
        'KNN填充': 'KNN Imputation',
        '线性回归填充': 'Linear Regression Imputation',  # 改为线性回归
        '插值填充': 'Interpolate Imputation'
    }

    # 筛选出CD_开头的列（输出列）
    output_cols = [col for col in columns_to_impute if col.startswith('CD_')]

    if not output_cols:
        print("没有找到输出列，跳过负值比例对比图")
        return

    # 创建图表
    fig, axes = plt.subplots(len(output_cols), 1, figsize=(12, 5 * len(output_cols)))
    if len(output_cols) == 1:
        axes = [axes]

    # 为每个输出列创建条形图
    for i, column in enumerate(output_cols):
        # 计算原始数据的负值比例
        original_valid = original_data[~original_data[column].isnull()][column]
        original_neg_ratio = (original_valid < 0).mean()

        # 计算每种方法的负值比例
        method_neg_ratios = {}
        for method_name, imputed_data in imputed_data_dict.items():
            # 计算所有数据的负值比例
            all_neg_ratio = (imputed_data[column] < 0).mean()

            # 计算仅填充值的负值比例
            filled_mask = original_data[column].isnull()
            if filled_mask.any():
                filled_neg_ratio = (imputed_data.loc[filled_mask, column] < 0).mean()
            else:
                filled_neg_ratio = np.nan

            method_neg_ratios[method_name] = {
                'all': all_neg_ratio,
                'filled': filled_neg_ratio
            }

        # 准备绘图数据
        methods = list(method_neg_ratios.keys())
        english_methods = [method_mapping.get(m, m) for m in methods]
        all_ratios = [method_neg_ratios[m]['all'] for m in methods]
        filled_ratios = [method_neg_ratios[m]['filled'] for m in methods]

        # 设置x位置
        x = np.arange(len(methods))
        width = 0.35

        # 绘制条形图
        bar1 = axes[i].bar(x - width / 2, all_ratios, width, label='All Values', color='skyblue')
        bar2 = axes[i].bar(x + width / 2, filled_ratios, width, label='Filled Values Only', color='orange')

        # 添加原始数据的负值比例参考线
        axes[i].axhline(y=original_neg_ratio, color='r', linestyle='-',
                        label=f'Original Negative Ratio: {original_neg_ratio:.2%}')

        # 设置图表属性
        axes[i].set_title(f'Negative Value Ratio Comparison - {column}',
                          fontsize=22, fontfamily='Times New Roman')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(english_methods, rotation=45, ha='right')
        axes[i].set_ylabel('Negative Value Ratio', fontsize=20, fontfamily='Times New Roman')
        axes[i].set_ylim(0, max(max(all_ratios), max(filled_ratios) if any(~np.isnan(filled_ratios)) else 0,
                                original_neg_ratio) * 1.1)

        # 添加数值标签
        def add_labels(bars, ratios):
            for bar, ratio in zip(bars, ratios):
                if not np.isnan(ratio):
                    height = bar.get_height()
                    axes[i].annotate(f'{ratio:.2%}',
                                     xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 3),  # 3点垂直偏移
                                     textcoords="offset points",
                                     ha='center', va='bottom',
                                     fontsize=14, fontfamily='Times New Roman')

        add_labels(bar1, all_ratios)
        add_labels(bar2, filled_ratios)

        # 添加图例
        axes[i].legend(fontsize=16, prop={'family': 'Times New Roman'})

        # 设置网格
        axes[i].grid(True, alpha=0.3)

        # 调整刻度标签字体
        for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(16)

    plt.tight_layout()
    plt.savefig('negative_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("开始执行程序...")

    try:
        # 读取数据文件
        data = pd.read_csv(r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv')
        print(f"文件读取成功！数据形状：{data.shape}")

        # 打印所有列名，确认列名正确性
        print(f"数据集列名: {data.columns.tolist()}")

        # 选择需要处理的列
        columns_to_check = [
            'pH', 'organicmatter', 'CEC', 'pHsoil', 'temperature',
            'Cdcontent',
            'CD_soil_change', 'CD_brown_rice_change',
            'CD_stem_change', 'CD_leaf_change',
            'CD_shell_change', 'CD_root_change'
        ]
        print(f"需要检查的列：{columns_to_check}")

        # 验证并获取可用的数值列
        print("\n开始验证数据列类型...")
        columns_to_impute = validate_numeric_columns(data, columns_to_check)
        print(f"可用的数值列：{columns_to_impute}")

        if not columns_to_impute:
            print("错误：没有可用的数值列进行处理")
            return

        # 预处理数据
        print("\n开始预处理数据...")
        processed_data = preprocess_data(data, columns_to_impute)
        print("预处理完成")

        # 打印初始统计信息
        print("\n打印初始统计信息...")
        print_initial_statistics(processed_data, columns_to_impute)

        # 比较所有方法
        print("\n开始比较不同填充方法...")
        comparison_results, imputed_data_dict = compare_methods(processed_data, columns_to_impute)
        print("方法比较完成")

        # 比较所有方法后添加验证步骤
        print("\n验证填充结果...")
        for method_name, imputed_data in imputed_data_dict.items():
            missing_cols = []
            for column in columns_to_impute:
                if imputed_data[column].isnull().any():
                    missing_count = imputed_data[column].isnull().sum()
                    missing_cols.append(f"{column}({missing_count})")

            if missing_cols:
                print(f"{method_name}仍有未填充的列: {', '.join(missing_cols)}")
            else:
                print(f"{method_name}所有列填充完成")

        # 生成总体对比图
        print("\n生成总体对比图...")
        compare_all_methods_plot(processed_data, imputed_data_dict, columns_to_impute)

        # 为每种方法单独生成详细对比图
        methods_english = {
            '均值填充': 'Mean Imputation',
            'KNN填充': 'KNN Imputation',
            '线性回归填充': 'Linear Regression Imputation',
            '插值填充': 'Interpolate Imputation'
        }

        for method_name, imputed_data in imputed_data_dict.items():
            english_name = methods_english.get(method_name, method_name)
            plot_imputation_comparison(
                processed_data,
                imputed_data,
                columns_to_impute,
                english_name,
                f'{english_name.lower().replace(" ", "_")}.pdf'
            )
        # 保存结果 - 同时保存KNN和线性回归的结果
        try:
            # 保存KNN填充结果
            knn_imputed = imputed_data_dict.get('KNN填充')
            if knn_imputed is not None:
                knn_imputed.to_csv(r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\knn_imputed_data_new.csv', index=False,
                                   encoding='utf-8-sig')
                print("已将KNN填充结果保存到 knn_imputed_data_new.csv")
            else:
                print("警告：未能找到KNN填充结果")

            # 保存线性回归填充结果
            lr_imputed = imputed_data_dict.get('线性回归填充')
            if lr_imputed is not None:
                lr_imputed.to_csv(r'C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new（111）.csv', index=False,
                                  encoding='utf-8-sig')
                print("已将线性回归填充结果保存到 lr_imputed_data_new.csv")
            else:
                print("警告：未能找到线性回归填充结果")
        except Exception as e:
            print(f"保存结果时出错：{e}")

    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()
        return

    print("程序执行完成！")


def validate_numeric_columns(data, columns):
    """验证并获取可用的数值列"""
    print("开始验证数值列...")
    numeric_columns = []
    for column in columns:
        print(f"检查列：{column}")
        try:
            pd.to_numeric(data[column], errors='raise')
            numeric_columns.append(column)
            print(f"列 {column} 是有效的数值列")
        except (ValueError, TypeError):
            print(f"警告：列 {column} 包含非数值数据，将被跳过")
    print(f"验证完成，找到 {len(numeric_columns)} 个有效数值列")
    return numeric_columns


def preprocess_data(data, columns):
    """预处理数据，转换为数值类型"""
    processed_data = data.copy()
    for column in columns:
        processed_data.loc[:, column] = pd.to_numeric(processed_data[column], errors='coerce')
    return processed_data


def print_initial_statistics(data, columns):
    """只打印填充前的有效数据统计量"""
    print("\n原始数据的有效值统计：")
    for column in columns:
        valid_data = data[column].dropna()
        if len(valid_data) > 0:
            stats = pd.Series({
                '有效数据量': len(valid_data),
                '均值': valid_data.mean(),
                '标准差': valid_data.std(),
                '最小值': valid_data.min(),
                '25%分位数': valid_data.quantile(0.25),
                '中位数': valid_data.quantile(0.5),
                '75%分位数': valid_data.quantile(0.75),
                '最大值': valid_data.max(),
                '缺失值数量': data[column].isnull().sum(),
                '负值比例': (valid_data < 0).mean()
            })
            print(f"\n{column}的统计信息：")
            print(stats.to_string())
        else:
            print(f"\n{column}: 没有有效数据")


def plot_imputation_comparison(original_data, imputed_data, columns, method_name, save_path=None):
    """
    绘制原始数据和填充后数据的分布对比图，使用Times New Roman字体，字号22
    """
    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn-whitegrid')
        except:
            pass  # 使用默认样式

    # 设置全局字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 22  # 设置基础字号为22

    # 创建子图
    n_cols = min(len(columns), 4)  # 最多4列
    n_rows = (len(columns) + n_cols - 1) // n_cols  # 计算需要的行数
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))

    # 在图表顶部添加填充方法名称
    fig.suptitle(f'{method_name}', fontsize=28, fontfamily='Times New Roman')

    # 将axes转换为扁平化数组以便迭代
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    # 为每个列创建直方图
    for i, column in enumerate(columns[:len(axes)]):
        # 获取非缺失值的掩码
        mask = ~original_data[column].isnull()

        # 绘制直方图
        axes[i].hist(original_data.loc[mask, column], bins=30, alpha=0.5,
                     label='Original', color='skyblue')
        axes[i].hist(imputed_data[column], bins=30, alpha=0.5,
                     label='Filled', color='orange')

        # 设置图表属性
        axes[i].set_title(f'{column}', fontsize=22, fontfamily='Times New Roman')

        # 移除x轴标签
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

        # 设置图例字体
        axes[i].legend(prop={'family': 'Times New Roman', 'size': 22})

        # 调整刻度标签字体
        for label in axes[i].get_xticklabels() + axes[i].get_yticklabels():
            label.set_fontfamily('Times New Roman')
            label.set_fontsize(22)

        # 调整布局
        axes[i].grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()