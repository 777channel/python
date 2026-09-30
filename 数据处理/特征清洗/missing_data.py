def process_missing_data(data, threshold=0.5, strategies=None):
    """
    处理缺失值，但不删除整列

    Parameters:
    -----------
    data : pandas DataFrame
        输入的数据框
    threshold : float
        缺失值占比的阈值，超过此阈值的列才会被特殊处理
    strategies : dict
        每列的处理策略，格式为 {'column_name': 'strategy'}
        strategy 可以是: 'mean', 'median', 'mode', 'random', 'end_dist'

    Returns:
    --------
    pandas DataFrame : 处理后的数据框
    dict : 每列使用的填充值
    """
    data_copy = data.copy(deep=True)
    fill_values = {}  # 存储每列的填充值

    # 检查每列的缺失值情况
    missing_stats = pd.concat([data.isnull().sum(), data.isnull().mean()],
                              axis=1,
                              keys=['missing_count', 'missing_ratio'])

    # 如果没有指定策略，创建默认策略
    if strategies is None:
        strategies = {}
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                strategies[col] = 'mean'
            else:
                strategies[col] = 'mode'

    # 处理每一列
    for column in data.columns:
        missing_ratio = missing_stats.loc[column, 'missing_ratio']

        if missing_ratio > 0:  # 只处理有缺失值的列
            strategy = strategies.get(column, 'mean')  # 获取该列的处理策略

            if missing_ratio > threshold:
                print(f"Warning: Column {column} has {missing_ratio:.2%} missing values")

            # 根据不同策略处理缺失值
            try:
                if strategy == 'mean':
                    fill_value = data_copy[column].mean()
                elif strategy == 'median':
                    fill_value = data_copy[column].median()
                elif strategy == 'mode':
                    fill_value = data_copy[column].mode()[0]
                elif strategy == 'random':
                    # 随机填充不需要存储填充值
                    random_sample = data_copy[column].dropna().sample(
                        data_copy[column].isnull().sum(),
                        random_state=42
                    )
                    random_sample.index = data_copy[data_copy[column].isnull()].index
                    data_copy.loc[data_copy[column].isnull(), column] = random_sample
                    continue
                elif strategy == 'end_dist':
                    fill_value = data_copy[column].mean() + 3 * data_copy[column].std()
                else:
                    print(f"Warning: Unknown strategy {strategy} for column {column}")
                    continue

                # 填充缺失值
                data_copy[column] = data_copy[column].fillna(fill_value)
                fill_values[column] = fill_value

            except Exception as e:
                print(f"Error processing column {column}: {str(e)}")
                continue

    return data_copy, fill_values


# 使用示例
def main():
    # 假设你有一个数据框 df
    df = pd.read_csv('your_data.csv')

    # 定义特定列的处理策略
    custom_strategies = {
        'age': 'median',
        'income': 'mean',
        'category': 'mode',
        'rating': 'random'
    }

    # 处理缺失值
    df_processed, fill_values = process_missing_data(
        data=df,
        threshold=0.5,  # 当某列缺失值超过50%时发出警告
        strategies=custom_strategies
    )

    # 打印填充值统计
    print("\n填充值统计:")
    for col, value in fill_values.items():
        print(f"{col}: {value}")

    return df_processed