import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import shap
import traceback
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 导入检查 ---
tabpfn_available = False 
pfn_interpret_available = False
interpretability = None
TabPFNRegressor = None

try:
    # 优先尝试从 tabpfn_extensions 导入 TabPFNRegressor 和 interpretability
    from tabpfn_extensions import interpretability, TabPFNRegressor as TPFNRegressorExt
    TabPFNRegressor = TPFNRegressorExt
    pfn_interpret_available = True
    tabpfn_available = True
    print("tabpfn_extensions.interpretability 模块导入成功。")
    print("TabPFNRegressor 从 'tabpfn_extensions' 库导入成功。")
except ImportError:
    print("警告: 未找到 tabpfn_extensions.interpretability 模块，SHAP分析将不可用。")
    try:
        # 如果 tabpfn_extensions 失败，尝试从标准 tabpfn 导入 TabPFNRegressor
        from tabpfn import TabPFNRegressor
        tabpfn_available = True
        print("TabPFNRegressor 从标准 'tabpfn' 库导入成功。")
    except ImportError:
        print("警告: TabPFNRegressor 未找到（无论是 tabpfn_extensions 还是标准 tabpfn）。TabPFN 模型将不可用。")


# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False # 保持负号正常显示
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

# --- 数据加载与绘图函数 ---
def load_and_process_data(file_path):
    """加载数据并对所有列进行初步数值转换（无法转换的为NaN），并移除BOM。"""
    try:
        # 尝试多种编码读取，确保兼容性
        read_successfully = False
        encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"数据从 '{file_path}' (使用 {enc} 编码) 加载成功。")
                read_successfully = True
                break
            except UnicodeDecodeError:
                continue # 尝试下一个编码
            except Exception as e:
                print(f"尝试使用 {enc} 编码读取文件 '{file_path}' 时发生错误: {e}")
                return None

        if not read_successfully:
            print(f"错误: 尝试所有常用编码后仍无法读取文件 '{file_path}'。请检查文件编码或损坏。")
            return None

    except Exception as e:
        print(f"加载数据时发生意外错误: {e}")
        return None

    # 移除列名开头的BOM字符
    df.columns = [col.lstrip('\ufeff') for col in df.columns]
    print("\n已移除列名开头的BOM字符。列名的大小写和空格将保持不变，请确保与代码中的列名精确匹配。")

    # 尝试将所有列转换为数值型，无法转换的变为NaN
    print("\n尝试将所有列转换为数值型...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")

    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                 r2_train, r2_test, model_name_display, target_col_name, output_dir='.'):
    """为单个模型创建实际值 vs. 预测值的 JointGrid 对比图。"""
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")

    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df])

    data_combined_df['True'] = pd.to_numeric(data_combined_df['True'], errors='coerce')
    data_combined_df['Predicted'] = pd.to_numeric(data_combined_df['Predicted'], errors='coerce')
    data_combined_df.dropna(subset=['True', 'Predicted'], inplace=True)

    if data_combined_df.empty:
        print(f"警告: 为模型 '{model_name_display}' 清洗后无有效数据可用于绘图。"); return

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
    g.plot_joint(sns.scatterplot, alpha=0.5, s=100, edgecolor='w', linewidth=1)
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=palette['Train'], label='Train Regression Line', ci=None)
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=palette['Test'], label='Test Regression Line', ci=None)
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

    ax_joint = g.ax_joint
    ax_joint.tick_params(axis='x', labelsize=18)
    ax_joint.tick_params(axis='y', labelsize=18)
    ax_joint.set_xlabel('True', fontsize=16, fontweight='bold')
    ax_joint.set_ylabel('Predicted', fontsize=16, fontweight='bold')

    ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax_joint.text(0.6, 0.95, f'Model: {model_name_display}', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    min_val = min(data_combined_df['True'].min(), data_combined_df['Predicted'].min())
    max_val = max(data_combined_df['True'].max(), data_combined_df['Predicted'].max())
    ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', label='x=y', lw=2)

    legend = ax_joint.legend(fontsize=20)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    # 修改主标题为英文
    g.fig.suptitle(f'{model_name_display} Model: Actual vs. Predicted Values for {target_col_name} (Natural Aging)', fontsize=22, fontweight='bold', y=1.02) # 使用 target_col_name

    os.makedirs(output_dir, exist_ok=True)
    # 保存路径保持不变（包含中文）
    save_path = os.path.join(output_dir, f'{target_col_name}_自然老化_{model_name_display}_JointGrid.pdf') # 使用 target_col_name
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(g.fig)
    print(f"JointGrid 图已保存至: {save_path}")

# --- 新增函数：保存SHAP值为CSV ---
def save_shap_values_to_csv(shap_values, feature_names, target_column_name, output_dir='./shap_csv'):
    """
    将SHAP值保存为CSV文件
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        else:
            shap_values_array = shap_values

        shap_values_df = pd.DataFrame(shap_values_array, columns=feature_names)

        csv_filename = f'shap_values_自然老化_{target_column_name}.csv'
        csv_filepath = os.path.join(output_dir, csv_filename)

        shap_values_df.to_csv(csv_filepath, index=False, encoding='utf-8')

        print(f"SHAP值已保存至CSV文件: {csv_filepath}")
        print(f"SHAP值CSV文件形状: {shap_values_df.shape}")
        print(f"前5行SHAP值预览:")
        print(shap_values_df.head())

        return csv_filepath

    except Exception as e:
        print(f"保存SHAP值到CSV文件时发生错误: {e}")
        traceback.print_exc()
        return None

# --- 新增函数：保存原始特征值和SHAP值的组合数据 ---
def save_features_and_shap_combined(X_test, shap_values, feature_names, target_column_name,
                                     y_test, y_pred, output_dir='./shap_features_combined'):
    """
    保存原始特征值和对应的SHAP值，用于绘制PDP风格的图
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        else:
            shap_values_array = shap_values

        if not isinstance(X_test, pd.DataFrame):
            # Attempt to reconstruct DataFrame with original column names if X_test is an array
            # This is a fallback and might not always preserve original names perfectly if X_test was a raw numpy array
            X_test_df = pd.DataFrame(X_test, columns=X_test.columns if hasattr(X_test, 'columns') else [f'col_{i}' for i in range(X_test.shape[1])])
        else:
            X_test_df = X_test.copy()

        X_test_df = X_test_df.reset_index(drop=True)

        combined_data = X_test_df.copy()

        # 将SHAP值按预处理后的特征名加入
        if len(feature_names) == shap_values_array.shape[1]:
            for i, feature_shap_name in enumerate(feature_names):
                shap_col_name = f'{feature_shap_name}_SHAP'
                combined_data[shap_col_name] = shap_values_array[:, i]
        else:
            print(f"警告: 原始特征数量 ({X_test_df.shape[1]}) 与SHAP特征数量 ({shap_values_array.shape[1]}) 或提供的特征名数量 ({len(feature_names)}) 不匹配，SHAP值可能无法正确匹配到原始特征。将按索引添加SHAP值。")
            for i in range(shap_values_array.shape[1]):
                shap_col_name = f'SHAP_Value_Feature_{i}'
                combined_data[shap_col_name] = shap_values_array[:, i]


        combined_data[f'{target_column_name}_true'] = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else y_test
        combined_data[f'{target_column_name}_pred'] = y_pred

        combined_data['sample_id'] = range(len(combined_data))

        combined_filename = f'features_shap_combined_自然老化_{target_column_name}.csv'
        combined_filepath = os.path.join(output_dir, combined_filename)
        combined_data.to_csv(combined_filepath, index=False, encoding='utf-8')

        print(f"\n=== 特征+SHAP组合数据保存成功 ===")
        print(f"文件路径: {combined_filepath}")
        print(f"数据形状: {combined_data.shape}")

        return combined_filepath, None

    except Exception as e:
        print(f"保存特征+SHAP组合数据时发生错误: {e}")
        traceback.print_exc()
        return None, None

# --- 主训练与评估函数 ---
def train_and_evaluate_autotabpfn(df, target_column_name):
    """
    为TabPFN模型进行训练和评估。
    保留分层抽样逻辑，确保所有输出列和分层列不被作为特征列。
    """
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练 TabPFN 模型（保留分层抽样）---")

    df_processed = df.copy()
    initial_target_nan_count = df_processed[target_column_name].isnull().sum()
    if initial_target_nan_count > 0:
        print(f"\n目标列 '{target_column_name}' 存在 {initial_target_nan_count} 个缺失值。")
        print(f"将删除包含这些缺失值的 {initial_target_nan_count} 行数据。")
        df_processed.dropna(subset=[target_column_name], inplace=True)
        print(f"删除缺失值行后，数据剩余 {len(df_processed)} 行。")
    else:
        print(f"\n目标列 '{target_column_name}' 没有缺失值。")

    # --- 分层抽样列的定义和处理 ---
    strata_col_name_original = 'T-Aging' # 原始分层列名
    strata_col_name_final = strata_col_name_original # 最终用于分层的列名，可能被更新为binned版本
    strata = None # 默认不分层

    if strata_col_name_original in df_processed.columns:
        print(f"\n尝试根据列 '{strata_col_name_original}' 进行分层抽样...")

        # 如果是数值型，且不是整数类型（即有小数点），则需要分箱
        if pd.api.types.is_numeric_dtype(df_processed[strata_col_name_original]) and \
           not pd.api.types.is_integer_dtype(df_processed[strata_col_name_original]):
            print(f"警告: 分层列 '{strata_col_name_original}' 是连续数值类型。将对其进行分箱处理以进行分层。")
            try:
                # 至少分2个箱子，最多10个，或者不超过唯一值数量
                q_bins = min(10, df_processed[strata_col_name_original].nunique())
                if q_bins < 2:
                    print(f"警告: 分层列 '{strata_col_name_original}' 唯一值数量过少 ({df_processed[strata_col_name_original].nunique()})，无法有效分箱。将退回普通随机抽样。")
                    strata = None
                else:
                    # 创建一个新的分箱列用于分层
                    strata_col_name_final = f'{strata_col_name_original}_binned_for_stratify'
                    df_processed[strata_col_name_final] = pd.qcut(
                        df_processed[strata_col_name_original],
                        q=q_bins,
                        duplicates='drop' # 允许丢弃重复的箱边界
                    )
                    print(f"已创建分箱列 '{strata_col_name_final}'。各类别数量:\n{df_processed[strata_col_name_final].value_counts()}")
            except Exception as e:
                print(f"错误: 自动分箱 '{strata_col_name_original}' 失败: {e}。将退回普通随机抽样。")
                strata = None

        # 检查最终用于分层的列 (可能是原列或分箱后的列) 是否有缺失值
        # 仅当 strata 尚未因分箱失败而设置为 None 时才检查并处理
        if strata is None and strata_col_name_final in df_processed.columns and df_processed[strata_col_name_final].isnull().any():
            print(f"警告: 分层列 '{strata_col_name_final}' 存在缺失值，将使用众数填充。")
            imputer_strata = SimpleImputer(strategy='most_frequent')
            # 确保填充后保持正确的类型，例如如果原始是分类的，填充后依然是分类
            filled_strata = imputer_strata.fit_transform(df_processed[[strata_col_name_final]].astype(str)).flatten()
            df_processed[strata_col_name_final] = pd.Series(filled_strata, index=df_processed.index)
            # 如果是分箱产生的分类，转换回 Categorical
            if pd.api.types.is_object_dtype(df_processed[strata_col_name_final]) and strata_col_name_final != strata_col_name_original:
                    df_processed[strata_col_name_final] = pd.Categorical(df_processed[strata_col_name_final])


        # 检查最终分层列的类别数量是否足够进行分层
        # 仅当 strata 仍然为 None 时才重新评估并设置
        if strata is None and strata_col_name_final in df_processed.columns and all(df_processed[strata_col_name_final].value_counts() >= 2):
            strata = df_processed[strata_col_name_final] # 赋值为用于分层的列 (可能是原列或分箱列)
            print(f"将使用 '{strata_col_name_final}' 列进行分层抽样。")
        else:
            if strata is None: # 仅在 strata 仍然为 None 时才打印此警告
                print(f"警告: 未找到或无法使用 '{strata_col_name_original}' 或其分箱版本进行分层，将使用普通随机抽样。")
    else:
        print(f"警告: 未找到用于分层的列 '{strata_col_name_original}'。将使用普通随机抽样。")


    # --- 定义要从特征中排除的列 ---
    # 这部分是本次最核心的修正

    # 1. 首先，获取所有潜在的输出列。这些列都不能作为特征。
    # global target_column_options 声明用于访问 main 函数中定义的 target_column_options
    global target_column_options
    all_output_cols_to_exclude = list(target_column_options) # 确保这是所有输出列的副本

    # 初始化最终排除列表，包含所有潜在的输出列
    final_cols_to_exclude = list(set(all_output_cols_to_exclude))

    # 2. 排除原始分层列 (如果存在且尚未被排除)
    if strata_col_name_original in df_processed.columns and strata_col_name_original not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_original)

    # 3. 排除分箱后的分层列 (如果存在且不同于原始列，并且尚未被排除)
    if strata_col_name_final != strata_col_name_original and strata_col_name_final in df_processed.columns and strata_col_name_final not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_final)

    # 从所有列中选择特征列
    # feature_columns = 所有 df_processed 的列 - final_cols_to_exclude 中的列
    feature_columns = [col for col in df_processed.columns if col not in final_cols_to_exclude]

    # --- 详细打印特征和目标列分配详情 ---
    print(f"\n--- 特征和目标列分配详情 ---")
    print(f"所有原始数据列: {list(df_processed.columns)}")
    print(f"将从特征中排除的列 (final_cols_to_exclude): {final_cols_to_exclude}")
    print(f"最终用于模型训练的特征列 (feature_columns): {feature_columns}")
    print(f"模型的目标列 (target_column_name): {target_column_name}")
    print(f"用于分层抽样的列 (strata.name): {strata.name if strata is not None else 'None (使用普通随机抽样)'}")
    print(f"-----------------------------")


    X = df_processed[feature_columns]
    y = df_processed[target_column_name]

    # 数据分割 - 使用分层抽样 (如果 strata 不为 None)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strata
    )

    # --- 新增功能：打印用于训练的所有列 ---
    print(f"\n--- 实际传递给预处理和模型训练的特征列 ---")
    print(f"训练集特征列 (X_train.columns): {list(X_train.columns)}")
    print(f"测试集特征列 (X_test.columns): {list(X_test.columns)}")
    print(f"---------------------------------------")

    print(f"\n数据分割完成:")
    print(f"训练集规模: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: {X_test.shape}, y_test: {y_test.shape}")

    print("定义预处理流程：包含缺失值处理、Log1p 变换、和独热编码（已去除标准化）...")

    # 获取 X_train 的列，保持原始大小写
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist() # 非数值型即视为分类

    transformers = []

    # 1. 处理需要 Log1p 变换的数值特征
    # 如果你的数据集中没有 'Cd-C' 或 'BET'，这些列表会是空的，不影响逻辑
    log_transform_cols = ['Cd-C', 'BET']
    log_transform_cols_actual = [col for col in log_transform_cols if col in numeric_columns]

    if log_transform_cols_actual:
        print(f"将对 {log_transform_cols_actual} 列进行 KNNImputer -> Log1p 变换 (无标准化)。")
        log_numeric_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')),
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            # ('scaler', StandardScaler()) # 已移除标准化
        ])
        transformers.append(('log_numeric_features', log_numeric_pipeline, log_transform_cols_actual))

    # 2. 处理其他数值特征 (KNNImputer，无标准化)
    other_numeric_cols = [col for col in numeric_columns if col not in log_transform_cols_actual]
    if other_numeric_cols:
        print(f"将对 {len(other_numeric_cols)} 个其他数值特征进行 KNNImputer 变换 (无标准化)。")
        numeric_pipeline = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')),
            # ('scaler', StandardScaler()) # 已移除标准化
        ])
        transformers.append(('other_numeric_features', numeric_pipeline, other_numeric_cols))

    # 3. 处理分类特征 (SimpleImputer + OneHotEncoder)
    if categorical_columns:
        print(f"将对 {len(categorical_columns)} 个分类特征进行 SimpleImputer (众数) -> OneHotEncoder 变换。")
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('categorical_features', categorical_pipeline, categorical_columns))

    if not transformers:
        print("警告: 没有找到任何可用于预处理的特征！请检查您的数据和列名。")
        return

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # 确保只有明确处理的特征进入模型
    )

    # 确保 TabPFNRegressor 在开始训练前可用
    if not tabpfn_available:
        print("错误: TabPFNRegressor 不可用。无法继续训练。");
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model_instance = AutoTabPFNRegressor(
        device=device,
        max_time=80,
    )
    # model_instance = TabPFNRegressor(device=device)

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model_instance)
    ])

    try:
        print("\n=== 开始拟合 TabPFN 模型 ===")
        # 警告 TabPFN 对数据规模的限制
        try:
            temp_X_train_transformed = preprocessor.fit_transform(X_train)
            if temp_X_train_transformed.shape[0] > 1024 or temp_X_train_transformed.shape[1] >= 100:
                print(f"警告: 预处理后的训练数据规模 ({temp_X_train_transformed.shape[0]} 样本, {temp_X_train_transformed.shape[1]} 特征) 可能超出 TabPFN 的建议限制 (1024 样本, 100 特征)。这可能导致性能下降或内存/时间问题。")
            del temp_X_train_transformed # 释放内存
        except Exception as e:
            print(f"预处理步骤测试失败 (不影响后续训练): {e}")


        full_pipeline.fit(X_train, y_train)

        print("=== TabPFN 模型拟合完成！===")
         # --- 在这里添加保存模型的代码 ---
        model_save_dir = './trained_models' # 定义保存模型的文件夹
        os.makedirs(model_save_dir, exist_ok=True) # 创建文件夹如果不存在

        # 构建模型文件名，包含目标列名以便区分
        model_filename = f'natural_aging_model_{target_column_name}.joblib'
        model_save_path = os.path.join(model_save_dir, model_filename)

        joblib.dump(full_pipeline, model_save_path)
        print(f"\n模型已成功保存至: {model_save_path}")
        # --- 保存模型的代码结束 ---
        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)

        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)


        print(f"训练集 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
        print(f"测试集 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

        plot_output_dir = f'./tabpfn_plots_自然老化_{target_column_name}'
        plot_single_model_jointgrid(
            y_train, y_pred_train, y_test, y_pred_test,
            r2_train, r2_test, "TabPFN", target_column_name, output_dir=plot_output_dir
        )

        print("\n开始为 TabPFN 进行 SHAP 分析...")
        if not pfn_interpret_available:
            print("错误: tabpfn_extensions 未安装，无法进行SHAP分析。");
            return

        print(f"使用 {X_test.shape[0]} 条测试数据进行SHAP分析...")

        fitted_preprocessor = full_pipeline.named_steps['preprocessor']
        fitted_model = full_pipeline.named_steps['regressor']
        X_test_transformed = fitted_preprocessor.transform(X_test)

        try:
            raw_feature_names = fitted_preprocessor.get_feature_names_out()
            # Clean feature names from ColumnTransformer output (e.g., 'numeric_features__colA' -> 'colA')
            cleaned_feature_names = [name.split('__')[-1] for name in raw_feature_names]
        except AttributeError:
            print("警告: 预处理器不支持 get_feature_names_out()，尝试从原始特征列获取名称。")
            cleaned_feature_names = X.columns.tolist()

        if len(cleaned_feature_names) != X_test_transformed.shape[1]:
            print(f"警告: 清理后的特征名数量 ({len(cleaned_feature_names)}) 与转换后的特征列数量 ({X_test_transformed.shape[1]}) 不匹配。SHAP 图可能不准确。请检查预处理步骤。将生成默认特征名。")
            cleaned_feature_names = [f'feature_{i}' for i in range(X_test_transformed.shape[1])]


        shap_values_obj = interpretability.shap.get_shap_values(
            estimator=fitted_model,
            test_x=X_test_transformed,
            attribute_names=cleaned_feature_names,
            algorithm="permutation"
        )

        # Ensure shap_values_obj is a numpy array for saving functions if it's a shap.Explanation object
        if hasattr(shap_values_obj, 'values'):
            shap_values_array_for_save = shap_values_obj.values
        else:
            shap_values_array_for_save = shap_values_obj

        print(f"\n保存SHAP值到CSV文件（目标列: {target_column_name}）...")
        csv_output_dir = f'./shap_csv_自然老化_{target_column_name}'
        saved_csv_path = save_shap_values_to_csv(
            shap_values=shap_values_array_for_save,
            feature_names=cleaned_feature_names,
            target_column_name=target_column_name,
            output_dir=csv_output_dir
        )

        if saved_csv_path:
            print(f"✓ SHAP值CSV文件保存成功: {saved_csv_path}")
        else:
            print("✗ SHAP值CSV文件保存失败")

        print(f"\n保存特征原始值+SHAP值组合数据（用于PDP图绘制）...")
        combined_output_dir = f'./features_shap_combined_自然老化_{target_column_name}'
        saved_combined_path, readme_path = save_features_and_shap_combined(
            X_test=X_test, # Pass original X_test here as we want original feature values for combined data
            shap_values=shap_values_array_for_save, # Pass numpy array of SHAP values
            feature_names=cleaned_feature_names, # Use cleaned feature names that match transformed data
            target_column_name=target_column_name,
            y_test=y_test,
            y_pred=y_pred_test,
            output_dir=combined_output_dir
        )

        if saved_combined_path:
            print(f"✓ 特征+SHAP组合数据保存成功: {saved_combined_path}")
        else:
            print("✗ 特征+SHAP组合数据保存失败")

        shap_plot_dir = f'./shap_plots_自然老化_{target_column_name}';
        os.makedirs(shap_plot_dir, exist_ok=True)

        with plt.rc_context({'font.size': 20, 'font.weight': 'bold'}):
            shap_summary_file = os.path.join(shap_plot_dir, f'自然老化_TabPFN_shap_summary_{target_column_name}.pdf')

            # For shap.summary_plot, pass the shap.Explanation object directly
            shap.summary_plot(shap_values_obj, X_test_transformed, feature_names=cleaned_feature_names,
                             show=False, cmap=plt.get_cmap('plasma'))

            plt.suptitle(f'TabPFN SHAP Summary Plot - {target_column_name} (Natural Aging)', fontsize=22, fontweight='bold', y=1.02)
            plt.savefig(shap_summary_file, dpi=600, bbox_inches='tight');
            plt.close()
            print(f"TabPFN SHAP summary plot 已保存至: {shap_summary_file}")

        print("计算并绘制特征重要性柱状图...")
        # Ensure shap_values_array is a numpy array for mean_abs_shap calculation
        if hasattr(shap_values_obj, 'values'):
            shap_values_array = shap_values_obj.values
        else:
            shap_values_array = shap_values_obj

        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': cleaned_feature_names,
            'Importance': mean_abs_shap
        })
        total_importance = feature_importance_df['Importance'].sum()
        if total_importance > 0:
            feature_importance_df['Percentage'] = (feature_importance_df['Importance'] / total_importance) * 100
        else:
            feature_importance_df['Percentage'] = 0
        feature_importance_df.sort_values(by='Percentage', ascending=False, inplace=True)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x='Percentage', y='Feature', data=feature_importance_df)

        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.1, p.get_y() + p.get_height() / 2., f'{width:.2f}%',
                                 va='center', ha='left', fontsize=20, fontweight='bold')

        plt.title(f'TabPFN Feature Importance (Mean Absolute SHAP) - {target_column_name} (Natural Aging)', fontsize=22, fontweight='bold')
        plt.xlabel('Importance (%)', fontsize=20, fontweight='bold')
        plt.ylabel('Feature', fontsize=20, fontweight='bold')

        for label in ax.get_yticklabels():
            label.set_fontsize(20)
            label.set_fontweight('bold')

        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for label in ax.get_xticklabels():
            label.set_fontsize(18)
            label.set_fontweight('bold')

        plt.xlim(0, feature_importance_df['Percentage'].max() * 1.15)
        plt.tight_layout()
        importance_plot_filename = os.path.join(shap_plot_dir, f'自然老化_TabPFN_feature_importance_bar_{target_column_name}.pdf')
        plt.savefig(importance_plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"特征重要性柱状图已保存至: {importance_plot_filename}")

    except Exception as e:
        print(f"TabPFN 模型主处理流程失败: {e}");
        traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
    # 请根据你的实际文件路径修改这里
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"

    df_original = load_and_process_data(file_path)

    if df_original is not None:
        # 关键：在这里定义所有你的输出列。代码会确保这些列都不会成为模型的特征。
        # 当你选择其中一个作为 target_column_name 时，其他列也会被排除。
        target_column_options = ['pH-S-CH', 'Cd-CH']

        while True:
            print("\n请选择要训练的目标列:")
            [print(f"{i + 1}. {col_name}") for i, col_name in enumerate(target_column_options)]
            print("0. 退出程序")
            try:
                user_choice_str = input(f"请输入选项编号 (0-{len(target_column_options)}): ")
                if not user_choice_str.strip(): continue
                user_choice = int(user_choice_str)
                if user_choice == 0:
                    print("程序退出。")
                    break
                elif 1 <= user_choice <= len(target_column_options):
                    selected_target_col = target_column_options[user_choice - 1]
                    train_and_evaluate_autotabpfn(df_original.copy(), selected_target_col)
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"主程序循环中发生了一个未预料的错误: {e}")
                traceback.print_exc()
    else:
        print("数据加载和初步处理失败，无法执行训练。")