import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 1. 全局绘图参数设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 12 
plt.rcParams['axes.labelsize'] = 14 
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12

# --- 2. 数据加载函数 (读取Excel) ---
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print(f"数据成功从 '{file_path}' 加载。")
    except FileNotFoundError:
        print(f"错误: 文件未找到于路径 '{file_path}'。请检查路径是否正确。")
        return None
    except Exception as e:
        print(f"错误: 读取 Excel 文件时发生错误。错误信息: {e}")
        return None
    df.columns = df.columns.str.strip()
    return df

# --- 3. 缺失值填充和绘图函数 ---
def plot_imputation_comparison(original_train_df, imputed_train_dfs, feature_name, output_dir):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
    axes = axes.flatten()
    method_order = ['Mean Imputation', 'Median Imputation', 'KNN Imputation', 'Random Forest Imputation']
    original_data = original_train_df[feature_name].dropna()
    all_data_for_feature = [original_data]
    for method in method_order:
        if method in imputed_train_dfs:
            all_data_for_feature.append(imputed_train_dfs[method][feature_name])
    if not any(not s.empty for s in all_data_for_feature):
        print(f"特征 '{feature_name}' 没有有效数据可供绘图。跳过。")
        plt.close(fig)
        return
    min_val = min(s.min() for s in all_data_for_feature if not s.empty)
    max_val = max(s.max() for s in all_data_for_feature if not s.empty)
    x_range_buffer = (max_val - min_val) * 0.1
    xlim_min = min_val - x_range_buffer
    xlim_max = max_val + x_range_buffer
    for i, method_name in enumerate(method_order):
        ax = axes[i]
        sns.histplot(original_data, kde=True, stat='density',
                     color='skyblue', label='Original Train Data', ax=ax,
                     edgecolor='black', line_kws={'linestyle': '--', 'linewidth': 1.5})
        if method_name in imputed_train_dfs:
            imputed_data = imputed_train_dfs[method_name][feature_name]
            sns.histplot(imputed_data, kde=True, stat='density',
                         color='gold', label='Imputed Train Data', ax=ax,
                         edgecolor='black', line_kws={'linestyle': '-', 'linewidth': 1.5})
        else:
            ax.text(0.5, 0.5, "Imputation Failed", ha='center', va='center', transform=ax.transAxes, color='red')
        ax.set_title(f'{method_name}', fontsize=16)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(xlim_min, xlim_max)
        ax.grid(True, linestyle=':', alpha=0.6)
    fig.suptitle(f'Imputation Method Comparison for Feature: {feature_name}', fontsize=22, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(output_dir, f'{feature_name}_Imputation_Comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"特征 '{feature_name}' 的对比图已保存至: {save_path}")

def run_imputation_pipeline(df, feature_cols, target_cols, strata_col, output_dir):
    print("\n--- 开始缺失值填充流程 ---")
    df_work = df.copy()

    for col in feature_cols + target_cols:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
        else:
            print(f"警告：列 '{col}' 在数据中不存在，请检查列名配置。")
            return
            
    X = df_work[feature_cols]
    y = df_work[target_cols]

    print(f"\n基于 '{strata_col}' 列进行分层抽样 (80% 训练集, 20% 测试集)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X[strata_col]
        )
        print("分层抽样成功。")
        print(f"训练集大小: {X_train.shape[0]} 行")
        print(f"测试集大小: {X_test.shape[0]} 行")
    except ValueError as e:
        print(f"错误: 分层抽样失败。原因: {e}")
        return

    cols_to_impute_and_plot = ['OM', 'CEC']
    actual_cols_needing_imputation = [col for col in cols_to_impute_and_plot if X_train[col].isnull().any()]
    if not actual_cols_needing_imputation:
        print("\n指定列 'OM' 和 'CEC' 在训练集中没有发现缺失值，无需填充。")
        return
    print(f"\n将在以下特征列中进行缺失值填充: {actual_cols_needing_imputation}")

    imputers = {
        'Mean Imputation': SimpleImputer(),
        'Median Imputation': SimpleImputer(strategy='median'),
        'KNN Imputation': KNNImputer(n_neighbors=5),
        'Random Forest Imputation': IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42), max_iter=10, random_state=42)
    }

    imputed_train_dfs = {}

    for name, imputer in imputers.items():
        print(f"\n正在执行: {name}...")
        X_train_imputed = X_train.copy()
        X_test_imputed = X_test.copy()
        try:
            X_train_imputed[feature_cols] = imputer.fit_transform(X_train[feature_cols])
            X_test_imputed[feature_cols] = imputer.transform(X_test[feature_cols])
            imputed_train_dfs[name] = X_train_imputed
            df_full_imputed = pd.concat([pd.concat([X_train_imputed, y_train], axis=1), pd.concat([X_test_imputed, y_test], axis=1)]).sort_index()
            output_filename = os.path.join(output_dir, f'data_imputed_{name.replace(" ", "_")}.csv')
            df_full_imputed.to_csv(output_filename, index=False, encoding='utf-8-sig')
            print(f"填充后的完整数据已保存至: {output_filename}")
        except Exception as e:
            print(f"错误: {name} 执行失败。错误信息: {e}")

    print("\n正在为 'OM' 和 'CEC' 生成对比图...")
    for feature in cols_to_impute_and_plot:
        if feature in actual_cols_needing_imputation:
            plot_imputation_comparison(
                original_train_df=X_train,
                imputed_train_dfs=imputed_train_dfs,
                feature_name=feature,
                output_dir=output_dir
            )

    print("\n--- 缺失值填充流程全部完成 ---")


# --- 主程序入口 ---
if __name__ == "__main__":
    
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\微塑料+cd训练版本.xlsx"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'MPs', 'Size', 'MPR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'MPs'
    
    # --- 已按您的要求修改 ---
    # 定义输出文件夹的路径和名称
    output_directory = r"机器学习\微塑料机器学习输出\微塑料+cd缺失值填充"
    # --- 修改结束 ---

    # 确保这个多层级的文件夹存在，如果不存在就创建它
    os.makedirs(output_directory, exist_ok=True)
    
    original_df = load_data(file_path)
    if original_df is not None:
        run_imputation_pipeline(
            df=original_df,
            feature_cols=feature_columns,
            target_cols=target_columns,
            strata_col=stratification_column,
            output_dir=output_directory
        )