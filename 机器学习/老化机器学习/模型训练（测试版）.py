import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 导入检查 ---
try:
    # AutoTabPFNRegressor 需要从 tabpfn_extensions 库导入
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    print("成功导入 AutoTabPFNRegressor。")
    autotabpfn_available = True
except ImportError:
    print("严重错误: 'tabpfn_extensions' 库或 AutoTabPFNRegressor 未找到。模型将不可用。")
    autotabpfn_available = False
    AutoTabPFNRegressor = None


# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# --- 数据加载函数 ---
def load_and_process_data(file_path):
    """加载数据并对所有列进行初步数值转换。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")
    df.info()
    return df

# --- [!!核心!!] 主训练与评估函数 ---
def train_with_autotabpfn(df, target_column_name, all_potential_targets):
    """
    使用 AutoTabPFN 模型进行训练和评估。
    """
    version_suffix = f"Natural_Aging_AutoTabPFN_{target_column_name}"
    print(f"\n{'='*25}\n   开始为目标 '{target_column_name}' 训练 AutoTabPFN 模型\n{'='*25}")
    print(f"--- 本次运行版本: {version_suffix} ---")

    # 1. 数据清洗与准备
    df_processed = df.copy()
    df_processed.dropna(subset=[target_column_name], inplace=True)
    
    strata_col_name = 'T-Aging'
    cols_to_exclude = list(all_potential_targets)
    if strata_col_name in df_processed.columns and strata_col_name not in cols_to_exclude:
        cols_to_exclude.append(strata_col_name)

    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]
    print(f"用于训练的特征数量: {len(feature_columns)}")

    # 2. 分层抽样
    strata = None
    if strata_col_name in df_processed.columns:
        strata_series = df_processed[strata_col_name]
        if all(strata_series.value_counts() >= 2):
            strata = strata_series
            print(f"已启用基于 '{strata_col_name}' 的分层抽样。")
        else:
            print(f"警告: '{strata_col_name}' 中存在样本数小于2的层级，无法进行分层抽样。")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strata
    )

    # 3. 预处理流程 (保持不变)
    n_samples = len(X_train)
    n_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
    preprocessor = KNNImputer(n_neighbors=n_neighbors)

    # 4. [!!核心修改!!] 模型实例化
    if not autotabpfn_available:
        print("错误: AutoTabPFNRegressor 不可用。无法继续训练。"); return

    # 您可以在这里调整最大训练时间（单位：秒）
    max_training_time = 300 # 5分钟
    
    print(f"实例化 AutoTabPFNRegressor，最大训练时间: {max_training_time} 秒。")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = AutoTabPFNRegressor(max_time=max_training_time, device=device)
    
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])

    # 5. 模型训练与评估
    try:
        print(f"\n=== 开始在 {device} 上拟合 AutoTabPFN 模型 (最多耗时 {max_training_time} 秒) ===")
        full_pipeline.fit(X_train, y_train)
        print("=== AutoTabPFN 模型拟合完成！===")
        
        r2_train = full_pipeline.score(X_train, y_train)
        r2_test = full_pipeline.score(X_test, y_test)
        
        print(f"\nAutoTabPFN - 最终性能 - 训练集 R²: {r2_train:.4f}, 测试集 R²: {r2_test:.4f}")
        
        # 保存模型
        model_save_dir = './saved_models'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'pipeline_{version_suffix}.joblib')
        joblib.dump(full_pipeline, model_save_path)
        print(f"已保存 AutoTabPFN 模型至: {model_save_path}")

    except Exception as e:
        print(f"模型主处理流程失败: {e}");
        traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"
    df_original = load_and_process_data(file_path)

    if df_original is not None:
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
                    print("程序退出。"); break
                elif 1 <= user_choice <= len(target_column_options):
                    selected_target_col = target_column_options[user_choice - 1]
                    # 调用新的训练函数
                    train_with_autotabpfn(df_original.copy(), selected_target_col, target_column_options)
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"主程序循环中发生了一个未预料的错误: {e}")
                traceback.print_exc()
    else:
        print("数据加载和初步处理失败，无法执行训练。")