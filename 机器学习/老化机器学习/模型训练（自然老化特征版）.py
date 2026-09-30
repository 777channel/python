import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import traceback
import warnings
warnings.filterwarnings('ignore') # 忽略警告

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
def train_and_fine_tune_xgboost(df, target_column_name, all_potential_targets):
    """
    仅使用XGBoost模型，并基于之前的最佳结果进行精细化的网格搜索调优。
    """
    print(f"\n{'='*25}\n   开始为目标 '{target_column_name}' 精细化调优XGBoost模型\n{'='*25}")

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

    # 3. [!!核心修改!!] 定义针对不同目标的精细化超参数网格
    # 网格的中心点是您上次运行找到的最佳参数
    target_specific_grids = {
        'Cd-CH': {
            'regressor__n_estimators': [800],
            'regressor__learning_rate': [0.05],
            'regressor__max_depth': [3],
            'regressor__subsample': [0.6],
            'regressor__colsample_bytree': [0.66],
            'regressor__gamma': [0],
            'regressor__reg_alpha': [0.2],
            'regressor__reg_lambda': [2.5]
        },
        'pH-S-CH': {
            'regressor__n_estimators': [800],
            'regressor__learning_rate': [0.01],
            'regressor__max_depth': [3],
            'regressor__subsample': [0.6, 0.7, 0.8],
            'regressor__colsample_bytree': [0.8],
            'regressor__gamma': [0],
            'regressor__reg_alpha': [ 0.2],
            'regressor__reg_lambda': [0.5]
        }
    }
    
    # 根据当前目标选择正确的参数网格
    param_grid = target_specific_grids.get(target_column_name)
    if not param_grid:
        print(f"错误: 未为目标 '{target_column_name}' 定义参数网格。")
        return

    # 4. 创建Pipeline并进行网格搜索
    n_samples = len(X_train)
    n_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
    
    pipeline = Pipeline(steps=[
        ('preprocessor', KNNImputer(n_neighbors=n_neighbors)),
        ('regressor', XGBRegressor(random_state=42, objective='reg:squarederror'))
    ])
    
    cross_validator = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=cross_validator,
        scoring='r2', 
        n_jobs=-1, 
        verbose=1
    )
    
    print(f"开始为 XGBoost 进行精细化网格搜索...")
    grid_search.fit(X_train, y_train)
    print("网格搜索完成！")
    
    print(f"最佳交叉验证 R² 分数: {grid_search.best_score_:.4f}")
    print(f"最佳参数: {grid_search.best_params_}")
    
    best_pipeline = grid_search.best_estimator_
    
    r2_train = best_pipeline.score(X_train, y_train)
    r2_test = best_pipeline.score(X_test, y_test)
    
    print(f"\nXGBoost - 最终性能 - 训练集 R²: {r2_train:.4f}, 测试集 R²: {r2_test:.4f}")
    
    # 保存最佳模型
    version_suffix = f"Natural_Aging_XGBoost-FineTuned_{target_column_name}"
    model_save_dir = './saved_models'
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f'pipeline_{version_suffix}.joblib')
    joblib.dump(best_pipeline, model_save_path)
    print(f"已保存最终优化后的 XGBoost 模型至: {model_save_path}")

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
                    train_and_fine_tune_xgboost(df_original.copy(), selected_target_col, target_column_options)
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"主程序循环中发生了一个未预料的错误: {e}")
                traceback.print_exc()
    else:
        print("数据加载和初步处理失败，无法执行训练。")