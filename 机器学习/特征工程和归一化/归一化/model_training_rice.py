import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import warnings
import traceback
import time
import pickle
import json # 用于将参数字典转换为字符串

warnings.filterwarnings('ignore')


def initialize_models():
    """
    初始化所有要测试的模型及其扩展后的超参数网格。
    """
    # (此函数内容与之前版本完全相同，保持不变)
    models = {
        'LR': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'positive': [False]
            }
        },
        'Bayesian Ridge': {
            'model': BayesianRidge(),
            'params': {
                'n_iter': [300, 500],
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4],
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': np.logspace(-2, 2, 5),
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
                'max_iter': [1000, 2000],
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
            'params': {
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 1.5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, verbosity=-1),
            'params': {
                'n_estimators': [200],
                'num_leaves': [20],
                'max_depth': [5],
                'learning_rate': [0.05],
                'reg_alpha': [0.3],
                'reg_lambda': [0.5, 1, 1.5],
                'subsample': [0.9],
                'colsample_bytree': [0.9]
            }
        },
        'RF': {
            'model': RandomForestRegressor(random_state=42, bootstrap=True, oob_score=True),
            'params': {
                'n_estimators': [100, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [0.6, 0.7, 0.8, 'sqrt'],
                'max_samples': [0.6, 0.7, 0.8]
            }
        },
        'GBDT': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 200, 400],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'min_samples_split': [2, 4],
                'min_samples_leaf': [1, 5, 10],
                'max_features': [0.7, 0.8, 'sqrt']
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(random_state=42, bootstrap=True, oob_score=True),
            'params': {
                'n_estimators': [100, 400, 600],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [0.7, 0.8, 0.9, 'sqrt'],
                'max_samples': [0.7, 0.8, 0.9]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [1.0, 5.0, 10.0, 20.0],
                'epsilon': [0.01, 0.05, 0.1],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear'],
            }
        }
    }
    return models


def find_best_params(model_name, model_info, X_train, y_train, cv=5):
    """
    使用网格搜索找到模型的最佳参数，并同时返回最佳模型和最佳参数字典。
    """
    model = model_info['model']
    param_grid = model_info['params']

    # 如果没有参数网格，直接返回默认模型和空参数字典
    if not any(param_grid.values()):
        model.fit(X_train, y_train)
        print(f"\n{model_name} 使用默认参数进行训练。")
        return model, {}

    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=0
    )

    try:
        grid_search.fit(X_train, y_train)
        print(f"\n{model_name} 的最佳参数已找到:")
        # 将参数字典格式化输出，更美观
        best_params_str = json.dumps(grid_search.best_params_, indent=2)
        print(best_params_str)
        print(f"对应的网格搜索交叉验证R² (GridSearchCV best_score_): {grid_search.best_score_:.4f}")
        # 返回最佳估计器和最佳参数字典
        return grid_search.best_estimator_, grid_search.best_params_
    except Exception as e:
        print(f"模型 {model_name} 的网格搜索失败: {str(e)}")
        print("将使用默认参数继续...")
        model.fit(X_train, y_train)
        return model, {}


def run_training_pipeline(df, target_col, all_target_cols, test_size=0.3, random_state=42, output_dir='.'):
    """
    为单个指定的目标列执行完整的训练流程，并将详细结果保存到CSV。
    """
    print(f"\n当前目标列: '{target_col}'")

    # 1. 定义特征和目标
    feature_cols = [c for c in df.columns if c not in all_target_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # 2. 数据清理和划分
    valid_mask = y.notna()
    X, y = X[valid_mask], y[valid_mask]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"数据划分完成 -> 训练集: {len(X_train)}样本, 测试集: {len(X_test)}样本")

    # --- 变更点：初始化用于存储详细结果的CSV文件 ---
    detailed_results_path = os.path.join(output_dir, f'{target_col}_detailed_training_log.csv')
    # 定义CSV文件的列名
    csv_columns = [
        'Model', 'Best_Params', 'GridSearch_CV_R2', 'RepeatedKFold_CV_R2_Mean', 'RepeatedKFold_CV_R2_Std',
        'R2_Test', 'RMSE_Test', 'MAE_Test', 'R2_Train'
    ]
    # 创建一个空的DataFrame并写入头部，如果文件不存在
    if not os.path.exists(detailed_results_path):
        pd.DataFrame(columns=csv_columns).to_csv(detailed_results_path, index=False)

    # 3. 训练和评估流程
    model_infos = initialize_models()
    # 这个列表仍然用于在流程结束后找到最佳模型进行保存
    final_results_list = [] 
    cv_obj = RepeatedKFold(n_splits=5, n_repeats=5, random_state=random_state)

    for name, model_info in model_infos.items():
        try:
            print(f"\n--- 正在为 '{target_col}' 训练模型: {name} ---")
            
            # find_best_params现在返回模型和参数
            best_model, best_params = find_best_params(name, model_info, X_train, y_train, cv=5)
            
            # 使用最佳模型进行更稳健的交叉验证
            print(f"正在对 {name} 的最佳配置进行 {cv_obj.get_n_splits()} 次重复交叉验证...")
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_obj, scoring='r2', error_score='raise')
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
            print(f"重复交叉验证 R² (均值 ± 标准差): {cv_mean:.4f} ± {cv_std:.4f}")

            # 在完整训练集上拟合最终模型
            best_model.fit(X_train, y_train)
            
            # 在测试集上评估
            y_test_pred = best_model.predict(X_test)
            r2_test = r2_score(y_test, y_test_pred)
            print(f"最终测试集 R²: {r2_test:.4f}")

            # --- 变更点：准备并将当前模型的结果写入CSV ---
            current_result_row = {
                'Model': name,
                'Best_Params': json.dumps(best_params), # 将参数字典转为字符串
                'GridSearch_CV_R2': best_model.grid_search_score_ if hasattr(best_model, 'grid_search_score_') else None, # 假设find_best_params返回的是GridSearchCV对象
                'RepeatedKFold_CV_R2_Mean': cv_mean,
                'RepeatedKFold_CV_R2_Std': cv_std,
                'R2_Test': r2_test,
                'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'MAE_Test': mean_absolute_error(y_test, y_test_pred),
                'R2_Train': r2_score(y_train, best_model.predict(X_train))
            }
            # 以追加模式写入CSV，不包含头部
            pd.DataFrame([current_result_row]).to_csv(detailed_results_path, mode='a', header=False, index=False)
            
            # 将模型对象也添加到内存中的结果列表，用于后续保存最佳模型
            current_result_row['Model_Object'] = best_model
            final_results_list.append(current_result_row)

        except Exception as e:
            print(f"为 '{target_col}' 训练模型 {name} 时出错: {str(e)}")
            traceback.print_exc()
            continue

    if not final_results_list:
        print(f"警告: '{target_col}' 的所有模型都未能成功训练。")
        return

    # 4. 从内存中的结果找到最佳模型并保存
    # 注意：这里我们使用内存中的列表，因为CSV中没有模型对象
    final_results_df = pd.DataFrame(final_results_list)
    # 按重复交叉验证的均值R2排序
    final_results_df = final_results_df.sort_values('RepeatedKFold_CV_R2_Mean', ascending=False)
    
    print(f"\n===== '{target_col}' 的模型性能最终排名 (基于重复CV) =====")
    print(final_results_df.drop(columns=['Model_Object', 'Best_Params'])) # 打印一个简洁的排名表
    print("=" * (45 + len(target_col)))
    print(f"详细的参数和指标已全部保存在: {detailed_results_path}")


    if not final_results_df.empty:
        best_result_row = final_results_df.iloc[0]
        best_model_name = best_result_row['Model']
        
        model_save_dict = {
            'model': best_result_row['Model_Object'],
            'feature_names': feature_cols,
            'metrics': best_result_row.drop('Model_Object').to_dict()
        }
        
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, f"{target_col}_{best_model_name}_best_model.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_save_dict, f)
        print(f"\n最佳模型 ({best_model_name}) 的.pkl文件已保存至: {model_path}")
    

def main(data_path, output_dir='results'):
    """
    主函数，提供交互式菜单让用户选择要训练的目标列。
    """
    overall_start_time = time.time()
    
    try:
        print(f"从 {data_path} 加载数据...")
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        return
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 1. 自动识别所有潜在的目标列
    all_target_cols = [col for col in df.columns if col.startswith('CD_')]
    
    if not all_target_cols:
        print("错误: 在文件中未找到任何以 'CD_' 开头的目标列。请检查列名。")
        return
        
    # 2.向用户展示可选的目标列
    print("\n" + "="*50)
    print("检测到以下可用目标列：")
    for i, col_name in enumerate(all_target_cols, 1):
        print(f"  {i}: {col_name}")
    print("="*50)

    # 3. 获取并验证用户的选择
    selected_target_col = None
    while True:
        try:
            choice = input(f"请输入要训练的目标列编号 (1-{len(all_target_cols)}): ")
            choice_num = int(choice)
            if 1 <= choice_num <= len(all_target_cols):
                selected_target_col = all_target_cols[choice_num - 1]
                break
            else:
                print(f"输入错误: 请输入 1 到 {len(all_target_cols)} 之间的数字。")
        except ValueError:
            print("输入错误: 请输入一个有效的数字。")
        except (KeyboardInterrupt, EOFError):
            print("\n操作已取消。")
            return

    # 4. 为选定的目标列执行训练
    print("#" * 80)
    print(f"### 您已选择: {selected_target_col}，现在开始训练流程... ###")
    print("#" * 80)
    
    target_specific_output_dir = os.path.join(output_dir, selected_target_col)
    os.makedirs(target_specific_output_dir, exist_ok=True)
    print(f"结果将保存在: {target_specific_output_dir}")

    run_training_pipeline(
        df=df.copy(),
        target_col=selected_target_col,
        all_target_cols=all_target_cols,
        output_dir=target_specific_output_dir
    )
    
    overall_time = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("所选目标列的训练流程已完成！")
    print(f"总运行时间: {overall_time:.2f} 秒。")
    print(f"所有结果均已保存在 '{target_specific_output_dir}' 文件夹中。")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为多个目标列自动执行机器学习模型训练与评估')
    parser.add_argument('--data', type=str, 
                        default=r'C:\Users\AMDYE\OneDrive\python\output\模型结果\fully_processed_data(4).csv',
                        help='包含特征和多个目标列的数据CSV文件路径')
    parser.add_argument('--output', type=str, default='forexample',
                        help='存放所有结果的根目录')
    args = parser.parse_args()

    main(args.data, output_dir=args.output)