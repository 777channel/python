import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import torch
import traceback
import warnings
warnings.filterwarnings('ignore')

# --- 导入检查 ---
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 从 'tabpfen' 库导入成功。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: 标准 TabPFN库 (tabpfen) 未找到。TabPFN 模型将不可用。")

# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# --- 数据加载函数 ---
def load_and_process_data(file_path):
    """加载数据，对特定列进行数值转换和初步检查。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"数据从 '{file_path}' (UTF-8编码) 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}")
        return None
    
    # 对指定的特征和目标列尝试进行数值转换
    cols_to_convert = ['BR', 'pH-S-CH', 'Cd-CH']
    for col_name in cols_to_convert:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    print(f"\n缺失值统计:")
    missing_info = df.isnull().sum()
    missing_info = missing_info[missing_info > 0]
    if len(missing_info) > 0:
        print(missing_info)
    else:
        print("没有发现缺失值。")
    
    return df

# --- 改进的缺失值处理函数 ---
def create_imputation_pipeline():
    """创建防止数据泄露的缺失值处理pipeline"""
    
    # 对数值特征使用KNN插补
    numeric_imputer = KNNImputer(n_neighbors=5, weights='uniform')
    
    # 对分类特征使用众数插补（如果有的话）
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    
    return numeric_imputer, categorical_imputer

# --- 改进1: 模型选择和超参数调优 ---
def get_models_with_hyperparameters():
    """定义模型及其超参数网格"""
    models_config = {
        "线性回归": {
            "model": LinearRegression(),
            "params": {}  # 线性回归没有超参数需要调优
        },
        "岭回归": {
            "model": Ridge(),
            "params": {
                'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        "随机森林": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5, 10]
            }
        },
        "梯度提升树 (GBDT)": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [50, 100, 150],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.2]
            }
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [50, 100, 150],
                'regressor__max_depth': [3, 5, 7],
                'regressor__learning_rate': [0.01, 0.1, 0.2]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, verbosity=-1),
            "params": {
                'regressor__n_estimators': [100],
                'regressor__num_leaves': [20],
                'regressor__learning_rate': [0.01]
            }
        },
        "极端随机树 (ExtraTrees)": {
            "model": ExtraTreesRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [50, 100, 200],
                'regressor__max_depth': [None, 10, 20]
            }
        },
        "支持向量回归 (SVR)": {
            "model": SVR(),
            "params": {
                'regressor__C': [0.1, 1, 10],
                'regressor__kernel': ['rbf', 'linear'],
                'regressor__gamma': ['scale', 'auto']
            }
        }
    }
    
    # 添加TabPFN（如果可用）
    if tabpfn_available and TabPFNRegressor is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models_config["TabPFN Regressor"] = {
            "model": TabPFNRegressor(device=device),
            "params": {}  # TabPFN通常不需要调参
        }
    
    return models_config

# --- 改进2: 更好的交叉验证评估函数 ---
def evaluate_model_with_cv(pipeline, param_grid, X_train, y_train, cv=10, scoring='r2'):
    """使用交叉验证进行模型评估和超参数调优"""
    
    if param_grid:  # 如果有参数需要调优
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, 
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        grid_search.fit(X_train, y_train)
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    else:  # 没有参数需要调优，直接交叉验证
        cv_results = cross_validate(
            pipeline, X_train, y_train, 
            cv=cv, 
            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            n_jobs=-1, 
            return_train_score=False
        )
        
        pipeline.fit(X_train, y_train)  # 训练最终模型
        
        return {
            'best_model': pipeline,
            'best_params': {},
            'best_cv_score': np.mean(cv_results['test_r2']),
            'cv_results': cv_results
        }

# --- 改进3: 结果记录和比较 ---
def compare_models_performance(results_dict):
    """比较所有模型的性能并选择最佳模型"""
    
    print("\n" + "="*80)
    print("模型性能比较 (基于交叉验证R²分数)")
    print("="*80)
    
    model_scores = []
    for model_name, result in results_dict.items():
        cv_score = result['best_cv_score']
        model_scores.append((model_name, cv_score, result))
        print(f"{model_name:20s}: CV R² = {cv_score:.4f}")
    
    # 按性能排序
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "-"*80)
    print("模型排名:")
    for i, (model_name, score, _) in enumerate(model_scores, 1):
        print(f"{i}. {model_name}: {score:.4f}")
    
    best_model_name, best_score, best_result = model_scores[0]
    print(f"\n最佳模型: {best_model_name} (CV R² = {best_score:.4f})")
    
    return best_model_name, best_result

# --- 修改后的模型对比绘图函数 ---
def plot_all_models_comparison(plotting_data_list, target_col, output_dir='.'):
    """为目标变量创建一个大图，包含所有模型的预测比较。显示交叉验证结果"""
    if not plotting_data_list:
        print(f"没有可用于绘图的结果: {target_col}")
        return

    n_rows, n_cols = 3, 3 # 固定为 3x3
    model_order = ['LR', 'Ridge', 'RF', 'GBDT', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SVR', 'TabPFN']
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 30))
    axes = axes.flatten()

    train_color = '#2aace2'
    test_color = '#f57a85'

    model_results_dict = {r['Model']: r for r in plotting_data_list}
    
    valid_results = [r for r in plotting_data_list if isinstance(r.get('CV_Score'), (int, float))]
    best_model_name = max(valid_results, key=lambda x: x['CV_Score'])['Model'] if valid_results else None

    i = -1
    for i, model_name_short in enumerate(model_order):
        if i >= len(axes): break
        ax = axes[i]
        
        if model_name_short in model_results_dict:
            r = model_results_dict[model_name_short]
            
            y_train, y_pred_train = np.array(r.get('Train_Values', [])), np.array(r.get('Train_Predictions', []))
            y_test, y_pred_test = np.array(r.get('Test_Values', [])), np.array(r.get('Test_Predictions', []))
            r2_train, rmse_train, mae_train = r.get('R2_Train', np.nan), r.get('RMSE_Train', np.nan), r.get('MAE_Train', np.nan)
            r2_test, rmse_test, mae_test = r.get('R2_Test', np.nan), r.get('RMSE_Test', np.nan), r.get('MAE_Test', np.nan)
            cv_score = r.get('CV_Score', np.nan) 
            if y_train.size > 0: ax.scatter(y_train, y_pred_train, alpha=1, edgecolor='w',facecolors=train_color, s=400)
            if y_test.size > 0: ax.scatter(y_test, y_pred_test, alpha=1,edgecolor='w',facecolors=test_color, s=400)
            
            all_y_values = np.concatenate([val for val in [y_train, y_pred_train, y_test, y_pred_test] if val.size > 0])
            if all_y_values.size > 0:
                min_val, max_val = np.nanmin(all_y_values), np.nanmax(all_y_values)
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.0)
                padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
                ax.set_xlim(min_val - padding, max_val + padding)
                ax.set_ylim(min_val - padding, max_val + padding)

            ax.text(0.05, 0.95, f"Train\nR²: {r2_train:.3f}\nRMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}", transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', color=train_color,alpha=1)
            ax.text(0.65, 0.27, f"Test\nR²: {r2_test:.3f}\nRMSE: {rmse_test:.3f}\nMAE: {mae_test:.3f}", transform=ax.transAxes, fontsize=28, fontweight='bold', va='top', color=test_color,alpha=1)
            
            model_label_char = chr(97 + i)
            is_best_text = "(Best)" if model_name_short == best_model_name else ""
            ax.set_title(f'({model_label_char}) {model_name_short}{is_best_text}\nCV R²={cv_score:.3f}', fontsize=30, fontweight='bold')
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(30); label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        else:
            ax.text(0.5, 0.5, f'({chr(97 + i)}) {model_name_short}\n无数据', ha='center', va='center', transform=ax.transAxes, fontsize=28); ax.axis('off')

    for j in range(i + 1, n_rows * n_cols):
        if j < len(axes): fig.delaxes(axes[j])

    fig.text(0.5, 0.02, 'Observed Values', ha='center', fontsize=30, fontweight='bold')
    fig.text(0.02, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=30,fontweight='bold')
    # 修改主标题，添加“自然老化”字样
    fig.suptitle(f'Comparison of  Models - {target_col}', fontsize=32, y=0.98, fontweight='bold')

    # 调整子图间距
    # fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95]) # 你可以选择保留或注释掉这行，然后手动调整
    plt.subplots_adjust(wspace=0.3, hspace=0.35, left=0.07, right=0.95, top=0.93, bottom=0.07) # 示例值，你可以根据需要调整

    os.makedirs(output_dir, exist_ok=True)
    # 修改保存路径，添加“自然老化”字样
    save_path = os.path.join(output_dir, f'{target_col}_all_models_comparison_optimized.pdf')
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"所有模型比较图已成功保存至 {save_path}")

# --- 改进的主训练与评估函数 ---
def train_and_evaluate_models(df, target_column_name):
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练模型 ---")
    df_processed = df.copy()

    # 首先删除目标变量为空的行
    df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.shape[0] < 10:
        print(f"数据过少 (少于10行)，无法进行有效训练。")
        return None

    print(f"删除目标变量缺失值后，剩余样本数: {df_processed.shape[0]}")

    # --- 1. 数据划分（使用分层抽样）---
    strata_col_name = 'T_aging'
    if strata_col_name not in df_processed.columns:
        print(f"警告: 未找到用于分层的列 '{strata_col_name}'。将使用普通随机抽样。")
        strata = None
    else:
        if all(df_processed[strata_col_name].value_counts() > 1):
            strata = df_processed[strata_col_name]
            print(f"将使用 '{strata_col_name}' 列进行分层抽样。各类别数量:\n{df_processed[strata_col_name].value_counts()}")
        else:
            print(f"警告: 分层列 '{strata_col_name}' 中存在样本数小于2的类别，无法进行分层。将使用普通随机抽样。")
            strata = None

    # 定义特征和目标
    potential_targets = ['pH-S-CH', 'Cd-CH']
    cols_to_exclude = potential_targets + [strata_col_name]
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude and col != target_column_name]
    
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]

    # 数据分割 - 重要：在插补之前分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=strata
    )
    print(f"\n数据分割完成:")
    print(f"训练集规模: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 检查训练集和测试集的缺失值情况
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    print(f"训练集缺失值总数: {train_missing}")
    print(f"测试集缺失值总数: {test_missing}")
    
    # --- 2. 防止数据泄露的缺失值处理 ---
    print("\n--- 开始缺失值处理 (防止数据泄露) ---")
    
    # 分别处理数值型和分类型特征
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"数值型特征: {len(numeric_columns)} 个")
    print(f"分类型特征: {len(categorical_columns)} 个")
    
    # 使用训练集拟合插补器
    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    if len(numeric_columns) > 0:
        # 对数值型特征使用KNN插补
        numeric_imputer = KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')
        X_train_numeric_imputed = numeric_imputer.fit_transform(X_train[numeric_columns])
        X_test_numeric_imputed = numeric_imputer.transform(X_test[numeric_columns])
        
        # 重新组装数据
        X_train_imputed[numeric_columns] = X_train_numeric_imputed
        X_test_imputed[numeric_columns] = X_test_numeric_imputed
        
        print(f"数值型特征插补完成")
    
    if len(categorical_columns) > 0:
        # 对分类型特征使用众数插补
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X_train_categorical_imputed = categorical_imputer.fit_transform(X_train[categorical_columns])
        X_test_categorical_imputed = categorical_imputer.transform(X_test[categorical_columns])
        
        # 重新组装数据
        X_train_imputed[categorical_columns] = X_train_categorical_imputed
        X_test_imputed[categorical_columns] = X_test_categorical_imputed
        
        print(f"分类型特征插补完成")
    
    # 验证插补结果
    train_missing_after = X_train_imputed.isnull().sum().sum()
    test_missing_after = X_test_imputed.isnull().sum().sum()
    print(f"插补后训练集缺失值: {train_missing_after}")
    print(f"插补后测试集缺失值: {test_missing_after}")
    
    # --- 3. 预处理器定义（在插补后的数据上） ---
    log_transform_cols = ['Cd-C', 'BET']
    log_transform_cols = [col for col in log_transform_cols if col in X_train_imputed.columns]
    cec_col_list = ['CEC'] if 'CEC' in X_train_imputed.columns else []
    other_numeric_cols = [col for col in X_train_imputed.columns if col not in log_transform_cols + cec_col_list]

    # 注意：这里不再需要插补步骤，因为已经在前面处理了
    log_pipeline = Pipeline(steps=[('log1p', FunctionTransformer(np.log1p, validate=False)), ('scaler_log', StandardScaler())])
    standard_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
    
    transformers_list = []
    if log_transform_cols: 
        transformers_list.append(('log_features', log_pipeline, log_transform_cols))
    if cec_col_list: 
        transformers_list.append(('cec_feature', standard_pipeline, cec_col_list))
    if other_numeric_cols: 
        transformers_list.append(('other_numeric_features', standard_pipeline, other_numeric_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

    # --- 4. 获取模型配置 ---
    models_config = get_models_with_hyperparameters()
    
    # 模型名称映射
    model_name_map = {
        "线性回归": "LR", "岭回归": "Ridge", "随机森林": "RF", 
        "梯度提升树 (GBDT)": "GBDT", "XGBoost": "XGBoost", "LightGBM": "LightGBM", 
        "极端随机树 (ExtraTrees)": "ExtraTrees", "支持向量回归 (SVR)": "SVR", 
        "TabPFN Regressor": "TabPFN"
    }

    # --- 5. 模型训练和评估 ---
    all_results = {}
    plotting_data_list = []
    
    for name, config in models_config.items():
        print(f"\n--- 正在处理模型: {name} ---")
        
        # 检查TabPFN的数据限制
        if name == "TabPFN Regressor":
            if X_train_imputed.shape[0] > 1024 or X_train_imputed.shape[1] >= 100:
                print(f"跳过TabPFN: 数据规模超出限制 (样本数: {X_train_imputed.shape[0]}, 特征数: {X_train_imputed.shape[1]})")
                continue
        
        try:
            # 创建pipeline
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('regressor', config['model'])
            ])
            
            # 模型评估和调优（使用插补后的训练数据）
            result = evaluate_model_with_cv(
                full_pipeline, 
                config['params'], 
                X_train_imputed, y_train, 
                cv=5, 
                scoring='r2'
            )
            
            all_results[name] = result
            best_model = result['best_model']
            
            # 训练集预测
            y_pred_train = best_model.predict(X_train_imputed)
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_train = mean_absolute_error(y_train, y_pred_train)
            
            # 测试集预测（使用插补后的测试数据）
            y_pred_test = best_model.predict(X_test_imputed)
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            # 打印结果
            print(f"最佳参数: {result['best_params']}")
            print(f"交叉验证R²: {result['best_cv_score']:.4f}")
            print(f"训练集 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
            print(f"测试集 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
            
            # 保存绘图数据（新增CV_Score字段）
            plotting_data_list.append({
                'Model': model_name_map.get(name, name),
                'Train_Values': y_train.copy(),
                'Train_Predictions': y_pred_train.copy(),
                'Test_Values': y_test.copy(),
                'Test_Predictions': y_pred_test.copy(),
                'R2_Train': r2_train,
                'RMSE_Train': rmse_train,
                'MAE_Train': mae_train,
                'R2_Test': r2_test,
                'RMSE_Test': rmse_test,
                'MAE_Test': mae_test,
                'CV_Score': result['best_cv_score']  # 新增交叉验证分数
            })
            
        except Exception as e:
            print(f"模型 {name} 处理失败: {e}")
            traceback.print_exc()
    
    # --- 6. 比较模型性能 ---
    if all_results:
        best_model_name, best_result = compare_models_performance(all_results)
        
        # 保存最佳模型信息
        print(f"\n最终推荐模型: {best_model_name}")
        print(f"最佳超参数: {best_result['best_params']}")
        print(f"交叉验证性能: {best_result['best_cv_score']:.4f}")
    
    # --- 7. 生成可视化结果 ---
    if plotting_data_list:
        plot_output_dir = f'./model_comparison_plots_{target_column_name}'
        plot_all_models_comparison(plotting_data_list, target_column_name, output_dir=plot_output_dir)
    else:
        print("没有模型成功完成评估。")
    
    return all_results

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
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
                    print("程序退出。")
                    break
                elif 1 <= user_choice <= len(target_column_options):
                    selected_target_col = target_column_options[user_choice - 1]
                    results = train_and_evaluate_models(df_original.copy(), selected_target_col)
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"主程序循环中发生了一个未预料的错误: {e}")
                traceback.print_exc()
    else:
        print("数据加载和初步处理失败，无法执行训练。")