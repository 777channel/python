import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
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
warnings.filterwarnings('ignore') # 忽略警告

# --- 导入检查 (确保TabPFN可用性) ---
tabpfn_available = False # 初始状态：假设TabPFN不可用
TabPFNRegressor = None

try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 从 'tabpfn' 库导入成功。")
except ImportError:
    print("警告: 标准 TabPFN库 (tabpfn) 未找到。TabPFN 模型将不可用。")


# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# --- 数据加载函数 ---
def load_and_process_data(file_path):
    """加载数据，对所有列进行数值转换和初步检查。"""
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
    df.columns = [col.lstrip('\ufeff') for col in df.columns] # 移除BOM
    print("\n已移除列名开头的BOM字符。列名的大小写和空格将保持不变，请确保与代码中的列名精确匹配。")

    # 将所有列转换为数值型，无法转换的变为NaN
    print("\n尝试将所有列转换为数值型...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")
                
    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    print(f"\n缺失值统计 (所有列):")
    missing_info = df.isnull().sum()
    missing_info_present = missing_info[missing_info > 0]
    if len(missing_info_present) > 0:
        print(missing_info_present)
    else:
        print("没有发现缺失值。")
    
    return df


# --- 模型选择和超参数调优 ---
def get_models_with_hyperparameters():
    """定义模型及其超参数网格"""
    models_config = {
        "线性回归": {
            "model": LinearRegression(),
            "params": {}
        },
        "岭回归": {
            "model": Ridge(),
            "params": {
                'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0] 
            }
        },
        "随机森林": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [500], 
                'regressor__max_depth': [10], 
                'regressor__min_samples_split': [2], 
                'regressor__min_samples_leaf': [1] 
            }
        },
        "梯度提升树 (GBDT)": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [300], 
                'regressor__max_depth': [4], 
                'regressor__learning_rate': [0.05], 
                'regressor__subsample': [0.7] 
            }
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse'), 
            "params": {
                'regressor__n_estimators': [200], 
                'regressor__max_depth': [5], 
                'regressor__learning_rate': [0.2], 
                'regressor__subsample': [0.7], 
                'regressor__colsample_bytree': [0.9], 
                'regressor__reg_alpha': [0.0], 
                'regressor__reg_lambda': [0.0, 0.1, 0.5, 1.0] 
            }
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, verbosity=-1),
            "params": {
                'regressor__n_estimators': [500], 
                'regressor__num_leaves': [40], 
                'regressor__learning_rate': [0.2], 
                'regressor__reg_alpha': [1.0], 
                'regressor__reg_lambda': [1.0], 
                'regressor__colsample_bytree': [0.9, 1.0], 
                'regressor__subsample': [1.0] 
            }
        },
        "极端随机树 (ExtraTrees)": {
            "model": ExtraTreesRegressor(random_state=42),
            "params": {
                'regressor__n_estimators': [500], 
                'regressor__max_depth': [5, 10, 15, 20], 
                'regressor__min_samples_split': [2, 5], 
                'regressor__min_samples_leaf': [1] 
            }
        },
        "支持向量回归 (SVR)": {
            "model": SVR(),
            "params": {
                'regressor__C': [0.01, 0.1], 
                'regressor__kernel': ['rbf', 'linear', 'poly'], 
                'regressor__gamma': ['scale', 'auto'] 
            }
        }
    }
    
    # 添加TabPFN（如果可用）
    if tabpfn_available and TabPFNRegressor is not None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models_config["TabPFN Regressor"] = {
            "model": TabPFNRegressor(device=device), 
            "params": {}
        }
    
    return models_config

# --- 交叉验证评估函数 ---
def evaluate_model_with_cv(pipeline, param_grid, X_train, y_train, cv=10, scoring='r2'):
    """使用交叉验证进行模型评估和超参数调优"""
    
    if param_grid:
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=cv, # 保持传入的cv值
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False # 只返回测试分数，避免误判过拟合
        )
        grid_search.fit(X_train, y_train)
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    else:
        cv_results = cross_validate(
            pipeline, X_train, y_train, 
            cv=cv, # 保持传入的cv值
            scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'],
            n_jobs=-1, 
            return_train_score=False
        )
        
        pipeline.fit(X_train, y_train)
        
        return {
            'best_model': pipeline,
            'best_params': {},
            'best_cv_score': np.mean(cv_results['test_r2']),
            'cv_results': cv_results
        }

# --- 结果记录和比较 ---
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
    
    model_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "-"*80)
    print("模型排名:")
    for i, (model_name, score, _) in enumerate(model_scores, 1):
        print(f"{i}. {model_name}: {score:.4f}")
    
    best_model_name, best_score, best_result = model_scores[0]
    print(f"\n最佳模型: {best_model_name} (CV R² = {best_score:.4f})")
    
    return best_model_name, best_result

# --- 模型对比绘图函数 ---
def plot_all_models_comparison(plotting_data_list, target_col, output_dir='.'):
    """为目标变量创建一个大图，包含所有模型的预测比较。显示交叉验证结果"""
    if not plotting_data_list:
        print(f"没有可用于绘图的结果: {target_col}")
        return

    n_rows, n_cols = 3, 3 # 固定为 3x3
    # 确保这里的模型名称与get_models_with_hyperparameters中的映射一致
    model_order = ['LR', 'Ridge', 'RF', 'GBDT', 'XGBoost', 'LightGBM', 'ExtraTrees', 'SVR', 'TabPFN'] 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 28))
    axes = axes.flatten()

    train_color = '#b4d4e1'
    test_color = '#f4ba8a'

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
            if y_train.size > 0: ax.scatter(y_train, y_pred_train, alpha=1, edgecolors='w', facecolors=train_color, s=400)
            if y_test.size > 0: ax.scatter(y_test, y_pred_test, alpha=1, edgecolors='w', facecolors=test_color, s=400)
            
            all_y_values = np.concatenate([val for val in [y_train, y_pred_train, y_test, y_pred_test] if val.size > 0])
            if all_y_values.size > 0:
                min_val, max_val = np.nanmin(all_y_values), np.nanmax(all_y_values)
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2.0)
                padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
                ax.set_xlim(min_val - padding, max_val + padding)
                ax.set_ylim(min_val - padding, max_val + padding)

            ax.text(0.05, 0.95, f"Train\nR²: {r2_train:.3f}\nRMSE: {rmse_train:.3f}\nMAE: {mae_train:.3f}", transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', color=train_color)
            ax.text(0.7, 0.2, f"Test\nR²: {r2_test:.3f}\nRMSE: {rmse_test:.3f}\nMAE: {mae_test:.3f}", transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', color=test_color)
            
            model_label_char = chr(97 + i)
            is_best_text = "(Best)" if model_name_short == best_model_name else ""
            ax.set_title(f'({model_label_char}) {model_name_short}{is_best_text}\nCV R²={cv_score:.3f}', fontsize=24, fontweight='bold')
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(22); label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        else:
            ax.text(0.5, 0.5, f'({chr(97 + i)}) {model_name_short}\n无数据', ha='center', va='center', transform=ax.transAxes, fontsize=22); ax.axis('off')

    for j in range(i + 1, n_rows * n_cols):
        if j < len(axes): fig.delaxes(axes[j])

    fig.text(0.5, 0.02, 'Observed Values', ha='center', fontsize=26, fontweight='bold')
    fig.text(0.02, 0.5, 'Predicted Values', va='center', rotation='vertical', fontsize=26, fontweight='bold')
    # 修改主标题，添加“自然老化”字样
    fig.suptitle(f'自然老化模型比较 - {target_col}', fontsize=28, y=0.96, fontweight='bold')

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    # 修改保存路径，添加“自然老化”字样
    save_path = os.path.join(output_dir, f'{target_col}_自然老化_all_models_comparison_optimized.pdf')
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"所有模型比较图已成功保存至 {save_path}")

# --- 主训练与评估函数 ---
def train_and_evaluate_models(df, target_column_name):
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练模型 ---")
    
    df_processed = df.copy()

    # --- 根据用户要求：如果目标列存在缺失值，则删除这些行 ---
    initial_target_nan_count = df_processed[target_column_name].isnull().sum()
    if initial_target_nan_count > 0:
        print(f"\n目标列 '{target_column_name}' 存在 {initial_target_nan_count} 个缺失值。")
        print(f"根据您的要求，将删除包含这些缺失值的 {initial_target_nan_count} 行数据。")
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
        
        # 检查原始分层列的类型
        # 如果是数值型，且不是整数类型（即有小数点），则需要分箱
        if pd.api.types.is_numeric_dtype(df_processed[strata_col_name_original]) and \
           not pd.api.types.is_integer_dtype(df_processed[strata_col_name_original]):
            print(f"警告: 分层列 '{strata_col_name_original}' 是连续数值类型。将对其进行分箱处理以进行分层。")
            try:
                q_bins = min(10, df_processed[strata_col_name_original].nunique()) # 最多10个箱子，防止箱子过多
                if q_bins < 2: # 确保至少能分2个箱子
                    print(f"警告: 分层列 '{strata_col_name_original}' 唯一值数量过少 ({df_processed[strata_col_name_original].nunique()})，无法有效分箱。将退回普通随机抽样。")
                    strata = None
                else:
                    # 创建一个新的分箱列用于分层
                    strata_col_name_final = f'{strata_col_name_original}_binned_for_stratify' 
                    df_processed[strata_col_name_final] = pd.qcut(
                        df_processed[strata_col_name_original], 
                        q=q_bins, 
                        duplicates='drop' 
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
            filled_strata = imputer_strata.fit_transform(df_processed[[strata_col_name_final]].astype(str)).flatten()
            df_processed[strata_col_name_final] = pd.Series(filled_strata, index=df_processed.index)
            if pd.api.types.is_object_dtype(df_processed[strata_col_name_final]):
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
    # 1. 所有潜在的输出列 (来自 target_column_options)
    # 2. 原始分层列 (strata_col_name_original)
    # 3. 最终用于分层的列 (strata_col_name_final)
    
    # 获取所有潜在的输出列，因为它们都不应是当前模型的特征
    global target_column_options # 声明使用全局变量
    all_output_cols_to_exclude = list(target_column_options) 

    # 初始化最终排除列表，包含所有潜在的输出列
    final_cols_to_exclude = list(set(all_output_cols_to_exclude)) # 使用 set 去重，确保唯一性
    
    # 排除原始分层列（如果存在且尚未被排除）
    if strata_col_name_original in df_processed.columns and strata_col_name_original not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_original)
    
    # 排除分箱后的分层列（如果存在且不同于原始列，并且尚未被排除）
    if strata_col_name_final != strata_col_name_original and strata_col_name_final in df_processed.columns and strata_col_name_final not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_final)

    # 从所有列中选择特征列
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
        X, y, 
        test_size=0.3, 
        random_state=42,
        stratify=strata 
    )

    # --- 新增功能：打印用于训练的所有列 ---
    print(f"\n--- 实际传递给预处理和模型训练的特征列 ---")
    print(f"训练集特征列 (X_train.columns): {list(X_train.columns)}")
    print(f"测试集特征列 (X_test.columns): {list(X_test.columns)}")
    print(f"---------------------------------------")
    
    print(f"数据分割完成:")
    print(f"训练集规模: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: {X_test.shape}, y_test: {y_test.shape}")
    
    # --- 2. 构建数据预处理 Pipeline (包含缺失值处理和标准化/独热编码) ---
    print("\n--- 开始构建数据预处理 Pipeline ---")
    
    # 获取 X_train 的列，保持原始大小写
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    # 定义数值特征的预处理步骤 (KNN 插补 + 标准化)
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')),
        ('scaler', StandardScaler())
    ])

    # 定义分类特征的预处理步骤 (众数插补 + 独热编码)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # 众数插补
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # handle_unknown='ignore' 处理测试集中出现的新类别
    ])

    transformers_list = []
    
    # 对 Cd-C 和 BET 进行 log1p 变换，其他数值列进行标准化
    # 这些列名必须与您CSV文件中的实际列名大小写完全匹配
    log_transform_cols = ['Cd-C', 'BET'] 
    log_transform_cols_actual = [col for col in log_transform_cols if col in numeric_columns]
    other_numeric_cols = [col for col in numeric_columns if col not in log_transform_cols_actual]
    
    if log_transform_cols_actual:
        transformers_list.append(('log_numeric', Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')),
            ('log1p', FunctionTransformer(np.log1p, validate=False)), 
            ('scaler', StandardScaler())
        ]), log_transform_cols_actual))
    
    if other_numeric_cols:
        transformers_list.append(('other_numeric', numeric_transformer, other_numeric_cols))
            
    if categorical_columns:
        transformers_list.append(('categorical', categorical_transformer, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers_list,
        remainder='drop' # 确保只有明确处理的特征进入模型
    )

    # --- 3. 获取模型配置 ---
    models_config = get_models_with_hyperparameters()
    
    # 模型名称映射 (确保与 plotting_data_list 中的 'Model' 键一致)
    model_name_map = {
        "线性回归": "LR", "岭回归": "Ridge", "随机森林": "RF", 
        "梯度提升树 (GBDT)": "GBDT", "XGBoost": "XGBoost", "LightGBM": "LightGBM", 
        "极端随机树 (ExtraTrees)": "ExtraTrees", "支持向量回归 (SVR)": "SVR", 
        "TabPFN Regressor": "TabPFN"
    }

    # --- 4. 模型训练和评估 --- 
    all_results = {}
    plotting_data_list = []
    
    for name, config in models_config.items():
        print(f"\n--- 正在处理模型: {name} ---")
        
        # 特别处理 TabPFN 的数据量限制
        if name == "TabPFN Regressor":
            # 临时转换一次训练数据，检查其维度是否超出TabPFN的限制
            # 注意：这里我们使用一个新的 preprocessor 实例，因为每次循环模型训练时，
            # preprocessor 都是在 pipeline 中被 fit_transform 的，
            # 直接使用 main preprocessor 可能会在多次调用中产生问题。
            # 或者更简单的方式是，如果 preprocessor 是在循环外部定义的，
            # 那么它的 fit 状态会保留。这里为了安全和清晰，我们假设需要检查原始特征列。
            # 但是由于 preprocessor 是 pipeline 的一部分，它在 fit 的时候才真正被fit_transform。
            # 实际 TabPFN 限制检查应该在预处理之后的数据上进行。
            # 更好的做法是在 Pipeline 运行前检查 `X_train` 的维度，或者在 Pipeline 内捕获 TabPFN 的错误。
            # 考虑到多模型循环，我们在这里做个预估性检查。
            
            # 使用一个临时的、与主preprocessor相同的结构来转换X_train
            temp_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            temp_categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            temp_transformers_list = []
            
            temp_log_transform_cols_actual = [col for col in log_transform_cols if col in temp_numeric_cols]
            temp_other_numeric_cols = [col for col in temp_numeric_cols if col not in temp_log_transform_cols_actual]

            if temp_log_transform_cols_actual:
                temp_transformers_list.append(('log_numeric', Pipeline(steps=[
                    ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1), weights='uniform')),
                    ('log1p', FunctionTransformer(np.log1p, validate=False)),
                    ('scaler', StandardScaler())
                ]), temp_log_transform_cols_actual))
            if temp_other_numeric_cols:
                temp_transformers_list.append(('other_numeric', numeric_transformer, temp_other_numeric_cols)) # Reuse numeric_transformer
            if temp_categorical_cols:
                temp_transformers_list.append(('categorical', categorical_transformer, temp_categorical_cols)) # Reuse categorical_transformer

            temp_preprocessor = ColumnTransformer(transformers=temp_transformers_list, remainder='drop')

            try:
                X_train_transformed_for_tabpfn = temp_preprocessor.fit_transform(X_train)
                if X_train_transformed_for_tabpfn.shape[0] > 1024 or X_train_transformed_for_tabpfn.shape[1] >= 100:
                    print(f"跳过TabPFN: 数据规模超出限制 (预处理后样本数: {X_train_transformed_for_tabpfn.shape[0]}, 特征数: {X_train_transformed_for_tabpfn.shape[1]})")
                    continue
            except Exception as e:
                print(f"TabPFN预检查数据转换失败: {e}。跳过TabPFN模型。")
                continue # 如果预处理失败，跳过这个模型
        
        try:
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), # 处理 X 的缺失值和缩放/编码
                ('regressor', config['model']) # 最终的回归模型
            ])
            
            result = evaluate_model_with_cv(
                full_pipeline, 
                config['params'], 
                X_train, y_train, 
                cv=5, # 默认5折，可在主函数调用时调整
                scoring='r2'
            )
            
            all_results[name] = result
            best_model = result['best_model']
            
            y_pred_train = best_model.predict(X_train) 
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_train = mean_absolute_error(y_train, y_pred_train)
            
            y_pred_test = best_model.predict(X_test) 
            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            print(f"最佳参数: {result['best_params']}")
            print(f"交叉验证R²: {result['best_cv_score']:.4f}")
            print(f"训练集 - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
            print(f"测试集 - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
            
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
                'CV_Score': result['best_cv_score']
            })
            
        except Exception as e:
            print(f"模型 {name} 处理失败: {e}")
            traceback.print_exc()
    
    # --- 5. 比较模型性能 ---
    if all_results:
        best_model_name, best_result = compare_models_performance(all_results)
        
        print(f"\n最终推荐模型: {best_model_name}")
        print(f"最佳超参数: {best_result['best_params']}")
        print(f"交叉验证性能: {best_result['best_cv_score']:.4f}")
    
    # --- 6. 生成可视化结果 ---
    if plotting_data_list:
        plot_output_dir = f'./model_comparison_plots_自然老化_{target_column_name}'
        plot_all_models_comparison(plotting_data_list, target_column_name, output_dir=plot_output_dir)
    else:
        print("没有模型成功完成评估。")
    
    return all_results

# --- 主程序 ---
if __name__ == "__main__":
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