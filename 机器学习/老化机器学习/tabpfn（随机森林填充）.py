import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer # KNNImputer 已不再主要用于数值填充，所以可以省略，但这里保留以防万一
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor # RandomForestRegressor 用于填充
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer, IterativeImputer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import torch
import traceback

import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 自定义随机森林缺失值填充器 ---
# 这个填充器会按列顺序迭代填充数值缺失值，使用 RandomForestRegressor 进行预测。
# 它是一个简化的实现，为了避免IterativeImputer。
# 对于生产环境或更复杂的缺失模式，IterativeImputer (需启用实验特性) 或 fancyimpute 库可能更适合。
class RandomForestImputer:
    def __init__(self, estimator=None, random_state=42, max_iter=10):
        if estimator is None:
            self.estimator = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        else:
            self.estimator = estimator
        self.random_state = random_state
        self.max_iter = max_iter
        self.feature_names_ = None # 用于存储fit时的特征名，以便transform时保持一致

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            # 如果输入不是DataFrame，尝试将其转换为DataFrame以获取列名
            # 这假设X_train的列顺序和类型在调用fit和transform时是固定的
            self.feature_names_ = [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            self.feature_names_ = X.columns.tolist()
            X_df = X.copy()
        
        # 仅对数值列进行fit操作的准备，实际填充在transform中进行
        self.numeric_cols_ = X_df.select_dtypes(include=np.number).columns.tolist()
        
        # 可以在这里做一些预分析，例如识别哪些列有缺失值，并确定填充顺序
        # 但实际的预测模型fit发生在transform的迭代中
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_ is None:
                raise RuntimeError("Imputer has not been fitted. Call fit() before transform().")
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()

        # 确保只处理数值列
        X_numeric = X_df[self.numeric_cols_].copy()
        missing_mask_numeric = X_numeric.isnull()

        # 如果没有数值列或没有缺失值，则直接返回
        if not self.numeric_cols_ or not missing_mask_numeric.any().any():
            return X_df

        # 将DataFrame转换为numpy数组以便于迭代处理
        X_np = X_numeric.values
        
        # 记录各列的原始索引
        col_to_idx = {col: i for i, col in enumerate(self.numeric_cols_)}
        
        # 按照缺失值数量升序排列要填充的列，通常有助于提高填充质量
        cols_to_impute_ordered = missing_mask_numeric.sum().sort_values(ascending=True).index.tolist()
        cols_to_impute_ordered = [col for col in cols_to_impute_ordered if missing_mask_numeric[col].any()] # 仅包含实际有缺失的列

        # 迭代填充过程
        for iteration in range(self.max_iter):
            filled_in_iter = 0
            for col_name_to_impute in cols_to_impute_ordered:
                idx_to_impute = col_to_idx[col_name_to_impute]
                
                # 当前列的缺失值掩码
                missing_rows_current_col = missing_mask_numeric[col_name_to_impute]

                if missing_rows_current_col.any():
                    # 确定用于预测的特征列 (所有其他数值列，但要确保它们不是全缺失)
                    # 排除当前要填充的列本身
                    feature_cols_for_pred = [
                        col for col in self.numeric_cols_ 
                        if col != col_name_to_impute and not X_numeric[col].isnull().all()
                    ]
                    
                    if not feature_cols_for_pred:
                        # 如果没有其他特征可以用于预测，回退到该列的均值填充
                        if iteration == 0: # 只在第一次迭代时警告
                            print(f"警告: 列 '{col_name_to_impute}' 没有独立特征进行随机森林填充，回退到均值填充。")
                        mean_val = np.nanmean(X_np[:, idx_to_impute])
                        X_np[missing_rows_current_col.values, idx_to_impute] = mean_val
                        filled_in_iter += np.sum(missing_rows_current_col)
                        continue

                    # 训练集：已观测到当前列值的行，特征为其他列
                    X_train_rf = X_np[~missing_rows_current_col.values][:, [col_to_idx[c] for c in feature_cols_for_pred]]
                    y_train_rf = X_np[~missing_rows_current_col.values][:, idx_to_impute]

                    # 预测集：当前列缺失值的行，特征为其他列
                    X_pred_rf = X_np[missing_rows_current_col.values][:, [col_to_idx[c] for c in feature_cols_for_pred]]

                    # 确保用于训练和预测的数据不包含NaN（暂时用均值填充，因为RF不能处理NaN）
                    # 在实际IterativeImputer中，是根据BayesianRidge或LinearRegression填充
                    # 这里为了兼容RF，先做一次临时的均值填充
                    X_train_rf_temp = np.nan_to_num(X_train_rf, nan=np.nanmean(X_train_rf)) # 填充训练特征的NaN
                    X_pred_rf_temp = np.nan_to_num(X_pred_rf, nan=np.nanmean(X_pred_rf)) # 填充预测特征的NaN

                    if X_train_rf_temp.shape[0] > 0 and X_train_rf_temp.shape[1] > 0:
                        try:
                            self.estimator.fit(X_train_rf_temp, y_train_rf)
                            predicted_values = self.estimator.predict(X_pred_rf_temp)
                            X_np[missing_rows_current_col.values, idx_to_impute] = predicted_values
                            filled_in_iter += np.sum(missing_rows_current_col)
                        except Exception as e:
                            if iteration == 0:
                                print(f"警告: 随机森林填充列 '{col_name_to_impute}' 预测失败 ({e})，回退到均值填充。")
                            mean_val = np.nanmean(X_np[:, idx_to_impute])
                            X_np[missing_rows_current_col.values, idx_to_impute] = mean_val
                            filled_in_iter += np.sum(missing_rows_current_col)
                    else:
                        if iteration == 0:
                            print(f"警告: 随机森林填充列 '{col_name_to_impute}' 没有足够数据进行训练，回退到均值填充。")
                        mean_val = np.nanmean(X_np[:, idx_to_impute])
                        X_np[missing_rows_current_col.values, idx_to_impute] = mean_val
                        filled_in_iter += np.sum(missing_rows_current_col)

            # 检查是否所有缺失值都已填充，或填充没有进展
            remaining_nans = np.sum(np.isnan(X_np))
            if remaining_nans == 0 or filled_in_iter == 0:
                if remaining_nans == 0:
                    print(f"所有数值缺失值在迭代 {iteration+1} 中填充完毕。")
                else:
                    print(f"随机森林填充在迭代 {iteration+1} 中收敛，剩余 {remaining_nans} 个未填充的缺失值。")
                break
            print(f"随机森林填充迭代 {iteration+1} 完成，剩余 {remaining_nans} 个缺失值。")

        # 将填充后的数值数据放回原DataFrame
        X_df[self.numeric_cols_] = pd.DataFrame(X_np, columns=self.numeric_cols_, index=X_df.index)
        return X_df

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

# --- 数据加载函数 (保持不变) ---
def load_and_process_data(file_path):
    """加载数据，对所有列进行数值转换和初步检查。"""
    try:
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
                continue
            except Exception as e:
                print(f"尝试使用 {enc} 编码读取文件 '{file_path}' 时发生错误: {e}")
                return None
        
        if not read_successfully:
            print(f"错误: 尝试所有常用编码后仍无法读取文件 '{file_path}'。请检查文件编码或损坏。")
            return None

    except Exception as e:
        print(f"加载数据时发生意外错误: {e}")
        return None
    
    df.columns = [col.lstrip('\ufeff') for col in df.columns] 
    print("\n已移除列名开头的BOM字符。列名的大小写和空格将保持不变，请确保与代码中的列名精确匹配。")

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


# --- 模型选择和超参数调优 (保持不变) ---
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

# --- 交叉验证评估函数 (保持不变) ---
def evaluate_model_with_cv(pipeline, param_grid, X_train, y_train, cv=10, scoring='r2'):
    """使用交叉验证进行模型评估和超参数调优"""
    
    if param_grid:
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
    else:
        cv_results = cross_validate(
            pipeline, X_train, y_train, 
            cv=cv, 
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

# --- 结果记录和比较 (保持不变) ---
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

# --- 模型对比绘图函数 (保持不变) ---
def plot_all_models_comparison(plotting_data_list, target_col, output_dir='.'):
    """为目标变量创建一个大图，包含所有模型的预测比较。显示交叉验证结果"""
    if not plotting_data_list:
        print(f"没有可用于绘图的结果: {target_col}")
        return

    n_rows, n_cols = 3, 3 
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
    print(f"所有模型比较图已成功保存至: {save_path}")

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
    
    global target_column_options 
    all_output_cols_to_exclude = list(target_column_options) 

    final_cols_to_exclude = list(set(all_output_cols_to_exclude)) 
    
    if strata_col_name_original in df_processed.columns and strata_col_name_original not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_original)
    
    if strata_col_name_final != strata_col_name_original and strata_col_name_final in df_processed.columns and strata_col_name_final not in final_cols_to_exclude:
        final_cols_to_exclude.append(strata_col_name_final)

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

    # --- 打印用于训练的所有列 ---
    print(f"\n--- 实际传递给预处理和模型训练的特征列 ---")
    print(f"训练集特征列 (X_train.columns): {list(X_train.columns)}")
    print(f"测试集特征列 (X_test.columns): {list(X_test.columns)}")
    print(f"---------------------------------------")
    
    print(f"数据分割完成:")
    print(f"训练集规模: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: {X_test.shape}, y_test: {y_test.shape}")
    
    # --- 构建数据预处理 Pipeline (核心修改为随机森林填充) ---
    print("\n--- 开始构建数据预处理 Pipeline ---")
    
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []

    # 定义随机森林填充器实例
    rf_imputer_instance = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42)

    # 1. 处理需要 Log1p 变换的数值特征 (先随机森林填充，再Log1p，再标准化)
    log_transform_cols = ['Cd-C', 'BET'] 
    log_transform_cols_actual = [col for col in log_transform_cols if col in numeric_columns]
    
    if log_transform_cols_actual:
        print(f"将对 {log_transform_cols_actual} 列进行 随机森林填充 -> Log1p -> StandardScaler 变换。")
        log_numeric_pipeline = Pipeline(steps=[
            ('imputer_rf', rf_imputer_instance), # 使用 IterativeImputer
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('log_numeric_features', log_numeric_pipeline, log_transform_cols_actual))

    # 2. 处理其他数值特征 (随机森林填充 + StandardScaler)
    other_numeric_cols = [col for col in numeric_columns if col not in log_transform_cols_actual]
    if other_numeric_cols:
        print(f"将对 {len(other_numeric_cols)} 个其他数值特征进行 随机森林填充 -> StandardScaler 变换。")
        numeric_pipeline = Pipeline(steps=[
            ('imputer_rf', rf_imputer_instance), # 使用 IterativeImputer
            ('scaler', StandardScaler())
        ])
        transformers.append(('other_numeric_features', numeric_pipeline, other_numeric_cols))
    
    # 3. 处理分类特征 (众数填充 + 独热编码)
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

    # 获取模型配置
    models_config = get_models_with_hyperparameters()
    
    # 模型名称映射
    model_name_map = {
        "线性回归": "LR", "岭回归": "Ridge", "随机森林": "RF", 
        "梯度提升树 (GBDT)": "GBDT", "XGBoost": "XGBoost", "LightGBM": "LightGBM", 
        "极端随机树 (ExtraTrees)": "ExtraTrees", "支持向量回归 (SVR)": "SVR", 
        "TabPFN Regressor": "TabPFN"
    }

    # 模型训练和评估 
    all_results = {}
    plotting_data_list = []
    
    for name, config in models_config.items():
        print(f"\n--- 正在处理模型: {name} ---")
        
        # 特别处理 TabPFN 的数据量限制
        # 注意：这里为了保持 TabPFN 预检查的独立性，它会根据当前管道的填充器进行预转换。
        if name == "TabPFN Regressor":
            temp_numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            temp_categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
            temp_transformers_list = []
            
            # 临时预处理器也使用 IterativeImputer(RandomForestRegressor)
            temp_rf_imputer_instance = IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42)

            temp_log_transform_cols_actual = [col for col in log_transform_cols if col in temp_numeric_cols]
            temp_other_numeric_cols = [col for col in temp_numeric_cols if col not in temp_log_transform_cols_actual]

            if temp_log_transform_cols_actual:
                temp_transformers_list.append(('log_numeric', Pipeline(steps=[
                    ('imputer_rf', temp_rf_imputer_instance),
                    ('log1p', FunctionTransformer(np.log1p, validate=False)),
                    ('scaler', StandardScaler())
                ]), temp_log_transform_cols_actual))
            if temp_other_numeric_cols:
                temp_transformers_list.append(('other_numeric', Pipeline(steps=[
                    ('imputer_rf', temp_rf_imputer_instance),
                    ('scaler', StandardScaler())
                ]), temp_other_numeric_cols))
            if temp_categorical_cols:
                temp_transformers_list.append(('categorical', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), temp_categorical_cols))

            temp_preprocessor = ColumnTransformer(transformers=temp_transformers_list, remainder='drop')

            try:
                X_train_transformed_for_tabpfn = temp_preprocessor.fit_transform(X_train)
                if X_train_transformed_for_tabpfn.shape[0] > 1024 or X_train_transformed_for_tabpfn.shape[1] >= 100:
                    print(f"跳过TabPFN: 数据规模超出限制 (预处理后样本数: {X_train_transformed_for_tabpfn.shape[0]}, 特征数: {X_train_transformed_for_tabpfn.shape[1]})")
                    continue
            except Exception as e:
                print(f"TabPFN预检查数据转换失败: {e}。跳过TabPFN模型。")
                continue
        
        try:
            full_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('regressor', config['model']) 
            ])
            
            result = evaluate_model_with_cv(
                full_pipeline, 
                config['params'], 
                X_train, y_train, 
                cv=5, 
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
    
    # 比较模型性能
    if all_results:
        best_model_name, best_result = compare_models_performance(all_results)
        
        print(f"\n最终推荐模型: {best_model_name}")
        print(f"最佳超参数: {best_result['best_params']}")
        print(f"交叉验证性能: {best_result['best_cv_score']:.4f}")
    
    # 生成可视化结果
    if plotting_data_list:
        plot_output_dir = f'./model_comparison_plots_自然老化_{target_column_name}'
        plot_all_models_comparison(plotting_data_list, target_column_name, output_dir=plot_output_dir)
    else:
        print("没有模型成功完成评估。")
    
    return all_results

# --- 主程序 ---
if __name__ == "__main__":
    # 在主程序中添加这行，启用 IterativeImputer (如果你的Python版本需要)
    from sklearn.experimental import enable_iterative_imputer 

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