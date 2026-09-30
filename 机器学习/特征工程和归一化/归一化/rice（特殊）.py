import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import RepeatedKFold, cross_val_score, GridSearchCV # Added GridSearchCV

import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import traceback
import time
import pickle

warnings.filterwarnings('ignore')

# --- Categorical Feature Encoding Functions ---
# (classify_varieties, classify_materials, encode_irrigation_method functions remain the same)
def classify_varieties(df, data_id="Dataset"):
    print(f"\n检查数据集 '{data_id}' 的品种信息...")
    df_encoded = df.copy()
    if '品种' not in df_encoded.columns:
        print(f"信息: '{data_id}' 数据中没有'品种'列。跳过品种编码。")
        return df_encoded 

    print(f"对数据集 '{data_id}' 的品种进行分类和编码...")
    df_encoded['品种'] = df_encoded['品种'].astype(str)
    df_encoded['品种类型'] = df_encoded['品种'].apply(lambda x: '低累积品种' if '低累积' in x else '普通品种')
    df_encoded['variety_binary'] = np.where(df_encoded['品种类型'] == '普通品种', 1.0, 0.0)
    df_encoded = df_encoded.drop(['品种', '品种类型'], axis=1, errors='ignore')
    print(f"'{data_id}': 'variety_binary' 列已创建。")
    return df_encoded

def classify_materials(df, data_id="Dataset"): 
    print(f"\n检查数据集 '{data_id}' 的原料信息...")
    df_encoded = df.copy()
    # CORRECTED: Check for '原料' (Yuanliao) if that's the actual column name in your CSV.
    # If it's 'ingredient', then the original check was fine.
    # Assuming '原料' based on your latest provided script snippet.
    if '原料' not in df_encoded.columns: 
        print(f"信息: '{data_id}' 数据中没有'原料' (或 'ingredient')列。跳过原料编码。")
        return df_encoded 
    
    print(f"对数据集 '{data_id}' 的原料进行分类和编码...")
    df_encoded['原料'] = df_encoded['原料'].astype(str) # Use the actual column name
    
    detailed_categories = {
        'rice_straw': ['水稻秸秆', '高硅水稻秸秆', '低硅水稻秸秆', '稻草'],
        'wheat_straw': ['小麦秸秆'], 'corn_straw': ['玉米秸秆', '玉米芯'],
        'other_crop_straw': ['花生秸秆', '油菜秸秆', '豆秸', '大蒜秸秆'],
        'nutshell': ['山胡桃果壳', '花生壳', '椰壳', '椰子壳', '稻壳'],
        'wood': ['木屑', '硬木', '竹'], 'animal_derived': ['蛋壳'], 'waste': ['污泥', '酒糟']
    }
    material_full_mapping = {
        'rice_straw': 0, 'wheat_straw': 1, 'corn_straw': 2, 'other_crop_straw': 3,
        'nutshell': 4, 'wood': 5, 'animal_derived': 6, 'waste': 7, 'other_mixed': 8
    }

    def detailed_category_map(material_string):
        for category, materials in detailed_categories.items():
            for material in materials:
                if material in material_string: return category
        return 'other_mixed'

    df_encoded['material_type'] = df_encoded['原料'].apply(detailed_category_map) # Use the actual column name
    df_encoded['material_type_code'] = df_encoded['material_type'].map(material_full_mapping).fillna(material_full_mapping['other_mixed']).astype(float)
    
    print(f"原料分类统计 for '{data_id}':\n{df_encoded['material_type'].value_counts(dropna=False)}")
    print(f"'{data_id}': 'material_type_code' 列已创建。")
    
    df_encoded = df_encoded.drop(['原料', 'material_type'], axis=1, errors='ignore') # Use the actual column name
    return df_encoded

def encode_irrigation_method(df, data_id="Dataset"):
    print(f"\n检查数据集 '{data_id}' 的灌溉方式信息...")
    df_encoded = df.copy()
    if 'Irrigationmethod' not in df_encoded.columns:
        print(f"信息: '{data_id}' 数据中没有'Irrigationmethod'列。跳过灌溉方式编码。")
        return df_encoded

    print(f"对数据集 '{data_id}' 的灌溉方式进行编码...")
    df_encoded['Irrigationmethod'] = df_encoded['Irrigationmethod'].astype(str)
    def map_irrigation(method_str):
        method_str_lower = method_str.lower()
        if '干湿交替' in method_str or 'awd' in method_str_lower: return 1.0
        if '长期淹水' in method_str or 'cf' in method_str_lower or 'continuous' in method_str_lower: return 0.0
        if '干湿交替灌溉' in method_str : return 1.0 # Fallback
        if '持续淹水灌溉' in method_str : return 0.0 # Fallback
        return -1.0 
    df_encoded['irrigation_type'] = df_encoded['Irrigationmethod'].apply(map_irrigation)
    df_encoded = df_encoded.drop('Irrigationmethod', axis=1, errors='ignore')
    print(f"'{data_id}': 'irrigation_type' 列已创建。")
    return df_encoded

# --- Data Preprocessing Function ---
def preprocess_data_custom(raw_df, data_id="Dataset"):
    print(f"\n对数据集 '{data_id}' 进行预处理 (无数值特征归一化)...")
    print(f"原始数据列名: {raw_df.columns.tolist()}") 
    
    cols_to_drop = [col for col in ['水稻种植方式', '地点'] if col in raw_df.columns]
    if cols_to_drop:
        raw_df = raw_df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")

    numeric_cols_to_check = ['pH', 'organicmatter', 'CEC', 'Cdcontent', 'temperature', 'pHsoil', 'Charcoal_application_ratio']
    
    for col in numeric_cols_to_check:
        if col in raw_df.columns:
            raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
            if raw_df[col].isnull().any():
                fill_value = raw_df[col].median(skipna=True)
                if pd.isna(fill_value): fill_value = raw_df[col].mean(skipna=True)
                if pd.isna(fill_value): fill_value = 0 
                raw_df[col].fillna(fill_value, inplace=True)
                print(f"警告: '{data_id}' 的 '{col}' 列缺失值已用中位数/均值 {fill_value:.4f} 填充。")

    df_processed = raw_df.copy()
    df_processed = classify_varieties(df_processed, data_id=data_id)
    df_processed = classify_materials(df_processed, data_id=data_id) 
    df_processed = encode_irrigation_method(df_processed, data_id=data_id)

    if 'Cdcontent' in df_processed.columns:
        if df_processed['Cdcontent'].isnull().any():
            fill_value_cd = df_processed['Cdcontent'].median(skipna=True)
            if pd.isna(fill_value_cd): fill_value_cd = df_processed['Cdcontent'].mean(skipna=True)
            if pd.isna(fill_value_cd): fill_value_cd = 1e-9
            df_processed['Cdcontent'].fillna(fill_value_cd, inplace=True)
            print(f"警告: '{data_id}' 的 'Cdcontent' 列在对数变换前再次检查并填充缺失值（如有）为 {fill_value_cd:.4f}。")
        
        df_processed['Cdcontent_non_negative'] = np.maximum(1e-9, df_processed['Cdcontent'])
        df_processed['log_Cdcontent'] = np.log(df_processed['Cdcontent_non_negative'])
        df_processed = df_processed.drop(columns=['Cdcontent_non_negative'])
        print(f"'{data_id}': 'log_Cdcontent' 列已创建。")
    else:
        print(f"信息: '{data_id}' 数据中缺少 'Cdcontent' 列，无法创建 'log_Cdcontent'。")
             
    print(f"\n'{data_id}' 预处理完成，最终数据框形状: {df_processed.shape}")
    print(f"预处理后数据列名: {df_processed.columns.tolist()}")
    return df_processed


# --- SPXY 划分函数 ---
def spxy_train_test_split(X, y, test_size=0.3, random_state=42, distance_weight=0.5, distance_metric='cosine'):
    print(f"SPXY划分: test_size={test_size}, random_state={random_state}, distance_weight={distance_weight}, distance_metric='{distance_metric}'")
    np.random.seed(random_state)
    X_scaled = StandardScaler().fit_transform(X)
    if y.std() == 0: y_scaled = y.copy()
    else: y_scaled = (y - y.mean()) / y.std()

    Dx = squareform(pdist(X_scaled, metric=distance_metric))
    Dy = squareform(pdist(y_scaled.values.reshape(-1, 1), metric='euclidean'))
    D = distance_weight * Dx + (1 - distance_weight) * Dy

    n_samples = len(X); test_size_n = int(n_samples * test_size); train_size_n = n_samples - test_size_n
    train_indexes, test_indexes_list = [], list(range(n_samples)) 
    sum_dist = D.sum(axis=1); max_dist_idx = np.argmax(sum_dist)
    train_indexes.append(max_dist_idx); test_indexes_list.remove(max_dist_idx)

    while len(train_indexes) < train_size_n:
        if not test_indexes_list:
            print("警告: SPXY 在填满训练集之前耗尽了测试索引。"); break
        min_dist_values = np.array([min(D[idx, j] for j in train_indexes) for idx in test_indexes_list])
        max_idx_in_test = np.argmax(min_dist_values); selected_idx = test_indexes_list[max_idx_in_test]
        train_indexes.append(selected_idx); test_indexes_list.pop(max_idx_in_test)

    X_train = X.iloc[train_indexes]; X_test = X.iloc[test_indexes_list] 
    y_train = y.iloc[train_indexes]; y_test = y.iloc[test_indexes_list]
    print(f"SPXY划分完成，训练集大小: {len(X_train)}，测试集大小: {len(X_test)}")
    return X_train, X_test, y_train, y_test

# --- XGBoost 和 GBDT 模型与参数网格初始化 (用于GridSearchCV) ---
def initialize_models_for_grid_search(random_state_model=42):
    """
    初始化 XGBoost 和 GBDT 模型及其用于网格搜索的参数范围。
    """
    models = {
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=random_state_model),
            'params': {
                'n_estimators': [300],
                'max_depth': [5],
                'learning_rate': [0.01],
                'reg_alpha': [0.2],
                'reg_lambda': [0.1],
                'subsample': [0.7],
                'colsample_bytree': [0.6],
                'min_child_weight': [1]
            }
        },
        'GBDT': {
            'model': GradientBoostingRegressor(random_state=random_state_model, init=None),
            'params': {
                'n_estimators': [400],
                'max_depth': [3],
                'learning_rate': [0.06],
                'subsample': [0.6],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'max_features': [0.6]
            }
        }
    }
    return models

# --- 模型训练函数 (使用GridSearchCV) ---
def train_model_with_gridsearch(model_name, model_config, X_train, y_train, 
                                 cv_folds=5, gridsearch_scoring='r2', global_random_state=42):
    print(f"\n对模型 {model_name} 执行 GridSearchCV...")
    
    # GridSearchCV itself uses cross-validation.
    # The cv parameter in GridSearchCV handles the splitting for hyperparameter tuning.
    # If you want RepeatedKFold for GridSearchCV's internal CV, you can pass it to the cv parameter.
    # For simplicity, we'll use integer cv for folds, n_jobs for parallelism.
    # random_state in GridSearchCV is for the case when the estimator is not a CV-splitter.
    # It's generally good to have random_state set in the models themselves for reproducibility.

    grid_search = GridSearchCV(
        estimator=model_config['model'], # Model instance
        param_grid=model_config['params'], # Parameter grid
        cv=cv_folds, # Number of CV folds for GridSearchCV
        scoring=gridsearch_scoring,
        n_jobs=-1, # Use all available cores
        verbose=1  # Print progress
    )
    
    best_model = None
    best_params = {}
    gridcv_best_score = np.nan

    try:
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        gridcv_best_score = grid_search.best_score_
        print(f"{model_name} - GridSearchCV 最佳参数: {best_params}")
        print(f"{model_name} - GridSearchCV 最佳 {gridsearch_scoring} 分数: {gridcv_best_score:.4f}")
        
        # For GBDT, ensure init is set if it was part of best_params (though usually not)
        # or if it's a fixed attribute of the best_estimator_
        if model_name == 'GBDT' and hasattr(best_model, 'init') and best_model.init is None:
            print(f"注意: {model_name} 最佳估计器的 init 为 None。")

    except Exception as e:
        print(f"模型 {model_name} 的 GridSearchCV 失败: {str(e)}")
        traceback.print_exc()
        # Fallback: fit with default parameters if GridSearchCV fails
        print(f"使用默认参数为 {model_name} 进行拟合。")
        default_model = model_config['model'] # Get a fresh default model
        default_model.fit(X_train, y_train)
        best_model = default_model
        # gridcv_best_score remains NaN, best_params empty
        
    return best_model, best_params, gridcv_best_score


# --- XGBoost 和 GBDT 模型训练、评估和保存函数 (使用GridSearchCV) ---
def train_and_compare_xgboost_gbdt(df_processed, target_col_name, feature_columns_to_use, 
                                   cv_folds_gridsearch=5, # Folds for GridSearchCV's internal CV
                                   global_random_state=42, 
                                   model_purpose_tag="XGB_GBDT_GridSearch", 
                                   output_dir_base='.'):
    if target_col_name not in df_processed.columns:
        print(f"错误: 目标列 '{target_col_name}' 不在提供的DataFrame中。")
        return None
    
    actual_feature_columns = [f_col for f_col in feature_columns_to_use if f_col in df_processed.columns]
    if not actual_feature_columns:
        print(f"错误: 指定的特征列在处理后的数据中均不存在 for '{target_col_name}'. 可用列: {df_processed.columns.tolist()}")
        return None
    if len(actual_feature_columns) < len(feature_columns_to_use):
        missing_feats = set(feature_columns_to_use) - set(actual_feature_columns)
        print(f"警告: 以下指定的特征在数据中缺失，将不会用于建模: {missing_feats}.")
    
    feature_columns_to_use_final = actual_feature_columns 

    print(f"\n{'='*20} 开始为目标 '{target_col_name}' (用途: {model_purpose_tag}) 进行 XGBoost 和 GBDT 模型训练与比较 (使用GridSearchCV) {'='*20}")
    
    safe_target_name_for_path = target_col_name.replace(" ", "_").replace("/", "_")
    fixed_spxy_test_size = 0.3
    fixed_spxy_distance_weight = 0.5
    fixed_spxy_distance_metric = 'euclidean'

    current_task_output_dir = os.path.join(output_dir_base, model_purpose_tag, 
                                           f"{safe_target_name_for_path}_spxy_{fixed_spxy_distance_metric}_w{fixed_spxy_distance_weight}_ts{fixed_spxy_test_size}")
    os.makedirs(current_task_output_dir, exist_ok=True)
    print(f"当前运行结果将保存在: {current_task_output_dir}")

    X = df_processed[feature_columns_to_use_final].copy()
    y = df_processed[target_col_name].copy()
    print(f"实际使用的特征 ({len(feature_columns_to_use_final)}): {feature_columns_to_use_final}")
    print(f"目标变量: {target_col_name}")

    if y.isnull().any():
        fill_value_y = y.mean(skipna=True); y.fillna(fill_value_y, inplace=True)
        print(f"警告: 目标列 '{target_col_name}' 的NaN值已用均值 {fill_value_y:.4f} 填充。")
    
    valid_mask = ~y.isna() & ~X.isna().any(axis=1); X = X[valid_mask]; y = y[valid_mask]

    if len(X) == 0 or len(y) == 0: print(f"错误: 移除NaN后没有有效数据用于目标 {target_col_name}"); return None
    if len(X) < max(10, cv_folds_gridsearch): # Check against GridSearchCV folds
        print(f"警告: 样本数量过少 ({len(X)})，SPXY或GridSearchCV可能不稳定。")
        if len(X) <=1: print(f"错误: 样本数量不足 ({len(X)}) 无法执行SPXY。"); return None
        if len(X) < cv_folds_gridsearch: print(f"错误: 样本数量 ({len(X)}) 少于GridSearchCV折数 ({cv_folds_gridsearch})。"); return None # Or adjust folds
    if len(np.unique(y)) < 2 : print(f"警告: 目标列 '{target_col_name}' 的唯一值少于2个。")

    X_train, X_test, y_train, y_test = spxy_train_test_split(
        X, y, test_size=fixed_spxy_test_size, random_state=global_random_state,        
        distance_weight=fixed_spxy_distance_weight, distance_metric=fixed_spxy_distance_metric  
    )
    
    if len(X_train) == 0 or len(X_test) == 0: print(f"错误: SPXY划分后训练集或测试集为空 for {target_col_name}。"); return None
    if len(X_train) < cv_folds_gridsearch: print(f"警告: 训练集大小 ({len(X_train)}) 小于GridSearchCV折数 ({cv_folds_gridsearch})。"); # Might need to adjust cv_folds_gridsearch

    models_to_search = initialize_models_for_grid_search(random_state_model=global_random_state)
    results_list = []

    for model_name, model_config in models_to_search.items():
        try:
            best_fitted_model, best_params, gridcv_score = train_model_with_gridsearch(
                model_name, model_config, X_train, y_train,
                cv_folds=cv_folds_gridsearch, global_random_state=global_random_state
            )
            
            if best_fitted_model is None: 
                print(f"模型 {model_name} 未能通过GridSearchCV成功训练，跳过。")
                continue

            y_train_pred = best_fitted_model.predict(X_train)
            y_test_pred = best_fitted_model.predict(X_test)

            r2_train = r2_score(y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_test = r2_score(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            mae_test = mean_absolute_error(y_test, y_test_pred)

            print(f"{model_name} - 训练集 (使用最佳参数) - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
            print(f"{model_name} - 测试集 (使用最佳参数) - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

            results_list.append({
                'Model': model_name, 'Target': target_col_name,
                'R2_Train': r2_train, 'RMSE_Train': rmse_train, 'MAE_Train': mae_train,
                'R2_Test': r2_test, 'RMSE_Test': rmse_test, 'MAE_Test': mae_test,
                'Model_Object': best_fitted_model, 
                'Best_Params': best_params,
                'GridCV_Best_Score': gridcv_score,
                'Features_Used': list(X_train.columns)
            })
        except Exception as e:
            print(f"处理模型 {model_name} (GridSearchCV) for {target_col_name} 时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    if not results_list: print(f"所有模型在目标 {target_col_name} 上GridSearchCV均失败。"); return None

    results_df = pd.DataFrame([res for res in results_list if 'Model_Object' in res]) 
    if results_df.empty: print(f"没有成功通过GridSearchCV训练的模型 for {target_col_name}"); return None
    
    results_df = results_df.sort_values('R2_Test', ascending=False)
    print(f"\n模型性能 for {target_col_name} (按测试集 R² 排序, GridSearchCV):\n{results_df[['Model', 'R2_Train', 'R2_Test', 'GridCV_Best_Score', 'Best_Params']]}")
    
    # Save performance CSV
    # Convert Best_Params dict to string for CSV compatibility
    results_df_to_save = results_df.copy()
    results_df_to_save['Best_Params'] = results_df_to_save['Best_Params'].astype(str)
    perf_csv_path = os.path.join(current_task_output_dir, f"performance_gridsearch_{safe_target_name_for_path}_{model_purpose_tag}.csv")
    results_df_to_save[['Model', 'Target', 'R2_Train', 'RMSE_Train', 'MAE_Train', 'R2_Test', 'RMSE_Test', 'MAE_Test', 'GridCV_Best_Score', 'Best_Params']].to_csv(perf_csv_path, index=False)
    print(f"性能指标已保存到: {perf_csv_path}")

    # Save the best model overall (based on R2_Test from this run)
    if not results_df.empty:
        best_model_row = results_df.iloc[0]
        best_model_name_overall = best_model_row['Model']
        
        best_result_obj = next((r for r in results_list if r['Model'] == best_model_name_overall and r['Target'] == target_col_name), None)
        
        if best_result_obj and best_result_obj['Model_Object'] is not None:
            best_model_to_save = best_result_obj['Model_Object']
            
            feature_importance_values = None
            if hasattr(best_model_to_save, 'feature_importances_'):
                feature_importance_values = best_model_to_save.feature_importances_.tolist()

            model_save_dict = {
                'model': best_model_to_save,
                'best_params': best_result_obj['Best_Params'],
                'feature_names': best_result_obj['Features_Used'], 
                'target_name': target_col_name,
                'model_purpose_tag': model_purpose_tag,
                'spxy_params': {'metric': fixed_spxy_distance_metric, 'weight': fixed_spxy_distance_weight,
                                'test_size': fixed_spxy_test_size, 'random_state': global_random_state},
                'gridsearch_cv_params': {'folds': cv_folds_gridsearch, 'scoring': 'r2'},
                'data_processing_notes': 'Custom preprocessing. XGB/GBDT with GridSearchCV. SPXY fixed strategy.',
                'metrics': {
                    'R2_Test': best_model_row['R2_Test'], 'R2_Train': best_model_row['R2_Train'],
                    'MAE_Test': best_model_row['MAE_Test'], 'RMSE_Test': best_model_row['RMSE_Test'],
                    'GridCV_Best_Score': best_model_row['GridCV_Best_Score']
                },
                'feature_importance': { 
                    'values': feature_importance_values, 'feature_names': best_result_obj['Features_Used'] 
                } if feature_importance_values else None
            }
            
            model_filename = f"{safe_target_name_for_path}_{best_model_name_overall}_{model_purpose_tag}_GridSearchCV.pkl"
            model_path = os.path.join(current_task_output_dir, model_filename)
            with open(model_path, 'wb') as f: pickle.dump(model_save_dict, f)
            print(f"最佳模型 ({best_model_name_overall}) for target '{target_col_name}' (GridSearchCV) 已保存至: {model_path}")
        else: print(f"无法找到最佳模型对象 for {target_col_name}。")
    else: print(f"没有有效的模型结果 for {target_col_name}，无法保存最佳模型。")
        
    return results_df

# --- 主函数 ---
def main(data_path, output_dir_main='xgb_gbdt_gridsearch_run', force_preprocess=False,
         cv_f_gridsearch=3, # Folds for GridSearchCV, keep it small for faster runs, e.g., 3 or 5
         random_s=42):
    try:
        os.makedirs(output_dir_main, exist_ok=True)
        base_data_filename = os.path.splitext(os.path.basename(data_path))[0]
        
        data_ready_for_training_filename = f"data_ready_for_training_{base_data_filename}.csv"
        data_ready_for_training_path = os.path.join(output_dir_main, data_ready_for_training_filename)
        preprocess_cache_dir = os.path.join(output_dir_main, "preprocess_internal_cache")
        os.makedirs(preprocess_cache_dir, exist_ok=True)
        preprocess_cache_path = os.path.join(preprocess_cache_dir, f"internal_cache_{base_data_filename}.csv")

        start_time = time.time()
        print(f"{'=' * 70}\n开始 XGBoost 和 GBDT GridSearchCV 模型训练流程\n{'=' * 70}")
        print(f"数据文件: {data_path}")
        print(f"输出目录: {output_dir_main}")
        print(f"全局随机种子: {random_s}")
        print(f"GridSearchCV 交叉验证折数: {cv_f_gridsearch}")

        processed_df_for_saving = None
        if not force_preprocess and os.path.exists(preprocess_cache_path):
            print(f"\n找到已缓存的内部预处理数据: {preprocess_cache_path}，直接加载...")
            try:
                processed_df_for_saving = pd.read_csv(preprocess_cache_path)
                print(f"已加载缓存的内部预处理数据，形状: {processed_df_for_saving.shape}")
            except Exception as e:
                print(f"读取内部缓存文件失败: {str(e)}。将重新预处理。"); processed_df_for_saving = None
        
        if processed_df_for_saving is None:
            print("\n执行数据预处理 (使用自定义流程)...")
            if not os.path.exists(data_path): print(f"错误: 找不到输入文件 {data_path}"); return
            try:
                try: df_loaded = pd.read_csv(data_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try: df_loaded = pd.read_csv(data_path, encoding='gbk')
                    except UnicodeDecodeError: df_loaded = pd.read_csv(data_path, encoding='latin1') 
                
                processed_df_for_saving = preprocess_data_custom(df_loaded, data_id=base_data_filename)
                if processed_df_for_saving is None or processed_df_for_saving.empty:
                    print("数据预处理失败或返回空数据框。"); return
                processed_df_for_saving.to_csv(preprocess_cache_path, index=False)
                print(f"内部预处理数据已缓存到 {preprocess_cache_path}")
            except Exception as e:
                print(f"预处理过程中出错: {str(e)}"); traceback.print_exc(); return
        
        if processed_df_for_saving is not None and not processed_df_for_saving.empty:
            processed_df_for_saving.to_csv(data_ready_for_training_path, index=False)
            print(f"\n预处理后的完整数据已保存至: {data_ready_for_training_path}")
        else: print("错误: 预处理步骤未能生成有效数据。"); return

        print(f"\n从 '{data_ready_for_training_path}' 加载数据进行模型训练...")
        try:
            df_for_training = pd.read_csv(data_ready_for_training_path)
            print(f"已加载用于训练的数据，形状: {df_for_training.shape}")
        except Exception as e: print(f"错误: 无法从 '{data_ready_for_training_path}' 加载预处理数据: {e}"); return

        CURRENT_TARGET_COLUMN = 'CD_brown_rice_change' 
        FEATURES_TO_USE = [
            'organicmatter', 'pH', 'pHsoil', 'temperature', 
            'Charcoal_application_ratio', 'log_Cdcontent', 'CEC',
            'variety_binary', 'material_type_code', 'irrigation_type'
        ]
        MODEL_PURPOSE_TAG = "Baseline_Cd_XGB_GBDT_GridSearchCV"

        print(f"\n--- 当前训练配置 (GridSearchCV) ---")
        print(f"目标列: {CURRENT_TARGET_COLUMN}")
        print(f"期望使用的特征: {FEATURES_TO_USE}")
        
        if CURRENT_TARGET_COLUMN not in df_for_training.columns:
            print(f"错误: 目标列 '{CURRENT_TARGET_COLUMN}' 在加载的预处理数据中不存在。"); return
        
        results_dataframe = train_and_compare_xgboost_gbdt(
            df_for_training.copy(), 
            target_col_name=CURRENT_TARGET_COLUMN, 
            feature_columns_to_use=FEATURES_TO_USE, 
            cv_folds_gridsearch=cv_f_gridsearch, 
            global_random_state=random_s,
            model_purpose_tag=MODEL_PURPOSE_TAG,
            output_dir_base=output_dir_main 
        )

        if results_dataframe is not None and not results_dataframe.empty:
            print(f"\n针对目标 '{CURRENT_TARGET_COLUMN}' 的GridSearchCV模型训练与比较成功。")
        else:
            print(f"\n针对目标 '{CURRENT_TARGET_COLUMN}' 的GridSearchCV模型训练与比较未成功或未产生结果。")

        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60); hours, minutes = divmod(minutes, 60)
        print(f"\n总运行时间: {int(hours)}时 {int(minutes)}分 {int(seconds)}秒")
        print(f"\n完整流程已完成! 结果文件已保存在 {output_dir_main} 目录内。")

    except Exception as e: print(f"主流程发生未捕获的错误: {str(e)}"); traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost/GBDT GridSearchCV训练流程')
    parser.add_argument('--data', type=str, 
                        default=r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\lr_imputed_data_new.csv", 
                        help='原始数据CSV文件路径')
    parser.add_argument('--output_dir', type=str, 
                        default='xgb_gbdt_gridsearch_results', 
                        help='输出结果的根目录')
    parser.add_argument('--force_preprocess', action='store_true', 
                        help='强制重新执行预处理')
    parser.add_argument('--cv_folds_gridsearch', type=int, default=3, # Default to 3 for faster runs
                        help="GridSearchCV的交叉验证折数 (默认: 3)")
    parser.add_argument('--random_state', type=int, default=42,
                        help="全局随机种子 (默认: 42)")
    
    args = parser.parse_args()

    main(data_path=args.data, 
         output_dir_main=args.output_dir,
         force_preprocess=args.force_preprocess,
         cv_f_gridsearch=args.cv_folds_gridsearch,
         random_s=args.random_state)