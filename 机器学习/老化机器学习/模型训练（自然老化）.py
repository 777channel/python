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
warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['axes.titlesize'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28
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
    
    # --- 根据用户要求：仅移除 BOM，不进行列名大小写转换和空格去除 ---
    df.columns = [col.lstrip('\ufeff') for col in df.columns] # 移除BOM
    print("\n已移除列名开头的BOM字符。列名的大小写和空格将保持不变，请确保与代码中的列名精确匹配。")

    # --- 将所有列转换为数值型，无法转换的变为NaN ---
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
            "model": xgb.XGBRegressor(random_state=42, objective='reg:squarederror', eval_metric='rmse'), # 移除 use_label_encoder
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
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [10, 20]
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
        # 直接交叉验证并训练最终模型
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
    fig.suptitle(f'Comparison of Natural Aging Models - {target_col}', fontsize=32, y=0.98, fontweight='bold')

    # 调整子图间距
    # fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.95]) # 你可以选择保留或注释掉这行，然后手动调整
    plt.subplots_adjust(wspace=0.3, hspace=0.35, left=0.07, right=0.95, top=0.93, bottom=0.07) # 示例值，你可以根据需要调整

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

    # 首先删除目标变量为空的行，这是必须的
    df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.shape[0] < 10:
        print(f"数据过少 (少于10行)，无法进行有效训练。")
        return None

    print(f"删除目标变量缺失值后，剩余样本数: {df_processed.shape[0]}")

    # --- 1. 数据划分（使用分层抽样）---
    # 这里的列名必须与您CSV文件中的实际列名大小写完全匹配
    strata_col_name = 'T-Aging' # 假设这是分层列名
    strata = None # 默认不分层

    if strata_col_name in df_processed.columns:
        # 检查分层列的类型
        # 如果是连续数值，需要分箱。is_numeric_dtype 返回True，is_integer_dtype 返回False
        if pd.api.types.is_numeric_dtype(df_processed[strata_col_name]) and \
           not pd.api.types.is_integer_dtype(df_processed[strata_col_name]):
            print(f"警告: 分层列 '{strata_col_name}' 是连续数值类型。将对其进行分箱处理以进行分层。")
            try:
                # 尝试用 qcut 进行等频分箱，确保每个箱子里有足够的样本
                # nunique() 获取唯一值的数量，确保 q 不超过唯一值数量
                q_bins = min(5, df_processed[strata_col_name].nunique()) 
                # 确保 q_bins 至少为 2，因为 pd.qcut(q=1) 会报错
                if q_bins < 2:
                    print(f"警告: 分层列 '{strata_col_name}' 唯一值数量过少 ({df_processed[strata_col_name].nunique()})，无法有效分箱。将退回普通随机抽样。")
                    strata = None
                else:
                    strata_binned_col_name = f'{strata_col_name}_binned'
                    df_processed[strata_binned_col_name] = pd.qcut(
                        df_processed[strata_col_name], 
                        q=q_bins, # 使用动态确定的箱子数量
                        duplicates='drop' # 如果出现重复的箱体边界（因为值重复），则删除重复项
                    )
                    strata_col_name = strata_binned_col_name # 更新分层列名
                    print(f"已创建分箱列 '{strata_col_name}'。各类别数量:\n{df_processed[strata_col_name].value_counts()}")
            except Exception as e:
                print(f"错误: 自动分箱 '{strata_col_name}' 失败: {e}。将退回普通随机抽样。")
                strata = None
        
        # 检查分层列是否有缺失值，有的话需要先填充 (在分箱后执行，因为分箱可能产生NaN)
        if df_processed[strata_col_name].isnull().any():
            print(f"警告: 分层列 '{strata_col_name}' 存在缺失值，将使用众数填充。")
            imputer_strata = SimpleImputer(strategy='most_frequent')
            # astype(str) 确保 SimpleImputer 能够处理 CategoricalDtype
            df_processed[strata_col_name] = imputer_strata.fit_transform(df_processed[[strata_col_name]].astype(str)) 
            # 填充后，如果原始是 CategoricalDtype，可能需要转换回来以保持类型一致性
            if pd.api.types.is_object_dtype(df_processed[strata_col_name]): # 填充后可能是object
                 df_processed[strata_col_name] = df_processed[strata_col_name].astype('category')
            
        # 检查分层列的类别数量是否足够进行分层（每个类别至少2个样本）
        if all(df_processed[strata_col_name].value_counts() >= 2):
            strata = df_processed[strata_col_name]
        else:
            print(f"警告: 分层列 '{strata_col_name}' 中存在样本数小于2的类别，无法进行分层。将使用普通随机抽样。")
            strata = None
    else:
        print(f"警告: 未找到用于分层的列 '{strata_col_name}'。将使用普通随机抽样。")


    # 定义特征和目标
    # 目标列名和排除列名必须与您CSV文件中的实际列名大小写完全匹配
    # '有效老化' 和 'm' 是不作为特征的列，需要从原始数据中排除
    cols_to_exclude_from_features = ['有效老化', 'M'] 
    
    # 确保目标列不被算作特征
    cols_to_exclude_from_features = [col for col in cols_to_exclude_from_features if col != target_column_name]

    # 所有非目标列和非排除列都是特征列
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude_from_features and col != target_column_name]
    
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
    
    # 对部分数值列进行 log1p 变换（例如 Cd-C, BET），其他数值列进行标准化
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

    # --- 3. 获取模型配置 --- (保持不变)
    models_config = get_models_with_hyperparameters()
    
    # 模型名称映射 (保持不变)
    model_name_map = {
        "线性回归": "LR", "岭回归": "Ridge", "随机森林": "RF", 
        "梯度提升树 (GBDT)": "GBDT", "XGBoost": "XGBoost", "LightGBM": "LightGBM", 
        "极端随机树 (ExtraTrees)": "ExtraTrees", "支持向量回归 (SVR)": "SVR", 
        "TabPFN Regressor": "TabPFN"
    }

    # --- 4. 模型训练和评估 --- (保持不变，因为X_train, X_test已经是正确数据)
    all_results = {}
    plotting_data_list = []
    
    for name, config in models_config.items():
        print(f"\n--- 正在处理模型: {name} ---")
        
        if name == "TabPFN Regressor":
            temp_preprocessor = ColumnTransformer(
                transformers=transformers_list,
                remainder='drop' 
            )
            X_train_transformed_shape = temp_preprocessor.fit_transform(X_train).shape
            
            if X_train_transformed_shape[0] > 1024 or X_train_transformed_shape[1] >= 100:
                print(f"跳过TabPFN: 数据规模超出限制 (预处理后样本数: {X_train_transformed_shape[0]}, 特征数: {X_train_transformed_shape[1]})")
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
    
    # --- 5. 比较模型性能 ---
    if all_results:
        best_model_name, best_result = compare_models_performance(all_results)
        
        print(f"\n最终推荐模型: {best_model_name}")
        print(f"最佳超参数: {best_result['best_params']}")
        print(f"交叉验证性能: {best_result['best_cv_score']:.4f}")
    
    # --- 6. 生成可视化结果 ---
    if plotting_data_list:
        # 修改输出目录名和图片文件名，添加“自然老化”字样
        plot_output_dir = f'./model_comparison_plots_自然老化_{target_column_name}'
        plot_all_models_comparison(plotting_data_list, target_column_name, output_dir=plot_output_dir)
    else:
        print("没有模型成功完成评估。")
    
    return all_results

# --- 主程序 ---
if __name__ == "__main__":
    # 请确保这个路径下的CSV文件是已经进行过特征工程（包括老化方式分列）的文件
    # !!! 您之前提供的路径是 '分列数据_特征工程后_v4.csv'，这里改为 'naturliche Aging（训练版本）-adam.csv'
    # 请确认您正在使用正确的、已经经过特征工程处理的CSV文件。
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"
    
    df_original = load_and_process_data(file_path)

    if df_original is not None:
        # 目标列名和特征列名必须与您CSV文件中的实际列名大小写和空格完全匹配
        target_column_options = ['pH-S-CH', 'Cd-CH'] # 这些是您想作为目标列的名称

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
                    # 直接使用用户选择的列名（保持原始大小写）
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