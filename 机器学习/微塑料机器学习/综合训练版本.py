# ---------------------------------------------------------------------------
# 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import seaborn as sns
import torch

# 尝试导入 TabPFNRegressor
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 成功导入。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: 'tabpfen' 库未找到。TabPFN 模型将不可用。")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. 全局绘图参数
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ---------------------------------------------------------------------------
# 2. 模型定义 (第二轮优化参数)
# ---------------------------------------------------------------------------
def get_models_with_hyperparameters():
    """
    【第二轮优化】根据第一轮结果，在最佳参数附近进行精细化和扩展搜索。
    """
    models_config = {
        "Linear Regression": {"model": LinearRegression(), "params": {}},
        "Ridge": {"model": Ridge(random_state=42), "params": {'alpha': [30.0, 50.0, 70.0, 100.0]}},
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {'n_estimators': [100, 200, 300], 'max_depth': [5, 8, 12], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [10, 20], 'max_features': ['sqrt', 'log2']}
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {'n_estimators': [300], 'learning_rate': [0.01], 'max_depth': [6], 'subsample': [0.8], 'min_samples_leaf': [5], 'max_features': ['sqrt']}
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
            "params": {'n_estimators': [500], 'learning_rate': [0.01], 'max_depth': [7], 'colsample_bytree': [0.8], 'subsample': [0.6], 'reg_alpha': [0.1], 'reg_lambda': [5], 'gamma': [0]}
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1),
            "params": {'n_estimators': [500], 'learning_rate': [0.01], 'num_leaves': [50], 'reg_alpha': [0.5], 'reg_lambda': [1], 'colsample_bytree': [0.8], 'subsample': [0.8]}
        },
        "Extra Trees": {
            "model": ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "params": {'n_estimators': [100, 300], 'max_depth': [9, 12, 15, None], 'min_samples_leaf': [2, 5, 10], 'min_samples_split': [5, 10, 20], 'max_features': ['sqrt']}
        },
        "SVR": {
            "model": SVR(),
            "params": {'C': [0.1, 0.5, 1.0, 2.0], 'kernel': ['rbf'], 'gamma': ['scale', 0.01, 0.05, 0.1]}
        }
    }

    for model_name, config in models_config.items():
        config['params'] = {f'regressor__estimator__{k}': v for k, v in config['params'].items()}

    if tabpfn_available:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"TabPFN 将使用设备: {device}")
        models_config["TabPFN"] = {"model": TabPFNRegressor(device=device), "params": {}}
    return models_config

# ---------------------------------------------------------------------------
# 3. 绘图与分析函数
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """计算并返回 RMSE, MAE, R²"""
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)


def plot_all_models_comparison(all_results, target_cols, output_dir):
    """
    为每个目标变量生成并保存一个包含所有模型对比的图。
    """
    print("\n--- 正在生成所有模型的对比图 ---")
    
    # 定义模型全称到缩写的映射
    model_name_map = {
        "Linear Regression": "LR", "Ridge": "Ridge", "Random Forest": "RF",
        "Gradient Boosting": "GBDT", "XGBoost": "XGBoost", "LightGBM": "LightGBM",
        "Extra Trees": "ExtraTrees", "SVR": "SVR", "TabPFN": "TabPFN"
    }

    # 为每个目标变量创建一个图
    for target_index, target_col in enumerate(target_cols):
        print(f"  - 正在为目标 '{target_col}' 准备绘图数据...")
        
        plotting_data_list = []
        for result in all_results:
            # 提取当前目标的数据
            y_train_target = result['Train_Values'][:, target_index]
            pred_train_target = result['Train_Predictions'][:, target_index]
            y_test_target = result['Test_Values'][:, target_index]
            pred_test_target = result['Test_Predictions'][:, target_index]
            
            # 为当前目标计算指标
            rmse_train, mae_train, r2_train = calculate_metrics(y_train_target, pred_train_target)
            rmse_test, mae_test, r2_test = calculate_metrics(y_test_target, pred_test_target)
            
            plotting_data_list.append({
                'Model': model_name_map.get(result['Model'], result['Model']),
                'CV_Score': result.get('CV_Score', np.nan),
                'Train_Values': y_train_target,
                'Train_Predictions': pred_train_target,
                'Test_Values': y_test_target,
                'Test_Predictions': pred_test_target,
                'R2_Train': r2_train, 'RMSE_Train': rmse_train, 'MAE_Train': mae_train,
                'R2_Test': r2_test, 'RMSE_Test': rmse_test, 'MAE_Test': mae_test,
            })
        
        # 调用核心绘图函数
        _generate_comparison_plot(plotting_data_list, target_col, output_dir)


def _generate_comparison_plot(plotting_data_list, target_col, output_dir='.'):
    """为单一目标变量创建一个大图，包含所有模型的预测比较。"""
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
        
        # 计算当前子图的行列位置
        current_row = i // n_cols
        current_col = i % n_cols
        
        if model_name_short in model_results_dict:
            r = model_results_dict[model_name_short]
            
            y_train, y_pred_train = np.array(r.get('Train_Values', [])), np.array(r.get('Train_Predictions', []))
            y_test, y_pred_test = np.array(r.get('Test_Values', [])), np.array(r.get('Test_Predictions', []))
            r2_train, rmse_train, mae_train = r.get('R2_Train', np.nan), r.get('RMSE_Train', np.nan), r.get('MAE_Train', np.nan)
            r2_test, rmse_test, mae_test = r.get('R2_Test', np.nan), r.get('RMSE_Test', np.nan), r.get('MAE_Test', np.nan)
            cv_score = r.get('CV_Score', np.nan) 
            
            if y_train.size > 0: ax.scatter(y_train, y_pred_train, alpha=1, edgecolor='w',facecolors=train_color, s=400, label='Train')
            if y_test.size > 0: ax.scatter(y_test, y_pred_test, alpha=1,edgecolor='w',facecolors=test_color, s=400, label='Test')
            
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
            is_best_text = " (Best)" if model_name_short == best_model_name else ""
            ax.set_title(f'({model_label_char}) {model_name_short}{is_best_text}\nCV R²={cv_score:.3f}', fontsize=28, fontweight='bold')
            
            # 为最后一行的所有子图添加 x 轴标签
            if current_row == n_rows - 1:
                ax.set_xlabel('Observed Values', fontsize=30, fontweight='bold')
            
            # 为第一列的所有子图添加 y 轴标签
            if current_col == 0:
                ax.set_ylabel('Predicted Values', fontsize=30, fontweight='bold')
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(30); label.set_fontweight('bold')
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        else:
            ax.text(0.5, 0.5, f'({chr(97 + i)}) {model_name_short}\n无数据', ha='center', va='center', transform=ax.transAxes, fontsize=28)
            ax.axis('off')

    for j in range(i + 1, n_rows * n_cols):
        if j < len(axes): fig.delaxes(axes[j])

    # 移除原来的全局标签
    safe_target_name = target_col.replace('Δ', 'delta_')
    fig.suptitle(f'{target_col}', fontsize=32, y=0.98, fontweight='bold')
    
    plt.subplots_adjust(wspace=0.3, hspace=0.35, left=0.07, right=0.95, top=0.93, bottom=0.07)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{safe_target_name}_all_models_comparison.pdf')
    
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  - 所有模型对 '{target_col}' 的比较图已成功保存至: {save_path}")

# ---------------------------------------------------------------------------
# 4. 主训练与评估流程
# ---------------------------------------------------------------------------
def compare_models(df, feature_cols, target_cols, strata_col, log_cols, impute_cols):
    X = df[feature_cols]
    y = df[target_cols]
    stratify_data = df[strata_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=stratify_data)

    cols_impute_and_log = [col for col in impute_cols if col in log_cols]
    cols_impute_only = [col for col in impute_cols if col not in log_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one'))]), cols_impute_and_log),
            ('impute_only', Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5))]), cols_impute_only)
        ],
        remainder='passthrough'
    )

    models_config = get_models_with_hyperparameters()
    all_results = []

    for name, config in models_config.items():
        print(f"\n--- 正在评估: {name} ---")
        try:
            if name == "TabPFN":
                if not tabpfn_available: continue
                pred_train = np.full(y_train.shape, np.nan)
                pred_test = np.full(y_test.shape, np.nan)
                cv_scores = []
                best_model = None # TabPFN doesn't have a single 'best_model' in this setup
                params = {}

                for i, target_col in enumerate(target_cols):
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', config['model'])])
                    # Note: TabPFN is not typically cross-validated this way, but we do it for a comparable score.
                    cv_scores_target = cross_val_score(pipeline, X_train, y_train.iloc[:, i], cv=5, scoring='r2')
                    cv_scores.append(np.mean(cv_scores_target))
                    pipeline.fit(X_train, y_train.iloc[:, i])
                    pred_train[:, i] = pipeline.predict(X_train)
                    pred_test[:, i] = pipeline.predict(X_test)
                cv_score = np.mean(cv_scores)
            else:
                multi_output_model = MultiOutputRegressor(config['model'])
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', multi_output_model)])

                grid_search = GridSearchCV(pipeline, config['params'], cv=5, scoring='r2', n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)

                best_model = grid_search.best_estimator_
                cv_score = grid_search.best_score_
                params = grid_search.best_params_
                pred_train = best_model.predict(X_train)
                pred_test = best_model.predict(X_test)

            print(f"  - {name} 完成评估。")
            print(f"  - 最佳平均交叉验证 R²: {cv_score:.4f}")
            if params:
                print(f"  - 最佳参数: {params}")

            print("  --- 性能指标详细报告 ---")
            for i, target_col in enumerate(target_cols):
                rmse_train, mae_train, r2_train = calculate_metrics(y_train.iloc[:, i], pred_train[:, i])
                print(f"    - 训练集 ({target_col}): R²={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")

                rmse_test, mae_test, r2_test = calculate_metrics(y_test.iloc[:, i], pred_test[:, i])
                print(f"    - 测试集 ({target_col}): R²={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")

            all_results.append({
                'Model': name, 'CV_Score': cv_score, 'Best_Params': params,
                'Best_Estimator': best_model if 'best_model' in locals() else None,
                'Train_Values': y_train.values, 'Train_Predictions': pred_train,
                'Test_Values': y_test.values, 'Test_Predictions': pred_test
            })

        except Exception as e:
            print(f"错误: 无法训练 {name}. 原因: {e}")
            traceback.print_exc()


    return all_results

# ---------------------------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 用户配置区 ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T','BET', 'pH-B', 'BR', 'MPs', 'Size', 'MPR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'type'
    log_transform_columns = ['CEC']
    impute_columns = ['OM', 'CEC', 'pH-S','BET']
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "合并分析模型对比优化"

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    model_results = compare_models(
        df=df, feature_cols=feature_columns, target_cols=target_columns,
        strata_col=stratification_column, log_cols=log_transform_columns,
        impute_cols=impute_columns
    )

    if model_results:
        analysis_output_dir = os.path.join(output_base_path, output_folder_name)
        os.makedirs(analysis_output_dir, exist_ok=True)

        # 调用新的绘图函数，它将为每个目标生成一个对比图
        plot_all_models_comparison(
            all_results=model_results, 
            target_cols=target_columns,
            output_dir=analysis_output_dir
        )

        best_model_result = max(model_results, key=lambda x: x['CV_Score'])
        print(f"\n{'='*30}\n最佳模型是: {best_model_result['Model']} (CV R² = {best_model_result['CV_Score']:.4f})")

    print("\n程序执行完毕。")