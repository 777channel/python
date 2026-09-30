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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import torch
import traceback
import warnings
warnings.filterwarnings('ignore')

# --- 导入TabPFN (如果可用) ---
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 成功导入。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: 'tabpfen' 库未找到。TabPFN 模型将不可用。请运行 'pip install tabpfen'")

# --- 1. 全局绘图参数 ---
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


# --- 2. 模型定义和扩展的超参数网格 ---
def get_models_with_hyperparameters():
    """定义所有模型及其对应的超参数网格。"""
    models_config = {
    "Extra Trees": {
        "model": ExtraTreesRegressor(random_state=42),
        "params": {
            # n_estimators: 在最优值400周围，探索更高和更低的数量。
            'regressor__n_estimators': [300, 400, 500],

            # max_depth: 在最优值10周围，探索更深和更浅的树。
            'regressor__max_depth': [8, 10, 12],

            # min_samples_leaf: 在最优值1周围，强制探索更严格的正则化选项(2, 3)。
            # 这是对抗过拟合的关键。
            'regressor__min_samples_leaf': [1, 2, 3],

            # min_samples_split: 在最优值2周围，探索更严格的正则化选项(3, 5)。
            'regressor__min_samples_split': [2, 3, 5],

            # max_features: 保持已证明的最优选择。
            'regressor__max_features': ['sqrt']
        }
    },

    # --- 以下模型参数固定，作为性能基准 ---
    "LightGBM": {
        # 固定为历史最优参数
        "model": lgb.LGBMRegressor(random_state=42, verbosity=-1),
        "params": {
            'regressor__learning_rate': [0.1],
            'regressor__n_estimators': [100],
            'regressor__num_leaves': [14],
            'regressor__reg_alpha': [0.1],
            'regressor__reg_lambda': [1]
        }
    },

    "Gradient Boosting": {
        # 固定为历史最优参数
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            'regressor__learning_rate': [0.1],
            'regressor__max_depth': [4],
            'regressor__max_features': ['log2'],
            'regressor__min_samples_leaf': [15],
            'regressor__n_estimators': [300],
            'regressor__subsample': [0.8]
        }
    },
    
    "SVR": {
        # 固定为历史最优参数
        "model": SVR(),
        "params": {
            'regressor__C': [250],
            'regressor__gamma': [1.0],
            'regressor__kernel': ['rbf']
        }
    },

    "Random Forest": {
        # 固定为历史最优参数
        "model": RandomForestRegressor(random_state=42),
        "params": {
            'regressor__max_depth': [8],
            'regressor__max_features': ['log2'],
            'regressor__min_samples_leaf': [3],
            'regressor__min_samples_split': [6],
            'regressor__n_estimators': [600]
        }
    },

    "XGBoost": {
        # 固定为历史最优参数
        "model": xgb.XGBRegressor(random_state=42, objective='reg:squarederror'),
        "params": {
            'regressor__gamma': [0],
            'regressor__learning_rate': [0.1],
            'regressor__max_depth': [4],
            'regressor__n_estimators': [500],
            'regressor__reg_alpha': [0.1],
            'regressor__reg_lambda': [1]
        }
    },
    
    "Ridge": {
        "model": Ridge(random_state=42),
        "params": {'regressor__alpha': [80.0]}
    },
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    }
}

    if tabpfn_available:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"TabPFN 将使用设备: {device}")
        models_config["TabPFN"] = {"model": TabPFNRegressor(device=device), "params": {}}
    return models_config

# --- 3. 模型对比绘图函数 (无变化) ---
def plot_all_models_comparison(plotting_data_list, target_col, data_source_name, output_dir):
    if not plotting_data_list: return
    n_models = len(plotting_data_list)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, n_rows * 8.5))
    axes = axes.flatten()
    plotting_data_list.sort(key=lambda x: x['CV_Score'], reverse=True)
    best_model_name = plotting_data_list[0]['Model']
    train_color = '#2aace2'
    test_color = '#f57a85'
    for i, r in enumerate(plotting_data_list):
        ax = axes[i]
        y_train, y_pred_train = r['Train_Values'], r['Train_Predictions']
        y_test, y_pred_test = r['Test_Values'], r['Test_Predictions']
        ax.scatter(y_train, y_pred_train, alpha=0.6, edgecolor='k', facecolors=train_color, s=100, label='Train')
        ax.scatter(y_test, y_pred_test, alpha=0.8, edgecolor='k', facecolors=test_color, s=120, label='Test')
        all_y = np.concatenate([y_train, y_pred_train, y_test, y_pred_test])
        min_val, max_val = all_y.min(), all_y.max()
        padding = (max_val - min_val) * 0.05
        ax.plot([min_val - padding, max_val + padding], [min_val - padding, max_val + padding], 'k--', lw=2)
        ax.set_xlim(min_val - padding, max_val + padding)
        ax.set_ylim(min_val - padding, max_val + padding)
        is_best_text = " (Best)" if r['Model'] == best_model_name else ""
        ax.set_title(f"{r['Model']}{is_best_text}\nCV R² = {r['CV_Score']:.3f}", fontweight='bold')
        ax.set_xlabel("Observed Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        ax.grid(True, linestyle=':')
        train_text = (f"Train Set\n"
                      f"  R²: {r['R2_Train']:.3f}\n"
                      f"  RMSE: {r['RMSE_Train']:.3f}\n"
                      f"  MAE: {r['MAE_Train']:.3f}")
        ax.text(0.05, 0.95, train_text, transform=ax.transAxes, fontsize=16, color='white',
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc=train_color, alpha=0.8))
        test_text = (f"Test Set\n"
                     f"  R²: {r['R2_Test']:.3f}\n"
                     f"  RMSE: {r['RMSE_Test']:.3f}\n"
                     f"  MAE: {r['MAE_Test']:.3f}")
        ax.text(0.95, 0.35, test_text, transform=ax.transAxes, fontsize=16, color='white',
                horizontalalignment='right', verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc=test_color, alpha=0.8))
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f'Model Comparison for Target: {target_col} (Data: {data_source_name})', fontsize=30, fontweight='bold')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'model_comparison_{target_col.replace("Δ", "delta")}_{data_source_name}.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n所有模型的对比图已保存至: {save_path}")

# --- 4. 主训练与评估流程 (已更新) ---
# 【修改】: 新增 impute_cols 参数
def train_and_evaluate_models(df, feature_cols, target_col, strata_col, log_cols, impute_cols):
    """
    【修改】: 运行完整的训练和评估流程。
    - 仅对 impute_cols (OM, CEC) 进行KNN填充。
    - 对 log_cols (包含OM, CEC) 进行log转换。
    - 不进行标准化。
    """
    print(f"\n{'='*30}\n为目标列: {target_col} 训练模型\n{'='*30}")
    X = df[feature_cols]
    y = df[target_col]
    
    print("正在进行70/30分层抽样...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=X[strata_col]
    )
    print(f"训练集: {X_train.shape[0]} 个样本 | 测试集: {X_test.shape[0]} 个样本")

    # 【修改】: 创建更精细的预处理管道
    # 管道1: 用于需要填充和log转换的列
    impute_and_log_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('log1p', FunctionTransformer(np.log1p, validate=False))
    ])
    
    # 管道2: 用于仅需要log转换的列 (无填充)
    log_only_transformer = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p, validate=False))
    ])
    
    # 确定每个管道应用于哪些列
    # 这里的 `impute_cols` 就是需要填充和log化的列
    cols_impute_and_log = impute_cols
    # 需要log化但不需要填充的列
    cols_log_only = [col for col in log_cols if col not in impute_cols]
    
    # 使用 ColumnTransformer 组合所有处理步骤
    # 未被指定的列将通过 remainder='passthrough' 保持原样
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', impute_and_log_transformer, cols_impute_and_log),
            ('log_only', log_only_transformer, cols_log_only)
        ],
        remainder='passthrough'
    )

    models_config = get_models_with_hyperparameters()
    plotting_data_list = []
    
    for name, config in models_config.items():
        print(f"\n--- 正在评估: {name} ---")
        try:
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', config['model'])])

            if name == "TabPFN":
                X_train_transformed_check = preprocessor.fit_transform(X_train, y_train)
                if X_train_transformed_check.shape[0] > 1024 or X_train_transformed_check.shape[1] > 100:
                    print(f"跳过TabPFN: 数据规模超出限制 (样本数: {X_train_transformed_check.shape[0]}, 特征数: {X_train_transformed_check.shape[1]})")
                    continue
                
                print(f"正在为 {name} 计算5折交叉验证分数...")
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
                cv_score = np.mean(cv_scores)
                
                pipeline.fit(X_train, y_train)
                best_model = pipeline
                best_params = {}
                print(f"  - 最佳交叉验证 R²: {cv_score:.4f}")

            else:
                print(f"正在为 {name} 运行GridSearchCV... 这可能需要一些时间。")
                grid_search = GridSearchCV(pipeline, config['params'], cv=5, scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                cv_score = grid_search.best_score_
                best_params = grid_search.best_params_
                print(f"  - 最佳交叉验证 R²: {cv_score:.4f}")
                print(f"  - 最佳参数: {best_params}")

            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            mae_train = mean_absolute_error(y_train, y_pred_train)

            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae_test = mean_absolute_error(y_test, y_pred_test)

            print(f"  - 训练集性能: R²={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
            print(f"  - 测试集性能: R²={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
            
            plotting_data_list.append({
                'Model': name, 'CV_Score': cv_score, 
                'R2_Train': r2_train, 'RMSE_Train': rmse_train, 'MAE_Train': mae_train,
                'R2_Test': r2_test, 'RMSE_Test': rmse_test, 'MAE_Test': mae_test,
                'Train_Values': y_train, 'Train_Predictions': y_pred_train,
                'Test_Values': y_test, 'Test_Predictions': y_pred_test,
            })
        except Exception as e:
            print(f"错误: 无法训练 {name}. 原因: {e}")
            traceback.print_exc()
            
    return plotting_data_list

# --- 5. 主程序入口 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\生物炭+cd训练版本.csv"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'pH-B','BR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'RM'
    
    log_transform_columns = ['CEC','Cd-C']  # 包含需要log转换的列
    # 【修改】: 新增列表，明确指定只对这两列进行填充
    impute_columns = ['OM', 'CEC','pH-S']
    
    try:
        df = pd.read_csv(file_path)
        print(f"成功从 '{file_path}' 加载数据。")
    except FileNotFoundError:
        print(f"错误: 文件未找到于路径 '{file_path}'。")
        exit()
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    while True:
        print("\n请选择要预测的目标列:")
        for i, col in enumerate(target_columns):
            print(f"{i + 1}: {col}")
        print("0: 退出程序")
        try:
            target_choice = int(input(f"请输入选项编号 (0-{len(target_columns)}): "))
            if target_choice == 0:
                break
            elif 1 <= target_choice <= len(target_columns):
                selected_target = target_columns[target_choice - 1]
                # 【修改】: 调用函数时传入新的 impute_columns 列表
                results = train_and_evaluate_models(
                    df=df, 
                    feature_cols=feature_columns, 
                    target_col=selected_target,
                    strata_col=stratification_column, 
                    log_cols=log_transform_columns,
                    impute_cols=impute_columns
                )
                if results:
                    base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
                    plot_folder_name = "生物炭+cd模型对比"
                    plot_output_dir = os.path.join(base_path, plot_folder_name)
                    data_source_name = "Targeted Imputation (OM, CEC)" # 更新图表标题
                    plot_all_models_comparison(
                        plotting_data_list=results, 
                        target_col=selected_target,
                        data_source_name=data_source_name, 
                        output_dir=plot_output_dir
                    )
            else:
                print("无效选项，请重试。")
        except ValueError:
            print("无效输入，请输入数字。")
    print("\n程序执行完毕。")