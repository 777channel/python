# ---------------------------------------------------------------------------
# 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import seaborn as sns
import torch

# ---------------------------------------------------------------------------
# 库可用性检查 (这是解决问题的关键部分)
# ---------------------------------------------------------------------------
try:
    import shap
    shap_available = True
    print("SHAP 库成功导入。")
except ImportError:
    shap_available = False
    print("警告: 'shap' 库未找到。SHAP 分析将不可用。请运行 'pip install shap'")

try:
    # --- MODIFIED START ---
    # 从 tabpfn_extensions 库导入 AutoTabPFNRegressor
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    from tabpfn_extensions import interpretability
    tabpfn_available = True
    print("AutoTabPFNRegressor 和其 SHAP 解释器成功导入。")
except ImportError:
    tabpfn_available = False
    AutoTabPFNRegressor = None # 定义一个占位符，以防后续代码意外引用
    print("警告: 'tabpfn-extensions' 库未找到。AutoTabPFNRegressor 模型将不可用。")
    print("请运行 'pip install tabpfn-extensions'。")
    # --- MODIFIED END ---

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
# 2. 模型定义
# ---------------------------------------------------------------------------
# --- MODIFIED START ---
def get_autotabpfn_regressor_config():
    """
    定义 AutoTabPFNRegressor 模型配置，在初始化时设定参数。
    """
    if not tabpfn_available:
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"AutoTabPFNRegressor 将使用设备: {device}")
    
    # 在模型初始化时直接传入 max_time 和 device 参数
    return {
        "model": AutoTabPFNRegressor(max_time=120, device=device)
    }
# --- MODIFIED END ---

# ---------------------------------------------------------------------------
# 3. 绘图与分析函数
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """计算 RMSE, MAE 和 R²"""
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)

def plot_feature_importance_with_percentages(shap_values_obj, feature_names, target_col, output_dir):
    """
    计算并绘制带百分比标签的 SHAP 特征重要性条形图。
    """
    mean_abs_shap = np.mean(np.abs(shap_values_obj.values), axis=0)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_abs_shap})
    total_importance = importance_df['Importance'].sum()
    if total_importance > 0:
        importance_df['Percentage'] = (importance_df['Importance'] / total_importance) * 100
    else:
        importance_df['Percentage'] = 0
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    
    plt.title(f'SHAP Feature Importance for Target: {target_col}', fontsize=24, fontweight='bold')
    plt.xlabel('Mean Absolute SHAP Value (Feature Importance)', fontsize=20, fontweight='bold')
    plt.ylabel('Feature', fontsize=20, fontweight='bold')

    max_bar_width = ax.get_xlim()[1]
    offset = max_bar_width * 0.01

    for i, bar in enumerate(ax.patches):
        percentage = importance_df['Percentage'].iloc[i]
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f'{percentage:.1f}%', va='center', ha='left', fontsize=16)

    plt.xlim(right=max_bar_width * 1.15)
    plt.tight_layout()
    
    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'shap_feature_importance_{safe_target_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 已为目标 '{target_col}' 保存带百分比的特征重要性图至: {save_path}")

def save_all_shap_data_to_csv(shap_values_obj, X_test_transformed_df, target_col, output_dir):
    """
    将所有特征的实际值和 SHAP 值合并并保存到一个 CSV 文件中。
    """
    print(f"    - 正在为目标 '{target_col}' 准备完整的 SHAP 数据...")
    
    values_df = X_test_transformed_df.copy().add_suffix('_Value')
    shap_df = pd.DataFrame(
        shap_values_obj.values, 
        columns=X_test_transformed_df.columns, 
        index=X_test_transformed_df.index
    ).add_suffix('_SHAP')
    
    combined_df = pd.concat([values_df, shap_df], axis=1)
    
    column_order = []
    for feature in X_test_transformed_df.columns:
        column_order.append(f'{feature}_Value')
        column_order.append(f'{feature}_SHAP')
    
    final_df = combined_df[column_order]

    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'all_shap_values_and_data_{safe_target_name}.csv')
    final_df.to_csv(save_path, index=False)
    print(f"    - 已将所有特征的 SHAP 值和数据保存至: {save_path}")

def analyze_and_plot_shap_for_autotabpfn(fitted_pipelines, X_test, output_dir):
    """
    使用 TabPFN 官方的 SHAP 解释器为每个目标进行 SHAP 分析、绘图并保存数据。
    """
    if not tabpfn_available or not shap_available:
        print("\nAutoTabPFNRegressor 或 SHAP 库不可用，跳过 SHAP 分析。")
        return
        
    print("\n--- 正在为 AutoTabPFNRegressor 执行 SHAP 分析 (使用官方解释器) ---")

    for target_col, pipeline in fitted_pipelines.items():
        print(f"\n  --- 正在为目标 '{target_col}' 计算 SHAP 值 ---")
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            # 从 AutoTabPFNRegressor 对象中提取最终训练好的 TabPFN 模型
            fitted_model = pipeline.named_steps['regressor'].model_
            X_test_transformed = preprocessor.transform(X_test)
            
            try:
                transformed_feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                transformed_feature_names = X_test.columns.tolist()

            X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=transformed_feature_names, index=X_test.index)

            print("    - 正在计算... 这可能需要一些时间。")
            shap_values = interpretability.shap.get_shap_values(
                estimator=fitted_model,
                test_x=X_test_transformed_df,
                algorithm="auto"
            )
            print("    - SHAP 值计算完成。")
            
            safe_target_name = target_col.replace("Δ", "delta")
            
            plt.figure()
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
            fig = plt.gcf()
            fig.suptitle(f'SHAP Summary for Target: {target_col}', fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_summary = os.path.join(output_dir, f'shap_summary_{safe_target_name}.pdf')
            fig.savefig(save_path_summary, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    - 已为目标 '{target_col}' 保存 SHAP Summary 图至: {save_path_summary}")

            plot_feature_importance_with_percentages(
                shap_values_obj=shap_values,
                feature_names=transformed_feature_names,
                target_col=target_col,
                output_dir=output_dir
            )

            save_all_shap_data_to_csv(
                shap_values_obj=shap_values,
                X_test_transformed_df=X_test_transformed_df,
                target_col=target_col,
                output_dir=output_dir
            )

        except Exception as e:
            print(f"为目标 '{target_col}' 生成 SHAP 图或数据时发生错误: {e}")
            traceback.print_exc()

# ---------------------------------------------------------------------------
# 4. 主训练与评估流程
# ---------------------------------------------------------------------------
def train_and_evaluate_autotabpfn_regressor(df, feature_cols, target_cols, strata_col, log_cols, impute_cols):
    """
    为每个目标训练并评估一个独立的 AutoTabPFNRegressor 模型。
    """
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=df[strata_col])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)), 
                ('log1p', FunctionTransformer(np.log1p, validate=False))
            ]), [col for col in impute_cols if col in log_cols]),
            ('impute_only', KNNImputer(n_neighbors=5), [col for col in impute_cols if col not in log_cols])
        ],
        remainder='passthrough',
        verbose_feature_names_out=True
    )
    
    fitted_pipelines = {}
    print(f"\n--- 正在评估: AutoTabPFNRegressor ---")
    
    for target_col in target_cols:
        print(f"\n  --- 正在为目标 '{target_col}' 训练模型 (初始化时间限制: 120s) ---")
        try:
            model_config = get_autotabpfn_regressor_config()
            if not model_config:
                print(f"    - 无法创建 AutoTabPFNRegressor 模型，跳过目标 '{target_col}'。")
                continue

            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_config['model'])])
            
            # --- MODIFIED START ---
            print(f"    - 执行5折交叉验证...")
            # 不需要 fit_params，因为 max_time 已经在模型初始化时设置
            cv_scores_target = cross_val_score(pipeline, X_train, y_train[target_col], cv=5, scoring='r2', n_jobs=-1)
            print(f"    - 平均交叉验证 R²: {np.mean(cv_scores_target):.4f}")

            print(f"    - 正在训练最终模型...")
            # 不需要 fit_params，直接调用 fit
            pipeline.fit(X_train, y_train[target_col])
            # --- MODIFIED END ---
            
            fitted_pipelines[target_col] = pipeline
            
            pred_train = pipeline.predict(X_train)
            pred_test = pipeline.predict(X_test)

            rmse_train, mae_train, r2_train = calculate_metrics(y_train[target_col], pred_train)
            print(f"    - 训练集 ({target_col}): R²={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
            rmse_test, mae_test, r2_test = calculate_metrics(y_test[target_col], pred_test)
            print(f"    - 测试集 ({target_col}): R²={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")

        except Exception as e:
            print(f"错误: 无法为目标 '{target_col}' 训练 AutoTabPFNRegressor 模型. 原因: {e}")
            traceback.print_exc()

    if not fitted_pipelines:
        return None, None
        
    return fitted_pipelines, X_test

# ---------------------------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 用户配置区 ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'pH-B', 'BR', 'MPs', 'Size', 'MPR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'type'
    log_transform_columns = ['CEC']
    impute_columns = ['OM', 'CEC', 'pH-S']
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "AutoTabPFNRegressor_分析_120s" # 更新输出文件夹名称
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    if not tabpfn_available:
        print("\nAutoTabPFNRegressor 库不可用，程序即将退出。")
        exit()

    trained_models, X_test_data = train_and_evaluate_autotabpfn_regressor(
        df=df, feature_cols=feature_columns, target_cols=target_columns,
        strata_col=stratification_column, log_cols=log_transform_columns,
        impute_cols=impute_columns
    )
    
    if trained_models:
        analysis_output_dir = os.path.join(output_base_path, output_folder_name)
        os.makedirs(analysis_output_dir, exist_ok=True)
        
        print(f"\n{'='*30}\n模型训练完成，开始 SHAP 分析\n{'='*30}")
        
        analyze_and_plot_shap_for_autotabpfn(
            fitted_pipelines=trained_models,
            X_test=X_test_data,
            output_dir=analysis_output_dir
        )

    print("\n程序执行完毕。")