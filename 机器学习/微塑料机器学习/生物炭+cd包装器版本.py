# ---------------------------------------------------------------------------
# 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import seaborn as sns

# 尝试导入 SHAP 库，如果失败则给出提示
try:
    import shap
    shap_available = True
    print("SHAP 库成功导入。")
except ImportError:
    shap_available = False
    print("警告: 'shap' 库未找到。SHAP 分析将不可用。请运行 'pip install shap'")

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. 全局绘图参数设置
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
def get_final_model_config():
    """
    返回最终模型的配置，使用已找到的最佳参数。
    """
    return {
        "model": ExtraTreesRegressor(
            random_state=42, 
            n_jobs=-1, 
            n_estimators=600,
            max_depth=16, 
            min_samples_leaf=1, 
            min_samples_split=2,
            max_features='sqrt'
        )
    }

# ---------------------------------------------------------------------------
# 3. 评价指标计算函数
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """计算 RMSE, MAE 和 R²"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ---------------------------------------------------------------------------
# 4. 特征重要性分析与绘图
# ---------------------------------------------------------------------------
def analyze_and_plot_feature_importance(best_model_pipeline, feature_names, target_cols, output_dir):
    """
    为每个输出目标提取并绘制模型的特征重要性（Gini Importance）。
    """
    print("\n--- 正在生成特征重要性图 ---")
    multi_output_regressor = best_model_pipeline.named_steps['regressor']
    
    for i, target_col in enumerate(target_cols):
        estimator = multi_output_regressor.estimators_[i]
        importances = estimator.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Feature Importance for Target: {target_col}', fontsize=22, fontweight='bold')
        plt.xlabel('Importance (Gini)', fontsize=18, fontweight='bold')
        plt.ylabel('Feature', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        safe_target_name = target_col.replace("Δ", "delta")
        save_path = os.path.join(output_dir, f'feature_importance_{safe_target_name}.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - 已为目标 '{target_col}' 保存特征重要性图至: {save_path}")

# ---------------------------------------------------------------------------
# 5. SHAP 分析与绘图 (使用 KernelExplainer)
# ---------------------------------------------------------------------------
def analyze_and_plot_shap(best_model_pipeline, X_train, X_test, target_cols, output_dir):
    """
    使用 KernelExplainer 为每个输出目标分别进行 SHAP 分析。
    """
    if not shap_available:
        print("\nSHAP 库不可用，跳过 SHAP 分析。")
        return
        
    print("\n--- 正在执行 SHAP 分析 (使用 KernelExplainer) ---")
    
    # 创建背景数据集 (只需一次)
    print("  - 正在创建 SHAP 背景数据摘要 (使用 k-means)...")
    if X_train.shape[0] > 200:
        background_data = shap.kmeans(X_train.sample(200, random_state=42), 50)
    else:
        k = max(1, X_train.shape[0] // 5)
        background_data = shap.kmeans(X_train, k)

    # 遍历每一个输出目标，并为其单独创建解释器和计算SHAP值
    for i, target_col in enumerate(target_cols):
        print(f"\n  --- 正在为目标 '{target_col}' (第 {i+1}/{len(target_cols)} 个) 计算 SHAP 值 ---")

        # 为当前目标创建一个只返回单列预测值的函数
        def predict_fn_single_output(x):
            x_df = pd.DataFrame(x, columns=X_train.columns)
            # 关键：从多输出预测中只选择当前目标(i)对应的列
            return best_model_pipeline.predict(x_df)[:, i]

        # 使用这个单输出函数创建 KernelExplainer
        explainer = shap.KernelExplainer(predict_fn_single_output, background_data)
        
        # 计算 SHAP 值。
        print(f"    - 正在计算... 警告：这将非常耗时！")
        shap_values = explainer.shap_values(X_test.to_numpy())
        print("    - SHAP 值计算完成。")
        
        # 绘图
        safe_target_name = target_col.replace("Δ", "delta")
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        fig = plt.gcf()
        fig.suptitle(f'SHAP Summary for Target: {target_col}', fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_path_summary = os.path.join(output_dir, f'shap_summary_{safe_target_name}.pdf')
        fig.savefig(save_path_summary, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    - 已为目标 '{target_col}' 保存 SHAP Summary 图至: {save_path_summary}")

        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        fig = plt.gcf()
        fig.suptitle(f'SHAP Mean Impact for Target: {target_col}', fontsize=20)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_path_bar = os.path.join(output_dir, f'shap_bar_{safe_target_name}.pdf')
        fig.savefig(save_path_bar, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    - 已为目标 '{target_col}' 保存 SHAP Bar 图至: {save_path_bar}")

# ---------------------------------------------------------------------------
# 6. 主训练与评估流程
# ---------------------------------------------------------------------------
def train_and_evaluate_final_model(df, feature_cols, target_cols, strata_col, log_cols, impute_cols):
    """
    训练并评估最终模型，并返回所需对象以供后续分析。
    """
    print(f"\n{'='*30}\n为目标列: {', '.join(target_cols)} 训练最终模型\n{'='*30}")
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=X[strata_col]
    )
    print(f"训练集: {X_train.shape[0]} 个样本 | 测试集: {X_test.shape[0]} 个样本")

    # 定义预处理器，确保 FunctionTransformer 包含 feature_names_out 参数
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one'))
            ]), [col for col in impute_cols if col in log_cols]),
            
            ('impute_only', KNNImputer(n_neighbors=5), [col for col in impute_cols if col not in log_cols]),
            
            ('log_only', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one'), 
             [col for col in log_cols if col not in impute_cols])
        ],
        remainder='passthrough'
    )

    final_model_config = get_final_model_config()
    
    print("\n--- 正在使用最佳参数训练最终模型 ---")
    try:
        base_model = final_model_config["model"]
        multi_output_model = MultiOutputRegressor(base_model)
        
        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', multi_output_model)
        ])

        final_pipeline.fit(X_train, y_train)
        print("  - 最终模型训练完成。")

        y_pred_train = final_pipeline.predict(X_train)
        y_pred_test = final_pipeline.predict(X_test)
        
        print("\n--- 各目标性能指标详细报告 ---")
        for i, target_col in enumerate(target_cols):
            rmse_train, mae_train, r2_train = calculate_metrics(y_train[target_col], y_pred_train[:, i])
            print(f"  - 训练集 ({target_col}): RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}, R² = {r2_train:.4f}")
            rmse_test, mae_test, r2_test = calculate_metrics(y_test[target_col], y_pred_test[:, i])
            print(f"  - 测试集 ({target_col}): RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}, R² = {r2_test:.4f}")
        
        return final_pipeline, X_train, X_test

    except Exception as e:
        print(f"错误: 无法训练最终模型. 原因: {e}")
        traceback.print_exc()
        return None, None, None

# ---------------------------------------------------------------------------
# 7. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 用户配置区 ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\生物炭+cd训练版本.csv"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'pH-B','BR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'RM'
    log_transform_columns = ['CEC']
    impute_columns = ['OM', 'CEC','pH-S']
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "最终模型分析"
    # --- 配置结束 ---
    
    # 加载数据
    try:
        df = pd.read_csv(file_path)
        print(f"成功从 '{file_path}' 加载数据。")
    except FileNotFoundError:
        print(f"错误: 文件未找到于路径 '{file_path}'。")
        exit()
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    # 训练并评估模型
    final_model, X_train, X_test = train_and_evaluate_final_model(
        df=df, 
        feature_cols=feature_columns, 
        target_cols=target_columns,
        strata_col=stratification_column, 
        log_cols=log_transform_columns,
        impute_cols=impute_columns
    )
    
    # 如果模型训练成功，则执行后续分析
    if final_model:
        analysis_output_dir = os.path.join(output_base_path, output_folder_name)
        os.makedirs(analysis_output_dir, exist_ok=True)
        
        # 执行特征重要性分析
        analyze_and_plot_feature_importance(
            best_model_pipeline=final_model,
            feature_names=feature_columns,
            target_cols=target_columns,
            output_dir=analysis_output_dir
        )
        
        # 执行 SHAP 分析
        analyze_and_plot_shap(
            best_model_pipeline=final_model,
            X_train=X_train,
            X_test=X_test,
            target_cols=target_columns,
            output_dir=analysis_output_dir
        )

    print("\n程序执行完毕。")