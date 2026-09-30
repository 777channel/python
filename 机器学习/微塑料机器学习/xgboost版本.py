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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import seaborn as sns
import joblib

# ---------------------------------------------------------------------------
# 库可用性检查
# ---------------------------------------------------------------------------
try:
    import shap
    shap_available = True
    print("SHAP 库成功导入。")
except ImportError:
    shap_available = False
    print("警告: 'shap' 库未找到。")

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    xgboost_available = True
    print("XGBoost 库成功导入。")
except ImportError:
    xgboost_available = False
    print("警告: 'xgboost' 库未找到。")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. 全局绘图参数
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
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
# 2. 辅助功能函数
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)

def plot_feature_importance_with_percentages(shap_values_obj, feature_names, target_col, output_dir):
    """绘制带百分比的特征重要性图"""
    mean_abs_shap = np.mean(np.abs(shap_values_obj.values), axis=0)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_abs_shap})
    total_importance = importance_df['Importance'].sum()
    importance_df['Percentage'] = (importance_df['Importance'] / total_importance) * 100 if total_importance > 0 else 0
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(14, 10))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'SHAP Feature Importance: {target_col}', fontsize=24, fontweight='bold')
    
    max_bar_width = ax.get_xlim()[1]
    for i, bar in enumerate(ax.patches):
        percentage = importance_df['Percentage'].iloc[i]
        ax.text(bar.get_width() + (max_bar_width * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{percentage:.1f}%', va='center', ha='left', fontsize=16)

    plt.tight_layout()
    safe_target = target_col.replace("Δ", "delta")
    plt.savefig(os.path.join(output_dir, f'shap_feature_importance_{safe_target}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

def save_all_shap_data_to_csv(shap_values_obj, X_transformed_df, target_col, output_dir):
    """保存特征值与对应的 SHAP 值"""
    values_df = X_transformed_df.reset_index(drop=True).add_suffix('_Value')
    shap_df = pd.DataFrame(shap_values_obj.values, columns=X_transformed_df.columns).add_suffix('_SHAP')
    
    column_order = []
    for feature in X_transformed_df.columns:
        column_order.append(f'{feature}_Value')
        column_order.append(f'{feature}_SHAP')
    
    combined_df = pd.concat([values_df, shap_df], axis=1)[column_order]
    safe_target = target_col.replace("Δ", "delta")
    combined_df.to_csv(os.path.join(output_dir, f'all_shap_values_and_data_{safe_target}.csv'), index=False)

def analyze_and_plot_shap_for_xgboost(fitted_pipelines, X_full, output_dir, feature_cols, impute_cols, log_cols):
    """执行 SHAP 分析并保存结果"""
    if not shap_available: return
    
    # 按照 ColumnTransformer 的处理顺序还原特征名
    group1 = [col for col in impute_cols if col in log_cols]
    group2 = [col for col in impute_cols if col not in log_cols]
    group3 = [col for col in feature_cols if col not in set(group1 + group2)]
    final_feature_names = group1 + group2 + group3

    for target_col, pipeline in fitted_pipelines.items():
        print(f"--- 正在分析目标 '{target_col}' ---")
        preprocessor = pipeline.named_steps['preprocessor']
        model = pipeline.named_steps['regressor']
        
        X_trans = preprocessor.transform(X_full)
        X_trans_df = pd.DataFrame(X_trans, columns=final_feature_names)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_trans_df)

        # 保存摘要图
        plt.figure()
        shap.summary_plot(shap_values, X_trans_df, show=False)
        plt.gcf().savefig(os.path.join(output_dir, f'shap_summary_{target_col.replace("Δ", "delta")}.pdf'), bbox_inches='tight')
        plt.close()

        # 绘制重要性百分比图
        plot_feature_importance_with_percentages(shap_values, final_feature_names, target_col, output_dir)
        
        # 保存 CSV 数据
        save_all_shap_data_to_csv(shap_values, X_trans_df, target_col, output_dir)

# ---------------------------------------------------------------------------
# 3. 核心训练函数 (已统一内部变量名)
# ---------------------------------------------------------------------------
def train_final_model(df, feature_cols, target_cols, strata_col, log_cols, impute_cols, save_dir):
    """使用固定最佳参数训练最终模型"""
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=df[strata_col])
    
    # 在函数内部直接使用参数传入的 log_cols 和 impute_cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)), 
                ('log1p', FunctionTransformer(np.log1p, validate=False))
            ]), [col for col in impute_cols if col in log_cols]),
            ('impute_only', KNNImputer(n_neighbors=5), [col for col in impute_cols if col not in log_cols])
        ], remainder='passthrough'
    )

    # 锁定最佳参数
    best_params = {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 7,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 5,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror'
    }

    best_pipelines = {}
    for target in target_cols:
        print(f"\n--- 正在构建目标 '{target}' 的模型 ---")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(**best_params))])
        pipeline.fit(X_train, y_train[target])
        
        # 记录结果
        best_pipelines[target] = pipeline
        
        # 评估性能
        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)
        _, _, r2_tr = calculate_metrics(y_train[target], pred_train)
        _, _, r2_te = calculate_metrics(y_test[target], pred_test)
        print(f"    训练集 R²: {r2_tr:.4f} | 测试集 R²: {r2_te:.4f}")

        # 保存模型
        safe_target = target.replace("Δ", "delta")
        joblib.dump(pipeline, os.path.join(save_dir, f'final_model_{safe_target}.joblib'))
        
    return best_pipelines, X

# ---------------------------------------------------------------------------
# 4. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 原生配置区 (已标准化变量名) ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    feature_cols = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T','BET', 'pH-B', 'BR', 'MPs', 'Size', 'MPR','type']
    target_cols = ['ΔpH', 'ΔCd']
    strata_col = 'type'
    log_cols = ['CEC']      # 统一名称：log_cols
    impute_cols = ['OM', 'CEC', 'pH-S','BET']  # 统一名称：impute_cols
    
    # 原始输出路径
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "XGBoost_HugeRange_CorrectNames"
    
    analysis_output_dir = os.path.join(output_base_path, output_folder_name)
    os.makedirs(analysis_output_dir, exist_ok=True)

    print(f"数据加载中: {file_path}")
    df_raw = pd.read_csv(file_path)

    # 1. 执行训练
    models, X_full_data = train_final_model(
        df=df_raw, 
        feature_cols=feature_cols, 
        target_cols=target_cols, 
        strata_col=strata_col, 
        log_cols=log_cols, 
        impute_cols=impute_cols, 
        save_dir=analysis_output_dir
    )

    # 2. 执行 SHAP 分析
    print(f"\n{'='*30}\n开始全数据集 SHAP 分析\n{'='*30}")
    analyze_and_plot_shap_for_xgboost(
        fitted_pipelines=models,
        X_full=X_full_data, 
        output_dir=analysis_output_dir,
        feature_cols=feature_cols,
        impute_cols=impute_cols,
        log_cols=log_cols
    )

    print(f"\n任务运行完毕。输出路径: {analysis_output_dir}")