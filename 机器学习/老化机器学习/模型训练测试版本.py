import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor # <--- 新增导入
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import traceback
from bayes_opt import BayesianOptimization
import shap
import warnings

# --- 全局设置 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# --- 数据加载函数 (不变) ---
def load_and_process_data(file_path):
    """从指定路径加载CSV文件，尝试多种编码，并将所有列转换为数值类型。"""
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
        if not read_successfully:
            print(f"错误: 尝试所有常用编码后仍无法读取文件 '{file_path}'。")
            return None
    except Exception as e:
        print(f"加载数据时发生意外错误: {e}")
        return None
    df.columns = [col.lstrip('\ufeff') for col in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- 通用SHAP绘图函数 (不变) ---
def perform_shap_plot(shap_values, data, feature_names, target_name, model_name, output_dir):
    """生成并保存SHAP摘要图和特征重要性图。"""
    print(f"\n--- 正在为 {model_name} 生成SHAP图 ---")
    os.makedirs(output_dir, exist_ok=True)
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {model_name} ({target_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_name}_{model_name}_SHAP_summary.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance for {model_name} ({target_name})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_name}_{model_name}_SHAP_importance.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"SHAP 图已成功保存至: {output_dir}")

# --- 独立的SHAP分析函数 (不变) ---
def run_shap_analysis(final_pipeline, X_train, X_test, target_column_name, model_name):
    """一个专门的函数，接收训练好的Pipeline和数据，执行SHAP分析。"""
    print("\n--- [步骤2/2] 开始进行SHAP分析 ---")
    final_model = final_pipeline.named_steps['regressor']
    preprocessor = final_pipeline.named_steps['preprocessor']
    X_test_processed = preprocessor.transform(X_test)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_processed)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names_out = X_train.columns.tolist()
    output_dir = f'./{model_name}_SHAP_analysis_{target_column_name}'
    perform_shap_plot(shap_values, X_test_processed, feature_names_out, target_column_name, model_name, output_dir)
    print("--- SHAP分析和绘图已完成 ---")

# --- 【新】随机森林训练、优化与分析主函数 ---
def train_and_optimize_random_forest(df, target_column_name):
    """
    专门针对随机森林模型，进行数据处理和贝叶斯优化，并返回训练结果。
    """
    print(f"\n{'='*80}")
    print(f"--- [步骤1/2] 开始为目标列 '{target_column_name}' 训练 随机森林 模型 ---")
    print(f"{'='*80}")
    
    df_processed = df.copy()
    df_processed.dropna(subset=[target_column_name], inplace=True)
    
    # 1. 定义特征与目标 (不变)
    all_possible_target_cols = ['pH-S-CH', 'Cd-CH']
    other_non_feature_cols = ['有效老化', 'M']
    strata_col_name = 'T-Aging'
    cols_to_exclude = list(set(all_possible_target_cols + other_non_feature_cols + [strata_col_name]))
    feature_columns = [col for col in df.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]
    print(f"\n已从特征集中排除以下列: {cols_to_exclude}")
    print(f"已选定目标 (y): '{target_column_name}' | 使用的特征 (X) 数量: {len(feature_columns)}")
    
    # 2. 分层抽样 (不变)
    strata = None
    if strata_col_name in df_processed.columns:
        strata_series = df_processed[strata_col_name].copy()
        if pd.api.types.is_numeric_dtype(strata_series) and not pd.api.types.is_integer_dtype(strata_series):
            q_bins = min(5, strata_series.nunique())
            if q_bins >= 2:
                strata_series = pd.qcut(strata_series, q=q_bins, labels=False, duplicates='drop')
        if hasattr(strata_series, 'value_counts') and strata_series.notna().all() and all(strata_series.value_counts() >= 2):
            strata = strata_series
            print(f"将使用 '{strata_col_name}' 列进行分层抽样。")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strata)
    
    # 3. 预处理流程 (不变)
    preprocessor = ColumnTransformer(transformers=[('num', Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())]), feature_columns)], remainder='passthrough')
    
    # 4. 随机森林的贝叶斯优化
    print("\n--- 开始随机森林的贝叶斯超参数优化 ---")
    pbounds = {
        'n_estimators': (100, 1000),
        'max_depth': (5, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20),
        'max_features': (0.1, 1.0)
    }

    def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        # 需将浮点型参数转换为整型
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            ))
        ])
        return np.mean(cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1))

    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, random_state=42, verbose=2)
    optimizer.maximize(init_points=10, n_iter=40)
    
    best_params = optimizer.max['params']
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])
    
    print(f"\n最佳交叉验证 R²: {optimizer.max['target']:.4f}\n最佳参数: {best_params}")
    
    # 5. 最终模型训练与评估
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1, **best_params))
    ])
    final_pipeline.fit(X_train, y_train)
    r2_test = r2_score(y_test, final_pipeline.predict(X_test))
    print(f"\n最终模型在测试集上的 R²: {r2_test:.4f}")
    print(f"--- 随机森林模型训练与优化完毕 ---")
    
    # 返回训练好的Pipeline和测试数据
    return final_pipeline, X_train, X_test

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv"
    df_original = load_and_process_data(file_path)

    if df_original is not None:
        target_column_options = ['pH-S-CH', 'Cd-CH']
        
        while True:
            print("\n请选择要训练的目标列:")
            [print(f"{i + 1}. {col_name}") for i, col_name in enumerate(target_column_options)]
            print("0. 退出程序")
            try:
                choice_target_str = input(f"请输入目标列编号 (0-{len(target_column_options)}): ")
                if not choice_target_str.strip():
                    print("输入为空，请重新输入。")
                    continue
                choice_target = int(choice_target_str)

                if choice_target == 0:
                    print("程序退出。")
                    break
                elif 1 <= choice_target <= len(target_column_options):
                    selected_target_col = target_column_options[choice_target - 1]
                    
                    # 步骤1: 调用随机森林的训练和优化函数
                    training_results = train_and_optimize_random_forest(df_original.copy(), selected_target_col)
                    
                    # 步骤2: 检查训练是否成功，然后调用SHAP分析函数
                    if training_results:
                        final_pipeline, X_train, X_test = training_results
                        run_shap_analysis(final_pipeline, X_train, X_test, selected_target_col, model_name="RandomForest")
                    
                    print(f"\n✅ 对 '{selected_target_col}' 的分析已全部完成。您可以选择新的目标列或输入0退出。")
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"处理过程中发生致命错误: {e}")
                traceback.print_exc()