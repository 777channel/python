import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 导入 KNNImputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import traceback
import joblib
import shap

# --- 导入检查 ---
try:
    from tabpfn_extensions import unsupervised, interpretability
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    tabpfn_available = True
    pfn_interpret_available = True
    print("TabPFN 及相关扩展库导入成功。")
except ImportError:
    tabpfn_available = False
    pfn_interpret_available = False
    TabPFNRegressor = None # Placeholder
    print("警告: 未找到 TabPFN 或 tabpfen_extensions 库。")

# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# --- 功能函数定义 ---

def load_and_process_data(file_path):
    """加载并对数据进行初步处理。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取文件时发生错误: {e}"); return None
    # 确保所有列名都存在
    cols_to_convert = ['BR', 'pH-S-CH', 'Cd-CH', 'T_aging']
    for col in cols_to_convert:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.info()
    return df

def clean_outliers_with_tabpfen(df, features_to_check, contamination=0.05):
    """使用 TabPFNUnsupervisedModel 来检测并移除异常值。"""
    print(f"开始异常值检测... 预计移除比例: {contamination:%}")
    
    if not features_to_check:
        print("警告: 未提供用于异常值检测的特征列，跳过此步骤。")
        return df

    # 在异常值检测的预处理中，也使用 KNNImputer
    print("正在使用 KNNImputer 填充缺失值以便进行异常值检测...")
    X_imputed = KNNImputer(n_neighbors=5).fit_transform(df[features_to_check])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"异常值检测将使用设备: {device}")

    unsupervised_model = unsupervised.TabPFNUnsupervisedModel(
        tabpfn_clf=TabPFNClassifier(n_estimators=4, device=device),
        tabpfn_reg=TabPFNRegressor(n_estimators=4, device=device)
    )
    
    exp_outlier = unsupervised.experiments.OutlierDetectionUnsupervisedExperiment(task_type="unsupervised")
    try:
        print("正在运行异常值检测实验 (这可能需要一些时间)...")
        results = exp_outlier.run(
            tabpfn=unsupervised_model,
            X=torch.from_numpy(X_imputed).float(),
            attribute_names=features_to_check
        )
        
        if 'outlier_scores' in results and len(results['outlier_scores']) > 0:
            outlier_scores = results['outlier_scores']
            threshold = np.percentile(outlier_scores, 100 - (contamination * 100))
            is_inlier = outlier_scores >= threshold
            n_outliers = len(is_inlier) - np.sum(is_inlier)
            print(f"检测到并移除了 {n_outliers} 个异常值。")
            return df.iloc[is_inlier].copy()
        else:
            print("警告: 未能计算出异常值得分或结果为空，跳过清洗。")
            return df

    except Exception as e:
        print(f"异常值清洗过程中发生错误: {e}")
        traceback.print_exc()
        print("将返回原始数据框。")
        return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred, r2_train, r2_test, model_name_display, output_dir='.'):
    """为单个模型生成训练集和测试集的实际值 vs. 预测值 JointGrid 图。"""
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")
    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df]).dropna()
    if data_combined_df.empty:
        print(f"警告: 清洗后无有效数据可用于绘图。")
        return
    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette={'Train': '#b4d4e1', 'Test': '#f4ba8a'}, marginal_ticks=True)
    g.plot_joint(sns.scatterplot, alpha=0.6, s=60, edgecolor='w', linewidth=0.5)
    all_values = pd.concat([data_combined_df['True'], data_combined_df['Predicted']])
    min_val, max_val = all_values.min(), all_values.max()
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', lw=2)
    g.ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=g.ax_joint.transAxes, ha='right', fontsize=14)
    g.ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=g.ax_joint.transAxes, ha='right', fontsize=14)
    g.fig.suptitle(f'{model_name_display}: Actual vs. Predicted', fontsize=20, fontweight='bold', y=1.02)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{model_name_display}_JointGrid.png'), dpi=300, bbox_inches='tight')
    plt.close(g.fig)
    print(f"JointGrid 图已保存至 '{output_dir}' 目录。")


def train_and_evaluate_model(df, target_column_name):
    """为TabPFN模型进行训练、评估和可解释性分析。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 进行完整流程 ---")
    
    # [!!修改!!] 将版本后缀恢复为 v4，将会覆盖之前的文件
    version_suffix = "v4_tabpfn_dynamic_device"
    print(f"本次运行文件后缀: {version_suffix} (注意: 将覆盖同名旧文件)")

    # 1. 异常值清洗
    potential_targets = ['pH-S-CH', 'Cd-CH']
    features_for_outlier = [col for col in df.columns if col not in potential_targets]
    df_cleaned = clean_outliers_with_tabpfen(df.copy(), features_to_check=features_for_outlier, contamination=0.05)
    
    # 2. 数据准备 (对于输出列，直接删除缺失行)
    df_cleaned.dropna(subset=[target_column_name], inplace=True)
    if df_cleaned.shape[0] < 20: 
        print("数据过少，无法进行有效训练。"); return
    
    # 分层抽样
    strata_col_name = 'T_aging'
    strata = None
    if strata_col_name in df_cleaned.columns and all(df_cleaned[strata_col_name].value_counts() > 1):
        strata = df_cleaned[strata_col_name]
        print(f"\n将使用 '{strata_col_name}' 列进行分层抽样。")
    
    cols_to_exclude = potential_targets
    feature_columns = [col for col in df_cleaned.columns if col not in cols_to_exclude and col != target_column_name]
    print(f"最终用于训练的特征 ({len(feature_columns)}个): {feature_columns}")
    X = df_cleaned[feature_columns]
    y = df_cleaned[target_column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strata)
    
    # 3. 预处理器
    # 使用 KNNImputer 填充特征中的缺失值
    preprocessor = KNNImputer(n_neighbors=5)

    # 4. 模型实例化与训练
    if not tabpfn_available: 
        print("错误: TabPFNRegressor 不可用。"); return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n模型训练将使用设备: {device}")
    
    model_instance = TabPFNRegressor(device=device)
    
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    
    try:
        print(f"开始拟合TabPFN模型...")
        full_pipeline.fit(X_train, y_train)
        
        r2_train = full_pipeline.score(X_train, y_train)
        r2_test = full_pipeline.score(X_test, y_test)
        print(f"模型在训练集上的 R²: {r2_train:.4f}")
        print(f"模型在测试集上的 R²: {r2_test:.4f}")

        # 模型保存
        model_dir = f'./saved_models_{target_column_name}_{version_suffix}'
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, f'tabpfn_pipeline_{version_suffix}.joblib')
        joblib.dump(full_pipeline, model_filename)
        print(f"训练好的模型已保存至: {model_filename}")

        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)

        # 绘图
        plot_output_dir = f'./tabpfn_plots_{target_column_name}_{version_suffix}'
        model_name_display = f"TabPFN_{version_suffix}"
        plot_single_model_jointgrid(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, model_name_display, plot_output_dir)
        
        # 5. SHAP 分析
        if pfn_interpret_available:
            print("\n--- 开始进行SHAP分析 ---")
            X_test_transformed = full_pipeline.named_steps['preprocessor'].transform(X_test)
            fitted_model = full_pipeline.named_steps['regressor']
            
            shap_values = interpretability.shap.get_shap_values(
                estimator=fitted_model, test_x=X_test_transformed,
                attribute_names=feature_columns, algorithm="permutation"
            )
            
            shap_plot_dir = f'./shap_plots_{target_column_name}_{version_suffix}'
            os.makedirs(shap_plot_dir, exist_ok=True)
            
            # Summary Plot
            with plt.rc_context({'font.weight': 'normal'}):
                plt.figure()
                shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_columns, show=False, cmap=plt.get_cmap('plasma'))
                plt.savefig(os.path.join(shap_plot_dir, f'TabPFN_shap_summary_{version_suffix}.pdf'), bbox_inches='tight')
                plt.close()
                print("SHAP Summary Plot 已保存。")
            
            # Importance Bar Plot
            if hasattr(shap_values, 'values'): shap_values_array = shap_values.values
            else: shap_values_array = shap_values
            mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
            imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': mean_abs_shap})
            total_imp = imp_df['Importance'].sum()
            imp_df['Percentage'] = (imp_df['Importance'] / total_imp) * 100 if total_imp > 0 else 0
            imp_df.sort_values('Percentage', ascending=False, inplace=True)
            
            plt.figure(figsize=(10, 8))
            ax = sns.barplot(x='Percentage', y='Feature', data=imp_df, palette='plasma_r')
            for p in ax.patches:
                ax.text(p.get_width() + 0.1, p.get_y() + p.get_height() / 2., f'{p.get_width():.2f}%', va='center')
            plt.title('SHAP Feature Importance'); plt.xlabel('Importance (%)'); plt.tight_layout()
            plt.savefig(os.path.join(shap_plot_dir, f'TabPFN_shap_importance_bar_{version_suffix}.pdf'), bbox_inches='tight')
            plt.close()
            print("SHAP 特征重要性柱状图已保存。")

        # --- 6. 置换特征重要性 (PFI) 分析 ---
        print("\n--- 开始进行置换特征重要性分析 ---")
        pfi_results = permutation_importance(full_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
        pfi_df = pd.DataFrame({'Feature': X.columns, 'Importance': pfi_results.importances_mean}).sort_values('Importance', ascending=False)
        pfi_plot_dir = f'./pfi_plots_{target_column_name}_{version_suffix}'
        os.makedirs(pfi_plot_dir, exist_ok=True)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=pfi_df[pfi_df.Importance > 0], palette='plasma')
        plt.title('Permutation Feature Importance'); plt.xlabel("Importance (R² Drop)"); plt.tight_layout()
        plt.savefig(os.path.join(pfi_plot_dir, f'TabPFN_pfi_importance_bar_{version_suffix}.pdf'), bbox_inches='tight')
        plt.close()
        print("置换特征重要性图已保存。")

    except Exception as e:
        print(f"模型主处理流程失败: {e}"); traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
    # 请确保您的CSV文件列名已更新为最终的缩写形式
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
    df_original = load_and_process_data(file_path)

    if df_original is not None:
        if 'T_aging' not in df_original.columns:
            print(f"\n错误: 您的数据文件中缺少 'T_aging' 列，无法进行分层抽样。")
        else:
            target_column_options = ['pH-S-CH', 'Cd-CH']
            while True:
                print("\n请选择要训练的目标列:")
                [print(f"{i + 1}. {col_name}") for i, col_name in enumerate(target_column_options)]
                print("0. 退出程序")
                try:
                    choice = input(f"请输入选项编号 (0-{len(target_column_options)}): ")
                    if not choice.strip(): continue
                    choice = int(choice)
                    if choice == 0: 
                        print("程序退出。")
                        break
                    elif 1 <= choice <= len(target_column_options):
                        target = target_column_options[choice - 1]
                        train_and_evaluate_model(df_original.copy(), target)
                    else: 
                        print("无效选项。")
                except ValueError: 
                    print("无效输入，请输入一个数字。")
                except Exception as e: 
                    print(f"主程序中发生错误: {e}")
                    traceback.print_exc()
    else:
        print("数据加载失败。")