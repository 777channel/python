import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import traceback
import joblib # 确保 joblib 已导入
import shap

# --- 导入检查 ---
try:
    from tabpfn_extensions import unsupervised,interpretability
    from tabpfn import TabPFNClassifier,TabPFNRegressor # For unsupervised model
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    tabpfn_available = True
    pfn_interpret_available = True
    print("TabPFN 及相关扩展库导入成功。")
except ImportError:
    tabpfn_available = False
    pfn_interpret_available = False
    AutoTabPFNRegressor = None # Placeholder
    print("警告: 未找到 TabPFN 或 tabpfen_extensions 库。")

# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# --- 功能函数定义 ---

def load_and_process_data(file_path):
    # ... (此函数保持不变) ...
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取文件时发生错误: {e}"); return None
    cols_to_convert = ['BR', 'pH-S-CH', 'Cd-CH', 'T_aging']
    for col in cols_to_convert:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df.info()
    return df

def clean_outliers_with_tabpfn(df, features_to_check, contamination=0.05):
    """
    使用 TabPFNUnsupervisedModel 来检测并移除异常值。
    最终修正版：使用了正确的键名 'outlier_scores'。

    参数:
    df (pd.DataFrame): 原始数据框。
    features_to_check (list): 用于检测异常值的特征列名列表。
    contamination (float): 预期数据中的异常值比例，用于设定阈值。

    返回:
    pd.DataFrame: 清洗掉异常值后的新数据框。
    """
    print(f"开始异常值检测... 预计移除比例: {contamination:%}")
    
    if not features_to_check:
        print("警告: 未提供用于异常值检测的特征列，跳过此步骤。")
        return df

    # 先填充NaN，因为异常值检测也不能处理NaN
    X_imputed = SimpleImputer(strategy='median').fit_transform(df[features_to_check])
    
    # 初始化用于无监督任务的TabPFN模型
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
        
        # ## 核心修正: 使用正确的键名 'outlier_scores' ##
        if 'outlier_scores' in results:
            outlier_scores = results['outlier_scores']
        else:
            print("警告: 异常值检测结果中未找到 'outlier_scores'，跳过清洗。")
            return df

        if len(outlier_scores) > 0:
            threshold = np.percentile(outlier_scores, contamination * 100)
            is_inlier = outlier_scores >= threshold
            n_outliers = len(is_inlier) - np.sum(is_inlier)
            print(f"检测到并移除了 {n_outliers} 个异常值。")
            
            # 使用 .iloc 来确保索引正确对齐
            return df.iloc[is_inlier].copy()
        else:
            print("警告: 未能计算出异常值得分。")
            return df

    except Exception as e:
        print(f"异常值清洗过程中发生错误: {e}")
        traceback.print_exc()
        print("将返回原始数据框。")
        return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred, r2_train, r2_test, model_name_display, output_dir='.'):
    # ... (此函数保持不变) ...
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")
    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df]).dropna()
    if data_combined_df.empty: print(f"警告: 清洗后无有效数据可用于绘图。"); return
    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette={'Train': '#b4d4e1', 'Test': '#f4ba8a'}, marginal_ticks=True)
    g.plot_joint(sns.scatterplot, alpha=0.6, s=60, edgecolor='w', linewidth=0.5)
    all_values = pd.concat([data_combined_df['True'], data_combined_df['Predicted']])
    min_val, max_val = all_values.min(), all_values.max()
    g.ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', lw=2)
    g.ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=g.ax_joint.transAxes, ha='right')
    g.ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=g.ax_joint.transAxes, ha='right')
    g.fig.suptitle(f'{model_name_display}: Actual vs. Predicted', fontsize=20, fontweight='bold', y=1.02)
    os.makedirs(output_dir, exist_ok=True); plt.savefig(os.path.join(output_dir, f'{model_name_display}_JointGrid.png'), dpi=300, bbox_inches='tight'); plt.close(g.fig)
    print(f"JointGrid 图已保存。")

def train_and_evaluate_model(df, target_column_name):
    """为AutoTabPFN模型进行训练、评估和可解释性分析。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 进行完整流程 ---")
    
    # 1. 异常值清洗
    potential_targets = ['pH-S-CH', 'Cd-CH']
    features_for_outlier = [col for col in df.columns if col not in potential_targets]
    df_cleaned = clean_outliers_with_tabpfn(df.copy(), features_to_check=features_for_outlier, contamination=0.05)
    
    # 2. 数据准备与分层抽样
    df_cleaned.dropna(subset=[target_column_name], inplace=True)
    if df_cleaned.shape[0] < 20: print("数据过少，无法进行有效训练。"); return

    strata_col_name = 'T_aging'
    strata = None
    if strata_col_name in df_cleaned.columns and all(df_cleaned[strata_col_name].value_counts() > 1):
        strata = df_cleaned[strata_col_name]; print(f"\n将使用 '{strata_col_name}' 列进行分层抽样。")
    
    cols_to_exclude = potential_targets + [strata_col_name]
    feature_columns = [col for col in df_cleaned.columns if col not in cols_to_exclude and col != target_column_name]
    X = df_cleaned[feature_columns]; y = df_cleaned[target_column_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strata)
    
    # 3. 预处理器（极简版，只做缺失值填充，如果后期需要对数化，这里需要扩展）
    # 如果您的"对数化"是指在X_test_transformed之外的额外操作，那么这里需要添加LogTransformer
    # 目前只包含SimpleImputer，所以shap_values.data将反映经过imputation但未Log转换的X_test
    # 如果您的数据已经包含对数化的列，那么这些列的原始值就是对数化的
    preprocessor = SimpleImputer(strategy='median')

    # 4. 模型实例化与训练
    if not tabpfn_available: print("错误: AutoTabPFNRegressor 不可用。"); return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = AutoTabPFNRegressor(max_time=120, device=device)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    
    try:
        print(f"开始拟合AutoTabPFN模型 (最多花费 {model_instance.max_time} 秒)...")
        full_pipeline.fit(X_train, y_train)
        
        r2_train = full_pipeline.score(X_train, y_train)
        r2_test = full_pipeline.score(X_test, y_test)
        print(f"模型在训练集上的 R²: {r2_train:.4f}")
        print(f"模型在测试集上的 R²: {r2_test:.4f}")

        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)
        plot_single_model_jointgrid(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, "AutoTabPFN", f'./autotabpfn_plots_{target_column_name}')
        
        # --- 5. SHAP 分析 (使用官方解释器) ---
        if pfn_interpret_available:
            print("\n--- 开始进行SHAP分析 ---")
            print(f"警告：将使用全部 {X_test.shape[0]} 条测试数据进行SHAP分析，这可能非常耗时。")
            
            # --- 关键修改点：获取SHAP值时传递原始X_test和预处理器 ---
            fitted_regressor = full_pipeline.named_steps['regressor']
            fitted_preprocessor = full_pipeline.named_steps['preprocessor']

            # 注意：tabpfn_extensions.interpretability.shap.get_shap_values
            # 内部会处理 preprocessor，将 X_test 转换为模型可接受的格式
            shap_values = interpretability.shap.get_shap_values(
                estimator=fitted_regressor,
                test_x=X_test, # 传入原始的X_test
                attribute_names=feature_columns,
                algorithm="permutation",
                preprocessor=fitted_preprocessor # 传入预处理器
            )
            
            # --- 保存 SHAP 值和原始特征值 ---
            shap_data_dir = f'./shap_data_{target_column_name}'
            os.makedirs(shap_data_dir, exist_ok=True)
            
            # 保存完整的 shap.Explanation 对象 (包含 .values, .data (原始特征), .base_values 等)
            shap_output_path = os.path.join(shap_data_dir, 'AutoTabPFN_shap_explanation.pkl')
            try:
                joblib.dump(shap_values, shap_output_path)
                print(f"完整的 SHAP Explanation 对象已保存为 joblib 文件: {shap_output_path}")
                
                # 为了方便，也可以单独保存 shap_values.values 和 shap_values.data
                # shap_values.values 是 SHAP 值数组
                np.save(os.path.join(shap_data_dir, 'AutoTabPFN_shap_values_array.npy'), shap_values.values)
                print(f"SHAP values 数组已保存为 NumPy 文件: {shap_data_dir}/AutoTabPFN_shap_values_array.npy")

                # shap_values.data 是用于计算SHAP值的原始特征数据 (经过预处理器但未Log转换的X_test)
                # 如果preprocessor没有Log转换，这里就是原始特征，如果preprocessor有Log转换，这里就是Log转换后的数据。
                # 由于您的preprocessor目前只有SimpleImputer，所以这里保存的是经过Imputation的原始尺度特征。
                pd.DataFrame(shap_values.data, columns=feature_columns).to_csv(
                    os.path.join(shap_data_dir, 'AutoTabPFN_original_features_for_shap.csv'), index=False
                )
                print(f"原始特征数据 (用于SHAP计算) 已保存为 CSV 文件: {shap_data_dir}/AutoTabPFN_original_features_for_shap.csv")

            except Exception as save_e:
                print(f"保存 SHAP Explanation 或原始特征数据时发生错误: {save_e}")
                traceback.print_exc()

            shap_plot_dir = f'./shap_plots_{target_column_name}'; os.makedirs(shap_plot_dir, exist_ok=True)
            
            # Summary Plot
            with plt.rc_context({'font.weight': 'normal'}): # SHAP图字体不加粗可能更清晰
                plt.figure(); shap.summary_plot(shap_values, X_test, feature_names=feature_columns, show=False, cmap=plt.get_cmap('plasma')); # 使用 X_test
                plt.savefig(os.path.join(shap_plot_dir, 'AutoTabPFN_shap_summary.pdf'), bbox_inches='tight'); plt.close()
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
            plt.savefig(os.path.join(shap_plot_dir, 'AutoTabPFN_shap_importance_bar.pdf'), bbox_inches='tight'); plt.close()
            print("SHAP 特征重要性柱状图已保存。")

            # --- SHAP Dependency Plots (可选，但为了示例，我将添加一个) ---
            print("\n--- 生成SHAP依赖图示例 (针对最重要的特征) ---")
            if not imp_df.empty:
                # 选取最重要的特征来绘制依赖图
                main_feature = imp_df.iloc[0]['Feature']
                plt.figure(figsize=(10, 6))
                # 注意：这里传递的是 shap_values 和原始 X_test。
                # shap.dependence_plot 能够从 shap_values 中提取 SHAP 值，
                # 并从 X_test 中获取特征的原始值，进行正确绘图。
                shap.dependence_plot(main_feature, shap_values.values, X_test, feature_names=feature_columns, show=False)
                plt.title(f'SHAP Dependence Plot for {main_feature}'); plt.tight_layout()
                plt.savefig(os.path.join(shap_plot_dir, f'AutoTabPFN_shap_dependence_{main_feature}.pdf'), bbox_inches='tight'); plt.close()
                print(f"SHAP 依赖图 ({main_feature}) 已保存。")
            else:
                print("特征重要性数据为空，跳过依赖图生成。")

        # --- 6. 置换特征重要性 (PFI) 分析 ---
        print("\n--- 开始进行置换特征重要性分析 ---")
        pfi_results = permutation_importance(full_pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
        pfi_df = pd.DataFrame({'Feature': X.columns, 'Importance': pfi_results.importances_mean}).sort_values('Importance', ascending=False)
        pfi_plot_dir = f'./pfi_plots_{target_column_name}'; os.makedirs(pfi_plot_dir, exist_ok=True)
        plt.figure(figsize=(10, 8)); sns.barplot(x='Importance', y='Feature', data=pfi_df[pfi_df.Importance > 0], palette='plasma')
        plt.title('Permutation Feature Importance'); plt.xlabel("Importance (R² Drop)"); plt.tight_layout()
        plt.savefig(os.path.join(pfi_plot_dir, 'AutoTabPFN_pfi_importance_bar.pdf'), bbox_inches='tight'); plt.close()
        print("置换特征重要性图已保存。")

    except Exception as e:
        print(f"模型主处理流程失败: {e}"); traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
    df_original = load_and_process_data(file_path)

    if df_original is not None:
        if 'T_aging' not in df_original.columns:
            print(f"\n错误: 您的数据文件中缺少 'T_age' 列，无法进行分层抽样。")
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
                    if choice == 0: print("程序退出。"); break
                    elif 1 <= choice <= len(target_column_options):
                        target = target_column_options[choice - 1]
                        train_and_evaluate_model(df_original.copy(), target)
                    else: print("无效选项。")
                except ValueError: print("无效输入，请输入一个数字。")
                except Exception as e: print(f"主程序中发生错误: {e}"); traceback.print_exc()
    else:
        print("数据加载失败。")