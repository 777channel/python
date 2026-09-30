import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import torch 
import joblib 
import shap   # 标准SHAP库

# 尝试导入TabPFN的特定解释性工具 (来自用户示例)
try:
    from tabpfn_extensions import interpretability
    pfn_interpret_available = True
    print("tabpfn_extensions.interpretability 模块导入成功。")
except ImportError:
    pfn_interpret_available = False
    interpretability = None # Placeholder
    print("警告: 未找到 tabpfn_extensions.interpretability 模块。")
    print("TabPFN特定的SHAP值计算方法 (interpretability.shap.get_shap_values) 将不可用。")
    print("请确保 tabpfn_extensions 已正确安装。")

# 设置全局绘图参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 28 
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

# Standard TabPFN Regressor
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 从 'tabpfn' 库导入成功。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: 标准 TabPFN库 (tabpfn) 未找到。TabPFN 模型将不可用。")
    print("请确保已安装 'tabpfn' 库: pip install tabpfn")


def load_and_process_data(file_path):
    """加载数据，对特定列进行数值转换和初步检查。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"数据从 '{file_path}' (UTF-8编码) 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None
    
    br_column_name = 'BR' 
    if br_column_name in df.columns:
        df[br_column_name] = pd.to_numeric(df[br_column_name], errors='coerce')
    else: print(f"警告: 特征列 '{br_column_name}' 未在数据中找到。")
    target_cols_to_convert = ['pH-S-CH', 'Cd-CH']
    for col_name in target_cols_to_convert:
        if col_name in df.columns: df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    df.info(); return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                                r2_train, r2_test, model_name_display, output_dir='.'):
    """为单个模型创建实际值 vs. 预测值的 JointGrid 对比图。"""
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")
    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df])
    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'} 
    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette=palette, marginal_ticks=True)
    g.plot_joint(sns.scatterplot, alpha=0.6, s=60, edgecolor='w', linewidth=0.5)
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Train'], label='Train Regression Line', line_kws={'linestyle':'--','linewidth':2})
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Test'], label='Test Regression Line', line_kws={'linestyle':':','linewidth':2})
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.7)
    ax_joint = g.ax_joint 
    for spine in ax_joint.spines.values(): spine.set_linewidth(1.5)
    g.ax_marg_x.spines['bottom'].set_linewidth(1.5); g.ax_marg_y.spines['left'].set_linewidth(1.5)
    for label in ax_joint.get_xticklabels() + ax_joint.get_yticklabels(): label.set_fontweight('bold'); label.set_fontsize(12)
    for label in g.ax_marg_x.get_xticklabels() + g.ax_marg_x.get_yticklabels(): label.set_fontweight('bold'); label.set_fontsize(10)
    for label in g.ax_marg_y.get_xticklabels() + g.ax_marg_y.get_yticklabels(): label.set_fontweight('bold'); label.set_fontsize(10)
    ax_joint.set_xlabel("Actual Values", fontsize=16, fontweight='bold')
    ax_joint.set_ylabel("Predicted Values", fontsize=16, fontweight='bold')
    ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=ax_joint.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor=palette['Train'], fc="white", alpha=0.8))
    ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=ax_joint.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor=palette['Test'], fc="white", alpha=0.8))
    ax_joint.text(0.95, 0.95, f'Model: {model_name_display}', transform=ax_joint.transAxes, fontsize=12, fontweight='bold', va='top', ha='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", fc="white", alpha=0.8))
    min_val_data = 0; max_val_data = 1
    if not data_combined_df.empty:
        vals = pd.concat([data_combined_df['True'], data_combined_df['Predicted']]).dropna()
        if not vals.empty: min_val_data, max_val_data = vals.min(), vals.max()
    ax_joint.plot([min_val_data, max_val_data], [min_val_data, max_val_data], c="black", alpha=0.5, linestyle='--', label='y=x (Ideal)', lw=2)
    handles, labels = ax_joint.get_legend_handles_labels(); unique_labels_dict = {}; final_handles, final_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels_dict: unique_labels_dict[label] = handle; final_handles.append(handle); final_labels.append(label)
    if final_handles: ax_joint.legend(final_handles, final_labels, fontsize=12, loc='upper left', frameon=True, facecolor='white', framealpha=0.7)
    g.fig.suptitle(f'{model_name_display}: Actual vs. Predicted Values', fontsize=20, fontweight='bold', y=1.02)
    g.fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    save_path_jointgrid = os.path.join(output_dir, f'{model_name_display}_JointGrid.png')
    plt.savefig(save_path_jointgrid, dpi=300, bbox_inches='tight'); print(f"JointGrid 图已保存至 {save_path_jointgrid}"); plt.show(); plt.close(g.fig)

def train_and_evaluate_models(df, target_column_name):
    """为标准TabPFN模型进行训练、评估、保存、特征重要性分析和绘图。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练标准 TabPFN 模型 ---")
    df_processed = df.copy()

    if target_column_name not in df_processed.columns: print(f"错误: 目标列 '{target_column_name}' 不存在。"); return None
    if not pd.api.types.is_numeric_dtype(df_processed[target_column_name]):
        df_processed[target_column_name] = pd.to_numeric(df_processed[target_column_name], errors='coerce')
        if not pd.api.types.is_numeric_dtype(df_processed[target_column_name]): print(f"错误: 目标列 '{target_column_name}' 无法转为数值。"); return None
    if df_processed[target_column_name].isnull().sum() > 0: df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.shape[0] < 10: print(f"数据过少。"); return None
    
    potential_targets = ['pH-S-CH', 'Cd-CH']
    feature_columns = [col for col in df_processed.columns if col not in potential_targets and col != target_column_name and col in df_processed.columns]
    if not feature_columns: print(f"错误: 无特征列。"); return None
    
    X = df_processed[feature_columns]; y = df_processed[target_column_name]
    if X.empty: print(f"错误: 特征集X为空。"); return None
    if len(y) < 10 : print(f"数据量过少 ({len(y)}条)。"); return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    if X_train.empty or X_test.empty or len(X_train) < 5 : print("训练/测试集数据不足。"); return None
    print(f"训练集规模: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    log_transform_cols = ['Cd-C', 'BET']; log_transform_cols = [col for col in log_transform_cols if col in X_train.columns] 
    cec_col_list = ['CEC'] if 'CEC' in X_train.columns else []
    other_numeric_cols = [col for col in X_train.columns if col not in log_transform_cols + cec_col_list]
    log_pipeline = Pipeline(steps=[('log1p', FunctionTransformer(np.log1p, validate=False)), ('scaler_log', StandardScaler())])
    cec_pipeline = Pipeline(steps=[('knn_imputer', KNNImputer()), ('scaler_cec', StandardScaler())])
    other_numeric_pipeline = Pipeline(steps=[('imputer_median', SimpleImputer()), ('scaler_numeric', StandardScaler())])
    transformers_list = []
    if log_transform_cols: transformers_list.append(('log_features', log_pipeline, log_transform_cols))
    if cec_col_list: transformers_list.append(('cec_feature', cec_pipeline, cec_col_list))
    if other_numeric_cols: transformers_list.append(('other_numeric_features', other_numeric_pipeline, other_numeric_cols))
    if not transformers_list:
        if X_train.shape[1] > 0: preprocessor = Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]); print("警告：回退到通用预处理。")
        else: print("错误：无特征列也无特定转换器。"); return None
    else: preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop')

    model_name_for_log = "TabPFN Regressor" 
    model_name_for_plot = "TabPFN" 
    model_instance = None

    if tabpfn_available and TabPFNRegressor is not None:
        if X_train.shape[0] <= 1024 and X_train.shape[1] < 100: 
            print(f"信息: 配置标准 TabPFNRegressor (使用默认参数)。")
            model_instance = TabPFNRegressor() 
        else:
            print(f"信息: 数据集大小不适合 TabPFN。"); return None
    else: 
        print("错误: TabPFNRegressor 不可用。"); return None
            
    results_summary_list = [] 
    print(f"\n--- 正在处理模型: {model_name_for_log} ---")
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    current_model_eval_metrics = {}

    try:
        print(f"开始对 {model_name_for_log} 进行5折交叉验证...")
        cv_results = cross_validate(full_pipeline, X_train, y_train, cv=5, scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'], n_jobs=1, return_train_score=False) 
        mean_cv_r2 = np.mean(cv_results['test_r2']); mean_cv_rmse = -np.mean(cv_results['test_neg_root_mean_squared_error']); mean_cv_mae = -np.mean(cv_results['test_neg_mean_absolute_error'])   
        current_model_eval_metrics.update({'CV R2 Mean': mean_cv_r2, 'CV RMSE Mean': mean_cv_rmse, 'CV MAE Mean': mean_cv_mae})
        print(f"交叉验证 - 平均R²: {mean_cv_r2:.4f}, RMSE: {mean_cv_rmse:.4f}, MAE: {mean_cv_mae:.4f}")

        print(f"开始在完整训练集上拟合 {model_name_for_log}...")
        full_pipeline.fit(X_train, y_train)
        
        model_save_dir = f'./saved_models_{target_column_name}'; os.makedirs(model_save_dir, exist_ok=True)
        model_filename = os.path.join(model_save_dir, f'{model_name_for_plot.replace(" ", "_")}_pipeline.joblib')
        joblib.dump(full_pipeline, model_filename); print(f"模型已保存至: {model_filename}")

        y_pred_train = full_pipeline.predict(X_train); r2_train = r2_score(y_train, y_pred_train); rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train)); mae_train = mean_absolute_error(y_train, y_pred_train)
        current_model_eval_metrics.update({'Train R2': r2_train, 'Train RMSE': rmse_train, 'Train MAE': mae_train})
        print(f"训练集   - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
        y_pred_test = full_pipeline.predict(X_test); r2_test = r2_score(y_test, y_pred_test); rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test)); mae_test = mean_absolute_error(y_test, y_pred_test)
        current_model_eval_metrics.update({'Test R2': r2_test, 'Test RMSE': rmse_test, 'Test MAE': mae_test})
        print(f"测试集   - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

        plot_output_dir_joint = f'./jointgrid_plots_{target_column_name}'
        plot_single_model_jointgrid(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, 
                                    model_name_for_plot, output_dir=plot_output_dir_joint)
        
        # --- SHAP 分析与特征重要性 ---
        print("\n开始进行SHAP分析和特征重要性提取...")
        original_feature_names = X_train.columns.tolist()

        try:
            # 检查 pfn_interpret_available 和 interpretability.shap.get_shap_values 是否存在
            if not (pfn_interpret_available and hasattr(interpretability, 'shap') and \
                    hasattr(interpretability.shap, 'get_shap_values')):
                print("警告: TabPFN 特定的 SHAP 工具 (interpretability.shap.get_shap_values) 不可用。")
                print("将尝试使用标准的 shap.Explainer (这对于TabPFN可能非常慢或不理想)。")
                
                if X_train.empty:
                     print("警告: X_train 为空，无法创建标准 SHAP Explainer。")
                     raise ValueError("X_train is empty, cannot proceed with SHAP.")

                explainer_background_data = shap.sample(X_train, min(100, X_train.shape[0]), random_state=42) 
                print("创建标准 SHAP Explainer (解释整个pipeline)...")
                def pipeline_predict_for_shap_original_features(data_df): return full_pipeline.predict(data_df)
                explainer = shap.Explainer(pipeline_predict_for_shap_original_features, explainer_background_data)
                
                n_shap_test_samples = min(50, X_test.shape[0])
                print(f"计算测试集 (前 {n_shap_test_samples} 个样本) 的SHAP值 (使用标准Explainer)...")
                if n_shap_test_samples > 0:
                    X_test_sample_for_shap = X_test.iloc[:n_shap_test_samples]
                    shap_values_output = explainer(X_test_sample_for_shap) # 返回 Explanation 对象
                else:
                    print("测试集样本过少，跳过SHAP值计算。"); raise ValueError("Test set too small for SHAP.")
            else: # 使用 interpretability.shap.get_shap_values
                n_shap_samples = min(50, X_test.shape[0])
                if n_shap_samples < 1: print("警告: 测试集样本过少，无法计算SHAP值。"); raise ValueError("Test set too small for SHAP.")
                
                X_test_sample_df_for_shap = X_test.iloc[:n_shap_samples].copy()
                print(f"使用 tabpfn_extensions.interpretability.shap.get_shap_values 计算 SHAP 值 (前 {n_shap_samples} 个测试样本)...")
                shap_values_output = interpretability.shap.get_shap_values(
                    estimator=full_pipeline, 
                    test_x=X_test_sample_df_for_shap, 
                    attribute_names=original_feature_names,
                    algorithm="permutation" 
                )
            
            # --- 后续SHAP值处理和绘图 (对两种方式获取的shap_values_output进行统一处理) ---
            shap_values_for_plot = None
            if hasattr(shap_values_output, 'values') and isinstance(shap_values_output.values, np.ndarray):
                shap_values_for_plot = shap_values_output.values
                # 如果是Explanation对象，它通常也包含 .data 和 .feature_names，summary_plot可以直接用整个对象
                # X_test_for_summary = shap_values_output # 可以直接传递 Explanation 对象
            elif isinstance(shap_values_output, np.ndarray):
                shap_values_for_plot = shap_values_output
                # X_test_for_summary = X_test_sample_df_for_shap # 需要原始特征值
            else:
                print(f"警告: SHAP值返回类型未知 ({type(shap_values_output)})，可能无法用于绘图。")

            if shap_values_for_plot is not None:
                # 确保维度和特征名数量匹配
                # (shap_values_for_plot.shape[1] == len(original_feature_names)) 检查已在前面加入

                print("生成标准 SHAP Summary Plot...")
                shap_plot_output_dir = f'./shap_plots_{target_column_name}'; os.makedirs(shap_plot_output_dir, exist_ok=True)
                shap_summary_filename = os.path.join(shap_plot_output_dir, f'{model_name_for_plot}_shap_summary.png')
                plt.figure() 
                with plt.rc_context({'font.size': 10, 'font.weight': 'normal'}): 
                    # 如果shap_values_output是Explanation对象，可以直接传递
                    # 如果是numpy array，则需要传递特征值X_test_sample_df_for_shap
                    if hasattr(shap_values_output, 'values'): #  shap.Explanation object
                        shap.summary_plot(shap_values_output, features=X_test_sample_df_for_shap, feature_names=original_feature_names, show=False, plot_size=None)
                    else: #  numpy array
                        shap.summary_plot(shap_values_for_plot, X_test_sample_df_for_shap, feature_names=original_feature_names, show=False, plot_size=None)
                    plt.title(f'SHAP Summary for {model_name_for_plot} on {target_column_name}', fontsize=14) 
                plt.savefig(shap_summary_filename, dpi=150, bbox_inches='tight')
                print(f"SHAP summary plot 已保存至: {shap_summary_filename}"); plt.show(); plt.close()

                feature_importances_shap = np.abs(shap_values_for_plot).mean(axis=0)
                if len(original_feature_names) == len(feature_importances_shap):
                    feature_importance_df = pd.DataFrame({'Feature': original_feature_names, 'Importance (Mean Abs SHAP)': feature_importances_shap}).sort_values(by='Importance (Mean Abs SHAP)', ascending=False)
                    feature_imp_filename = os.path.join(model_save_dir, f'{model_name_for_plot}_feature_importances_shap.csv')
                    feature_importance_df.to_csv(feature_imp_filename, index=False); print(f"基于SHAP的特征重要性已保存至: {feature_imp_filename}")
                    importance_plot_output_dir = f'./feature_importance_plots_{target_column_name}'; os.makedirs(importance_plot_output_dir, exist_ok=True)
                    importance_plot_filename = os.path.join(importance_plot_output_dir, f'{model_name_for_plot}_feature_importance_bar.png')
                    plt.figure(figsize=(12, max(6, len(original_feature_names) * 0.45))); num_features_to_plot = min(20, len(feature_importance_df))
                    sns.barplot(x='Importance (Mean Abs SHAP)', y='Feature', data=feature_importance_df.head(num_features_to_plot), palette='viridis_r')
                    plt.title(f'Top {num_features_to_plot} Feature Importance for {model_name_for_plot}\n(Target: {target_column_name}, Mean Absolute SHAP)')
                    plt.xlabel('Mean Absolute SHAP Value'); plt.ylabel('Feature')
                    plt.gca().tick_params(axis='y', labelsize=plt.rcParams['font.size']*0.5 if plt.rcParams['font.size'] > 15 else 10)
                    plt.gca().tick_params(axis='x', labelsize=plt.rcParams['font.size']*0.5 if plt.rcParams['font.size'] > 15 else 10)
                    plt.tight_layout(); plt.savefig(importance_plot_filename, dpi=150, bbox_inches='tight')
                    print(f"特征重要性条形图已保存至: {importance_plot_filename}"); plt.show(); plt.close()
                else: print("警告: SHAP特征名与重要性值数量不匹配。")
            else: print("警告:未能生成有效的SHAP数值数组用于绘图。")
        except ImportError: print("SHAP 分析因 tabpfn_extensions.interpretability 模块导入失败而被跳过（回退也失败）。")
        except Exception as e_shap_outer: print(f"SHAP分析或特征重要性流程失败: {e_shap_outer}"); import traceback; traceback.print_exc()
        
    except Exception as e_main: 
        print(f"模型 {model_name_for_log} 主处理流程失败: {e_main}"); import traceback; traceback.print_exc() 
        for metric_key in ['CV R2 Mean','CV RMSE Mean','CV MAE Mean','Train R2','Train RMSE','Train MAE','Test R2','Test RMSE','Test MAE']:
            current_model_eval_metrics[metric_key] = 'Error'
        current_model_eval_metrics['Error_Info'] = str(e_main)
    
    results_summary_list.append({'Model_Full_Name': model_name_for_log, **current_model_eval_metrics})
    print("\n--- 模型评估完成 ---"); 
    if results_summary_list:
        temp_results_for_df = {item.pop('Model_Full_Name'): item for item in results_summary_list}
        results_df = pd.DataFrame(temp_results_for_df).T
        cols_order = ['CV R2 Mean','CV RMSE Mean','CV MAE Mean','Train R2','Train RMSE','Train MAE','Test R2','Test RMSE','Test MAE','Error_Info']
        present_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[present_cols]
        print("\n评估结果总结:"); pd.set_option('display.float_format', lambda x: f'{x:.4f}' if isinstance(x, float) else x); print(results_df)
        return results_df
    else: print("模型未能成功完成评估。"); return None

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
    df_original = load_and_process_data(file_path) 
    if df_original is not None:
        target_column_options = ['pH-S-CH', 'Cd-CH'] 
        while True:
            print("\n请选择要训练的目标列:"); [print(f"{i+1}. {col_name}") for i, col_name in enumerate(target_column_options)]; print("0. 退出程序")
            try:
                user_choice_str = input(f"请输入选项编号 (0-{len(target_column_options)}): ")
                if not user_choice_str.strip(): print("未输入选项，请重新输入。"); continue
                user_choice = int(user_choice_str)
                if user_choice == 0: print("程序退出。"); break
                elif 1 <= user_choice <= len(target_column_options):
                    selected_target_col = target_column_options[user_choice - 1]
                    print(f"\n您已选择训练目标列: {selected_target_col}")
                    if selected_target_col not in df_original.columns: print(f"错误: 选择的目标列 '{selected_target_col}' 不在数据文件中。"); continue 
                    train_and_evaluate_models(df_original.copy(), selected_target_col) 
                else: print("无效选项，请输入列表中的有效编号。")
            except ValueError: print("无效输入，请输入一个数字编号。")
            except Exception as e: print(f"主程序循环中发生了一个未预料的错误: {e}"); import traceback; traceback.print_exc()
    else: print("数据加载和初步处理失败，无法执行训练。")