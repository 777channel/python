import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.linear_model import LinearRegression, Ridge # Not used if only AutoTabPFN
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor # Not used
# from sklearn.svm import SVR # Not used
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# import xgboost as xgb # Not used
# import lightgbm as lgb # Not used
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import torch # <--- 确保导入 torch 用于 CUDA 检测
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 28
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
# AutoTabPFN Regressor from tabpfn_extensions
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    autotabpfn_available = True
    print("AutoTabPFNRegressor 从 'tabpfn_extensions' 库导入成功。")
except ImportError:
    autotabpfn_available = False
    AutoTabPFNRegressor = None
    print("警告: 'tabpfn_extensions' 库未找到或 AutoTabPFNRegressor 无法导入。AutoTabPFN 模型将不可用。")
    print("请确保已安装: pip install tabpfn[full_extensions] 或者 pip install tabpfn-extensions")


def load_and_process_data(file_path):
    """加载数据，对特定列进行数值转换和初步检查。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"数据从 '{file_path}' (UTF-8编码) 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}")
        return None
    
    br_column_name = 'BR' 
    if br_column_name in df.columns:
        original_br_type = df[br_column_name].dtype
        nan_before_br = df[br_column_name].isnull().sum()
        df[br_column_name] = pd.to_numeric(df[br_column_name], errors='coerce')
        nan_after_br = df[br_column_name].isnull().sum()
        print(f"\n特征列 '{br_column_name}' 已尝试从 {original_br_type} 转换为数值类型。")
        if nan_after_br > nan_before_br:
            print(f"  转换过程中 '{br_column_name}' 列新增了 {nan_after_br - nan_before_br} 个缺失值 (NaN)。")
    else:
        print(f"警告: 特征列 '{br_column_name}' 未在数据中找到。")

    target_cols_to_convert = ['pH-S-CH', 'Cd-CH']
    print("\n尝试将指定的目标列转换为数值类型：")
    for col_name in target_cols_to_convert:
        if col_name in df.columns:
            original_type = df[col_name].dtype
            print(f" - 目标列 '{col_name}' (原始类型: {original_type}):")
            nan_before = df[col_name].isnull().sum()
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            nan_after = df[col_name].isnull().sum()
            if pd.api.types.is_numeric_dtype(df[col_name]):
                 print(f"   成功转换为数值类型 ({df[col_name].dtype}).")
            else:
                 print(f"   未能完全转换为标准数值类型，当前类型: {df[col_name].dtype}.")

            if nan_after > nan_before:
                print(f"   转换过程中 '{col_name}' 列新增了 {nan_after - nan_before} 个缺失值 (NaN)。")
            print(f"   '{col_name}' 当前缺失值总数: {nan_after}")
        else:
            print(f" - 警告: 目标列 '{col_name}' 未在数据中找到，无法转换。")
    
    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred, 
                                r2_train, r2_test, model_name_display, target_col_name, output_dir='.'):
    """
    为单个模型创建实际值 vs. 预测值的 JointGrid 对比图 (已更新样式)。
    """
    print(f"\n正在为模型 '{model_name_display}' 和目标 '{target_col_name}' 生成 JointGrid 图...")

    data_train_df = pd.DataFrame({
        'True': y_train_true,
        'Predicted': y_train_pred,
        'Data Set': 'Train'
    })
    data_test_df = pd.DataFrame({
        'True': y_test_true,
        'Predicted': y_test_pred,
        'Data Set': 'Test'
    })
    data_combined_df = pd.concat([data_train_df, data_test_df])

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'} 

    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", 
                      height=10, palette=palette, marginal_ticks=True)

    g.plot_joint(sns.scatterplot, alpha=0.6, s=60, edgecolor='w', linewidth=0.5)
    
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, 
                color=palette['Train'], label='Train Regression Line', line_kws={'linestyle':'--','linewidth':2})
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, 
                color=palette['Test'], label='Test Regression Line', line_kws={'linestyle':':','linewidth':2})
    
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.7)

    ax_joint = g.ax_joint 
    ax_marg_x = g.ax_marg_x
    ax_marg_y = g.ax_marg_y

    # --- 样式修改 ---
    # 1. 坐标轴线加粗 (spines)
    for spine in ax_joint.spines.values():
        spine.set_linewidth(1.5)
    # 边缘图的坐标轴线通常只显示部分，这里也尝试加粗主要的外边框
    ax_marg_x.spines['bottom'].set_linewidth(1.5)
    ax_marg_y.spines['left'].set_linewidth(1.5)
    # 通常边缘图的其他spines是隐藏的，如果需要显示并加粗，可以取消注释下面两行
    # for spine_direction in ['top', 'right']: # 隐藏marg_x的上和右边框
    #     ax_marg_x.spines[spine_direction].set_visible(False)
    # for spine_direction in ['bottom', 'right']: # 隐藏marg_y的下和右边框
    #    ax_marg_y.spines[spine_direction].set_visible(False)


    # 2. 坐标轴刻度标签加粗 (tick labels)
    for label in ax_joint.get_xticklabels() + ax_joint.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(12) # 保持与您示例一致的字体大小或稍作调整
    for label in ax_marg_x.get_xticklabels() + ax_marg_x.get_yticklabels(): # Y ticks for marg_x are counts
        label.set_fontweight('bold')
        label.set_fontsize(10)
    for label in ax_marg_y.get_xticklabels() + ax_marg_y.get_yticklabels(): # X ticks for marg_y are counts
        label.set_fontweight('bold')
        label.set_fontsize(10)
    
    # 3. 坐标轴标题加粗和英文 (axis labels)
    ax_joint.set_xlabel("Actual Values", fontsize=16, fontweight='bold')
    ax_joint.set_ylabel("Predicted Values", fontsize=16, fontweight='bold')


    # 4. 图中文字全英文并调整位置
    # R2 文本
    ax_joint.text(0.95, 0.15, f'Train $R^2$ = {r2_train:.3f}', 
            transform=ax_joint.transAxes, fontsize=12, fontweight='bold', # 使用您示例的字体大小
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=palette['Train'], facecolor="white", alpha=0.8))
    ax_joint.text(0.95, 0.05, f'Test $R^2$ = {r2_test:.3f}', 
            transform=ax_joint.transAxes, fontsize=12, fontweight='bold', # 使用您示例的字体大小
            verticalalignment='bottom', horizontalalignment='right', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor=palette['Test'], facecolor="white", alpha=0.8))
    
    # 模型名称文本移到右上角
    ax_joint.text(0.95, 1, f'Model: {model_name_display}', # 英文 "Model:"
            transform=ax_joint.transAxes, fontsize=12, fontweight='bold', # 使用您示例的字体大小
            verticalalignment='top', horizontalalignment='right', # 改为右上角对齐
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.8))

    min_val_data = 0; max_val_data = 1 # Default values
    if not data_combined_df['True'].empty: min_val_data = data_combined_df['True'].min()
    if not data_combined_df['True'].empty: max_val_data = data_combined_df['True'].max()
    if not data_combined_df['Predicted'].empty: 
        min_val_data = min(min_val_data, data_combined_df['Predicted'].min())
        max_val_data = max(max_val_data, data_combined_df['Predicted'].max())
    
    ax_joint.plot([min_val_data, max_val_data], [min_val_data, max_val_data], 
                  c="black", alpha=0.5, linestyle='--', label='y=x (Ideal)', lw=2) # 英文标签
    
    # 图例 (只包含回归线和 y=x 线)
    handles, labels = ax_joint.get_legend_handles_labels()
    unique_labels_dict = {}
    final_handles, final_labels = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels_dict: 
            unique_labels_dict[label] = handle
            final_handles.append(handle)
            final_labels.append(label)
    if final_handles: 
        ax_joint.legend(final_handles, final_labels, fontsize=12, loc='upper left', frameon=True, facecolor='white', framealpha=0.7)
    # --- 样式修改结束 ---

    g.fig.suptitle(f'{model_name_display} on {target_col_name}: Actual vs. Predicted', fontsize=20, fontweight='bold', y=1.02)
    g.fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col_name}_{model_name_display}_JointGrid.pdf')
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"JointGrid 图已保存至 {save_path}")
    plt.show()
    plt.close(g.fig)

def train_and_evaluate_models(df, target_column_name, 
                              autotabpfn_max_time=120, user_preferred_device='cuda'): # Renamed to user_preferred_device
    """为AutoTabPFN模型进行训练、评估和绘图，并进行CUDA检测。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练 AutoTabPFN 模型 ---")
    df_processed = df.copy()

    # ... (数据有效性检查和预处理，与您脚本中一致) ...
    if target_column_name not in df_processed.columns:
        print(f"错误: 目标列 '{target_column_name}' 在提供的DataFrame中不存在。")
        return None
    if not pd.api.types.is_numeric_dtype(df_processed[target_column_name]):
        print(f"错误: 目标列 '{target_column_name}' 不是数值类型 (当前类型: {df_processed[target_column_name].dtype})。请确保它已正确转换。")
        df_processed[target_column_name] = pd.to_numeric(df_processed[target_column_name], errors='coerce')
        if not pd.api.types.is_numeric_dtype(df_processed[target_column_name]):
             print(f"   再次尝试转换后，目标列 '{target_column_name}' 仍然不是数值类型。无法继续。")
             return None
        print(f"   目标列 '{target_column_name}' 在函数内被再次强制转换为数值类型。")
    print(f"\n目标列 '{target_column_name}' 的描述性统计 (转换和缺失值处理前):")
    print(df_processed[[target_column_name]].describe().T)
    initial_target_nan_count = df_processed[target_column_name].isnull().sum()
    if initial_target_nan_count > 0:
        print(f"注意: 目标列 '{target_column_name}' 中存在 {initial_target_nan_count} 个缺失值。这些行将被删除。")
        df_processed.dropna(subset=[target_column_name], inplace=True)
        print(f"删除目标列缺失值的行后，剩余数据: {df_processed.shape[0]} 行")
    if df_processed.shape[0] < 10:
        print(f"数据过少 (少于10行)，无法为目标 '{target_column_name}' 进行有效训练。")
        return None
    potential_targets = ['pH-S-CH', 'Cd-CH']
    all_cols_for_features = [col for col in df_processed.columns if col not in potential_targets and col != target_column_name]
    feature_columns = [col for col in all_cols_for_features if col in df_processed.columns]
    if not feature_columns: print(f"错误: 没有可用的特征列来训练目标 '{target_column_name}'。"); return None
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]
    if X.empty: print(f"错误: 特征集 X 为空，无法为目标 '{target_column_name}' 进行训练。"); return None
    min_samples_for_split_cv = 10 
    if len(y) < min_samples_for_split_cv : print(f"数据量过少 ({len(y)}条)，无法进行有效的训练集/测试集分割和5折交叉验证。"); return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"训练集规模: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集规模: X_test: {X_test.shape}, y_test: {y_test.shape}")
    if X_train.empty or X_test.empty or len(X_train) < 5 : print("训练集或测试集数据不足，无法进行5折交叉验证。"); return None
    
    log_transform_cols = ['Cd-C', 'BET'] 
    log_transform_cols = [col for col in log_transform_cols if col in X_train.columns] 
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

    model_name_for_log = "AutoTabPFN Regressor"
    model_name_for_plot = "AutoTabPFN"
    model_instance = None

    if autotabpfn_available and AutoTabPFNRegressor is not None:
        # --- 新增：CUDA 检测逻辑 ---
        actual_device_to_use = 'cpu' # 默认为CPU
        if user_preferred_device == 'cuda':
            print("信息: 用户选择使用 CUDA。正在检查 CUDA 可用性...")
            if torch.cuda.is_available():
                actual_device_to_use = 'cuda'
                print("信息: CUDA 可用，AutoTabPFN 将尝试在 GPU 上运行。")
            else:
                print("警告: 用户选择使用 CUDA，但系统未检测到可用的 CUDA 设备或 PyTorch CUDA 版本不匹配。AutoTabPFN 将在 CPU 上运行。")
        else:
            print("信息: AutoTabPFN 将在 CPU 上运行 (用户选择或默认)。")
        # --- CUDA 检测逻辑结束 ---

        print(f"信息: 配置 AutoTabPFNRegressor (max_time={autotabpfn_max_time}s, device='{actual_device_to_use}')。")
        model_instance = AutoTabPFNRegressor(
            max_time=autotabpfn_max_time, 
            device=actual_device_to_use, # 使用检测后的设备
        )
    else:
        print("错误: AutoTabPFNRegressor 不可用。无法继续。")
        return None

    if model_instance is None: # Should not happen if AutoTabPFN is available
        print(f"未能初始化模型 {model_name_for_log}。")
        return None
            
    results_summary_list = [] 
    print(f"\n--- 正在处理模型: {model_name_for_log} ---")
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    current_model_eval_metrics = {}

    try:
        print(f"开始对 {model_name_for_log} 进行5折交叉验证... (这可能耗时较长，特别是max_time较大时)")
        cv_results = cross_validate(full_pipeline, X_train, y_train, cv=5, 
                                    scoring=['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error'], 
                                    n_jobs=1, return_train_score=False) 
        mean_cv_r2 = np.mean(cv_results['test_r2'])
        mean_cv_rmse = -np.mean(cv_results['test_neg_root_mean_squared_error']) 
        mean_cv_mae = -np.mean(cv_results['test_neg_mean_absolute_error'])   
        current_model_eval_metrics['CV R2 Mean'] = mean_cv_r2
        current_model_eval_metrics['CV RMSE Mean'] = mean_cv_rmse
        current_model_eval_metrics['CV MAE Mean'] = mean_cv_mae
        print(f"交叉验证 - 平均R²: {mean_cv_r2:.4f}, 平均RMSE: {mean_cv_rmse:.4f}, 平均MAE: {mean_cv_mae:.4f}")

        print(f"开始在完整训练集上拟合 {model_name_for_log}...")
        full_pipeline.fit(X_train, y_train)
        
        y_pred_train = full_pipeline.predict(X_train)
        r2_train = r2_score(y_train, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae_train = mean_absolute_error(y_train, y_pred_train)
        current_model_eval_metrics['Train R2'] = r2_train
        current_model_eval_metrics['Train RMSE'] = rmse_train
        current_model_eval_metrics['Train MAE'] = mae_train
        print(f"训练集   - R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")

        y_pred_test = full_pipeline.predict(X_test)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        current_model_eval_metrics['Test R2'] = r2_test
        current_model_eval_metrics['Test RMSE'] = rmse_test
        current_model_eval_metrics['Test MAE'] = mae_test
        print(f"测试集   - R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

        plot_output_dir = f'./autotabpfn_performance_plots_{target_column_name}'
        plot_single_model_jointgrid(
            y_train_true=y_train, y_train_pred=y_pred_train,
            y_test_true=y_test, y_test_pred=y_pred_test,
            r2_train=r2_train, r2_test=r2_test,
            model_name_display=model_name_for_plot, 
            target_col_name=target_column_name,
            output_dir=plot_output_dir
        )
        
    except Exception as e:
        print(f"模型 {model_name_for_log} 处理失败: {e}")
        import traceback; traceback.print_exc() 
        for metric_key in ['CV R2 Mean','CV RMSE Mean','CV MAE Mean','Train R2','Train RMSE','Train MAE','Test R2','Test RMSE','Test MAE']:
            current_model_eval_metrics[metric_key] = 'Error'
        current_model_eval_metrics['Error_Info'] = str(e)
    
    results_summary_list.append({'Model_Full_Name': model_name_for_log, **current_model_eval_metrics})

    print("\n--- 模型评估完成 ---")
    if results_summary_list:
        temp_results_for_df = {item.pop('Model_Full_Name'): item for item in results_summary_list}
        results_df = pd.DataFrame(temp_results_for_df).T
        cols_order = ['CV R2 Mean','CV RMSE Mean','CV MAE Mean','Train R2','Train RMSE','Train MAE','Test R2','Test RMSE','Test MAE','Error_Info'] # Changed Error to Error_Info if that's the key
        present_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[present_cols]
        print("\n评估结果总结:")
        pd.set_option('display.float_format', lambda x: f'{x:.4f}' if isinstance(x, float) else x)
        print(results_df)
        return results_df
    else: print("模型未能成功完成评估。"); return None

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
    df_original = load_and_process_data(file_path) 
    if df_original is not None:
        target_column_options = ['pH-S-CH', 'Cd-CH'] 
        autotabpfn_max_time_input = 120 
        user_preferred_device_input = 'cpu' 
        try:
            time_str = input(f"请输入 AutoTabPFN 的最大调整时间（秒，例如 120）[默认 {autotabpfn_max_time_input}]: ")
            if time_str.strip(): autotabpfn_max_time_input = int(time_str)
            device_str = input(f"请选择 AutoTabPFN 使用的设备 ('cpu' 或 'cuda') [默认 {user_preferred_device_input}]: ").lower()
            if device_str.strip() in ['cuda', 'gpu']: user_preferred_device_input = 'cuda'
            elif device_str.strip() == 'cpu': user_preferred_device_input = 'cpu'
            elif device_str.strip(): print(f"无法识别的设备 '{device_str}', 将使用默认 '{user_preferred_device_input}'.")
        except ValueError: print(f"输入时间无效，将使用默认时间 {autotabpfn_max_time_input} 秒。")
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
                    train_and_evaluate_models(df_original.copy(), selected_target_col, 
                                              autotabpfn_max_time=autotabpfn_max_time_input, 
                                              user_preferred_device=user_preferred_device_input) 
                else: print("无效选项，请输入列表中的有效编号。")
            except ValueError: print("无效输入，请输入一个数字编号。")
            except Exception as e: print(f"主程序循环中发生了一个未预料的错误: {e}"); import traceback; traceback.print_exc()
    else: print("数据加载和初步处理失败，无法执行训练。")