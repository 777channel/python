import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import shap
import traceback

# --- 导入检查 ---
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
    print("TabPFNRegressor 库导入成功。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: TabPFNRegressor 未找到。")

# --- 全局绘图参数 ---
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

# --- 数据加载与绘图函数 ---
def load_and_process_data(file_path):
    """加载数据并对指定列进行数值转换。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None
    
    cols_to_convert = ['BR', 'pH-S-CH', 'Cd-CH']
    for col_name in cols_to_convert:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    print("\n数据类型信息 (df.info()):")
    df.info()
    return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                  r2_train, r2_test, model_name_display, output_dir='.'):
    """为单个模型创建实际值 vs. 预测值的 JointGrid 对比图。"""
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")

    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df])
    data_combined_df.dropna(subset=['True', 'Predicted'], inplace=True)

    if data_combined_df.empty:
        print(f"警告: 为模型 '{model_name_display}' 清洗后无有效数据可用于绘图。"); return

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
    g.plot_joint(sns.scatterplot, alpha=0.5, s=100, edgecolor='w', linewidth=1)
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Train'], label='Train Regression Line', ci=None)
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Test'], label='Test Regression Line', ci=None)
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

    ax_joint = g.ax_joint
    ax_joint.tick_params(axis='x', labelsize=18)
    ax_joint.tick_params(axis='y', labelsize=18)
    ax_joint.set_xlabel('True', fontsize=16, fontweight='bold')
    ax_joint.set_ylabel('Predicted', fontsize=16, fontweight='bold')

    ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=ax_joint.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=ax_joint.transAxes, fontsize=20, fontweight='bold', verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax_joint.text(0.05, 0.95, f'Model: {model_name_display}', transform=ax_joint.transAxes, fontsize=20, fontweight='bold', verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    min_val = min(data_combined_df['True'].min(), data_combined_df['Predicted'].min())
    max_val = max(data_combined_df['True'].max(), data_combined_df['Predicted'].max())
    ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', label='x=y', lw=2)
    
    legend = ax_joint.legend(fontsize=20)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    g.fig.suptitle(f'{model_name_display}: Actual vs. Predicted Values', fontsize=22, fontweight='bold', y=1.02)
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{model_name_display}_JointGrid.pdf')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(g.fig)
    print(f"JointGrid 图已保存至: {save_path}")

def save_shap_values_to_csv(shap_values, feature_names, target_column_name, output_dir='./shap_csv'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        else:
            shap_values_array = shap_values
        
        shap_values_df = pd.DataFrame(shap_values_array, columns=feature_names)
        csv_filename = f'shap_values_{target_column_name}_(测试).csv'
        csv_filepath = os.path.join(output_dir, csv_filename)
        shap_values_df.to_csv(csv_filepath, index=False, encoding='utf-8')
        
        print(f"SHAP值已保存至CSV文件: {csv_filepath}")
        return csv_filepath
    except Exception as e:
        print(f"保存SHAP值到CSV文件时发生错误: {e}")
        return None

def save_features_and_shap_combined(X_test, shap_values, feature_names, target_column_name, y_test, y_pred, output_dir='./shap_features_combined'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(shap_values, 'values'):
            shap_values_array = shap_values.values
        else:
            shap_values_array = shap_values
        
        X_test_df = X_test.copy().reset_index(drop=True)
        combined_data = X_test_df.copy()
        
        for i, feature in enumerate(feature_names):
            combined_data[f'{feature}_SHAP'] = shap_values_array[:, i]
            
        combined_data[f'{target_column_name}_true'] = y_test.reset_index(drop=True)
        combined_data[f'{target_column_name}_pred'] = y_pred
        combined_data['sample_id'] = range(len(combined_data))
        
        combined_filename = f'features_shap_combined_{target_column_name}_(测试).csv'
        combined_filepath = os.path.join(output_dir, combined_filename)
        combined_data.to_csv(combined_filepath, index=False, encoding='utf-8')
        print(f"特征+SHAP组合数据保存成功: {combined_filepath}")
        
        return combined_filepath
    except Exception as e:
        print(f"保存特征+SHAP组合数据时发生错误: {e}")
        return None

def train_and_evaluate_tabpfn(df, target_column_name):
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练 TabPFN 模型 ---")
    
    # 1. 数据准备和划分
    df_processed = df.copy().dropna(subset=[target_column_name])
    if df_processed.shape[0] < 10: print("数据过少，无法训练。"); return

    potential_targets = ['pH-S-CH', 'Cd-CH']
    strata_col_name = 'T_aging'
    cols_to_exclude = potential_targets + [strata_col_name]
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]
    
    strata = df_processed[strata_col_name] if strata_col_name in df_processed.columns and all(df_processed[strata_col_name].value_counts() > 1) else None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=strata)
    print(f"训练集规模: {X_train.shape}, 测试集规模: {X_test.shape}")

    # 2. 预处理器定义
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_features', Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))]), numeric_features)
        ],
        remainder='drop'
    )

    # 3. TabPFN模型实例化和训练
    if not tabpfn_available: print("错误: TabPFNRegressor 不可用。"); return
    
    # <--- 修改部分：强制使用CPU ---
    device = 'cpu'
    print(f"使用设备: {device}")
    model_instance = TabPFNRegressor(device=device)
    
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])
    
    try:
        print("开始在训练集上拟合 TabPFN...")
        full_pipeline.fit(X_train, y_train)
        
        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        print(f"训练集 - R²: {r2_train:.4f}, 测试集 - R²: {r2_test:.4f}")
        
        model_display_name = "TabPFN_(测试)"
        plot_output_dir = f'./tabpfn_plots_{target_column_name}_(测试)'
        plot_single_model_jointgrid(y_train, y_pred_train, y_test, y_pred_test, r2_train, r2_test, model_display_name, plot_output_dir)
        
        model_save_dir = './trained_models_(测试)'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'artificial_aging_model_{target_column_name}_(测试).joblib')
        joblib.dump(full_pipeline, model_save_path)
        print(f"\n模型已成功保存至: {model_save_path}")

        # SHAP 分析部分
        print("\n--- 7. SHAP 分析 (使用官方shap库) ---")

        fitted_preprocessor = full_pipeline.named_steps['preprocessor']
        fitted_model = full_pipeline.named_steps['regressor']
        
        # <--- 修改部分：移除了所有设备切换的逻辑，因为模型本身就在CPU上 ---
        
        X_train_transformed = fitted_preprocessor.transform(X_train)
        X_test_transformed = fitted_preprocessor.transform(X_test)
        
        try:
            raw_feature_names = fitted_preprocessor.get_feature_names_out()
            cleaned_feature_names = [name.split('__')[-1] for name in raw_feature_names]
        except AttributeError:
            cleaned_feature_names = X_train.columns.tolist()

        X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=cleaned_feature_names)
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=cleaned_feature_names)

        print("创建官方SHAP解释器 (KernelExplainer)...")
        background_data = shap.sample(X_train_transformed_df, 50)

        def predict_fn_for_shap(x_np):
            x_df = pd.DataFrame(x_np, columns=cleaned_feature_names)
            with torch.no_grad():
                 predictions = fitted_model.predict(x_df)
            return predictions
            
        explainer = shap.KernelExplainer(predict_fn_for_shap, background_data)
        
        print("正在为测试集计算SHAP值（在CPU上运行）...")
        shap_values = explainer(X_test_transformed_df)

        shap_plot_dir = f'./shap_plots_{target_column_name}_(测试)'
        os.makedirs(shap_plot_dir, exist_ok=True)
        
        # 绘制并保存SHAP瀑布图
        print("\n绘制SHAP瀑布图 (以第一个测试样本为例)...")
        sample_index_to_explain = 0 
        shap_explanation_for_one_sample = shap_values[sample_index_to_explain]

        plt.figure()
        shap.waterfall_plot(shap_explanation_for_one_sample, show=False)
        
        waterfall_filename = os.path.join(shap_plot_dir, f'shap_waterfall_sample_{sample_index_to_explain}_(测试).pdf')
        plt.savefig(waterfall_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP 瀑布图已保存至: {waterfall_filename}")

        # 为瀑布图创建并保存专属说明文档
        readme_waterfall_content = f"""瀑布图说明文档
========================

文件: {os.path.basename(waterfall_filename)}
对应目标列: {target_column_name}

此SHAP瀑布图展示了模型对 **测试集(X_test)** 中的第 **{sample_index_to_explain}** 个样本 (索引从0开始) 进行预测时的完整决策过程。

- **E[f(X)] (底部基准值)**: 模型对所有样本的平均预测值。
- **红色条**: 该特征使最终预测值 **增加**。
- **蓝色条**: 该特征使最终预测值 **减少**。
- **f(x) (顶部最终值)**: 模型对该样本的最终预测输出。

您可以修改代码中的 `sample_index_to_explain` 变量来查看其他样本的解释。
"""
        readme_waterfall_filename = os.path.join(shap_plot_dir, f'README_waterfall_sample_{sample_index_to_explain}_(测试).txt')
        with open(readme_waterfall_filename, 'w', encoding='utf-8') as f:
            f.write(readme_waterfall_content)
        
        print(f"✓ 瀑布图说明文档已保存至: {readme_waterfall_filename}")

        # 绘制 Summary Plot
        with plt.rc_context({'font.size': 20, 'font.weight': 'bold'}):
            shap_summary_file = os.path.join(shap_plot_dir, f'TabPFN_shap_summary_(测试).pdf')
            shap.summary_plot(shap_values, X_test_transformed_df, show=False, cmap=plt.get_cmap('plasma'))
            plt.savefig(shap_summary_file, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"SHAP summary plot 已保存至: {shap_summary_file}")

        csv_output_dir = f'./shap_csv_{target_column_name}_(测试)'
        save_shap_values_to_csv(shap_values, cleaned_feature_names, target_column_name, csv_output_dir)
        
        combined_output_dir = f'./features_shap_combined_{target_column_name}_(测试)'
        saved_combined_path = save_features_and_shap_combined(
            X_test, shap_values, cleaned_feature_names, target_column_name, y_test, y_pred_test, combined_output_dir
        )
        if saved_combined_path:
            print(f"✓ 特征+SHAP组合数据保存成功: {saved_combined_path}")
        else:
            print("✗ 特征+SHAP组合数据保存失败")
            
    except Exception as e:
        print(f"TabPFN 模型主处理流程失败: {e}")
        traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
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
                    user_choice_str = input(f"请输入选项编号 (0-{len(target_column_options)}): ")
                    if not user_choice_str.strip(): continue
                    user_choice = int(user_choice_str)
                    if user_choice == 0: print("程序退出。"); break
                    elif 1 <= user_choice <= len(target_column_options):
                        selected_target_col = target_column_options[user_choice - 1]
                        train_and_evaluate_tabpfn(df_original.copy(), selected_target_col)
                    else: print("无效选项。")
                except ValueError: print("无效输入，请输入一个数字。")
                except Exception as e: print(f"主程序循环中发生了一个未预料的错误: {e}"); traceback.print_exc()
    else:
        print("数据加载失败。")