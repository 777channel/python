import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
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
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 导入检查 ---
# 根据您的工作代码进行调整
try:
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions import interpretability
    print("成功导入 TabPFNRegressor 和 interpretability。")
    tabpfn_available = True
    pfn_interpret_available = True
except ImportError:
    print("警告: 未能从 'tabpfn_extensions' 导入。SHAP 分析将不可用。")
    pfn_interpret_available = False
    try:
        from tabpfn import TabPFNRegressor
        print("成功从标准 'tabpfn' 库导入 TabPFNRegressor。")
        tabpfn_available = True
    except ImportError:
        print("严重错误: 'tabpfen' 库未找到。模型将不可用。")
        tabpfn_available = False
        TabPFNRegressor = None


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
    """加载数据并对所有列进行初步数值转换（无法转换的为NaN），并移除BOM。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None

    print("\n尝试将所有列转换为数值型...")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")

    print("\n转换后数据类型信息 (df.info()):")
    df.info()
    return df

def plot_single_model_jointgrid(y_train_true, y_train_pred, y_test_true, y_test_pred,
                                r2_train, r2_test, model_name_display, target_col_name, output_dir='.'):
    """为单个模型创建实际值 vs. 预测值的 JointGrid 对比图。"""
    print(f"\n正在为模型 '{model_name_display}' 生成 JointGrid 图...")

    data_train_df = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    data_test_df = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    data_combined_df = pd.concat([data_train_df, data_test_df])

    if data_combined_df.empty:
        print(f"警告: 为模型 '{model_name_display}' 清洗后无有效数据可用于绘图。"); return

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
    g.plot_joint(sns.scatterplot, alpha=0.5, s=100, edgecolor='w', linewidth=1)
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=palette['Train'], label='Train Regression Line', ci=None)
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint,
                color=palette['Test'], label='Test Regression Line', ci=None)
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

    ax_joint = g.ax_joint
    ax_joint.tick_params(axis='x', labelsize=18)
    ax_joint.tick_params(axis='y', labelsize=18)
    ax_joint.set_xlabel('True', fontsize=16, fontweight='bold')
    ax_joint.set_ylabel('Predicted', fontsize=16, fontweight='bold')

    ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
    ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax_joint.text(0.6, 0.95, f'Model: {model_name_display}', transform=ax_joint.transAxes, fontsize=20,fontweight='bold',
                  verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    min_val = min(data_combined_df['True'].min(), data_combined_df['Predicted'].min())
    max_val = max(data_combined_df['True'].max(), data_combined_df['Predicted'].max())
    ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', label='x=y', lw=2)

    legend = ax_joint.legend(fontsize=20)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    g.fig.suptitle(f'{model_name_display} Model: Actual vs. Predicted Values for {target_col_name}', fontsize=22, fontweight='bold', y=1.02)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col_name}_{model_name_display}_JointGrid.pdf')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(g.fig)
    print(f"JointGrid 图已保存至: {save_path}")

# 省略了 save_shap_values_to_csv 和 save_features_and_shap_combined 函数，因为它们与绘图颜色无关

# --- 主训练与评估函数 ---
def train_and_evaluate_model(df, target_column_name, all_potential_targets):
    """为TabPFN模型进行训练和评估，并强制排除所有潜在目标和分层列。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练模型 ---")

    df_processed = df.copy()
    
    # 清洗目标列中的缺失值
    df_processed.dropna(subset=[target_column_name], inplace=True)
    
    # 定义需要排除的列
    strata_col_name = 'T-Aging' # 假设您的分层列名为 T-Aging
    cols_to_exclude = list(all_potential_targets)
    if strata_col_name in df_processed.columns and strata_col_name not in cols_to_exclude:
        cols_to_exclude.append(strata_col_name)

    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]

    # 分层抽样
    strata = None
    if strata_col_name in df_processed.columns:
        strata_series = df_processed[strata_col_name]
        if all(strata_series.value_counts() >= 2):
            strata = strata_series
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strata
    )

    # 预处理流程
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    log_transform_cols = ['Cd-C', 'BET']
    log_transform_cols_actual = [col for col in log_transform_cols if col in numeric_columns]
    other_numeric_cols = [col for col in numeric_columns if col not in log_transform_cols_actual]

    log_numeric_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1))),
        ('log1p', FunctionTransformer(np.log1p, validate=False)),
    ])
    numeric_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1))),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_numeric_features', log_numeric_pipeline, log_transform_cols_actual),
            ('other_numeric_features', numeric_pipeline, other_numeric_cols)
        ],
        remainder='drop'
    )

    # 模型实例化与训练
    if not tabpfn_available:
        print("错误: TabPFNRegressor 不可用。无法继续训练。"); return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = TabPFNRegressor(device=device)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])

    try:
        print("\n=== 开始拟合 TabPFN 模型 ===")
        full_pipeline.fit(X_train, y_train)
        print("=== TabPFN 模型拟合完成！===")
        
        y_pred_train = full_pipeline.predict(X_train)
        y_pred_test = full_pipeline.predict(X_test)

        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"训练集 - R²: {r2_train:.4f}")
        print(f"测试集 - R²: {r2_test:.4f}")
        
        plot_output_dir = f'./model_plots_{target_column_name}'
        plot_single_model_jointgrid(
            y_train, y_pred_train, y_test, y_pred_test,
            r2_train, r2_test, "TabPFN", target_column_name, output_dir=plot_output_dir
        )
        
        # --- SHAP 分析 ---
        print("\n开始为 TabPFN 进行 SHAP 分析...")
        if not pfn_interpret_available:
            print("错误: tabpfn_extensions 未安装，无法进行SHAP分析。"); return

        fitted_preprocessor = full_pipeline.named_steps['preprocessor']
        fitted_model = full_pipeline.named_steps['regressor']
        X_test_transformed = fitted_preprocessor.transform(X_test)
        
        # 提取特征名
        try:
            raw_feature_names = fitted_preprocessor.get_feature_names_out()
            cleaned_feature_names = [name.split('__')[-1] for name in raw_feature_names]
        except AttributeError:
            cleaned_feature_names = []
            for name, trans, original_cols in fitted_preprocessor.transformers_:
                if trans != 'drop': cleaned_feature_names.extend(original_cols)
        
        shap_values_obj = interpretability.shap.get_shap_values(
            estimator=fitted_model,
            test_x=X_test_transformed,
            attribute_names=cleaned_feature_names,
            algorithm="permutation"
        )
        
        shap_plot_dir = f'./shap_plots_{target_column_name}'
        os.makedirs(shap_plot_dir, exist_ok=True)
        
        # --- SHAP Summary Plot (摘要图) ---
        with plt.rc_context({'font.size': 20, 'font.weight': 'bold'}):
            shap_summary_file = os.path.join(shap_plot_dir, f'TabPFN_shap_summary_{target_column_name}.pdf')
            
            # ★★★ 修改点：移除 cmap 参数以使用默认红蓝色 ★★★
            shap.summary_plot(shap_values_obj, X_test_transformed, feature_names=cleaned_feature_names, show=False)
            
            plt.title(f'SHAP Summary Plot - {target_column_name}', fontsize=22, fontweight='bold', y=1.02)
            plt.savefig(shap_summary_file, dpi=600, bbox_inches='tight');
            plt.close()
            print(f"TabPFN SHAP summary plot 已保存至: {shap_summary_file}")

        # --- 特征重要性柱状图 ---
        print("计算并绘制特征重要性柱状图...")
        if hasattr(shap_values_obj, 'values'):
            shap_values_array = shap_values_obj.values
        else:
            shap_values_array = shap_values_obj

        mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': cleaned_feature_names,
            'Importance': mean_abs_shap
        })
        total_importance = feature_importance_df['Importance'].sum()
        feature_importance_df['Percentage'] = (feature_importance_df['Importance'] / total_importance) * 100 if total_importance > 0 else 0
        feature_importance_df.sort_values(by='Percentage', ascending=False, inplace=True)

        plt.figure(figsize=(12, 8))
        
        # ★★★ 修改点：不指定 palette 参数以使用默认颜色 ★★★
        ax = sns.barplot(x='Percentage', y='Feature', data=feature_importance_df)

        for p in ax.patches:
            width = p.get_width()
            ax.text(width + 0.1, p.get_y() + p.get_height() / 2., f'{width:.2f}%',
                    va='center', ha='left', fontsize=20, fontweight='bold')

        plt.title(f'Feature Importance (Mean Absolute SHAP) - {target_column_name}', fontsize=22, fontweight='bold')
        plt.xlabel('Importance (%)', fontsize=20, fontweight='bold')
        plt.ylabel('Feature', fontsize=20, fontweight='bold')
        
        # 省略其他样式代码...
        plt.tight_layout()
        importance_plot_filename = os.path.join(shap_plot_dir, f'TabPFN_feature_importance_bar_{target_column_name}.pdf')
        plt.savefig(importance_plot_filename, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"特征重要性柱状图已保存至: {importance_plot_filename}")

    except Exception as e:
        print(f"TabPFN 模型主处理流程失败: {e}");
        traceback.print_exc()

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
                user_choice_str = input(f"请输入选项编号 (0-{len(target_column_options)}): ")
                if not user_choice_str.strip(): continue
                user_choice = int(user_choice_str)
                if user_choice == 0:
                    print("程序退出。"); break
                elif 1 <= user_choice <= len(target_column_options):
                    selected_target_col = target_column_options[user_choice - 1]
                    # 将所有潜在目标传递给函数
                    train_and_evaluate_model(df_original.copy(), selected_target_col, target_column_options)
                else:
                    print("无效选项，请输入列表中的有效编号。")
            except ValueError:
                print("无效输入，请输入一个数字编号。")
            except Exception as e:
                print(f"主程序循环中发生了一个未预料的错误: {e}")
                traceback.print_exc()
    else:
        print("数据加载和初步处理失败，无法执行训练。")