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
import joblib  # <--- 新增：导入 joblib 用于保存模型

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
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions import interpretability
    # 这个变量在这里被定义
    tabpfn_available = True
    print("TabPFN 和其 SHAP 解释器成功导入。")
except ImportError:
    # 或者在这里被定义
    tabpfen_available = False
    TabPFNRegressor = None # 定义一个占位符，以防后续代码意外引用
    print("警告: 'tabpfen' 或 'tabpfen-extensions' 库未找到。TabPFN 模型将不可用。")
    print("请运行 'pip install tabpfn tabpfen-extensions'。")

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
def get_tabpfn_config():
    """
    定义 TabPFN 模型配置。
    """
    # 在这里安全地使用 tabpfn_available 变量
    if not tabpfn_available:
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"TabPFN 将使用设备: {device}")
    
    return {
        "model": TabPFNRegressor(device=device)
    }

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
    按照 ColumnTransformer 实际输出的顺序保存，不做重排序。
    """
    print(f"    - 正在为目标 '{target_col}' 准备完整的 SHAP 数据...")
    
    # 确保索引一致
    X_test_transformed_df_reset = X_test_transformed_df.reset_index(drop=True)
    
    # 创建特征值 DataFrame（添加 _Value 后缀）
    values_df = X_test_transformed_df_reset.copy().add_suffix('_Value')
    
    # 创建 SHAP 值 DataFrame（添加 _SHAP 后缀）
    shap_df = pd.DataFrame(
        shap_values_obj.values, 
        columns=X_test_transformed_df_reset.columns
    ).add_suffix('_SHAP')
    
    # 验证行数和列数
    print(f"    - 特征值数据形状: {values_df.shape}")
    print(f"    - SHAP 值数据形状: {shap_df.shape}")
    
    if len(values_df) != len(shap_df):
        print(f"    警告: 特征值行数 ({len(values_df)}) 与 SHAP 值行数 ({len(shap_df)}) 不匹配！")
        return
    
    if values_df.shape[1] != shap_df.shape[1]:
        print(f"    警告: 特征值列数 ({values_df.shape[1]}) 与 SHAP 值列数 ({shap_df.shape[1]}) 不匹配！")
        return
    
    # 按照 transformed 数据的实际顺序交替排列：Feature_Value, Feature_SHAP
    column_order = []
    for feature in X_test_transformed_df_reset.columns:
        column_order.append(f'{feature}_Value')
        column_order.append(f'{feature}_SHAP')
    
    # 合并数据
    combined_df = pd.concat([values_df, shap_df], axis=1)
    final_df = combined_df[column_order]
    
    # 输出前几行用于验证
    print(f"    - 数据预览（前3行，前8列）:")
    print(final_df.iloc[:3, :8])
    print(f"    - 最终数据形状: {final_df.shape}")
    print(f"    - 列名（前10个）: {list(final_df.columns[:10])}")

    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'all_shap_values_and_data_{safe_target_name}.csv')
    final_df.to_csv(save_path, index=False)
    print(f"    - 已将所有特征的 SHAP 值和数据保存至: {save_path}")


def analyze_and_plot_shap_for_tabpfn(fitted_pipelines, X_test, output_dir):
    """
    使用 TabPFN 官方的 SHAP 解释器为每个目标进行 SHAP 分析、绘图并保存数据。
    """
    if not tabpfn_available or not shap_available:
        print("\nTabPFN 或 SHAP 库不可用，跳过 SHAP 分析。")
        return
        
    print("\n--- 正在为 TabPFN 执行 SHAP 分析 (使用官方解释器) ---")

    for target_col, pipeline in fitted_pipelines.items():
        print(f"\n  --- 正在为目标 '{target_col}' 计算 SHAP 值 ---")
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            fitted_model = pipeline.named_steps['regressor']
            
            # 重置索引以确保对齐
            X_test_reset = X_test.reset_index(drop=True)
            
            # Transform 数据
            X_test_transformed = preprocessor.transform(X_test_reset)
            
            print(f"    - 原始 X_test 形状: {X_test_reset.shape}")
            print(f"    - Transformed 数据形状: {X_test_transformed.shape}")
            print(f"    - 原始特征列: {list(X_test_reset.columns)}")
            
            # 获取 transformed 后的特征名（保持 ColumnTransformer 的原始顺序）
            try:
                # 使用 sklearn 1.0+ 的 API
                transformed_feature_names = list(preprocessor.get_feature_names_out())
                print(f"    - get_feature_names_out() 返回的特征名: {transformed_feature_names}")
            except AttributeError:
                # 如果没有这个方法，手动构建
                print(f"    - get_feature_names_out() 不可用，手动构建特征名")
                transformed_feature_names = []
                
                # 遍历 transformers
                for name, trans, cols in preprocessor.transformers_:
                    if name == 'remainder':
                        # remainder 的列
                        remainder_cols = [c for c in X_test_reset.columns if c not in 
                                        sum([t[2] for t in preprocessor.transformers_[:-1]], [])]
                        transformed_feature_names.extend(remainder_cols)
                    else:
                        # 其他 transformer 的列
                        transformed_feature_names.extend(cols)
                
                print(f"    - 手动构建的特征名: {transformed_feature_names}")

            # 验证列数是否匹配
            if X_test_transformed.shape[1] != len(transformed_feature_names):
                print(f"    警告: transformed 数据列数 ({X_test_transformed.shape[1]}) "
                      f"与特征名数量 ({len(transformed_feature_names)}) 不匹配！")
                print(f"    将使用默认列名")
                transformed_feature_names = [f'feature_{i}' for i in range(X_test_transformed.shape[1])]

            # 创建 DataFrame
            X_test_transformed_df = pd.DataFrame(
                X_test_transformed, 
                columns=transformed_feature_names
            )
            
            # 输出每个特征的统计信息
            print(f"    - Transformed 数据统计（前5个特征）:")
            for i, col in enumerate(transformed_feature_names[:5]):
                vals = X_test_transformed_df.iloc[:, i]
                print(f"      列{i} ({col}): min={vals.min():.3f}, max={vals.max():.3f}, "
                      f"mean={vals.mean():.3f}, std={vals.std():.3f}")

            print("    - 正在计算 SHAP 值... 这可能需要一些时间。")
            shap_values = interpretability.shap.get_shap_values(
                estimator=fitted_model,
                test_x=X_test_transformed_df,
                algorithm="auto"
            )
            print("    - SHAP 值计算完成。")
            print(f"    - SHAP 值数组形状: {shap_values.values.shape}")
            
            # 输出 SHAP 值统计（前5个特征）
            print(f"    - SHAP 值统计（前5个特征）:")
            for i, col in enumerate(transformed_feature_names[:5]):
                shap_vals = shap_values.values[:, i]
                print(f"      列{i} ({col}): min={shap_vals.min():.3f}, max={shap_vals.max():.3f}, "
                      f"mean={shap_vals.mean():.3f}")
            
            safe_target_name = target_col.replace("Δ", "delta")
            
            # 绘制 SHAP Summary Plot
            plt.figure()
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
            fig = plt.gcf()
            fig.suptitle(f'SHAP Summary for Target: {target_col}', fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_summary = os.path.join(output_dir, f'shap_summary_{safe_target_name}.pdf')
            fig.savefig(save_path_summary, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    - 已保存 SHAP Summary 图至: {save_path_summary}")

            # 绘制特征重要性图
            plot_feature_importance_with_percentages(
                shap_values_obj=shap_values,
                feature_names=transformed_feature_names,
                target_col=target_col,
                output_dir=output_dir
            )

            # 保存 SHAP 数据到 CSV
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
def train_and_evaluate_tabpfn(df, feature_cols, target_cols, strata_col, log_cols, impute_cols, save_dir): # <--- 修改：增加 save_dir 参数
    """
    为每个目标训练、评估并保存一个独立的 TabPFN 模型。
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
    print(f"\n--- 正在评估: TabPFN ---")
    
    for target_col in target_cols:
        print(f"\n  --- 正在为目标 '{target_col}' 训练模型 ---")
        try:
            model_config = get_tabpfn_config()
            if not model_config:
                print(f"    - 无法创建 TabPFN 模型，跳过目标 '{target_col}'。")
                continue

            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_config['model'])])
            
            print(f"    - 执行5折交叉验证...")
            cv_scores_target = cross_val_score(pipeline, X_train, y_train[target_col], cv=5, scoring='r2', n_jobs=-1)
            print(f"    - 平均交叉验证 R²: {np.mean(cv_scores_target):.4f}")

            print(f"    - 正在训练最终模型...")
            pipeline.fit(X_train, y_train[target_col])
            fitted_pipelines[target_col] = pipeline
            
            # --- 新增代码: 保存训练好的模型 ---
            if save_dir:
                safe_target_name = target_col.replace("Δ", "delta")
                model_save_path = os.path.join(save_dir, f'tabpfn_model_{safe_target_name}.joblib')
                joblib.dump(pipeline, model_save_path)
                print(f"    - 模型已保存至: {model_save_path}")
            # --- 新增代码结束 ---
            
            pred_train = pipeline.predict(X_train)
            pred_test = pipeline.predict(X_test)

            rmse_train, mae_train, r2_train = calculate_metrics(y_train[target_col], pred_train)
            print(f"    - 训练集 ({target_col}): R²={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
            rmse_test, mae_test, r2_test = calculate_metrics(y_test[target_col], pred_test)
            print(f"    - 测试集 ({target_col}): R²={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
                                                                                                                        
        except Exception as e:
            print(f"错误: 无法为目标 '{target_col}' 训练 TabPFN 模型. 原因: {e}")
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
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'type','pH-B','BET','BR','MPs', 'Size', 'MPR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'type'
    log_transform_columns = ['CEC','Cd-C']
    impute_columns = ['OM', 'CEC', 'pH-S','BET']
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "TabPFN_shap分析版本"
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    # --- 修改: 提前创建输出目录 ---
    analysis_output_dir = os.path.join(output_base_path, output_folder_name)
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"所有输出文件将保存到: {analysis_output_dir}")

    # 在这里安全地使用 tabpfn_available 变量
    if not tabpfn_available:
        print("\nTabPFN 库不可用，程序即将退出。")
        exit()

    trained_models, X_test_data = train_and_evaluate_tabpfn(
        df=df, feature_cols=feature_columns, target_cols=target_columns,
        strata_col=stratification_column, log_cols=log_transform_columns,
        impute_cols=impute_columns,
        save_dir=analysis_output_dir  # <--- 修改: 传入保存路径
    )
    
    if trained_models:
        print(f"\n{'='*30}\n模型训练与保存完成，开始 SHAP 分析\n{'='*30}")
        
        analyze_and_plot_shap_for_tabpfn(
            fitted_pipelines=trained_models,
            X_test=X_test_data,
            output_dir=analysis_output_dir
        )

    print("\n程序执行完毕。")