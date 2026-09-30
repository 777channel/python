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
import warnings
import traceback
import shap

# 忽略所有警告，使输出更整洁
warnings.filterwarnings('ignore')

# --- 导入检查 ---
try:
    from tabpfn import TabPFNRegressor
    from tabpfn_extensions import interpretability
    print("成功导入 TabPFNRegressor 和 interpretability。")
    tabpfn_available = True
    pfn_interpret_available = True
except ImportError:
    print("警告: 未能从 'tabpfn_extensions' 导入。SHAP 分析将不可用。")
    pfn_interpret_available = False
    # Fallback to standard tabpfn
    try:
        from tabpfn import TabPFNRegressor
        print("成功从标准 'tabpfn' 库导入 TabPFNRegressor。")
        tabpfn_available = True
    except ImportError:
        print("严重错误: 'tabpfn' 库未找到。模型将不可用。")
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
    """加载数据，移除BOM，并尝试将所有列转为数值型。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}"); return None

    print("\n尝试将所有列转换为数值型...")
    for col in df.columns:
        # 修复：应该是 df[col] 而不是 col
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
    
    if data_combined_df.isnull().values.any():
        print("绘图数据中存在NaN值，将进行清理。")
        data_combined_df.dropna(inplace=True)

    if data_combined_df.empty:
        print(f"警告: 清洗后无有效数据可用于为模型 '{model_name_display}' 绘图。"); return

    palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
    g = sns.JointGrid(data=data_combined_df, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
    g.plot_joint(sns.scatterplot, alpha=0.5, s=100, edgecolor='w', linewidth=1)
    sns.regplot(data=data_train_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Train'], label='Train Regression Line', ci=None)
    sns.regplot(data=data_test_df, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color=palette['Test'], label='Test Regression Line', ci=None)
    g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)

    ax_joint = g.ax_joint
    ax_joint.tick_params(axis='x', labelsize=18); ax_joint.tick_params(axis='y', labelsize=18)
    ax_joint.set_xlabel('True Values', fontsize=16, fontweight='bold')
    ax_joint.set_ylabel('Predicted Values', fontsize=16, fontweight='bold')
    ax_joint.text(0.95, 0.15, f'Train $R^2 = {r2_train:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold', va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white"))
    ax_joint.text(0.95, 0.05, f'Test $R^2 = {r2_test:.3f}$', transform=ax_joint.transAxes, fontsize=20,fontweight='bold', va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white"))
    ax_joint.text(0.05, 0.95, f'Model: {model_name_display}', transform=ax_joint.transAxes, fontsize=20,fontweight='bold', va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", ec="black", fc="white"))

    min_val = min(data_combined_df['True'].min(), data_combined_df['Predicted'].min())
    max_val = max(data_combined_df['True'].max(), data_combined_df['Predicted'].max())
    ax_joint.plot([min_val, max_val], [min_val, max_val], c="black", alpha=0.5, linestyle='--', label='1:1 Line', lw=2)
    
    legend = ax_joint.legend(fontsize=20)
    for text in legend.get_texts(): text.set_fontweight('bold')
        
    g.fig.suptitle(f'{model_name_display} Model: Actual vs. Predicted for {target_col_name}', fontsize=22, fontweight='bold', y=1.02)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{target_col_name}_{model_name_display}_JointGrid.pdf')
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close(g.fig)
    print(f"JointGrid 图已保存至: {save_path}")

def plot_shap_feature_importance_with_percentage(shap_values_obj, feature_names, target_column_name, shap_plot_dir):
    """绘制带百分比标签的SHAP特征重要性图"""
    print("正在生成带百分比的SHAP特征重要性图...")
    
    # 处理SHAP Explanation对象，提取实际的SHAP值
    if hasattr(shap_values_obj, 'values'):
        # 如果是Explanation对象，提取values
        shap_values_array = shap_values_obj.values
        print("检测到SHAP Explanation对象，已提取values数组")
    else:
        # 如果已经是数组
        shap_values_array = shap_values_obj
        print("使用提供的SHAP值数组")
    
    print(f"SHAP值数组形状: {shap_values_array.shape}")
    
    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.mean(np.abs(shap_values_array), axis=0)
    
    # 计算百分比
    total_importance = np.sum(mean_abs_shap)
    percentages = (mean_abs_shap / total_importance) * 100
    
    # 创建特征重要性数据框并按重要性排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap,
        'percentage': percentages
    }).sort_values('importance', ascending=True)
    
    # 创建图形
    plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
    
    # 绘制水平条形图
    bars = plt.barh(range(len(importance_df)), importance_df['importance'], 
                   color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 添加百分比标签
    for i, (bar, percentage, importance) in enumerate(zip(bars, importance_df['percentage'], importance_df['importance'])):
        # 在条形图末端添加百分比文本
        plt.text(importance + max(importance_df['importance']) * 0.01, i, 
                f'{percentage:.1f}%', 
                va='center', ha='left', fontweight='bold', fontsize=12)
    
    # 设置y轴标签
    plt.yticks(range(len(importance_df)), importance_df['feature'], fontweight='bold', fontsize=12)
    
    # 设置标题和轴标签
    plt.title(f'SHAP Feature Importance with Percentages - {target_column_name}', 
              fontweight='bold', fontsize=16, pad=20)
    plt.xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=14)
    plt.ylabel('Features', fontweight='bold', fontsize=14)
    
    # 调整x轴范围以容纳百分比标签
    plt.xlim(0, max(importance_df['importance']) * 1.15)
    
    # 添加网格
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    importance_plot_path = os.path.join(shap_plot_dir, f'SHAP_feature_importance_with_percentage_{target_column_name}.pdf')
    plt.savefig(importance_plot_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"带百分比的SHAP特征重要性图已保存至: {importance_plot_path}")
    
    # 打印特征重要性排序结果
    print(f"\n特征重要性排序 (降序):")
    sorted_df = importance_df.sort_values('importance', ascending=False)
    for idx, row in sorted_df.iterrows():
        print(f"{row['feature']}: {row['percentage']:.2f}%")

# --- 主训练与评估函数 ---
def train_and_evaluate_model(df, target_column_name, all_potential_targets):
    """为TabPFN模型进行训练和评估，并强制排除所有潜在目标和分层列。"""
    print(f"\n--- 开始为目标列 '{target_column_name}' 训练模型 ---")

    df_current = df.copy() # 使用df_current来避免修改原始df
    
    print(f"数据处理前，总行数: {len(df_current)}")
    # 注意：这里不再进行异常值清洗，但仍保留目标列缺失值删除，以确保训练数据的有效性
    df_current.dropna(subset=[target_column_name], inplace=True)
    print(f"根据目标列 '{target_column_name}' 清除缺失值后，剩余行数: {len(df_current)}")
    
    strata_col_name = 'T_aging'
    # 确保所有潜在的目标列和分层列都被排除在特征之外
    cols_to_exclude = list(all_potential_targets)
    if strata_col_name in df_current.columns and strata_col_name not in cols_to_exclude:
        cols_to_exclude.append(strata_col_name)

    feature_columns = [col for col in df_current.columns if col not in cols_to_exclude]
    print(f"最终用于模型训练的特征列 ({len(feature_columns)}个): {feature_columns}")

    X = df_current[feature_columns]
    y = df_current[target_column_name]

    strata = None
    if strata_col_name in df_current.columns:
        strata_series = df_current[strata_col_name]
        if all(strata_series.value_counts() >= 2):
            strata = strata_series
            print(f"\n成功启用分层抽样，分层依据: '{strata_col_name}'")
        else:
            print(f"\n警告: 分层列 '{strata_col_name}' 中存在样本数小于2的类别，退回普通随机抽样。")
    else:
        print(f"\n警告: 未在数据中找到分层列 '{strata_col_name}'，使用普通随机抽样。")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=strata
    )
    print(f"数据按 7:3 划分完成: 训练集规模={X_train.shape}, 测试集规模={X_test.shape}")

    print("\n定义预处理流程：KNN填充 + 特定列Log1p变换...")
    log_transform_cols = ['BET', 'Cd-C'] 
    
    numeric_columns = X_train.select_dtypes(include=np.number).columns.tolist()
    log_transform_cols_actual = [col for col in log_transform_cols if col in numeric_columns]
    other_numeric_cols = [col for col in numeric_columns if col not in log_transform_cols_actual]

    log_numeric_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1))),
        ('log1p', FunctionTransformer(np.log1p, validate=False))
    ])
    if log_transform_cols_actual:
        print(f"将对 {log_transform_cols_actual} 列进行 KNNImputer -> Log1p 变换。")

    other_numeric_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=min(5, len(X_train)-1)))
    ])
    if other_numeric_cols:
        print(f"将对 {len(other_numeric_cols)} 个其他数值特征进行 KNNImputer 变换。")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('log_features', log_numeric_pipeline, log_transform_cols_actual),
            ('other_numeric', other_numeric_pipeline, other_numeric_cols)
        ],
        remainder='drop' # 仅处理指定列，其余丢弃
    )
    
    if not tabpfn_available:
        print("错误: TabPFN不可用。终止训练。"); return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    model_instance = TabPFNRegressor(device=device)
    full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_instance)])

    try:
        print("=== 开始拟合模型 ===")
        full_pipeline.fit(X_train, y_train)
        print("=== 模型拟合完成！ ===")
        
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
        
        model_save_dir = './trained_models'
        os.makedirs(model_save_dir, exist_ok=True)
        model_filename = f'model_{target_column_name}.joblib'
        joblib.dump(full_pipeline, os.path.join(model_save_dir, model_filename))
        print(f"\n模型已成功保存至: {os.path.join(model_save_dir, model_filename)}")

        # --- SHAP 可解释性分析 ---
        print("\n--- 开始 SHAP 可解释性分析 ---")
        if not pfn_interpret_available:
            print("错误: `tabpfn_extensions` 未找到。无法执行 SHAP 分析。")
            return

        shap_plot_dir = f'./shap_plots_{target_column_name}'
        os.makedirs(shap_plot_dir, exist_ok=True)
        
        fitted_preprocessor = full_pipeline.named_steps['preprocessor']
        fitted_model = full_pipeline.named_steps['regressor']
        
        # X_test_transformed 是经过预处理器处理后的数据，包含Log1p转换
        X_test_transformed = fitted_preprocessor.transform(X_test)
        
        # 尝试提取转换后的特征名
        cleaned_feature_names = []
        try:
            feature_names_transformed = fitted_preprocessor.get_feature_names_out()
            cleaned_feature_names = [name.split('__')[-1] for name in feature_names_transformed]
            print("成功使用 get_feature_names_out() 提取特征名。")
        except AttributeError:
            print("警告: get_feature_names_out() 不可用 (可能 scikit-learn 版本较旧)。正在手动重构特征名...")
            # 手动重构特征名逻辑，以防 get_feature_names_out 不可用
            # 这部分通常比较复杂，且依赖于具体的 ColumnTransformer 配置
            # 简单起见，如果 ColumnTransformer 内部是针对列名操作的，可以直接用 X.columns
            # 但是由于有 drop 和不同的 pipeline，最好能用 get_feature_names_out
            # 这里简化为直接使用 X_test 的原始特征名，并假定其顺序匹配 (这可能不总是安全的)
            cleaned_feature_names = X_test.columns.tolist()
            # 进一步考虑 Log1p 转换后名称可能会变，但当前 get_feature_names_out() 处理已经过滤掉了前缀
            # 这里的 fallback 逻辑需要更鲁棒，但为了不增加复杂性，先保持原样。
            # 如果实际运行发现特征名错误，需要在这里加强处理。

        # 最终校验特征名数量
        if len(cleaned_feature_names) != X_test_transformed.shape[1]:
            print(f"警告: 提取的特征名数量 ({len(cleaned_feature_names)}) 与转换后的数据列数 ({X_test_transformed.shape[1]}) 不匹配。将使用通用特征名。")
            cleaned_feature_names = [f'feature_{i}' for i in range(X_test_transformed.shape[1])]
        
        print(f"最终用于SHAP分析的特征名 ({len(cleaned_feature_names)}个): {cleaned_feature_names}")
        
        print("正在使用 `interpretability.shap.get_shap_values` 计算 SHAP 值...")
        # `test_x` 必须是模型实际接收的形式，即经过预处理（包括Log1p）的数据
        shap_values_obj = interpretability.shap.get_shap_values(
            estimator=fitted_model,
            test_x=X_test_transformed, # 传入经过预处理和Log1p转换的数据
            attribute_names=cleaned_feature_names,
            algorithm="permutation"
        )
        print("SHAP 值计算完成。")

        # --- ★★★ 核心修改：保存 SHAP 值和原始特征值到同一个CSV ★★★ ---
        shap_data_output_dir = f'./shap_data_for_{target_column_name}'
        os.makedirs(shap_data_output_dir, exist_ok=True)

        # 1. 保存完整的 shap_values_obj (shap.Explanation 对象) - 保留pickle格式便于后续使用
        shap_explanation_path = os.path.join(shap_data_output_dir, f'shap_explanation_obj_{target_column_name}.pkl')
        joblib.dump(shap_values_obj, shap_explanation_path)
        print(f"完整的 SHAP Explanation 对象已保存至: {shap_explanation_path}")

        # 2. 将SHAP值和原始特征值合并保存到同一个CSV文件
        print("正在合并SHAP值和原始特征值到同一个CSV文件...")
        
        # 提取SHAP值数组
        if hasattr(shap_values_obj, 'values'):
            shap_values_array = shap_values_obj.values
        else:
            shap_values_array = shap_values_obj
        
        # 创建SHAP值的DataFrame，列名加上前缀以区分
        shap_df = pd.DataFrame(shap_values_array, columns=[f'SHAP_{name}' for name in cleaned_feature_names])
        
        # 重置所有数据的索引，确保能正确合并
        X_test_reset = X_test.reset_index(drop=True)
        shap_df_reset = shap_df.reset_index(drop=True)
        
        # 合并数据：原始特征值 + SHAP值
        combined_df = pd.concat([
            X_test_reset,           # 原始特征值
            shap_df_reset           # SHAP值
        ], axis=1)
        
        # 保存合并后的数据
        combined_csv_path = os.path.join(shap_data_output_dir, f'combined_features_shap_{target_column_name}.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"合并的原始特征值和SHAP值已保存至: {combined_csv_path}")
        
        # 打印保存的数据概览
        print(f"保存的CSV文件包含：")
        print(f"  - 原始特征列数: {len(cleaned_feature_names)}")
        print(f"  - SHAP值列数: {len(cleaned_feature_names)} (以'SHAP_'开头)")
        print(f"  - 总行数: {len(combined_df)}")
        print(f"  - 总列数: {len(combined_df.columns)}")
        
        # --- ★★★ 核心修改结束 ★★★ ---

        print("正在生成 SHAP Summary Plot...")
        plt.figure()
        # Summary plot 使用 shap_values_obj，它内部会使用 .data 作为特征值
        # 此时 .data 包含 Log1p 转换后的值
        shap.summary_plot(shap_values_obj, show=False) 
        summary_plot_path = os.path.join(shap_plot_dir, f'SHAP_summary_plot_{target_column_name}.pdf')
        plt.savefig(summary_plot_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"SHAP Summary Plot 已保存至: {summary_plot_path}")

        # 使用新的带百分比的特征重要性图函数
        # 该函数也基于 shap_values_obj 中的 .values
        plot_shap_feature_importance_with_percentage(shap_values_obj, cleaned_feature_names, target_column_name, shap_plot_dir)

    except Exception as e:
        print(f"\n模型主处理流程失败: {e}");
        traceback.print_exc()

# --- 主程序 ---
if __name__ == "__main__":
    file_path = r'C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv'
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