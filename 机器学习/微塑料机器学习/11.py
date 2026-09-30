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
import joblib

# ---------------------------------------------------------------------------
# 禁用posthog和backoff日志 (避免干扰输出)
# ---------------------------------------------------------------------------
import os
import logging

os.environ['POSTHOG_DISABLED'] = '1'  # 禁用posthog
logging.getLogger('posthog').setLevel(logging.CRITICAL)  # 屏蔽posthog日志
logging.getLogger('backoff').setLevel(logging.CRITICAL)  # 屏蔽backoff日志

# ---------------------------------------------------------------------------
# 库可用性检查
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
    tabpfn_available = True
    print("TabPFN 和其 SHAP 解释器成功导入。")
except ImportError:
    tabpfn_available = False
    TabPFNRegressor = None
    print("警告: 'tabpfn' 或 'tabpfn-extensions' 库未找到。TabPFN 模型将不可用。")
    print("请运行 'pip install tabpfn tabpfn-extensions'。")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. 全局绘图参数
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['legend.fontsize'] = 28
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ---------------------------------------------------------------------------
# 2. 模型定义
# ---------------------------------------------------------------------------
def get_tabpfn_config():
    """定义 TabPFN 模型配置。"""
    if not tabpfn_available:
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"TabPFN 将使用设备: {device}")
    
    return {
        "model": TabPFNRegressor(device=device)
    }

# ---------------------------------------------------------------------------
# 3. 基础绘图与分析函数
# ---------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred):
    """计算 RMSE, MAE 和 R²"""
    return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)

def plot_feature_importance_with_percentages(shap_values_obj, feature_names, target_col, output_dir):
    """计算并绘制带百分比标签的 SHAP 特征重要性条形图。"""
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
    """将所有特征的实际值和 SHAP 值合并并保存到一个 CSV 文件中。"""
    print(f"    - 正在为目标 '{target_col}' 准备完整的 SHAP 数据...")
    
    values_df = X_test_transformed_df.copy().add_suffix('_Value')
    shap_df = pd.DataFrame(
        shap_values_obj.values, 
        columns=X_test_transformed_df.columns, 
        index=X_test_transformed_df.index
    ).add_suffix('_SHAP')
    
    combined_df = pd.concat([values_df, shap_df], axis=1)
    
    column_order = []
    for feature in X_test_transformed_df.columns:
        column_order.append(f'{feature}_Value')
        column_order.append(f'{feature}_SHAP')
    
    final_df = combined_df[column_order]

    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'all_shap_values_and_data_{safe_target_name}.csv')
    final_df.to_csv(save_path, index=False)
    print(f"    - 已将所有特征的 SHAP 值和数据保存至: {save_path}")

# ---------------------------------------------------------------------------
# 4. 新增: SHAP依赖图 (MPR×BR交互) + 拟合曲线
# ---------------------------------------------------------------------------
def plot_shap_dependence_interaction(shap_values_obj, X_test_transformed_df, target_col, output_dir):
    """
    绘制MPR和BR的SHAP依赖图,展示它们的交互效应,并添加LOWESS拟合曲线。
    
    子图A: MPR的SHAP值,颜色显示BR的调节作用
    子图B: BR的SHAP值,颜色显示MPR的调节作用
    """
    print(f"    - 正在为目标 '{target_col}' 绘制 SHAP 依赖交互图...")
    
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    feature_names = X_test_transformed_df.columns.tolist()
    
    # 查找MPR和BR在特征列表中的位置
    mpr_idx = None
    br_idx = None
    
    for idx, name in enumerate(feature_names):
        if 'MPR' in name:
            mpr_idx = idx
        if 'BR' in name:
            br_idx = idx
    
    if mpr_idx is None or br_idx is None:
        print(f"    - 警告: 未找到 MPR 或 BR 特征,跳过依赖图绘制")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图A: MPR的SHAP (颜色=BR)
    ax = axes[0]
    shap.dependence_plot(
        ind=mpr_idx,
        shap_values=shap_values_obj.values,
        features=X_test_transformed_df,
        interaction_index=br_idx,
        ax=ax,
        show=False
    )
    
    # 添加LOWESS拟合曲线
    mpr_values = X_test_transformed_df.iloc[:, mpr_idx].values
    mpr_shap = shap_values_obj.values[:, mpr_idx]
    
    # 排序以便绘制平滑曲线
    sort_idx = np.argsort(mpr_values)
    mpr_sorted = mpr_values[sort_idx]
    shap_sorted = mpr_shap[sort_idx]
    
    # LOWESS拟合
    lowess_result = lowess(shap_sorted, mpr_sorted, frac=0.3)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1], 
            color='black', linewidth=3, linestyle='--', 
            label='LOWESS Fit', alpha=0.8)
    
    ax.set_xlabel('MPR Value', fontsize=30, fontweight='bold')
    ax.set_ylabel('SHAP Value for MPR', fontsize=30, fontweight='bold')
    ax.set_title('(A) MPR Effect Moderated by BR', fontsize=30, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    
    # 子图B: BR的SHAP (颜色=MPR)
    ax = axes[1]
    shap.dependence_plot(
        ind=br_idx,
        shap_values=shap_values_obj.values,
        features=X_test_transformed_df,
        interaction_index=mpr_idx,
        ax=ax,
        show=False
    )
    
    # 添加LOWESS拟合曲线
    br_values = X_test_transformed_df.iloc[:, br_idx].values
    br_shap = shap_values_obj.values[:, br_idx]
    
    # 排序
    sort_idx = np.argsort(br_values)
    br_sorted = br_values[sort_idx]
    shap_sorted = br_shap[sort_idx]
    
    # LOWESS拟合
    lowess_result = lowess(shap_sorted, br_sorted, frac=0.3)
    ax.plot(lowess_result[:, 0], lowess_result[:, 1], 
            color='black', linewidth=3, linestyle='--', 
            label='LOWESS Fit', alpha=0.8)
    
    ax.set_xlabel('BR Value', fontsize=20, fontweight='bold')
    ax.set_ylabel('SHAP Value for BR', fontsize=20, fontweight='bold')
    ax.set_title('(B) BR Effect Moderated by MPR', fontsize=22, fontweight='bold')
    ax.legend(fontsize=16, loc='best')
    
    plt.tight_layout()
    
    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'shap_dependence_MPR_BR_{safe_target_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 已保存 SHAP 依赖交互图至: {save_path}")

# ---------------------------------------------------------------------------
# 5. 新增: 分条件偏依赖热图 (BR×MPR在9个pH-S×Cd-C条件下)
# ---------------------------------------------------------------------------
def plot_conditional_pdp_heatmaps(fitted_pipeline, X_test, target_col, output_dir):
    """\n    在9个pH-S×Cd-C条件组合下,绘制BR×MPR的偏依赖热图。\n    \n    关键修复: 在原始特征空间设定条件值,然后转换\n    \n    pH-S: 5, 6, 7\n    Cd-C: 1, 2, 3\n    总共: 3×3 = 9个子图\n    """
    print(f"    - 正在为目标 '{target_col}' 绘制9个条件PDP热图...")
    
    # 获取preprocessor和model
    preprocessor = fitted_pipeline.named_steps['preprocessor']
    model = fitted_pipeline.named_steps['regressor']
    
    # 在原始X_test中找到列名
    original_feature_names = X_test.columns.tolist()
    
    # 查找关键特征在原始数据中的列名
    br_col_orig = 'BR' if 'BR' in original_feature_names else None
    mpr_col_orig = 'MPR' if 'MPR' in original_feature_names else None
    ph_col_orig = 'pH-S' if 'pH-S' in original_feature_names else None
    cd_col_orig = 'Cd-C' if 'Cd-C' in original_feature_names else None
    
    if not all([br_col_orig, mpr_col_orig, ph_col_orig, cd_col_orig]):
        print(f"    - 警告: 在原始数据中未找到所有必需特征")
        print(f"    - 找到的特征: BR={br_col_orig}, MPR={mpr_col_orig}, pH-S={ph_col_orig}, Cd-C={cd_col_orig}")
        return
    
    # 创建基准样本(使用原始特征空间的中位数)
    baseline_original = X_test.median()
    
    # 定义固定的条件值(在原始特征空间)
    ph_levels = [5, 6, 7]
    cd_levels = [1, 2, 3]
    
    # 创建BR×MPR网格(在原始特征空间)
    br_range = np.linspace(X_test[br_col_orig].quantile(0.05), 
                           X_test[br_col_orig].quantile(0.95), 50)
    mpr_range = np.linspace(X_test[mpr_col_orig].quantile(0.05), 
                            X_test[mpr_col_orig].quantile(0.95), 50)
    BR_mesh, MPR_mesh = np.meshgrid(br_range, mpr_range)
    
    # 创建3×3布局
    fig, axes = plt.subplots(3, 3, figsize=(26, 24))
    
    for i, ph_val in enumerate(ph_levels):
        for j, cd_val in enumerate(cd_levels):
            ax = axes[i, j]
            
            print(f"      - 正在计算 pH-S={ph_val}, Cd-C={cd_val} 的响应面...")
            
            # 计算这个条件下的响应面
            predictions_mesh = np.zeros_like(BR_mesh)
            
            for row in range(BR_mesh.shape[0]):
                for col in range(BR_mesh.shape[1]):
                    # 在原始特征空间构造样本
                    X_sample_original = baseline_original.copy()
                    X_sample_original[br_col_orig] = BR_mesh[row, col]
                    X_sample_original[mpr_col_orig] = MPR_mesh[row, col]
                    X_sample_original[ph_col_orig] = ph_val
                    X_sample_original[cd_col_orig] = cd_val
                    
                    # 转换为DataFrame并通过pipeline预测
                    X_sample_df = X_sample_original.to_frame().T
                    
                    # 预测(pipeline会自动预处理)
                    try:
                        predictions_mesh[row, col] = fitted_pipeline.predict(X_sample_df)[0]
                    except:
                        predictions_mesh[row, col] = np.nan
            
            # 检查是否有有效预测
            if np.all(np.isnan(predictions_mesh)):
                print(f"      - 警告: pH-S={ph_val}, Cd-C={cd_val} 的预测全部失败")
                continue
            
            # 绘制热图
            levels = np.linspace(np.nanmin(predictions_mesh), np.nanmax(predictions_mesh), 20)
            contour = ax.contourf(BR_mesh, MPR_mesh, predictions_mesh, 
                                  levels=levels, cmap='RdYlGn_r', alpha=0.9)
            
            # 添加等高线
            try:
                contour_lines = ax.contour(BR_mesh, MPR_mesh, predictions_mesh, 
                                           levels=10,
                                           colors='black', linewidths=0.8, alpha=0.3)
                ax.clabel(contour_lines, inline=True, fontsize=11, fmt='%.2f')
            except:
                pass
            
            # 特别标注零线(如果存在)
            try:
                if np.nanmin(predictions_mesh) < 0 < np.nanmax(predictions_mesh):
                    zero_line = ax.contour(BR_mesh, MPR_mesh, predictions_mesh, 
                                           levels=[0], colors='red', linewidths=3)
                    ax.clabel(zero_line, inline=True, fontsize=14, fmt='ΔCd=0')
            except:
                pass
            
            # 叠加实际样本点(在原始空间筛选)
            tolerance_ph = 0.5
            tolerance_cd = 0.5
            
            mask = ((X_test[ph_col_orig] >= ph_val - tolerance_ph) & 
                    (X_test[ph_col_orig] <= ph_val + tolerance_ph) &
                    (X_test[cd_col_orig] >= cd_val - tolerance_cd) & 
                    (X_test[cd_col_orig] <= cd_val + tolerance_cd))
            
            subset = X_test[mask]
            if len(subset) > 0:
                ax.scatter(subset[br_col_orig], subset[mpr_col_orig], 
                          c='black', s=15, alpha=0.5, edgecolor='white', linewidth=0.5)
            
            # 设置标签和标题
            ax.set_xlabel('Biochar Rate (BR, %)', fontsize=28, fontweight='bold')
            ax.set_ylabel('Microplastic Rate (MPR, %)', fontsize=28, fontweight='bold')
            ax.set_title(f'pH-S={ph_val}, Cd-C={cd_val} mg/kg\n(n={len(subset)} samples)',
                        fontsize=28, fontweight='bold')
            
            # 设置坐标轴范围
            ax.set_xlim([BR_mesh.min(), BR_mesh.max()])
            ax.set_ylim([MPR_mesh.min(), MPR_mesh.max()])
            
            # 设置x，y轴刻度数字字体大小和加粗
            for label in ax.get_xticklabels():
                label.set_fontsize(28)
                label.set_fontweight('bold')
            for label in ax.get_yticklabels():
                label.set_fontsize(28)
                label.set_fontweight('bold')
            
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label(f'Predicted {target_col}', fontsize=28, fontweight='bold')
            
            # --- 修改颜色条刻度数字的样式 ---
            cbar.ax.tick_params(labelsize=28) # 设置基础大小
            
            # 强制设置字体大小和加粗 (与刚才修复坐标轴的方法保持一致)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(28)
                label.set_fontweight('bold')
    
    plt.tight_layout()
    
    safe_target_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'conditional_pdp_BR_MPR_9panels_{safe_target_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 已保存9面板条件PDP热图至: {save_path}")

# ---------------------------------------------------------------------------
# 6. 整合所有SHAP和PDP分析
# ---------------------------------------------------------------------------
def analyze_and_plot_shap_for_tabpfn(fitted_pipelines, X_test, output_dir):
    """
    使用 TabPFN 官方的 SHAP 解释器为每个目标进行完整分析:
    1. SHAP summary
    2. SHAP 特征重要性
    3. SHAP 依赖图 (MPR×BR交互)
    4. 条件PDP热图 (BR×MPR在不同pH和Cd条件下)
    """
    if not tabpfn_available or not shap_available:
        print("\nTabPFN 或 SHAP 库不可用,跳过 SHAP 分析。")
        return
        
    print("\n--- 正在为 TabPFN 执行完整的 SHAP 和 PDP 分析 ---")

    for target_col, pipeline in fitted_pipelines.items():
        print(f"\n  --- 正在为目标 '{target_col}' 进行分析 ---")
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            fitted_model = pipeline.named_steps['regressor']
            X_test_transformed = preprocessor.transform(X_test)
            
            try:
                transformed_feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                transformed_feature_names = X_test.columns.tolist()

            X_test_transformed_df = pd.DataFrame(X_test_transformed, 
                                                 columns=transformed_feature_names, 
                                                 index=X_test.index)

            # === 1. 计算SHAP值 ===
            print("    - 正在计算 SHAP 值... 这可能需要一些时间。")
            shap_values = interpretability.shap.get_shap_values(
                estimator=fitted_model,
                test_x=X_test_transformed_df,
                algorithm="auto"
            )
            print("    - SHAP 值计算完成。")
            
            safe_target_name = target_col.replace("Δ", "delta")
            
            # === 2. SHAP Summary Plot ===
            plt.figure()
            shap.summary_plot(shap_values, X_test_transformed_df, show=False)
            fig = plt.gcf()
            fig.suptitle(f'SHAP Summary for Target: {target_col}', fontsize=20)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            save_path_summary = os.path.join(output_dir, f'shap_summary_{safe_target_name}.pdf')
            fig.savefig(save_path_summary, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    - 已保存 SHAP Summary 图")

            # === 3. SHAP 特征重要性 ===
            plot_feature_importance_with_percentages(
                shap_values_obj=shap_values,
                feature_names=transformed_feature_names,
                target_col=target_col,
                output_dir=output_dir
            )

            # === 4. 保存SHAP数据 ===
            save_all_shap_data_to_csv(
                shap_values_obj=shap_values,
                X_test_transformed_df=X_test_transformed_df,
                target_col=target_col,
                output_dir=output_dir
            )

            # === 5. 新增: SHAP依赖交互图 ===
            plot_shap_dependence_interaction(
                shap_values_obj=shap_values,
                X_test_transformed_df=X_test_transformed_df,
                target_col=target_col,
                output_dir=output_dir
            )

            # === 6. 新增: 条件PDP热图 ===
            plot_conditional_pdp_heatmaps(
                fitted_pipeline=pipeline,
                X_test=X_test,
                target_col=target_col,
                output_dir=output_dir
            )

        except Exception as e:
            print(f"为目标 '{target_col}' 生成分析时发生错误: {e}")
            traceback.print_exc()

# ---------------------------------------------------------------------------
# 7. 主训练与评估流程
# ---------------------------------------------------------------------------
def train_and_evaluate_tabpfn(df, feature_cols, target_cols, strata_col, log_cols, impute_cols, save_dir):
    """为每个目标训练、评估并保存一个独立的 TabPFN 模型。"""
    X = df[feature_cols]
    y = df[target_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=42, 
                                                        stratify=df[strata_col])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)), 
                ('log1p', FunctionTransformer(np.log1p, validate=False))
            ]), [col for col in impute_cols if col in log_cols]),
            ('impute_only', KNNImputer(n_neighbors=5), 
             [col for col in impute_cols if col not in log_cols])
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
                print(f"    - 无法创建 TabPFN 模型,跳过目标 '{target_col}'。")
                continue

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), 
                ('regressor', model_config['model'])
            ])
            
            print(f"    - 执行5折交叉验证...")
            cv_scores_target = cross_val_score(pipeline, X_train, y_train[target_col], 
                                              cv=5, scoring='r2', n_jobs=-1)
            print(f"    - 平均交叉验证 R²: {np.mean(cv_scores_target):.4f}")

            print(f"    - 正在训练最终模型...")
            pipeline.fit(X_train, y_train[target_col])
            fitted_pipelines[target_col] = pipeline
            
            # 保存模型
            if save_dir:
                safe_target_name = target_col.replace("Δ", "delta")
                model_save_path = os.path.join(save_dir, f'tabpfn_model_{safe_target_name}.joblib')
                joblib.dump(pipeline, model_save_path)
                print(f"    - 模型已保存至: {model_save_path}")
            
            pred_train = pipeline.predict(X_train)
            pred_test = pipeline.predict(X_test)

            rmse_train, mae_train, r2_train = calculate_metrics(y_train[target_col], pred_train)
            print(f"    - 训练集: R²={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
            rmse_test, mae_test, r2_test = calculate_metrics(y_test[target_col], pred_test)
            print(f"    - 测试集: R²={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
                                                                                                                        
        except Exception as e:
            print(f"错误: 无法为目标 '{target_col}' 训练 TabPFN 模型. 原因: {e}")
            traceback.print_exc()

    if not fitted_pipelines:
        return None, None
        
    return fitted_pipelines, X_test

# ---------------------------------------------------------------------------
# 8. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 用户配置区 ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'pH-B','BET','BR','MPs', 'Size', 'MPR']
    target_columns = ['ΔpH', 'ΔCd']
    stratification_column = 'type'
    log_transform_columns = ['CEC']
    impute_columns = ['OM', 'CEC', 'pH-S']
    output_base_path = r"机器学习\微塑料机器学习\微塑料机器学习输出"
    output_folder_name = "TabPFN_完整分析版本"
    
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载数据: {len(df)} 行")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        exit()

    # 创建输出目录
    analysis_output_dir = os.path.join(output_base_path, output_folder_name)
    os.makedirs(analysis_output_dir, exist_ok=True)
    print(f"所有输出文件将保存到: {analysis_output_dir}")

    if not tabpfn_available:
        print("\nTabPFN 库不可用,程序即将退出。")
        exit()

    # 训练模型
    trained_models, X_test_data = train_and_evaluate_tabpfn(
        df=df, 
        feature_cols=feature_columns, 
        target_cols=target_columns,
        strata_col=stratification_column, 
        log_cols=log_transform_columns,
        impute_cols=impute_columns,
        save_dir=analysis_output_dir
    )
    
    # 完整的SHAP和PDP分析
    if trained_models:
        print(f"\n{'='*60}\n模型训练完成,开始完整的 SHAP 和 PDP 分析\n{'='*60}")
        
        analyze_and_plot_shap_for_tabpfn(
            fitted_pipelines=trained_models,
            X_test=X_test_data,
            output_dir=analysis_output_dir
        )

    print("\n" + "="*60)
    print("程序执行完毕。")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  - SHAP summary 图")
    print(f"  - SHAP 特征重要性图")
    print(f"  - SHAP 依赖交互图 (MPR×BR, 两个方向)")
    print(f"  -条件PDP热图 (BR×MPR在4个pH×Cd条件下)")
    print(f"  - SHAP值和数据CSV文件")
    print(f"  - 训练好的模型文件")
    print(f"\n所有文件保存在: {analysis_output_dir}")