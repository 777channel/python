# ---------------------------------------------------------------------------
# 最终版（已修复 IndexError）：独立的 SHAP 依赖图绘制脚本
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import os
import warnings
from matplotlib.ticker import MaxNLocator

# --- 全局设置 ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'

# --- 特征名称映射 ---
feature_name_map = {
    'pH-S': 'pH-S', 'OM': 'OM', 'CEC': 'CEC', 'Cd-C': 'Cd-C', 'RM': 'RM', 
    'T': 'T', 'pH-B': 'pH-B', 'BR': 'BR', 'MPs': 'MPs', 'Size': 'Size', 
    'MPR': 'MPR'
}

# --- 核心函数: 模型拟合与选择 ---
def find_best_fit(x, y, max_poly_degree=3):
    """尝试多种模型拟合并基于AIC返回最佳模型。"""
    n = len(y)
    if n < 4: return None
    candidate_models = {}
    if max_poly_degree >= 1: candidate_models['Linear'] = {'func': lambda x, a, b: a * x + b, 'p0': [1, 0], 'k': 2}
    if max_poly_degree >= 2: candidate_models['Quadratic'] = {'func': lambda x, a, b, c: a * x**2 + b * x + c, 'p0': [1, 1, 0], 'k': 3}
    if max_poly_degree >= 3: candidate_models['Cubic'] = {'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 'p0': [1, 1, 1, 0], 'k': 4}
    if np.all(x > 0) and np.min(x) > 1e-6: candidate_models['Logarithmic'] = {'func': lambda x, a, b: a * np.log(x) + b, 'p0': [1, 0], 'k': 2}
    best_model, min_aic = None, float('inf')
    for name, model_info in candidate_models.items():
        try:
            if name in ['Linear', 'Quadratic', 'Cubic']:
                params = np.polyfit(x, y, model_info['k'] - 1)
                y_pred = np.polyval(params, x)
            else:
                params, _ = curve_fit(model_info['func'], x, y, p0=model_info['p0'], maxfev=10000)
                y_pred = model_info['func'](x, *params)
            rss = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (rss / ss_tot) if ss_tot > 0 else 0
            k = model_info['k']
            aic = n * np.log(rss / n) + 2 * k if rss > 0 else -np.inf
            if aic < min_aic:
                min_aic = aic
                best_model = {'name': name, 'params': params, 'r_squared': r_squared, 'aic': aic, 'func': model_info.get('func'), 'degree': model_info.get('k', 0) - 1}
        except (RuntimeError, np.linalg.LinAlgError): continue
    return best_model

def format_equation(model_info):
    """格式化数学方程字符串。"""
    name, params = model_info['name'], model_info['params']
    try:
        if name == 'Linear': return f"$y = {params[0]:.2g}x {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}$"
        elif name == 'Quadratic': return f"$y = {params[0]:.2g}x^2 {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}x {'+' if params[2] >= 0 else '-'} {abs(params[2]):.2g}$"
        elif name == 'Cubic':
            line1 = f"$y = {params[0]:.2g}x^3 {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}x^2$"
            line2 = f"${'+' if params[2] >= 0 else '-'} {abs(params[2]):.2g}x {'+' if params[3] >= 0 else '-'} {abs(params[3]):.2g}$"
            return f"{line1}\n{line2}"
        elif name == 'Logarithmic': return f"$y = {params[0]:.2g} \\ln(x) {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}$"
    except (IndexError, TypeError): return ""
    return ""

# --- 主绘图函数 ---
def plot_pdp_with_best_fit(csv_file_path, ncols=4, n_bootstraps=100, random_seed=42, save_as_pdf=True, max_poly_degree=3, exclude_features=None, feature_name_map=None):
    """从指定的CSV文件绘制高级依赖图。"""
    print(f"正在读取文件: {csv_file_path}")
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取文件失败: {e}"); return

    shap_columns = [col for col in data.columns if col.endswith('_SHAP')]
    feature_shap_pairs = []
    exclude_list = exclude_features if exclude_features is not None else []
    
    for shap_col in shap_columns:
        value_col = shap_col.replace('_SHAP', '_Value')
        original_feature_name = shap_col.replace('_SHAP', '')
        if value_col in data.columns and original_feature_name not in exclude_list:
            feature_shap_pairs.append((value_col, shap_col, original_feature_name))
    
    if not feature_shap_pairs: print("错误: 未找到有效的 '..._Value' 和 '..._SHAP' 列对。"); return
    
    print(f"找到 {len(feature_shap_pairs)} 对特征进行绘图。最大多项式次数: {max_poly_degree}")
    
    output_dir = os.path.dirname(csv_file_path) or '.'
    np.random.seed(random_seed)
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", ['#ADD8E6', '#D8BFD8', '#DA70D6', '#FF00FF'])
    
    total_plots = len(feature_shap_pairs)
    actual_nrows = (total_plots + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(25, 6 * actual_nrows))
    gs = GridSpec(actual_nrows, ncols, figure=fig, left=0.04, right=0.85, top=0.92, bottom=0.12, wspace=0.35, hspace=0.55)
    
    # ------------------- FIX START -------------------
    # 1. 创建一个完整的子图网格
    axes = [fig.add_subplot(gs[i, j]) for i in range(actual_nrows) for j in range(ncols)]
    # -------------------- FIX END --------------------
    
    scatter_plots = []
    for idx, (value_col, shap_col, original_feature_name) in enumerate(feature_shap_pairs):
        ax = axes[idx]
        feature_display_name = feature_name_map.get(original_feature_name, original_feature_name)

        valid_indices = data[value_col].notna() & data[shap_col].notna()
        X_data, Y_data = data.loc[valid_indices, value_col].values, data.loc[valid_indices, shap_col].values
        
        if len(X_data) < 4:
            ax.set_title(f"{feature_display_name}", fontsize=22, fontweight='bold')
            ax.axis('off'); continue

        best_fit = find_best_fit(X_data, Y_data, max_poly_degree=max_poly_degree)
        
        vmin, vmax = np.min(X_data), np.max(X_data)
        scatter_plot = ax.scatter(X_data, Y_data, s=60, c=X_data, cmap=custom_cmap, alpha=0.7, edgecolor='none', zorder=2, vmin=vmin, vmax=vmax)
        scatter_plots.append(scatter_plot)
        
        if best_fit:
            x_fit = np.linspace(vmin, vmax, 200)
            y_fit = np.polyval(best_fit['params'], x_fit) if best_fit['name'] in ['Linear', 'Quadratic', 'Cubic'] else best_fit['func'](x_fit, *best_fit['params'])
            ax.plot(x_fit, y_fit, color='#A52A2A', linestyle='-', linewidth=3, label='Trendline', zorder=3)
            
            boot_preds = []
            if n_bootstraps > 0:
                x_eval = np.linspace(vmin, vmax, 100)
                for _ in range(n_bootstraps):
                    indices = np.random.choice(len(X_data), len(X_data), replace=True)
                    try:
                        if best_fit['name'] in ['Linear', 'Quadratic', 'Cubic']:
                            if len(np.unique(X_data[indices])) < best_fit['degree'] + 1: continue
                            params = np.polyfit(X_data[indices], Y_data[indices], best_fit['degree'])
                            boot_preds.append(np.polyval(params, x_eval))
                        elif best_fit['name'] == 'Logarithmic':
                            if not np.all(X_data[indices] > 0): continue
                            params, _ = curve_fit(best_fit['func'], X_data[indices], Y_data[indices], p0=best_fit['params'], maxfev=5000)
                            boot_preds.append(best_fit['func'](x_eval, *params))
                    except (RuntimeError, np.linalg.LinAlgError): continue
            
            if len(boot_preds) > 10:
                ci_lower, ci_upper = np.percentile(np.array(boot_preds), [2.5, 97.5], axis=0)
                ax.fill_between(x_eval, ci_lower, ci_upper, color='salmon', alpha=0.35, label='95% CI', zorder=1)

            ymin, ymax = ax.get_ylim(); y_range = ymax - ymin if ymax > ymin else 1.0
            ax.set_ylim(ymin - y_range * 0.30, ymax + y_range * 0.05)
            
            eq_str = format_equation(best_fit)
            r2_y_pos, eq_y_pos, eq_fs = (0.20, 0.03, 16) if best_fit['name'] == 'Cubic' else (0.12, 0.03, 18)
            ax.text(0.97, r2_y_pos, f"$R^2$ = {best_fit['r_squared']:.2f}", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='right', va='bottom')
            if eq_str: ax.text(0.97, eq_y_pos, eq_str, transform=ax.transAxes, fontsize=eq_fs, fontweight='bold', ha='right', va='bottom')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{feature_display_name}', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.locator_params(axis='y', nbins=6)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
        ax.set_xlabel('Feature Values', fontsize=22, fontweight='bold')
        
        if original_feature_name == 'BR': ax.xaxis.set_major_locator(MaxNLocator(3))
        else: ax.locator_params(axis='x', nbins=6)

        if idx % ncols == 0: ax.set_ylabel('SHAP Values', fontsize=22, fontweight='bold')
        else: ax.set_ylabel('')

        if ax.get_legend_handles_labels()[0]: ax.legend(loc='upper left', fontsize=18, frameon=False)
        ax.text(0.97, 0.97, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=22, fontweight='bold', ha='right', va='top')
        print(f"  - 已绘制: {feature_display_name}, 最佳拟合模型: {best_fit['name'] if best_fit else 'None'}")
    
    # ------------------- FIX START -------------------
    # 2. 将网格中未使用的子图关闭
    for idx in range(total_plots, len(axes)):
        axes[idx].axis('off')
    # -------------------- FIX END --------------------
    
    if scatter_plots and total_plots > 0:
        last_row_start_index = (actual_nrows - 1) * ncols
        # 确保 last_row_start_index 不会超出 axes 列表范围
        last_row_start_index = min(last_row_start_index, len(axes) - 1)
        
        first_ax_pos, last_ax_pos = axes[0].get_position(), axes[last_row_start_index].get_position()
        cbar_bottom, cbar_height = last_ax_pos.y0, first_ax_pos.y1 - last_ax_pos.y0
        cbar_ax = fig.add_axes([gs.right + 0.015, cbar_bottom, 0.02, cbar_height])
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Feature Values', rotation=270, labelpad=25, fontsize=22, fontweight='bold')
        cbar.outline.set_visible(False)
        cbar.set_ticks([])
        cbar_ax.text(1.8, 0, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')
        cbar_ax.text(1.8, 1, 'High', ha='center', va='top', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')

    file_basename = os.path.splitext(os.path.basename(csv_file_path))[0]
    if save_as_pdf:
        pdf_filename = f'Advanced_Dependence_Plots_from_{file_basename}.pdf'
        pdf_filepath = os.path.join(output_dir, pdf_filename)
        plt.savefig(pdf_filepath, format='pdf', bbox_inches='tight', dpi=600)
        print(f"\n高级依赖图已保存至: {pdf_filepath}")
    
    plt.close('all')

# =============================================================================
# --- 程序主入口：在这里直接使用您提供的文件路径 ---
# =============================================================================
if __name__ == "__main__":
    
    # --- 文件路径配置区 ---
    csv_file_deltapH = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_特征重要性版本\all_shap_values_and_data_deltaCd.csv"
    csv_file_deltaCd = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_特征重要性版本\all_shap_values_and_data_deltapH.csv"

    # --- 任务1: 为 deltapH 绘图 ---
    if os.path.exists(csv_file_deltapH):
        print("="*60)
        print("正在为目标 'deltapH' 生成依赖图...")
        plot_pdp_with_best_fit(
            csv_file_path=csv_file_deltapH,
            ncols=4, 
            max_poly_degree=2,
            feature_name_map=feature_name_map
        )
    else:
        print(f"\n警告: 文件未找到，请检查路径: {csv_file_deltapH}")

    # --- 任务2: 为 deltaCd 绘图 ---
    if os.path.exists(csv_file_deltaCd):
        print("\n" + "="*60)
        print("正在为目标 'deltaCd' 生成依赖图...")
        plot_pdp_with_best_fit(
            csv_file_path=csv_file_deltaCd,
            ncols=4, 
            max_poly_degree=2,
            feature_name_map=feature_name_map
        )
    else:
        print(f"\n警告: 文件未找到，请检查路径: {csv_file_deltaCd}")
        
    print("\n所有绘图任务完成。")