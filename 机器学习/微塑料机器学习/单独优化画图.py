# ---------------------------------------------------------------------------
# 按 Type 分类的 SHAP 依赖图绘制脚本
# 1. 逻辑: 仅使用设定范围内的数据进行拟合及绘图 (数据截断)
# 2. 视觉: LOWESS 曲线 + Bootstrapping CI
# 3. 文字: 二次方程拟合 (R2, 公式)
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
import statsmodels.api as sm 

# --- 全局设置 ---
warnings.filterwarnings("ignore")
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
    'MPR': 'MPR','BET':'BET'
}

# =============================================================================
# --- [配置 1] 自定义每个 Type 要显示的特征列表 ---
# =============================================================================
FEATURES_BY_TYPE = {
    # --- Type 1: 你想显示的特征 ---
    1: ['pH-S', 'OM', 'Cd-C','MPs', 'Size', 'MPR'],
    # --- Type 2: 你想显示的特征 ---
    2:['pH-S', 'OM', 'Cd-C','RM','pH-B', 'T','BET', 'BR'],
    # --- Type 3: 你想显示的特征 ---
    3: ['pH-S', 'OM', 'Cd-C','CEC','RM','pH-B', 'T','BET', 'BR', 'MPs', 'Size', 'MPR']
}

# =============================================================================
# --- [配置 2] X轴范围配置区域 (数据将被截断在此范围内) ---
# =============================================================================
FEATURE_XLIMS_BY_TYPE = {
    # --- Type 1 范围设置 ---
    1: {
        'MPR': (0, 0.05),       
        'Size': (0, 260),
        # 'pH-S': (4, 9)
    },
    
    # --- Type 2 范围设置 ---
    2: {
        # 'pH-B': (7, 12),  
        'BET': (0, 100),  
        'BR' :(0,0.1)    
    },
    
    # --- Type 3 范围设置 ---
    3: {
        'MPR': (0, 0.055), 
        'Size': (0, 100),  
        'BET': (0, 25), 
        'Size': (28, 52),
    }
}
TYPE_LABELS = {
    1: 'Type 1 (MPs Only)',
    2: 'Type 2 (Biochar Only)',
    3: 'Type 3 (Combined)'
}

# =============================================================================
# --- 辅助函数 ---
# =============================================================================
def find_best_fit_equation(x, y, max_poly_degree=2):
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
                best_model = {'name': name, 'params': params, 'r_squared': r_squared}
        except: continue
    return best_model

def format_equation(model_info):
    if not model_info: return ""
    name, params = model_info['name'], model_info['params']
    try:
        if name == 'Linear': return f"$y = {params[0]:.2g}x {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}$"
        elif name == 'Quadratic': return f"$y = {params[0]:.2g}x^2 {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}x {'+' if params[2] >= 0 else '-'} {abs(params[2]):.2g}$"
        elif name == 'Cubic':
            line1 = f"$y = {params[0]:.2g}x^3 {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}x^2$"
            line2 = f"${'+' if params[2] >= 0 else '-'} {abs(params[2]):.2g}x {'+' if params[3] >= 0 else '-'} {abs(params[3]):.2g}$"
            return f"{line1}\n{line2}"
        elif name == 'Logarithmic': return f"$y = {params[0]:.2g} \\ln(x) {'+' if params[1] >= 0 else '-'} {abs(params[1]):.2g}$"
    except: return ""
    return ""

# --- 主绘图函数 ---
def plot_pdp_by_type(csv_file_path, type_value, ncols=4, feature_xlims=None, features_to_show=None): 
    print(f"\n{'='*60}")
    print(f"正在为 {TYPE_LABELS[type_value]} 绘制 SHAP 依赖图 (数据截断模式)...")
    
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取文件失败: {e}"); return
    
    if 'type_Value' not in data.columns:
        print("错误: 数据中未找到 'type_Value' 列！"); return
    
    data_filtered = data[data['type_Value'] == type_value].copy()
    
    if features_to_show is None:
        print(f"错误: 未指定 Type={type_value} 要显示的特征列表。"); return

    feature_shap_pairs = []
    for feature in features_to_show:
        value_col = f"{feature}_Value"
        shap_col = f"{feature}_SHAP"
        if value_col in data_filtered.columns and shap_col in data_filtered.columns:
            if (data_filtered[value_col] != 0).any():
                feature_shap_pairs.append((value_col, shap_col, feature))
    
    if not feature_shap_pairs:
        print("没有有效的特征可绘制。"); return
    
    output_dir = os.path.dirname(csv_file_path) or '.'
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", ['#ADD8E6', '#D8BFD8', '#DA70D6', '#FF00FF'])
    
    total_plots = len(feature_shap_pairs)
    actual_nrows = (total_plots + ncols - 1) // ncols
    
    fig = plt.figure(figsize=(25, 6 * actual_nrows))
    gs = GridSpec(actual_nrows, ncols, figure=fig, left=0.04, right=0.85, 
                  top=0.92, bottom=0.12, wspace=0.35, hspace=0.55)
    
    axes = [fig.add_subplot(gs[i, j]) for i in range(actual_nrows) for j in range(ncols)]
    scatter_plots = []
    
    lowess = sm.nonparametric.lowess
    lowess_frac = 0.88 
    n_bootstraps = 1000
    np.random.seed(42)

    for idx, (value_col, shap_col, original_feature_name) in enumerate(feature_shap_pairs):
        ax = axes[idx]
        feature_display_name = feature_name_map.get(original_feature_name, original_feature_name)

        # 1. 获取原始数据
        valid_indices = data_filtered[value_col].notna() & data_filtered[shap_col].notna()
        X_data = data_filtered.loc[valid_indices, value_col].values
        Y_data = data_filtered.loc[valid_indices, shap_col].values
        
        # =====================================================================
        # --- [核心修改] 数据过滤/截断 ---
        # 如果定义了范围，直接剔除范围外的数据，不再参与后续任何计算
        # =====================================================================
        if feature_xlims and original_feature_name in feature_xlims:
            x_min_limit, x_max_limit = feature_xlims[original_feature_name]
            
            # 创建掩码
            mask = (X_data >= x_min_limit) & (X_data <= x_max_limit)
            
            # 只有当范围内有足够数据时才应用过滤，否则保留全量并警告
            if np.sum(mask) >= 5:
                X_data = X_data[mask]
                Y_data = Y_data[mask]
                print(f"  -> [{feature_display_name}] 数据已截断至 [{x_min_limit}, {x_max_limit}] (剩余点数: {len(X_data)})")
                
                # 强制设置 X 轴范围，以防万一散点没铺满
                ax.set_xlim(x_min_limit, x_max_limit)
            else:
                print(f"  -> [警告] [{feature_display_name}] 范围内数据点过少 ({np.sum(mask)})，保持使用全量数据。")
        # =====================================================================

        # 2. 绘制散点 (此时 X_data 已经是截断后的数据了)
        vmin, vmax = np.min(X_data), np.max(X_data)
        scatter_plot = ax.scatter(X_data, Y_data, s=60, c=X_data, cmap=custom_cmap, 
                                 alpha=0.7, edgecolor='none', zorder=2, vmin=vmin, vmax=vmax)
        scatter_plots.append(scatter_plot)

        # 生成平滑 X 轴点 (使用当前数据的范围)
        x_smooth = np.linspace(X_data.min(), X_data.max(), 200)

        # 3. 计算 LOWESS (基于截断数据)
        try:
            sort_idx = np.argsort(X_data)
            X_sorted, Y_sorted = X_data[sort_idx], Y_data[sort_idx]
            smooth_res = lowess(Y_sorted, X_sorted, frac=lowess_frac, it=0)
            y_lowess_main = np.interp(x_smooth, smooth_res[:, 0], smooth_res[:, 1])
            
            ax.plot(x_smooth, y_lowess_main, color='#A52A2A', linestyle='-', linewidth=3, 
                   label='LOWESS', zorder=3)
            
            # Bootstrapping (基于截断数据)
            boot_preds = []
            for _ in range(n_bootstraps):
                indices = np.random.choice(len(X_data), len(X_data), replace=True)
                X_boot, Y_boot = X_data[indices], Y_data[indices]
                sort_idx_boot = np.argsort(X_boot)
                X_boot, Y_boot = X_boot[sort_idx_boot], Y_boot[sort_idx_boot]
                
                if len(X_boot) < 3: continue
                try:
                    boot_res = lowess(Y_boot, X_boot, frac=lowess_frac, it=0)
                    unique_x, unique_idx = np.unique(boot_res[:, 0], return_index=True)
                    unique_y = boot_res[unique_idx, 1]
                    if len(unique_x) > 1:
                        y_boot_interp = np.interp(x_smooth, unique_x, unique_y)
                        boot_preds.append(y_boot_interp)
                except: continue

            if len(boot_preds) > 10:
                boot_preds_arr = np.array(boot_preds)
                ci_lower = np.percentile(boot_preds_arr, 2.5, axis=0)
                ci_upper = np.percentile(boot_preds_arr, 97.5, axis=0)
                ax.fill_between(x_smooth, ci_lower, ci_upper, color='salmon', alpha=0.35, 
                              label='95% CI', zorder=1)

        except Exception as e:
            print(f"  - LOWESS Error: {e}")

        # 4. 计算拟合方程 (基于截断数据，R2 仅反映当前范围内的拟合程度)
        best_fit_eq = find_best_fit_equation(X_data, Y_data, max_poly_degree=2)
        
        if best_fit_eq:
            eq_str = format_equation(best_fit_eq)
            r2_y_pos, eq_y_pos, eq_fs = (0.12, 0.03, 18)
            ax.text(0.97, r2_y_pos, f"$R^2$ = {best_fit_eq['r_squared']:.2f}", 
                   transform=ax.transAxes, fontsize=20, fontweight='bold', ha='right', va='bottom')
            if eq_str: 
                ax.text(0.97, eq_y_pos, eq_str, transform=ax.transAxes, fontsize=eq_fs, 
                       fontweight='bold', ha='right', va='bottom')

        # 5. 样式设置
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - y_range * 0.1, ymax + y_range * 0.1)
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{feature_display_name}', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.locator_params(axis='y', nbins=6)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
        ax.set_xlabel('Feature Values', fontsize=22, fontweight='bold')

        if original_feature_name == 'BR': 
            ax.xaxis.set_major_locator(MaxNLocator(3))
        else: 
            ax.locator_params(axis='x', nbins=6)

        if idx % ncols == 0: 
            ax.set_ylabel('SHAP Values', fontsize=22, fontweight='bold')
        else: 
            ax.set_ylabel('')

        if ax.get_legend_handles_labels()[0]: 
            ax.legend(loc='upper left', fontsize=18, frameon=False)
            
        ax.text(0.97, 0.97, f'({chr(97 + idx)})', transform=ax.transAxes, 
               fontsize=22, fontweight='bold', ha='right', va='top')
    
    for idx in range(total_plots, len(axes)): axes[idx].axis('off')
    
    if scatter_plots and total_plots > 0:
        last_row_start_index = min((actual_nrows - 1) * ncols, len(axes) - 1)
        cbar_bottom = axes[last_row_start_index].get_position().y0
        cbar_height = axes[0].get_position().y1 - cbar_bottom
        cbar_ax = fig.add_axes([gs.right + 0.015, cbar_bottom, 0.02, cbar_height])
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Feature Values', rotation=270, labelpad=25, fontsize=22, fontweight='bold')
        cbar.outline.set_visible(False)
        cbar.set_ticks([])
        cbar_ax.text(1.8, 0, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')
        cbar_ax.text(1.8, 1, 'High', ha='center', va='top', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')

    file_basename = os.path.splitext(os.path.basename(csv_file_path))[0]
    pdf_filename = f'SHAP_Dependence_{file_basename}_Type{type_value}.pdf'
    pdf_filepath = os.path.join(output_dir, pdf_filename)
    plt.savefig(pdf_filepath, format='pdf', bbox_inches='tight', dpi=600)
    print(f"\n{TYPE_LABELS[type_value]} 的依赖图已保存至: {pdf_filepath}")
    plt.close('all')

# =============================================================================
# --- 程序主入口 ---
# =============================================================================
if __name__ == "__main__":
    
    csv_file_deltaCd = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\XGBoost_HugeRange_CorrectNames\all_shap_values_and_data_deltaCd.csv"
    csv_file_deltapH = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\XGBoost_HugeRange_CorrectNames\all_shap_values_and_data_deltapH.csv"

    for csv_file, target_name in [(csv_file_deltaCd, 'deltaCd'), (csv_file_deltapH, 'deltapH')]:
        if os.path.exists(csv_file):
            print("\n" + "="*60)
            print(f"正在处理目标变量: {target_name}")
            print("="*60)
            
            for type_val in [1, 2, 3]:
                current_xlims = FEATURE_XLIMS_BY_TYPE.get(type_val, {})
                current_features = FEATURES_BY_TYPE.get(type_val, [])
                
                if current_features:
                    plot_pdp_by_type(
                        csv_file_path=csv_file,
                        type_value=type_val,
                        ncols=4, 
                        feature_xlims=current_xlims,
                        features_to_show=current_features
                    )
        else:
            print(f"\n警告: 文件未找到: {csv_file}")