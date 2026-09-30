# ---------------------------------------------------------------------------
# 按 Type 分类的 SHAP 依赖图绘制脚本
# 视觉：LOWESS 曲线 + Bootstrapping 置信区间
# 文字：多项式/对数拟合的近似方程 + R2
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
import statsmodels.api as sm # 引入 statsmodels 用于 LOWESS

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
# --- [用户指定] X轴范围配置区域 ---
# =============================================================================
FEATURE_XLIMS_BY_TYPE = {
    # --- Type 1: 单独微塑料 + 土壤 ---
    1: {
        'MPR': (0, 0.025),       
        'Size': (0, 250),
        'Cd-C': (0, 60),
        'pH-S': (5, 7.5)
        # 'pH-S': (4, 9)
    },
    
    # --- Type 2: 单独生物炭 + 土壤 ---
    2: {
        'BET': (0, 20),       
        # 'T': (300, 700),
        # 'pH-S': (5, 8)         
    },
    
    # --- Type 3: 复合施加 ---
    3: {
        'MPR': (0, 0.055),       
        # 'Size': (0, 1000),
        # 'BR': (0, 5),
        # 'pH-S': (4, 10)
    }
}

# --- 特征分组定义 ---
SOIL_FEATURES = ['pH-S', 'OM', 'CEC', 'Cd-C']
BIOCHAR_FEATURES = ['RM', 'T', 'pH-B', 'BR','BET']
MP_FEATURES = ['MPs', 'Size', 'MPR']

TYPE_FEATURE_MAP = {
    1: MP_FEATURES + SOIL_FEATURES,        
    2: BIOCHAR_FEATURES + SOIL_FEATURES,   
    3: MP_FEATURES + BIOCHAR_FEATURES + SOIL_FEATURES 
}

TYPE_LABELS = {
    1: 'Type 1 (MPs Only)',
    2: 'Type 2 (Biochar Only)',
    3: 'Type 3 (Combined)'
}

# =============================================================================
# --- 辅助函数: 计算近似方程 (仅用于显示文字) ---
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
def plot_pdp_by_type(csv_file_path, type_value, ncols=4, feature_xlims=None): 
    print(f"\n{'='*60}")
    print(f"正在为 {TYPE_LABELS[type_value]} 绘制 SHAP 依赖图...")
    
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except Exception as e:
        print(f"读取文件失败: {e}"); return
    
    data_filtered = data[data['type_Value'] == type_value].copy()
    if len(data_filtered) < 5: return
    
    features_to_plot = TYPE_FEATURE_MAP[type_value]
    feature_shap_pairs = []
    for feature in features_to_plot:
        v_col, s_col = f"{feature}_Value", f"{feature}_SHAP"
        if v_col in data_filtered.columns and s_col in data_filtered.columns:
            if (data_filtered[v_col] != 0).any():
                feature_shap_pairs.append((v_col, s_col, feature))
    
    if not feature_shap_pairs: return
    
    output_dir = os.path.dirname(csv_file_path) or '.'
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", ['#ADD8E6', '#D8BFD8', '#DA70D6', '#FF00FF'])
    total_plots = len(feature_shap_pairs)
    actual_nrows = (total_plots + ncols - 1) // ncols
    fig = plt.figure(figsize=(25, 6 * actual_nrows))
    gs = GridSpec(actual_nrows, ncols, figure=fig, left=0.04, right=0.85, top=0.92, bottom=0.12, wspace=0.35, hspace=0.55)
    axes = [fig.add_subplot(gs[i, j]) for i in range(actual_nrows) for j in range(ncols)]
    scatter_plots = []
    
    lowess = sm.nonparametric.lowess
    lowess_frac = 0.88
    n_bootstraps = 1000
    np.random.seed(42)

    for idx, (value_col, shap_col, original_feature_name) in enumerate(feature_shap_pairs):
        ax = axes[idx]
        feature_display_name = feature_name_map.get(original_feature_name, original_feature_name)

        valid_indices = data_filtered[value_col].notna() & data_filtered[shap_col].notna()
        X_raw = data_filtered.loc[valid_indices, value_col].values
        Y_raw = data_filtered.loc[valid_indices, shap_col].values
        
        # --- 核心修改：判断是否指定了范围，否则使用全部数据 ---
        if feature_xlims and original_feature_name in feature_xlims:
            x_min_limit, x_max_limit = feature_xlims[original_feature_name]
            mask = (X_raw >= x_min_limit) & (X_raw <= x_max_limit)
            X_fit, Y_fit = X_raw[mask], Y_raw[mask]
            x_plot_min, x_plot_max = x_min_limit, x_max_limit
        else:
            X_fit, Y_fit = X_raw, Y_raw
            x_plot_min, x_plot_max = np.min(X_raw), np.max(X_raw)
        # --------------------------------------------------

        # 1. 绘制基础散点 (绘制全部 X_raw，但会被后续的 set_xlim 裁剪)
        scatter_plot = ax.scatter(X_raw, Y_raw, s=60, c=X_raw, cmap=custom_cmap, 
                                  alpha=0.7, edgecolor='none', zorder=2)
        scatter_plots.append(scatter_plot)

        # 确保拟合数据点足够
        if len(X_fit) >= 3:
            # 拟合线绘制的采样点范围严格限制在 X_fit 范围内
            x_smooth = np.linspace(X_fit.min(), X_fit.max(), 200)

            # 2. 计算 LOWESS 曲线 (严格基于 X_fit)
            try:
                sort_idx = np.argsort(X_fit)
                X_s, Y_s = X_fit[sort_idx], Y_fit[sort_idx]
                smooth_res = lowess(Y_s, X_s, frac=lowess_frac, it=0)
                ux, ui = np.unique(smooth_res[:, 0], return_index=True)
                y_lowess_main = np.interp(x_smooth, ux, smooth_res[ui, 1])
                ax.plot(x_smooth, y_lowess_main, color='#A52A2A', lw=3, label='LOWESS', zorder=3)
                
                # Bootstrap CI (严格基于 X_fit)
                boot_preds = []
                for _ in range(n_bootstraps):
                    b_idx = np.random.choice(len(X_fit), len(X_fit), replace=True)
                    X_b, Y_b = X_fit[b_idx], Y_fit[b_idx]
                    if len(np.unique(X_b)) < 3: continue
                    sb = np.argsort(X_b)
                    rb = lowess(Y_b[sb], X_b[sb], frac=lowess_frac, it=0)
                    bux, bui = np.unique(rb[:, 0], return_index=True)
                    boot_preds.append(np.interp(x_smooth, bux, rb[bui, 1]))

                if len(boot_preds) > 10:
                    preds_arr = np.array(boot_preds)
                    ax.fill_between(x_smooth, np.percentile(preds_arr, 2.5, axis=0), 
                                    np.percentile(preds_arr, 97.5, axis=0), color='salmon', alpha=0.35, zorder=1)
            except Exception as e:
                print(f"  - Fitting Error for {feature_display_name}: {e}")

            # 3. 近似方程 (严格基于 X_fit)
            best_fit_eq = find_best_fit_equation(X_fit, Y_fit, max_poly_degree=2)
            if best_fit_eq:
                ax.text(0.97, 0.12, f"$R^2$ = {best_fit_eq['r_squared']:.2f}", 
                        transform=ax.transAxes, fontsize=20, fontweight='bold', ha='right', va='bottom')
                eq_str = format_equation(best_fit_eq)
                if eq_str: ax.text(0.97, 0.03, eq_str, transform=ax.transAxes, fontsize=18, 
                                   fontweight='bold', ha='right', va='bottom')

        # 4. 设置轴样式和范围
        ax.set_xlim(x_plot_min, x_plot_max)
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax > ymin else 1.0
        ax.set_ylim(ymin - y_range * 0.1, ymax + y_range * 0.1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(feature_display_name, fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.set_xlabel('Feature Values', fontsize=22, fontweight='bold')
        if idx % ncols == 0: ax.set_ylabel('SHAP Values', fontsize=22, fontweight='bold')
        if ax.get_legend_handles_labels()[0]: ax.legend(loc='upper left', fontsize=18, frameon=False)
        ax.text(0.97, 0.97, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=22, fontweight='bold', ha='right', va='top')
    
    for idx in range(total_plots, len(axes)): axes[idx].axis('off')
    
    # 颜色条和保存逻辑
    if scatter_plots and total_plots > 0:
        last_row_start_index = min((actual_nrows - 1) * ncols, len(axes) - 1)
        cbar_bottom = axes[last_row_start_index].get_position().y0
        cbar_height = axes[0].get_position().y1 - cbar_bottom
        cbar_ax = fig.add_axes([gs.right + 0.015, cbar_bottom, 0.02, cbar_height])
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax)
        cbar.set_label('Feature Values', rotation=270, labelpad=25, fontsize=22, fontweight='bold')
        cbar.outline.set_visible(False); cbar.set_ticks([])
        cbar_ax.text(1.8, 0, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')
        cbar_ax.text(1.8, 1, 'High', ha='center', va='top', transform=cbar_ax.transAxes, fontsize=22, fontweight='bold')

    file_basename = os.path.splitext(os.path.basename(csv_file_path))[0]
    pdf_filepath = os.path.join(output_dir, f'SHAP_Dependence_{file_basename}_Type{type_value}.pdf')
    plt.savefig(pdf_filepath, format='pdf', bbox_inches='tight', dpi=600)
    plt.close('all')

if __name__ == "__main__":
    csv_file_deltaCd = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\XGBoost_HugeRange_CorrectNames\all_shap_values_and_data_deltaCd.csv"
    csv_file_deltapH = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出\XGBoost_HugeRange_CorrectNames\all_shap_values_and_data_deltapH.csv"
    for csv_file, target_name in [(csv_file_deltaCd, 'deltaCd'), (csv_file_deltapH, 'deltapH')]:
        if os.path.exists(csv_file):
            for type_val in [1, 2, 3]:
                current_xlims = FEATURE_XLIMS_BY_TYPE.get(type_val, {})
                plot_pdp_by_type(csv_file, type_val, ncols=4, feature_xlims=current_xlims)