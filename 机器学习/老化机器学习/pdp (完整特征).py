import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import os
import warnings
from matplotlib.ticker import MaxNLocator

# --- Global Settings ---
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.default'] = 'regular'


# --- Feature Name Mapping ---
feature_name_map = {
    'BR': 'BR',
    'pH-B': 'pH-B',
    'BET': 'BET',
    'FT-D': 'FT-D',
    'pH-S': 'pH-S',
    'Cd-C': 'Cd-C',
    'T': 'T',
    'RM': 'RM',
    'DW-D': 'DW-D',
    'CEC': 'CEC',
    'C-T':'C-T',
    'C-W':'C-W',
    'C-D':'C-D',
}

# --- 定义特征顺序 ---
FEATURE_ORDER = {
    'Cd-CH': ['pH-S', 'pH-B', 'T', 'BET', 'C-D', 'C-W', 'C-T', 'Cd-C', 'CEC', 'RM', 'FT-D', 'BR', 'DW-D'],
    'pH-S-CH': ['pH-S', 'C-T', 'pH-B', 'CEC', 'BET', 'BR', 'RM', 'C-W', 'Cd-C', 'T', 'C-D', 'FT-D', 'DW-D']
}

# --- Core Functions: Model Fitting and Selection ---

def find_best_fit(x, y, max_poly_degree=3):
    n = len(y)
    if n < 4: return None

    candidate_models = {}
    if max_poly_degree >= 1:
        candidate_models['Linear'] = {'func': lambda x, a, b: a * x + b, 'p0': [1, 0], 'k': 2}
    if max_poly_degree >= 2:
        candidate_models['Quadratic'] = {'func': lambda x, a, b, c: a * x**2 + b * x + c, 'p0': [1, 1, 0], 'k': 3}
    if max_poly_degree >= 3:
        candidate_models['Cubic'] = {'func': lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 'p0': [1, 1, 1, 0], 'k': 4}
    
    if np.all(x > 0) and np.min(x) > 1e-6:
        candidate_models['Logarithmic'] = {'func': lambda x, a, b: a * np.log(x) + b, 'p0': [1, 0], 'k': 2}

    best_model = None
    min_aic = float('inf')

    for name, model_info in candidate_models.items():
        try:
            if name in ['Linear', 'Quadratic', 'Cubic']:
                degree = model_info['k'] - 1
                params = np.polyfit(x, y, degree)
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
                best_model = {
                    'name': name, 'params': params, 'r_squared': r_squared, 'aic': aic,
                    'func': model_info.get('func'), 'degree': model_info.get('k', 0) - 1
                }
        except (RuntimeError, np.linalg.LinAlgError):
            continue

    return best_model

def format_equation(model_info):
    name = model_info['name']
    params = model_info['params']
    try:
        if name == 'Linear':
            sign = '+' if params[1] >= 0 else '-'
            return f"$y = {params[0]:.2g}x {sign} {abs(params[1]):.2g}$"
        elif name == 'Quadratic':
            b_sign = '+' if params[1] >= 0 else '-'
            c_sign = '+' if params[2] >= 0 else '-'
            return f"$y = {params[0]:.2g}x^2 {b_sign} {abs(params[1]):.2g}x {c_sign} {abs(params[2]):.2g}$"
        elif name == 'Cubic':
            b_sign = '+' if params[1] >= 0 else '-'
            c_sign = '+' if params[2] >= 0 else '-'
            d_sign = '+' if params[3] >= 0 else '-'
            line1 = f"$y = {params[0]:.2g}x^3 {b_sign} {abs(params[1]):.2g}x^2$"
            line2 = f"${c_sign} {abs(params[2]):.2g}x {d_sign} {abs(params[3]):.2g}$"
            return f"{line1}\n{line2}"
        elif name == 'Logarithmic':
            sign = '+' if params[1] >= 0 else '-'
            return f"$y = {params[0]:.2g} \\ln(x) {sign} {abs(params[1]):.2g}$"
    except (IndexError, TypeError):
        return ""
    return ""

def detect_target_from_path(csv_file_path):
    if 'Cd-CH' in csv_file_path:
        return 'Cd-CH'
    elif 'pH-S-CH' in csv_file_path:
        return 'pH-S-CH'
    else:
        return None

# --- Main Plotting Function ---
def plot_pdp_with_best_fit(csv_file_path, ncols=4, nrows=3, 
                           n_bootstraps=100, random_seed=42, save_as_pdf=True,
                           max_poly_degree=3, exclude_features=None,
                           feature_name_map=None):
    print(f"Reading file: {csv_file_path}")
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except Exception as e:
        print(f"Failed to read file: {e}"); return

    target_type = detect_target_from_path(csv_file_path)
    if target_type is None:
        print("Warning: Cannot detect target type from file path. Using default ordering.")
        target_type = 'Cd-CH'
    
    print(f"Detected target type: {target_type}")
    
    # 【修改处1】：去掉 [:12] 切片，获取全部特征顺序
    desired_order = FEATURE_ORDER.get(target_type, FEATURE_ORDER['Cd-CH'])
    print(f"Using feature order: {desired_order}")

    shap_columns = [col for col in data.columns if col.endswith('_SHAP')]
    all_feature_shap_pairs = []
    
    exclude_list = exclude_features if exclude_features is not None else []
    
    for shap_col in shap_columns:
        feature_name_raw = shap_col.replace('_SHAP', '')
        if feature_name_raw in data.columns and feature_name_raw not in exclude_list:
            all_feature_shap_pairs.append((feature_name_raw, shap_col))
    
    # 【修改处2】：根据完整顺序重新排列特征对
    feature_shap_pairs = []
    for feature_name in desired_order:
        for feature_raw, shap_col in all_feature_shap_pairs:
            if feature_raw == feature_name:
                feature_shap_pairs.append((feature_raw, shap_col))
                break
    
    if not feature_shap_pairs: 
        print("Error: No valid feature-SHAP pairs found"); 
        return
    
    # 【修改处3】：更新打印提示信息
    print(f"Found {len(feature_shap_pairs)} feature-SHAP pairs to plot. Max polynomial degree set to: {max_poly_degree}")
    print("Features in order:", [pair[0] for pair in feature_shap_pairs])
    
    output_dir = os.path.dirname(csv_file_path) if os.path.dirname(csv_file_path) else '.'
    np.random.seed(random_seed)
    colors_for_cmap = ['#ADD8E6', '#D8BFD8', '#DA70D6', '#FF00FF']
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", colors_for_cmap)
    
    total_plots = len(feature_shap_pairs)
    # 自动计算行数
    actual_nrows = (total_plots + ncols - 1) // ncols
    
    # 根据实际行数调整画布高度 (如果特征很多，建议将 18 调大，例如 actual_nrows * 6)
    fig = plt.figure(figsize=(25, actual_nrows * 5.5))
    
    gs = GridSpec(actual_nrows, ncols, figure=fig, left=0.04, right=0.85, top=0.94, bottom=0.08, wspace=0.35, hspace=0.45)
    axes = [fig.add_subplot(gs[i, j]) for i in range(actual_nrows) for j in range(ncols)]
    
    scatter_plots = []
    
    for idx, (feature_name_raw, shap_col) in enumerate(feature_shap_pairs):
        ax = axes[idx]
        feature_display_name = feature_name_map.get(feature_name_raw, feature_name_raw)

        valid_indices = np.isfinite(data[feature_name_raw]) & np.isfinite(data[shap_col])
        X_data, Y_data = data.loc[valid_indices, feature_name_raw].values, data.loc[valid_indices, shap_col].values
        
        if len(X_data) < 4:
            ax.set_title(f"{feature_display_name}", fontsize=22, fontweight='bold')
            ax.axis('off')
            continue

        best_fit = find_best_fit(X_data, Y_data, max_poly_degree=max_poly_degree)
        
        current_vmin, current_vmax = np.min(X_data), np.max(X_data)
        scatter_plot = ax.scatter(X_data, Y_data, s=60, c=X_data, cmap=custom_cmap, 
                                  alpha=0.7, edgecolor='none', zorder=2, vmin=current_vmin, vmax=current_vmax)
        scatter_plots.append(scatter_plot)
        
        if best_fit:
            x_fit = np.linspace(current_vmin, current_vmax, 200)
            if best_fit['name'] in ['Linear', 'Quadratic', 'Cubic']:
                y_fit = np.polyval(best_fit['params'], x_fit)
            else:
                y_fit = best_fit['func'](x_fit, *best_fit['params'])
            
            ax.plot(x_fit, y_fit, color='#A52A2A', linestyle='-', linewidth=3, label='Trendline', zorder=3)
            
            plot_ci = False
            if n_bootstraps > 0:
                x_eval = np.linspace(current_vmin, current_vmax, 100)
                boot_preds = []
                for i in range(n_bootstraps):
                    indices = np.random.choice(len(X_data), len(X_data), replace=True)
                    X_boot, Y_boot = X_data[indices], Y_data[indices]
                    try:
                        if best_fit['name'] in ['Linear', 'Quadratic', 'Cubic']:
                            if len(np.unique(X_boot)) < best_fit['degree'] + 1: continue
                            boot_params = np.polyfit(X_boot, Y_boot, best_fit['degree'])
                            boot_preds.append(np.polyval(boot_params, x_eval))
                        elif best_fit['name'] == 'Logarithmic':
                            if not np.all(X_boot > 0): continue
                            boot_params, _ = curve_fit(best_fit['func'], X_boot, Y_boot, p0=best_fit['params'], maxfev=5000)
                            boot_preds.append(best_fit['func'](x_eval, *boot_params))
                    except (RuntimeError, np.linalg.LinAlgError):
                        continue
                if len(boot_preds) > 10:
                    boot_preds_array = np.array(boot_preds)
                    ci_lower = np.percentile(boot_preds_array, 2.5, axis=0)
                    ci_upper = np.percentile(boot_preds_array, 97.5, axis=0)
                    plot_ci = True

            if plot_ci:
                ax.fill_between(x_eval, ci_lower, ci_upper, color='salmon', alpha=0.35, label='95% CI', zorder=1)

            current_ymin, current_ymax = ax.get_ylim()
            y_range = current_ymax - current_ymin
            if y_range == 0: y_range = 1.0
            new_ymin = current_ymin - y_range * 0.30
            new_ymax = current_ymax + y_range * 0.05
            ax.set_ylim(new_ymin, new_ymax)

            equation_str = format_equation(best_fit)
            if best_fit['name'] == 'Cubic':
                r2_y_pos, eq_y_pos, eq_fontsize = 0.20, 0.03, 16
            else:
                r2_y_pos, eq_y_pos, eq_fontsize = 0.12, 0.03, 18

            ax.text(0.97, r2_y_pos, f"$R^2$ = {best_fit['r_squared']:.2f}", 
                    transform=ax.transAxes, fontsize=20, fontweight='bold',
                    horizontalalignment='right', verticalalignment='bottom')
            if equation_str:
                ax.text(0.97, eq_y_pos, equation_str,
                        transform=ax.transAxes, fontsize=eq_fontsize, fontweight='bold',
                        horizontalalignment='right', verticalalignment='bottom')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{feature_display_name}', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.locator_params(axis='y', nbins=6)
        for spine in ax.spines.values(): spine.set_linewidth(1.2)
        
        ax.set_xlabel('Feature Values', fontsize=22, fontweight='bold')

        if feature_name_raw == 'BR':
            ax.xaxis.set_major_locator(MaxNLocator(3))
        else:
            ax.locator_params(axis='x', nbins=6)

        if idx % ncols == 0:
            ax.set_ylabel('SHAP Values', fontsize=22, fontweight='bold')
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=True) 

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper left', fontsize=18, frameon=False)
        
        ax.text(0.97, 0.97, f'({chr(97 + idx)})',
                transform=ax.transAxes, fontsize=22, fontweight='bold',
                horizontalalignment='right', verticalalignment='top')
        
        print(f"Completed: {feature_name_raw}, Best fit model: {best_fit['name'] if best_fit else 'None'}")

    for idx in range(total_plots, len(axes)): 
        axes[idx].axis('off')
    
    # --- 颜色条逻辑 ---
    if scatter_plots:
        first_ax_pos = axes[0].get_position()
        last_ax_pos = axes[(actual_nrows-1) * ncols].get_position()
        
        cbar_bottom = last_ax_pos.y0 
        cbar_top = first_ax_pos.y1
        cbar_height = cbar_top - cbar_bottom

        cbar_ax = fig.add_axes([gs.right + 0.01, cbar_bottom, 0.02, cbar_height])
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax, orientation='vertical')
        cbar.set_label('Feature Values', rotation=270, labelpad=20, fontsize=22, fontweight='bold')
        cbar.outline.set_visible(False)
        cbar.set_ticks([]) 
        
        cbar_ax.text(1.7, 0, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes, 
                     fontsize=22, fontweight='bold', color='black')
        cbar_ax.text(1.7, 1, 'High', ha='center', va='top', transform=cbar_ax.transAxes, 
                     fontsize=22, fontweight='bold', color='black')

    file_basename = os.path.splitext(os.path.basename(csv_file_path))[0]
    
    if save_as_pdf:
        pdf_filename = f'SHAP依赖图完整特征_{file_basename}.png'
        pdf_filepath = os.path.join(output_dir, pdf_filename)
        plt.savefig(pdf_filepath, format='png', bbox_inches='tight', dpi=600)
        print(f"\nPDP plot saved: {pdf_filepath}")
    
    plt.close()

if __name__ == "__main__":
    DEGREE_LIMIT = 2

    csv_file_ph = r'C:\Users\AMDYE\OneDrive\python\机器学习\features_shap_combined_Cd-CH\features_shap_combined_Cd-CH.csv'
    if os.path.exists(csv_file_ph):
        print("=== Analyzing pH-S-CH Data ===")
        plot_pdp_with_best_fit(
            csv_file_path=csv_file_ph,
            ncols=4, nrows=3, n_bootstraps=100, save_as_pdf=True,
            max_poly_degree=DEGREE_LIMIT,
            feature_name_map=feature_name_map
        )
    
    csv_file_cd = r'C:\Users\AMDYE\OneDrive\python\机器学习\features_shap_combined_pH-S-CH\features_shap_combined_pH-S-CH.csv'
    if os.path.exists(csv_file_cd):
        print("=== Analyzing Cd-CH Data ===")
        plot_pdp_with_best_fit(
            csv_file_path=csv_file_cd,
            ncols=4, nrows=3, n_bootstraps=100, save_as_pdf=True,
            max_poly_degree=DEGREE_LIMIT,
            feature_name_map=feature_name_map
        )