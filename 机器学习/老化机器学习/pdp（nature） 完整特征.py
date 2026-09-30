import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import os
import warnings
from matplotlib.ticker import MaxNLocator
from numpy.polynomial import Polynomial  # 引入更稳定的多项式工具

# --- Global Settings ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 屏蔽拟合不稳定的警告
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

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
    'Cd-C': 'Cd-I',
    'T': 'T',
    'RM': 'RM',
    'T-A': 'T-A',
    'DW-D': 'DW-D',
    'C-30%-40%-D': 'C-L-D',
    'C-100%-D': 'C-H-D',
    'CEC': 'CEC',
    'C-T':'C-T',
    'C-W':'C-W',
    'C-D':'C-D',
}

# --- Core Functions: Model Fitting ---
def find_best_fit_stable(x, y, max_poly_degree=2):
    """
    使用 Polynomial.fit，它比 np.polyfit 更稳健，能自动处理数值量级差。
    """
    n = len(y)
    if n < 4: return None
    best_model = None
    min_aic = float('inf')
    
    # 尝试从 1 阶到 max_poly_degree 的多项式
    for deg in range(1, max_poly_degree + 1):
        try:
            # Polynomial.fit 会自动进行 domain scaling，消除 RankWarning
            p = Polynomial.fit(x, y, deg)
            y_pred = p(x)
            rss = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (rss / ss_tot) if ss_tot > 0 else 0
            aic = n * np.log(rss / n) + 2 * (deg + 1) if rss > 0 else -np.inf
            
            if aic < min_aic:
                min_aic = aic
                best_model = {'name': f'Poly-{deg}', 'model': p, 'r_squared': r_squared, 'degree': deg}
        except: continue
    return best_model

def format_equation_poly(model_info):
    """提取 Polynomial 对象的系数并格式化"""
    p = model_info['model'].convert() # 转换回标准多项式系数 [a, b, c...]
    params = p.coef[::-1] # 反转为 [x^n, ..., x^0]
    deg = model_info['degree']
    try:
        if deg == 1:
            return f"$y = {params[0]:.2g}x {'+' if params[1]>=0 else '-'} {abs(params[1]):.2g}$"
        elif deg == 2:
            return f"$y = {params[0]:.2g}x^2 {'+' if params[1]>=0 else '-'} {abs(params[1]):.2g}x {'+' if params[2]>=0 else '-'} {abs(params[2]):.2g}$"
    except: return ""
    return ""

# --- Main Plotting Function ---
def plot_pdp_with_best_fit(csv_file_path, ncols=4, 
                           n_bootstraps=100, random_seed=42, save_as_pdf=True,
                           max_poly_degree=2, exclude_features=None,
                           feature_name_map=None):
    print(f"Reading file: {csv_file_path}")
    try:
        data = pd.read_csv(csv_file_path, encoding='utf-8')
    except Exception as e:
        print(f"Failed to read file: {e}"); return

    shap_columns = [col for col in data.columns if col.endswith('_SHAP')]
    feature_shap_pairs = []
    exclude_list = exclude_features if exclude_features is not None else []
    
    for shap_col in shap_columns:
        feature_name_raw = shap_col.replace('_SHAP', '')
        if feature_name_raw in data.columns and feature_name_raw not in exclude_list:
            feature_shap_pairs.append((feature_name_raw, shap_col))
    
    if not feature_shap_pairs: return
    
    # --- 动态行数与画布高度设置 ---
    total_plots = len(feature_shap_pairs)
    actual_nrows = (total_plots + ncols - 1) // ncols
    # 高度自适应：每行 5.5 个单位，保证图片不被挤压
    fig = plt.figure(figsize=(25, actual_nrows * 5.5))
    
    gs = GridSpec(actual_nrows, ncols, figure=fig, left=0.05, right=0.88, top=0.94, bottom=0.08, wspace=0.35, hspace=0.5)
    axes = [fig.add_subplot(gs[i, j]) for i in range(actual_nrows) for j in range(ncols)]
    
    np.random.seed(random_seed)
    custom_cmap = LinearSegmentedColormap.from_list("my_custom_cmap", ['#ADD8E6', '#D8BFD8', '#DA70D6', '#FF00FF'])
    scatter_plots = []
    
    for idx, (feature_name_raw, shap_col) in enumerate(feature_shap_pairs):
        ax = axes[idx]
        feature_display_name = feature_name_map.get(feature_name_raw, feature_name_raw)
        valid = np.isfinite(data[feature_name_raw]) & np.isfinite(data[shap_col])
        X_data, Y_data = data.loc[valid, feature_name_raw].values, data.loc[valid, shap_col].values
        
        if len(X_data) < 4:
            ax.set_title(f"{feature_display_name}", fontsize=22, fontweight='bold')
            ax.axis('off'); continue

        # 拟合
        best_fit = find_best_fit_stable(X_data, Y_data, max_poly_degree=max_poly_degree)
        
        # 散点图
        sc = ax.scatter(X_data, Y_data, s=60, c=X_data, cmap=custom_cmap, alpha=0.7, edgecolor='none', zorder=2)
        scatter_plots.append(sc)
        
        if best_fit:
            x_range = np.linspace(X_data.min(), X_data.max(), 200)
            ax.plot(x_range, best_fit['model'](x_range), color='#A52A2A', lw=3, zorder=3)
            
            # Bootstrap 置信区间
            if n_bootstraps > 0:
                boot_preds = []
                for _ in range(n_bootstraps):
                    indices = np.random.choice(len(X_data), len(X_data), replace=True)
                    if len(np.unique(X_data[indices])) > best_fit['degree']:
                        try:
                            p_boot = Polynomial.fit(X_data[indices], Y_data[indices], best_fit['degree'])
                            boot_preds.append(p_boot(x_range))
                        except: continue
                if boot_preds:
                    ax.fill_between(x_range, np.percentile(boot_preds, 2.5, axis=0), 
                                   np.percentile(boot_preds, 97.5, axis=0), color='salmon', alpha=0.35, zorder=1)

            # 标注 R2 和 公式
            ax.text(0.95, 0.15, f"$R^2 = {best_fit['r_squared']:.2f}$", transform=ax.transAxes, fontsize=18, ha='right', fontweight='bold')
            ax.text(0.95, 0.05, format_equation_poly(best_fit), transform=ax.transAxes, fontsize=14, ha='right', fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{feature_display_name}', fontsize=22, fontweight='bold')
        ax.set_xlabel('Feature Values', fontsize=20, fontweight='bold')
        if idx % ncols == 0: ax.set_ylabel('SHAP Values', fontsize=20, fontweight='bold')
        ax.text(0.05, 0.95, f'({chr(97 + idx)})', transform=ax.transAxes, fontsize=20, fontweight='bold', va='top')
        ax.tick_params(labelsize=18)

    # 隐藏多余子图
    for idx in range(total_plots, len(axes)): axes[idx].axis('off')

    # --- 颜色条自适应 ---
    if scatter_plots:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) # 放置在右侧
        cbar = fig.colorbar(scatter_plots[0], cax=cbar_ax)
        cbar.set_label('Feature Value Intensity', fontsize=20, fontweight='bold')
        cbar.set_ticks([]) 
        cbar_ax.text(1.5, 0, 'Low', ha='center', va='bottom', transform=cbar_ax.transAxes, fontsize=18, fontweight='bold')
        cbar_ax.text(1.5, 1, 'High', ha='center', va='top', transform=cbar_ax.transAxes, fontsize=18, fontweight='bold')

    plt.savefig(csv_file_path.replace('.csv', '.png'), bbox_inches='tight', dpi=600)
    print(f"Plot saved: {csv_file_path}")
    plt.close()

if __name__ == "__main__":
    DEGREE_LIMIT = 2
    # 填入您的路径
    files = [
        r"C:\Users\AMDYE\OneDrive\python\机器学习\features_shap_combined_自然老化_pH-S-CH\features_shap_combined_自然老化_pH-S-CH.csv",
        r"C:\Users\AMDYE\OneDrive\python\机器学习\features_shap_combined_自然老化_Cd-CH\features_shap_combined_自然老化_Cd-CH.csv"
    ]
    
    for f in files:
        if os.path.exists(f):
            plot_pdp_with_best_fit(f, ncols=4, max_poly_degree=DEGREE_LIMIT, feature_name_map=feature_name_map)