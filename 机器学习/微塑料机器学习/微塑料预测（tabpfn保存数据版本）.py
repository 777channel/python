# ---------------------------------------------------------------------------
# 1. 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import warnings
from scipy.ndimage import gaussian_filter

# 使用标准 TabPFNRegressor
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
except ImportError:
    tabpfn_available = False
    print("错误: 未找到 tabpfn 库，请运行 pip install tabpfn")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 2. 全局绘图参数
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 32
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.default'] = 'rm'
# ---------------------------------------------------------------------------
# 3. 核心函数：高精绘图 + 区间节点数据导出
# ---------------------------------------------------------------------------
def plot_and_save_interval_data(pipeline, target_col, output_dir):
    """
    1. 绘制 100x100 高精 PDF 热力图。
    2. 按区间节点（MPR步长0.5%, Size步长50）采样导出 CSV。
    """
    print(f"    - 正在处理 {target_col}: 绘图(100x100) + 节点采样导出...")
    
    fixed_params = {
        'OM': 15, 'CEC': 15, 'type': 1, 'RM': 0, 
        'T': 0, 'BET': 0, 'pH-B': 0, 'BR': 0, 'MPs': 1
    }
    ph_s_values = [5, 6, 7]
    cd_c_values = [1, 2, 3]

    # --- A. 绘图配置 (保持高分辨率 100x100) ---
    mpr_range_plot = np.linspace(0.00, 0.05, 100) 
    size_range_plot = np.linspace(0, 500, 100)
    MPR_mesh_plot, Size_mesh_plot = np.meshgrid(mpr_range_plot, size_range_plot)

    # --- B. 采样配置 (区间节点采样) ---
    # MPR: 0, 0.005, 0.01 ... 0.05 (对应 0%, 0.5%, 1%...5%)
    mpr_save_points = np.arange(0, 0.055, 0.005) 
    # Size: 0, 50, 100, 150 ... 500 (每个 100um 区间的开头和中间)
    size_save_points = np.arange(0, 550, 50)
    MPR_save_mesh, Size_save_mesh = np.meshgrid(mpr_save_points, size_save_points)

    is_cd = 'Cd' in target_col
    scale_factor = 100 if is_cd else 1
    unit_label = "(%)" if is_cd else ""

    csv_rows = []
    fig, axes = plt.subplots(3, 3, figsize=(28, 26))

    for i, ph_val in enumerate(ph_s_values):
        for j, cd_val in enumerate(cd_c_values):
            ax = axes[i, j]
            
            # 1. 计算高精绘图数据
            grid_plot = []
            for r in range(MPR_mesh_plot.shape[0]):
                for c in range(MPR_mesh_plot.shape[1]):
                    s = fixed_params.copy()
                    s['MPR'] = MPR_mesh_plot[r, c]
                    s['Size'] = Size_mesh_plot[r, c]
                    s['pH-S'] = ph_val
                    s['Cd-C'] = cd_val
                    grid_plot.append(s)
            
            preds_plot = pipeline.predict(pd.DataFrame(grid_plot))
            Z = preds_plot.reshape(MPR_mesh_plot.shape) * scale_factor

            # 2. 计算区间节点采样数据用于导出
            grid_save = []
            for r in range(MPR_save_mesh.shape[0]):
                for c in range(MPR_save_mesh.shape[1]):
                    s = fixed_params.copy()
                    s['MPR'] = MPR_save_mesh[r, c]
                    s['Size'] = Size_save_mesh[r, c]
                    s['pH-S'] = ph_val
                    s['Cd-C'] = cd_val
                    grid_save.append(s)
            
            df_save = pd.DataFrame(grid_save)
            save_preds = pipeline.predict(df_save) * scale_factor
            df_save['Predicted_Value'] = save_preds
            df_save['Target'] = target_col
            csv_rows.append(df_save)

            # 3. 绘图渲染
            Z_smoothed = gaussian_filter(Z, sigma=1.2)
            cp = ax.contourf(MPR_mesh_plot * 100, Size_mesh_plot, Z_smoothed, 
                             levels=25, cmap='plasma', alpha=0.9)
            
            ax.set_xticks([1, 2, 3, 4, 5])
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label(f'Pred {target_col} {unit_label}', fontsize=24, fontweight='bold')
            # 颜色轴显示一位小数
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            
            ax.set_title(f'pH-S={ph_val}, Cd-C={cd_val}', fontsize=28, fontweight='bold')
            ax.set_xlabel('MPR (%)', fontsize=26, fontweight='bold')
            ax.set_ylabel('Size (μm)', fontsize=26, fontweight='bold')

    # --- 统一保存 ---
    safe_name = target_col.replace("Δ", "delta")
    
    # 保存 CSV (包含各区间节点数据)
    final_csv_df = pd.concat(csv_rows, ignore_index=True)
    csv_path = os.path.join(output_dir, f'interval_node_data_{safe_name}.csv')
    final_csv_df.to_csv(csv_path, index=False)
    
    # 保存高精 PDF
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'high_res_plot_{safe_name}.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 完成！CSV(1089行) 与 PDF 已保存在同一路径。")

# ---------------------------------------------------------------------------
# 4. 主程序 (TabPFN + CUDA + R2)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not tabpfn_available: exit()

    # 路径配置
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_dir = r"机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_Interval_Sampling"
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'BET', 'pH-B', 'BR', 'MPs', 'Size', 'MPR']
    target_cols = ['ΔpH', 'ΔCd']
    strata_col = 'type'
    impute_cols = ['OM', 'CEC', 'pH-S', 'BET']
    log_cols = ['CEC']

    # 1. 数据处理
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_cols], 
        test_size=0.3, random_state=42, stratify=df[strata_col]
    )

    # 2. 预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)), 
                ('log1p', FunctionTransformer(np.log1p))
            ]), [c for c in impute_cols if c in log_cols]),
            ('impute_only', KNNImputer(n_neighbors=5), [c for c in impute_cols if c not in log_cols])
        ], remainder='passthrough'
    )

    # 3. 初始化标准 TabPFN
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- 正在启动标准 TabPFN (设备: {device}) ---")
    tabpfn_model = TabPFNRegressor(device=device)

    for target in target_cols:
        print(f"\n处理目标: {target}")
        pipeline = Pipeline([('preprocessor', preprocessor), ('regressor', tabpfn_model)])
        pipeline.fit(X_train, y_train[target])
        
        # 显式定义并打印 R2
        train_r2 = r2_score(y_train[target], pipeline.predict(X_train))
        test_r2 = r2_score(y_test[target], pipeline.predict(X_test))
        print(f"    >>> 训练集 R²: {train_r2:.4f} | 测试集 R²: {test_r2:.4f}")

        # 执行绘图与区间采样导出
        plot_and_save_interval_data(pipeline, target, output_dir)

    print(f"\n任务全部执行完毕。输出目录: {output_dir}")