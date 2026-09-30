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

# 导入 TabPFN
try:
    from tabpfn import TabPFNRegressor
    tabpfn_available = True
except ImportError:
    tabpfn_available = False
    print("错误: 请先安装 tabpfn (pip install tabpfn)")

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 2. 全局绘图参数 (超大字号 + Times New Roman + 加粗)
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
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
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['mathtext.default'] = 'rm'
# ---------------------------------------------------------------------------
# 3. 核心 PDP 绘图与数据导出函数
# ---------------------------------------------------------------------------
def plot_tabpfn_pdp_with_data_save(pipeline, X_test_raw, target_col, output_dir):
    """
    绘制 9 宫格 PDP 热图并导出区间节点采样数据。
    """
    print(f"    - 正在处理 {target_col} 的 PDP 任务 (绘图 + 节点采样导出)...")

    ph_inputs = [5, 6, 7]
    cd_inputs = [1, 2, 3]

    # --- A. 绘图用：高分辨率网格 (100x100) ---
    br_range_plot = np.linspace(0.00, 0.05, 100) 
    mpr_range_plot = np.linspace(0.00, 0.05, 100)
    BR_mesh_plot, MPR_mesh_plot = np.meshgrid(br_range_plot, mpr_range_plot)

    # --- B. 采样用：区间节点网格 (步长 0.5%) ---
    # 0, 0.005, 0.01 ... 0.05
    br_save_points = np.arange(0, 0.055, 0.005)
    mpr_save_points = np.arange(0, 0.055, 0.005)
    BR_save_mesh, MPR_save_mesh = np.meshgrid(br_save_points, mpr_save_points)

    # 其余特征使用中位数作为基准
    baseline = X_test_raw.median()
    fig, axes = plt.subplots(3, 3, figsize=(28, 26))

    is_cd = 'Cd' in target_col
    scale_factor = 100 if is_cd else 1
    unit_label = "(%)" if is_cd else ""

    csv_rows = []

    for i, ph_val in enumerate(ph_inputs):
        for j, cd_val in enumerate(cd_inputs):
            ax = axes[i, j]
            
            # 1. 计算绘图数据
            grid_plot = []
            for r in range(BR_mesh_plot.shape[0]):
                for c in range(BR_mesh_plot.shape[1]):
                    sample = baseline.copy()
                    sample['BR'] = BR_mesh_plot[r, c]
                    sample['MPR'] = MPR_mesh_plot[r, c]
                    sample['pH-S'] = ph_val
                    sample['Cd-C'] = cd_val
                    grid_plot.append(sample)
            
            preds_plot = pipeline.predict(pd.DataFrame(grid_plot))
            Z = preds_plot.reshape(BR_mesh_plot.shape) * scale_factor
            Z_smoothed = gaussian_filter(Z, sigma=1.2)

            # 2. 计算区间节点采样数据
            grid_save = []
            for r in range(BR_save_mesh.shape[0]):
                for c in range(BR_save_mesh.shape[1]):
                    sample = baseline.copy()
                    sample['BR'] = BR_save_mesh[r, c]
                    sample['MPR'] = MPR_save_mesh[r, c]
                    sample['pH-S'] = ph_val
                    sample['Cd-C'] = cd_val
                    grid_save.append(sample)
            
            df_save = pd.DataFrame(grid_save)
            save_preds = pipeline.predict(df_save) * scale_factor
            df_save['Predicted_Value'] = save_preds
            df_save['Target'] = target_col
            csv_rows.append(df_save)

            # 3. 渲染热图
            cp = ax.contourf(BR_mesh_plot * 100, MPR_mesh_plot * 100, Z_smoothed, 
                             levels=25, cmap='RdYlGn_r', alpha=0.9)
            
            ax.set_xticks([0,1, 2, 3, 4, 5])
            ax.set_yticks([0,1, 2, 3, 4, 5])
            
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label(f'{target_col} {unit_label}', fontsize=24, fontweight='bold')
            fmt = '%.2f' if is_cd else '%.3f'
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            
            ax.set_title(f'pH-S={ph_val}, Cd-C={cd_val} $mg·kg^{{-1}}$', fontsize=28, fontweight='bold')
            ax.set_xlabel('BR (%)', fontsize=26, fontweight='bold')
            ax.set_ylabel('MPR (%)', fontsize=26, fontweight='bold')

    # --- 统一保存 ---
    safe_name = target_col.replace("Δ", "delta")
    
    # 保存采样 CSV (9组合 * 121点 = 1089行)
    final_csv_df = pd.concat(csv_rows, ignore_index=True)
    csv_path = os.path.join(output_dir, f'pdp_interval_data_{safe_name}.csv')
    final_csv_df.to_csv(csv_path, index=False)
    
    # 保存 PDF 图像
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, f'pdp_plot_high_res_{safe_name}.png')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"    - 完成！CSV 与 PDF 已同步导出至: {output_dir}")

# ---------------------------------------------------------------------------
# 4. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not tabpfn_available: exit()

    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_dir = r"机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_PDP_Sampling"
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'BET', 'pH-B', 'BR', 'MPs', 'Size', 'MPR']
    target_cols = ['ΔpH', 'ΔCd']
    strata_col = 'type'
    impute_cols = ['OM', 'CEC', 'pH-S', 'BET']
    log_cols = ['CEC']

    # 1. 加载数据与分层抽样
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_cols], 
        test_size=0.3, random_state=42, stratify=df[strata_col]
    )

    # 2. 预处理器 Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_log', Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)), 
                ('log1p', FunctionTransformer(np.log1p))
            ]), [c for c in impute_cols if c in log_cols]),
            ('impute_only', KNNImputer(n_neighbors=5), [c for c in impute_cols if c not in log_cols])
        ], remainder='passthrough'
    )

    # 3. 初始化 TabPFN (CUDA 加速)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- 正在启动 TabPFN (设备: {device}) ---")
    tabpfn_model = TabPFNRegressor(device=device)

    for target in target_cols:
        print(f"\n--- 正在处理目标: {target} ---")
        pipeline = Pipeline([
            ('preprocessor', preprocessor), 
            ('regressor', tabpfn_model)
        ])
        
        pipeline.fit(X_train, y_train[target])
        
        # 打印 R2 指标
        tr_r2 = r2_score(y_train[target], pipeline.predict(X_train))
        te_r2 = r2_score(y_test[target], pipeline.predict(X_test))
        print(f"    >>> 训练集 R²: {tr_r2:.4f} | 测试集 R²: {te_r2:.4f}")

        # 调用集成了导出逻辑的绘图函数
        plot_tabpfn_pdp_with_data_save(pipeline, X_test, target, output_dir)

    print(f"\n[任务全部完成] 输出目录: {output_dir}")