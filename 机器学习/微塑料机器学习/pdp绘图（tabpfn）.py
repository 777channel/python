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

# ---------------------------------------------------------------------------
# 3. 核心 PDP 绘图函数 (TabPFN 专用 + 自动颜色定标)
# ---------------------------------------------------------------------------
def plot_tabpfn_pdp_auto_scale(pipeline, X_test_raw, target_col, output_dir):
    """
    绘制 9 宫格 TabPFN PDP 热图。
    颜色条范围由系统自定义（自动定标），不再手动设定。
    """
    print(f"    - 正在绘制 {target_col} 的 9 宫格 TabPFN PDP 热图 (自动颜色定标)...")

    # 定义背景特征输入的特定水平
    ph_inputs = [5, 6, 7]
    cd_inputs = [1, 2, 3]

    # 1. 设定固定的计算网格范围 (0% 到 5%)
    br_range = np.linspace(0.00, 0.05, 100) 
    mpr_range = np.linspace(0.00, 0.05, 100)
    BR_mesh, MPR_mesh = np.meshgrid(br_range, mpr_range)

    # 其余特征使用中位数基准
    baseline = X_test_raw.median()
    fig, axes = plt.subplots(3, 3, figsize=(28, 26))

    is_cd = 'Cd' in target_col
    scale_factor = 100 if is_cd else 1
    unit_label = "(%)" if is_cd else ""

    for i, ph_val in enumerate(ph_inputs):
        for j, cd_val in enumerate(cd_inputs):
            ax = axes[i, j]
            
            # 2. 批量构造网格样本以加速并行预测
            grid_points = []
            for r in range(BR_mesh.shape[0]):
                for c in range(BR_mesh.shape[1]):
                    sample = baseline.copy()
                    sample['BR'] = BR_mesh[r, c]
                    sample['MPR'] = MPR_mesh[r, c]
                    sample['pH-S'] = ph_val
                    sample['Cd-C'] = cd_val
                    grid_points.append(sample)
            
            # 调用 TabPFN 进行 CUDA 加速预测
            preds = pipeline.predict(pd.DataFrame(grid_points))
            Z = preds.reshape(BR_mesh.shape) * scale_factor

            # 3. 应用高斯平滑使图像更圆润
            Z_smoothed = gaussian_filter(Z, sigma=1.2)

            # 4. 绘图 (不再设定 vmin/vmax，让系统自定义颜色范围)
            cp = ax.contourf(BR_mesh * 100, MPR_mesh * 100, Z_smoothed, 
                             levels=25, cmap='RdYlGn_r', alpha=0.9)
            
            # 颜色条格式化
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label(f'Pred {target_col} {unit_label}', fontsize=24, fontweight='bold')
            fmt = '%.2f' if is_cd else '%.3f'
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(24)
                t.set_fontweight('bold')
            
            # 标题与轴标签
            ax.set_title(f'pH-S={ph_val}, Cd-C={cd_val}', fontsize=28, fontweight='bold')
            ax.set_xlabel('Biochar Rate (%)', fontsize=26, fontweight='bold')
            ax.set_ylabel('Microplastic Rate (%)', fontsize=26, fontweight='bold')

    plt.tight_layout()
    safe_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'pdp_tabpfn_auto_{safe_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 图像已保存至: {save_path}")

# ---------------------------------------------------------------------------
# 4. 主程序入口 (TabPFN + CUDA 加速 + 分层抽样)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not tabpfn_available:
        exit()

    # --- 统一变量定义 ---
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_dir = r"机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_Final"
    os.makedirs(output_dir, exist_ok=True)

    feature_cols = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'BET', 'pH-B', 'BR', 'MPs', 'Size', 'MPR']
    target_cols = ['ΔpH', 'ΔCd']
    strata_col = 'type'
    impute_cols = ['OM', 'CEC', 'pH-S', 'BET']
    log_cols = ['CEC']

    # 1. 加载数据与执行分层抽样逻辑
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
    print(f"--- 正在初始化 TabPFN (运行设备: {device}) ---")
    tabpfn_model = TabPFNRegressor(device=device)

    # 4. 循环训练并执行分析
    for target in target_cols:
        print(f"\n--- 正在处理目标: {target} ---")
        
        # 封装模型
        pipeline = Pipeline([
            ('preprocessor', preprocessor), 
            ('regressor', tabpfn_model)
        ])
        
        # 训练 TabPFN (TabPFN 作为零样本模型，训练过程极快)
        pipeline.fit(X_train, y_train[target])
        
        # 调用自动定标的绘图函数
        plot_tabpfn_pdp_auto_scale(pipeline, X_test, target, output_dir)

    print(f"\n[任务全部完成] TabPFN PDP 图像已生成。输出目录: {output_dir}")