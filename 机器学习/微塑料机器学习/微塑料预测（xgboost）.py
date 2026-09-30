# ---------------------------------------------------------------------------
# 1. 导入所需库
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
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
import torch
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
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# ---------------------------------------------------------------------------
# 3. 场景预测绘图函数 (MPR x Size, plasma 色系, 高斯平滑)
# ---------------------------------------------------------------------------
def plot_xgb_scenario_prediction(pipeline, target_col, output_dir):
    """
    根据特定参数输入进行全量预测。
    X轴: MPR (0-5%), Y轴: Size (0-200μm), MPs 固定为 1。
    """
    print(f"    - 正在生成 {target_col} 的 9 宫格预测图 (XGBoost + 平滑)...")
    
    # 您指定的固定输入参数
    fixed_params = {
        'OM': 15, 'CEC': 15, 'type': 1, 'RM': 0, 
        'T': 0, 'BET': 0, 'pH-B': 0, 'BR': 0, 'MPs': 1
    }
    ph_s_values = [5, 6, 7]
    cd_c_values = [1, 2, 3]

    # 1. 设定网格分辨率 (100x100)
    mpr_range = np.linspace(0.00, 0.05, 100) 
    size_range = np.linspace(0, 200, 100)
    MPR_mesh, Size_mesh = np.meshgrid(mpr_range, size_range)

    is_cd = 'Cd' in target_col
    scale_factor = 100 if is_cd else 1
    unit_label = "(%)" if is_cd else ""

    fig, axes = plt.subplots(3, 3, figsize=(28, 26))

    for i, ph_val in enumerate(ph_s_values):
        for j, cd_val in enumerate(cd_c_values):
            ax = axes[i, j]
            
            # 2. 构造预测网格样本
            grid_points = []
            for r in range(MPR_mesh.shape[0]):
                for c in range(MPR_mesh.shape[1]):
                    sample = fixed_params.copy()
                    sample['MPR'] = MPR_mesh[r, c]
                    sample['Size'] = Size_mesh[r, c]
                    sample['pH-S'] = ph_val
                    sample['Cd-C'] = cd_val
                    grid_points.append(sample)
            
            # 批量预测
            preds = pipeline.predict(pd.DataFrame(grid_points))
            Z = preds.reshape(MPR_mesh.shape) * scale_factor

            # 3. 应用高斯平滑消除树模型的方块感
            Z_smoothed = gaussian_filter(Z, sigma=1.2)

            # 4. 绘图 (plasma 色系)
            cp = ax.contourf(MPR_mesh * 100, Size_mesh, Z_smoothed, 
                             levels=25, cmap='plasma', alpha=0.9)
            
            cbar = plt.colorbar(cp, ax=ax)
            cbar.set_label(f'Pred {target_col} {unit_label}', fontsize=24, fontweight='bold')
            fmt = '%.2f' if is_cd else '%.3f'
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(24)
                t.set_fontweight('bold')
            
            ax.set_title(f'pH-S={ph_val}, Cd-C={cd_val}', fontsize=28, fontweight='bold')
            ax.set_xlabel('MPR (%)', fontsize=26, fontweight='bold')
            ax.set_ylabel('Particle Size (μm)', fontsize=26, fontweight='bold')

    plt.tight_layout()
    save_name = target_col.replace("Δ", "delta")
    save_path = os.path.join(output_dir, f'xgb_scenario_smoothed_{save_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    - 图像已保存至: {save_path}")

# ---------------------------------------------------------------------------
# 4. 主程序 (XGBoost 训练 + R2 打印 + 预测)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 路径配置
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_dir = r"机器学习\微塑料机器学习\微塑料机器学习输出\XGBoost_Scenario_Plasma_Smoothed"
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

    # 3. XGBoost 最佳参数配置
    best_params = {
        'n_estimators': 500,
        'learning_rate': 0.01,
        'max_depth': 7,
        'reg_alpha': 0.1,
        'reg_lambda': 5,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'auto' # 自动开启 GPU 加速
    }

    for target in target_cols:
        print(f"\n--- 正在为目标训练 XGBoost 模型: {target} ---")
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor), 
            ('regressor', XGBRegressor(**best_params))
        ])
        
        # 训练模型
        pipeline.fit(X_train, y_train[target])
        
        # --- 打印训练集与测试集 R2 ---
        train_r2 = r2_score(y_train[target], pipeline.predict(X_train))
        test_r2 = r2_score(y_test[target], pipeline.predict(X_test))
        print(f"    >>> 训练集 R²: {train_r2:.4f}")
        print(f"    >>> 测试集 R²: {test_r2:.4f}")

        # 4. 执行绘图
        plot_xgb_scenario_prediction(pipeline, target, output_dir)

    print(f"\n[完成] 任务全部执行完毕。文件保存路径: {output_dir}")