import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import warnings
import torch # 导入torch以检查CUDA可用性
import joblib # 导入joblib用于保存模型

warnings.filterwarnings('ignore')

try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    print("成功导入 AutoTabPFNRegressor。")
except ImportError:
    print("错误：无法从 tabpfn_extensions.post_hoc_ensembles.sklearn_interface 导入 AutoTabPFNRegressor。")
    print("请确保 tabpfn_extensions 已正确安装，并且其依赖 hyperopt 库也已安装 ('pip install hyperopt')。")
    raise ImportError("AutoTabPFNRegressor 未找到或其依赖 hyperopt 未安装。")


plt.ioff()
# 设置字体样式
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def plot_actual_vs_predicted(y_train, y_train_pred, y_test, y_test_pred, title='', save_path=None):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train, y_train_pred, color='blue', alpha=0.6, label='train') # 已更改
    plt.scatter(y_test, y_test_pred, color='red', alpha=0.6, label='test')   # 已更改

    if y_train.size > 0 and y_test.size > 0 :
        all_actual = np.concatenate([y_train, y_test])
        all_pred = np.concatenate([y_train_pred, y_test_pred])
        if all_actual.size > 0 and all_pred.size > 0:
            min_val = min(min(all_actual), min(all_pred))
            max_val = max(max(all_actual), max(all_pred))
            if min_val == max_val :
                min_val -= 0.1 
                max_val += 0.1
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal prediction line') # 已更改
    else:
        print(f"警告: 训练集或测试集为空，无法绘制理想预测线 for {title}")

    plt.title(f'{title} Actual vs Predict') # 已更改
    plt.xlabel('Actual')                   # 已更改
    plt.ylabel('Predict')                  # 已更改
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'weight': 'bold'})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600) # dpi 已更改
        print(f"图表已保存至: {save_path}")
    plt.close()

def run_regression_for_target(X_train_full, X_test_full, y_train_target, y_test_target, target_name, base_output_dir, device):
    print(f"\n=== 开始处理目标列: {target_name} ===")
    
    safe_target_name = "".join(c if c.isalnum() else "_" for c in target_name)
    output_dir_target = os.path.join(base_output_dir, f"target_{safe_target_name}")
    os.makedirs(output_dir_target, exist_ok=True)

    auto_tabpfn_regressor = AutoTabPFNRegressor(
        max_time=200, # 总调优时间（秒）- 已更改
        device=device,
        # verbose=1 # 可以设置详细程度，0, 1, 2... 根据库的文档
        # seed=42 # 为了结果的可复现性
    )
    
    print(f"开始训练和自动调优 AutoTabPFNRegressor for {target_name} (max_time={auto_tabpfn_regressor.max_time}s, device='{auto_tabpfn_regressor.device}')...")
    try:
        auto_tabpfn_regressor.fit(X_train_full, y_train_target)
        print(f"AutoTabPFNRegressor for {target_name} 训练和调优完成。")

        # --- 使用 joblib 保存训练好的模型 ---
        model_filename = os.path.join(output_dir_target, f'autotabpfn_model_{safe_target_name}.joblib')
        joblib.dump(auto_tabpfn_regressor, model_filename)
        print(f"模型已为目标 '{target_name}' 保存至: {model_filename}")
        # ------------------------------------
        
        y_train_pred = auto_tabpfn_regressor.predict(X_train_full)
        y_test_pred = auto_tabpfn_regressor.predict(X_test_full)
        
        train_r2 = r2_score(y_train_target, y_train_pred)
        test_r2 = r2_score(y_test_target, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train_target, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_target, y_test_pred))
        train_mae = mean_absolute_error(y_train_target, y_train_pred)
        test_mae = mean_absolute_error(y_test_target, y_test_pred)
        
        print(f"\nAutoTabPFNRegressor for {target_name} 性能:")
        print(f"训练集 - R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"测试集 - R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        results_file = os.path.join(output_dir_target, f'autotabpfn_regressor_results_{safe_target_name}.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"AutoTabPFNRegressor 性能 for target: {target_name}\n")
            f.write(f"max_time: {auto_tabpfn_regressor.max_time}s\n")
            f.write(f"device: {auto_tabpfn_regressor.device}\n")
            f.write(f"metric_optimized: r2\n")
            f.write(f"训练集 - R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}\n")
            f.write(f"测试集 - R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}\n")
        
        scatter_path = os.path.join(output_dir_target, f'autotabpfn_regressor_actual_vs_predicted_{safe_target_name}.png')
        plot_actual_vs_predicted(
            y_train_target, y_train_pred, 
            y_test_target, y_test_pred, 
            title=f'AutoTabPFN Regressor for {target_name}\n(max_time={auto_tabpfn_regressor.max_time}s, {device.upper()})', 
            save_path=scatter_path
        )
        
        print(f"目标 {target_name} 的结果已保存到目录: {output_dir_target}")
        
    except Exception as e:
        print(f"AutoTabPFNRegressor for {target_name} 训练或评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    base_output_dir = r'C:\Users\AMDYE\OneDrive\python\output\autotabpfn_multi_target_results_v2'
    os.makedirs(base_output_dir, exist_ok=True)
    
    data_path = r"C:\Users\AMDYE\OneDrive\python\xgb_gbdt_gridsearch_results\data_ready_for_training_lr_imputed_data_new.csv" 
    print(f"加载数据: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件未找到于路径 {data_path}。请确保文件存在或路径正确。")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    target_column_names = [
        # 'CD_soil_change',
        'CD_brown_rice_change', 
        # 'CD_stem_change',
        # 'CD_leaf_change',
        # 'CD_shell_change',
        # 'CD_root_change'
    ]
    print(f"已定义的目标输出列 ({len(target_column_names)}列): {target_column_names}")

    feature_column_names = [
        'pHsoil', 'organicmatter', 'CEC', 'temperature', 'pH',
        'Charcoal_application_ratio', 'variety_binary', 'material_type_code',
        'irrigation_type', 'log_Cdcontent'
    ]
    print(f"已定义的输入特征列 ({len(feature_column_names)}列): {feature_column_names}")

    missing_targets = [col for col in target_column_names if col not in df.columns]
    if missing_targets:
        print(f"错误: 以下目标列在CSV文件中未找到: {missing_targets}")
        return
    missing_features = [col for col in feature_column_names if col not in df.columns]
    if missing_features:
        print(f"错误: 以下特征列在CSV文件中未找到: {missing_features}")
        return

    X_df = df[feature_column_names].copy()
    X = X_df.to_numpy()
    
    indices = np.arange(df.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=0.33, random_state=42) 

    X_train_full = X[train_indices]
    X_test_full = X[test_indices]

    print(f"整体特征训练集大小: {X_train_full.shape}, 测试集大小: {X_test_full.shape}")
    
    if torch.cuda.is_available():
        device = 'cuda'
        print("检测到CUDA，将使用GPU进行计算。")
    else:
        device = 'cpu'
        print("未检测到CUDA，将使用CPU进行计算。")

    for target_name in target_column_names:
        y_series_target = df[target_name].copy()
        y_target = y_series_target.to_numpy()
        
        y_train_target = y_target[train_indices]
        y_test_target = y_target[test_indices]
        
        if X_train_full.shape[0] != y_train_target.shape[0]:
            print(f"错误：对于目标 {target_name}，X_train 和 y_train 的样本数不匹配！")
            print(f"X_train_full shape: {X_train_full.shape}, y_train_target shape: {y_train_target.shape}")
            continue

        run_regression_for_target(X_train_full, X_test_full, y_train_target, y_test_target, target_name, base_output_dir, device)
        
    print("\n所有目标列处理完毕。")

if __name__ == "__main__":
    main()