import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# ---------------------------------------------------------------------------
# 1. 用户配置区
# ---------------------------------------------------------------------------

# --- 定义所有需要生成的图形 ---
PLOTS_TO_GENERATE = [
    {
        'model_path': r'机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_测试版本\tabpfn_model_deltaCd.joblib',
        'y_axis_label': r'Predicted ΔCd',
        'row_label_unit': r'(mg$\cdot$kg$^{-1}$)',
        'output_filename': 'model_prediction_grid_deltaCd_fixed_size.pdf',
        'y_lims': [
            (-0.1, 0.3),
            (-0.1, 0.2),
            (-0.2, 0.2)
        ]
    },
    {
        'model_path': r'机器学习\微塑料机器学习\微塑料机器学习输出\TabPFN_测试版本\tabpfn_model_deltapH.joblib',
        'y_axis_label': 'Predicted ΔpH',
        'output_filename': 'model_prediction_grid_deltapH_fixed_size.pdf',
    }
]

# 原始训练数据中的所有特征列名
FEATURE_COLUMNS = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'pH-B', 'BR','BET', 'MPs', 'Size', 'MPR', 'type']

# ---------------------------------------------------------------------------
# 2. 全局绘图参数设置
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['mathtext.fontset'] = 'stix'

# ---------------------------------------------------------------------------
# 3. 计算函数 (无变动)
# ---------------------------------------------------------------------------
def calculate_and_print_metrics(all_predictions, y_label, mps_values, ph_s_values, cd_c_values, mpr_range):
    """
    根据模型预测结果，计算并打印指定的变化率和差异。
    """
    mpr_target_value = 0.01
    try:
        mpr_index_for_comparison = np.where(np.isclose(mpr_range, mpr_target_value))[0][0]
    except IndexError:
        print(f"错误: 在 mpr_range 中找不到目标值 {mpr_target_value}。无法进行计算。")
        return

    mpr_value_for_comparison = mpr_range[mpr_index_for_comparison]
    
    print("\n" + "="*80)
    print(f"开始计算 '{y_label}' 的指标 (在 MPR = {mpr_value_for_comparison} 时):")
    print("="*80)

    # --- 计算 1: 固定pH，Cd浓度的变化率 ---
    print("\n>>> 分析1: 固定 pH, 镉浓度 (Cd-C) 的影响")
    print("-" * 50)
    for mps in mps_values:
        print(f"\n--- 微塑料类型 (MPs) = {mps} ---")
        for ph in ph_s_values:
            try:
                pred_at_cd1 = all_predictions[(mps, ph, cd_c_values[0])][mpr_index_for_comparison]
                pred_at_cd2 = all_predictions[(mps, ph, cd_c_values[1])][mpr_index_for_comparison]
                pred_at_cd3 = all_predictions[(mps, ph, cd_c_values[2])][mpr_index_for_comparison]
                
                change_1_to_2 = pred_at_cd2 - pred_at_cd1
                change_2_to_3 = pred_at_cd3 - pred_at_cd2
                
                print(f"  在 pH = {ph}:")
                print(f"    - Cd-C 从 {cd_c_values[0]} 变为 {cd_c_values[1]}, {y_label} 变化量: {change_1_to_2:+.4f}")
                print(f"    - Cd-C 从 {cd_c_values[1]} 变为 {cd_c_values[2]}, {y_label} 变化量: {change_2_to_3:+.4f}")

            except KeyError:
                print(f"  在 pH = {ph}: 数据不足，无法计算。")

    # --- 计算 2: 固定Cd浓度，pH的差距 ---
    print("\n>>> 分析2: 固定镉浓度 (Cd-C), pH 的影响")
    print("-" * 50)
    for mps in mps_values:
        print(f"\n--- 微塑料类型 (MPs) = {mps} ---")
        for cd in cd_c_values:
            try:
                pred_at_ph5 = all_predictions[(mps, ph_s_values[0], cd)][mpr_index_for_comparison]
                pred_at_ph6 = all_predictions[(mps, ph_s_values[1], cd)][mpr_index_for_comparison]
                pred_at_ph7 = all_predictions[(mps, ph_s_values[2], cd)][mpr_index_for_comparison]

                diff_7_vs_6 = pred_at_ph7 - pred_at_ph6
                diff_7_vs_5 = pred_at_ph7 - pred_at_ph5
                
                print(f"  在 Cd-C = {cd}:")
                print(f"    - pH=7 vs pH=6, {y_label} 差异: {diff_7_vs_6:+.4f}")
                print(f"    - pH=7 vs pH=5, {y_label} 差异: {diff_7_vs_5:+.4f}")

            except KeyError:
                print(f"  在 Cd-C = {cd}: 数据不足，无法计算。")
    
    print("\n" + "="*80)
    print("指标计算完成。")
    print("="*80 + "\n")


# ---------------------------------------------------------------------------
# 4. 主绘图函数 (已更新)
# ---------------------------------------------------------------------------
def plot_prediction_grid_detailed(model_path, feature_cols, output_path, y_label, y_lims=None, row_label_unit=''):
    """
    加载模型，进行预测，计算指标，并绘制网格折线图。
    """
    try:
        print(f"正在加载模型: {model_path}")
        model = joblib.load(model_path)
        print("模型加载成功。")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到 '{model_path}'。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # --- 参数定义 ---
    fixed_params = {'OM': 15, 'CEC': 15, 'type': 1, 'RM': 0, 'T': 0,'BET':0, 'pH-B': 0, 'BR': 0, 'Size': 100}
    ph_s_values = [5, 6, 7]
    cd_c_values = [1, 2, 3]
    mps_values = [1, 4, 6]
    mpr_range = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    color_map = {1: 'C0', 4: 'C1', 6: 'C2'}

    # ★★ 修改点 1: 在这里自定义图例标签 ★★
    # 将数字映射到您想要的文本。确保键与 mps_values 中的值一致。
    mps_label_map = {
        1: 'PP',
        4: 'PS',
        6: 'PVC'
    }
    
    # --- 步骤 1: 生成并存储所有预测数据 ---
    print(f"开始为 '{y_label}' 生成所有预测数据...")
    all_predictions = {}
    for cd_val in cd_c_values:
        for ph_val in ph_s_values:
            for mps_val in mps_values:
                predict_df = pd.DataFrame(index=range(len(mpr_range)))
                for col, val in fixed_params.items():
                    if col in feature_cols:
                        predict_df[col] = val
                predict_df['pH-S'] = ph_val
                predict_df['Cd-C'] = cd_val
                predict_df['MPs'] = mps_val
                predict_df['MPR'] = mpr_range
                
                if 'CEC' in predict_df.columns:
                    predict_df['CEC'] = np.log(predict_df['CEC'])

                predict_df = predict_df[feature_cols]
                predictions = model.predict(predict_df)
                all_predictions[(mps_val, ph_val, cd_val)] = predictions
    print("所有预测数据生成完毕。")

    # --- 步骤 2: 调用函数进行数值计算和打印 ---
    calculate_and_print_metrics(all_predictions, y_label, mps_values, ph_s_values, cd_c_values, mpr_range)

    # --- 步骤 3: 使用已生成的数据进行绘图 ---
    print(f"开始为 '{y_label}' 绘图...")
    fig, axs = plt.subplots(len(cd_c_values), len(ph_s_values), figsize=(18, 16), sharex=True, sharey='row')
    if len(cd_c_values) == 1 and len(ph_s_values) == 1:
        axs = np.array([[axs]])
    elif len(cd_c_values) == 1:
        axs = axs.reshape(1, -1)
    elif len(ph_s_values) == 1:
        axs = axs.reshape(-1, 1)

    for i, cd_val in enumerate(cd_c_values):
        for j, ph_val in enumerate(ph_s_values):
            ax = axs[i, j]
            for mps_val in mps_values:
                predictions = all_predictions.get((mps_val, ph_val, cd_val))
                if predictions is not None:
                    ax.plot(mpr_range, predictions, marker='o', linestyle='-', color=color_map[mps_val], linewidth=2.5)
            
            ax.tick_params(axis='both', which='major', labelsize=20)
            
            for tick in ax.get_xticklabels():
                tick.set_fontweight('bold')
            for tick in ax.get_yticklabels():
                tick.set_fontweight('bold')
            
            if i == 0:
                ax.set_title(f'pH-S = {ph_val}', fontsize=22, fontweight='bold', pad=10)
                
            if j == 0:
                full_label = f"Cd-C = {cd_val} {row_label_unit}".strip()
                ax.set_ylabel(full_label, fontsize=22, fontweight='bold', labelpad=10)
                
            ax.grid(True, linestyle='--', alpha=0.6)

            if y_lims and i < len(y_lims):
                ax.set_ylim(y_lims[i])
            
    fig.text(0.5, 0.06, 'MPR', ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.04, 0.5, y_label, ha='center', va='center', rotation='vertical', fontsize=24, fontweight='bold')
    
    # ★★ 修改点 2: 更新图例生成方式，使用自定义标签并移除标题 ★★
    # 使用 mps_label_map 获取标签
    mps_legend_elements = [
        Line2D([0], [0], color=color_map[mps], lw=2.5, marker='o', label=mps_label_map.get(mps, f'MPs = {mps}')) 
        for mps in mps_values
    ]
    # 在 fig.legend() 调用中移除 title 和 title_fontsize 参数
    fig.legend(handles=mps_legend_elements, 
               loc='upper center',
               bbox_to_anchor=(0.5, 0.95),
               ncol=len(mps_values),
               prop={'weight': 'bold', 'size': 24},
               frameon=False)

    plt.tight_layout(rect=[0.05, 0.08, 0.95, 0.90])
    
    try:
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"图形已成功保存至: {output_path}\n")
    except Exception as e:
        print(f"保存图形时发生错误: {e}\n")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. 主程序入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"自动化绘图和计算开始，将处理 {len(PLOTS_TO_GENERATE)} 个模型。\n")
    
    for plot_config in PLOTS_TO_GENERATE:
        model_file = plot_config['model_path']
        if not os.path.exists(model_file):
            print(f"警告: 找不到模型文件 '{model_file}'。跳过此任务。\n")
            continue
        
        plot_prediction_grid_detailed(
            model_path=model_file,
            feature_cols=FEATURE_COLUMNS,
            output_path=plot_config['output_filename'],
            y_label=plot_config['y_axis_label'],
            y_lims=plot_config.get('y_lims'),
            row_label_unit=plot_config.get('row_label_unit', '')
        )
        
    print("所有任务已完成。")

    FEATURES_BY_TYPE = {
    # --- Type 1: 你想显示的特征 ---
    1: ['pH-S', 'OM', 'Cd-C','MPs', 'Size', 'MPR'],
    # --- Type 2: 你想显示的特征 ---
    2:['pH-S', 'OM', 'Cd-C','RM','pH-B', 'T','BET', 'BR'],
    # --- Type 3: 你想显示的特征 ---
    3: ['pH-S', 'OM', 'Cd-C','CEC','RM','pH-B', 'T','BET', 'BR', 'MPs', 'Size', 'MPR']
}
    FEATURE_XLIMS_BY_TYPE = {
    # --- Type 1 范围设置 ---
    1: {
        'MPR': (0, 0.05),       
        'Size': (0, 100),
        # 'pH-S': (4, 9)
    },
    
    # --- Type 2 范围设置 ---
    2: {
        # 'pH-B': (7, 12),       
    },
    
    # --- Type 3 范围设置 ---
    3: {
        'MPR': (0, 0.05), 
        'Size': (0, 100),  
        'BET': (0, 100), 
    }
}