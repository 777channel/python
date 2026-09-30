import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def quantify_mp_effect_grid_scan(model_path, feature_cols, output_dir, target_label):
    """
    全网格扫描微塑料与生物炭的交互作用
    范围：BR 0-3% (6个点), MPR 0-3% (6个点)
    """
    
    # -----------------------------------------------------------
    # 0. 加载模型与设定参数
    # -----------------------------------------------------------
    print(f"[{target_label}] 正在加载模型...")
    model_pipeline = joblib.load(model_path)
    
    # 固定参数
    base_conditions = {
        'OM': 15.0, 'RM': 1, 'T': 400, 'pH-B': 9.6, 'BET': 3.6,
        # 'type', 'BR', 'CEC', 'Cd-C', 'MPs', 'Size', 'MPR' 在循环中生成
    }
    
    # 微塑料固定参数 (当 MPR > 0 时使用)
    mp_settings = {'MPs': 1, 'Size': 100} 
    
    # 土壤情景 (9种)
    soil_conditions = [
        {'pH-S': 5, 'Cd-C': 1}, {'pH-S': 5, 'Cd-C': 2}, {'pH-S': 5, 'Cd-C': 3},
        {'pH-S': 6, 'Cd-C': 1}, {'pH-S': 6, 'Cd-C': 2}, {'pH-S': 6, 'Cd-C': 3},
        {'pH-S': 7, 'Cd-C': 1}, {'pH-S': 7, 'Cd-C': 2}, {'pH-S': 7, 'Cd-C': 3},
    ]
    
    # -----------------------------------------------------------
    # 1. 设定严格的梯度 (您指定的6个点)
    # -----------------------------------------------------------
    # 生物炭比例梯度 (%)
    br_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    # 微塑料比例梯度 (%) - 这里是实验组
    mpr_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    print(f"计算梯度: BR={br_points}, MPR={mpr_points}")
    print("正在执行全网格计算 (含 Type逻辑修正 + 对数化处理)...")
    
    all_results = []
    
    # -----------------------------------------------------------
    # 2. 网格循环预测
    # -----------------------------------------------------------
    for soil_cond in soil_conditions:
        
        # [预处理] 对数化 (CEC 固定15, Cd-C 动态)
        val_CEC_log = np.log(15.0)
        val_Cd_log  = np.log(soil_cond['Cd-C'])
        
        for br in br_points:
            
            # --- 第一步：计算当前 BR 下的“零微塑料基准” (Baseline) ---
            # 条件：MPR=0, Type=2, MPs=0, Size=0
            base_input = {
                'pH-S': soil_cond['pH-S'],
                'OM': base_conditions['OM'],
                'CEC': val_CEC_log,         # Log
                'Cd-C': val_Cd_log,         # Log
                'RM': base_conditions['RM'],
                'T': base_conditions['T'],
                'pH-B': base_conditions['pH-B'],
                'BET': base_conditions['BET'],
                'BR': br,
                'type': 2,                  # 关键：单独生物炭
                'MPR': 0,                   # 关键
                'MPs': 0,                   # 关键
                'Size': 0                   # 关键
            }
            
            # 预测基准值
            df_base = pd.DataFrame([base_input])[feature_cols]
            baseline_pred = model_pipeline.predict(df_base)[0]
            
            # --- 第二步：遍历每一个微塑料浓度梯度 ---
            for mpr in mpr_points:
                
                # 条件：MPR>0, Type=3, MPs=Fixed, Size=Fixed
                exp_input = base_input.copy()
                exp_input['type'] = 3           # 关键：复合施加
                exp_input['MPR'] = mpr          # 当前浓度
                exp_input['MPs'] = mp_settings['MPs']
                exp_input['Size'] = mp_settings['Size']
                
                # 预测实验值
                df_exp = pd.DataFrame([exp_input])[feature_cols]
                current_pred = model_pipeline.predict(df_exp)[0]
                
                # 计算差异
                abs_effect = current_pred - baseline_pred
                epsilon = 1e-6
                rel_effect = (abs_effect) / (abs(baseline_pred) + epsilon) * 100
                
                # 保存这一行详细数据
                row = {
                    'pH-S': soil_cond['pH-S'],
                    'Cd-C': soil_cond['Cd-C'],
                    'BR': br,
                    'MPR': mpr,
                    'Baseline_Pred': baseline_pred,
                    'Current_Pred': current_pred,
                    'Abs_Effect': abs_effect,
                    'Rel_Effect_Pct': rel_effect
                }
                all_results.append(row)
    
    # -----------------------------------------------------------
    # 3. 保存与统计
    # -----------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    
    # 保存完整数据 (324行)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, '1_full_grid_data.csv')
    results_df.to_csv(save_path, index=False)
    
    # 生成按 MPR 聚合的统计表 (用于观察浓度趋势)
    stats_by_mpr = results_df.groupby('MPR')[['Abs_Effect', 'Rel_Effect_Pct']].mean().reset_index()
    stats_path = os.path.join(output_dir, '2_stats_by_mpr_concentration.csv')
    stats_by_mpr.to_csv(stats_path, index=False)
    
    print(f"[{target_label}] 计算完成，共生成 {len(results_df)} 个数据点。")
    print(f"结果已保存至: {output_dir}")
    
    # 简单的文本报告生成
    generate_report(results_df, stats_by_mpr, output_dir, target_label)
    
    return results_df

def generate_report(df, stats_mpr, output_dir, target_label):
    """
    生成针对 0-3% 梯度的分析报告
    """
    is_cd = "Cd" in target_label
    metric = "镉固定量" if is_cd else "pH改变值"
    
    text = f"微塑料与生物炭交互作用分析报告 (0-3% 梯度扫描)\n"
    text += "="*60 + "\n\n"
    
    text += "1. 浓度-效应趋势分析\n"
    text += "-"*30 + "\n"
    for _, row in stats_mpr.iterrows():
        text += f"  - MPR = {row['MPR']}% 时，平均影响值为 {row['Abs_Effect']:.4f} (相对变化 {row['Rel_Effect_Pct']:.2f}%)\n"
    
    # 找出影响最大的点
    max_row = df.loc[df['Abs_Effect'].abs().idxmax()]
    text += "\n2. 最大干扰点识别\n"
    text += "-"*30 + "\n"
    text += f"  在以下条件下，微塑料的干扰最严重：\n"
    text += f"  - 土壤环境: pH={max_row['pH-S']}, Cd={max_row['Cd-C']}\n"
    text += f"  - 施加比例: 生物炭 BR={max_row['BR']}%, 微塑料 MPR={max_row['MPR']}%\n"
    text += f"  - 导致 {metric} 变化了 {max_row['Abs_Effect']:.4f}\n"
    
    with open(os.path.join(output_dir, '3_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

# ============================================================================
# 主程序
# ============================================================================
if __name__ == "__main__":
    
    # 路径配置
    base_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\微塑料机器学习\微塑料机器学习输出"
    # 模型路径
    models = {
        'ΔCd': os.path.join(base_dir, r"TabPFN_shap分析版本\tabpfn_model_deltaCd.joblib"),
        'ΔpH': os.path.join(base_dir, r"TabPFN_shap分析版本\tabpfn_model_deltapH.joblib")
    }
    
    # 特征列 (必须与训练时一致)
    feature_columns = ['pH-S', 'OM', 'CEC', 'Cd-C', 'RM', 'T', 'type', 'pH-B', 'BET', 'BR', 'MPs', 'Size', 'MPR']
    
    # 输出目录
    output_base = os.path.join(base_dir, "MPs_Grid_Scan_0to3")
    
    try:
        for label, model_file in models.items():
            sub_dir = os.path.join(output_base, label.replace('Δ', 'delta'))
            quantify_mp_effect_grid_scan(model_file, feature_columns, sub_dir, label)
            
        print("\n所有分析结束。")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()