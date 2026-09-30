import numpy as np
import pandas as pd
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# --- 常量 ---
TARGET_CD = 0.2  # mg/kg

# --- 固定土壤/生物炭参数 (用于特定分析运行) ---
# 设置你当前分析想要使用的具体固定值。
FIXED_ORGANIC_MATTER = 10.0 # 当前分析场景的有机质值
FIXED_CEC = 10.0           # 当前分析场景的CEC值

# --- 固定的生物炭参数 (用于当前分析) ---
FIXED_BIOCHAR_MATERIAL_CODE = 0    # 直接使用编码值，例如0代表水稻秸秆
FIXED_BIOCHAR_TEMP = 400           # 示例：温度400°C
FIXED_BIOCHAR_PH = 10             # 示例：pH 9.6
FIXED_BIOCHAR_RATE = 3          # 示例：施用率3% (0.03)

# --- 归一化参数 (基于你的模型训练过程) ---
# 土壤 pH (Min-Max)
SOIL_PH_MIN_TRAINING = 4.32
SOIL_PH_MAX_TRAINING = 8.07

# 有机质 (OM) (Z-score)
OM_MEAN_TRAINING = 17.83
OM_STD_TRAINING = 10.04

# CEC (Z-score)
CEC_MEAN_TRAINING = 13.8
CEC_STD_TRAINING = 6.99

# 初始土壤镉浓度 (先对数，然后Z-score)
LOG_CD_MEAN_TRAINING = 0.3505061158219177
LOG_CD_STD_TRAINING = 1.000154132458402

# 生物炭类型 (直接使用编码，无需Min-Max参数，但可定义有效范围)
VALID_MATERIAL_CODES = list(range(8)) # [0, 1, 2, 3, 4, 5, 6, 7]

# 生物炭温度 (Min-Max)
TEMPERATURE_MIN_TRAINING = 300.0
TEMPERATURE_MAX_TRAINING = 800.0

# 生物炭 pH (Min-Max)
PH_BIOCHAR_MIN_TRAINING = 4.98
PH_BIOCHAR_MAX_TRAINING = 10.4

# 生物炭施炭比 (Ratio) (Min-Max) - 假设为实际比例
RATIO_MIN_TRAINING = 0.5
RATIO_MAX_TRAINING = 6

# 模型1输出反归一化参数
MODEL1_SCALE = 0.9252
MODEL1_OFFSET = 0.73490

# 容差
TOLERANCE_PCT = 50.0

# --- 工具函数 (归一化, 输入准备) ---
def normalize_input_data(item):
    """归一化土壤和初始Cd特征，使用训练时的参数"""
    norm_item = {}
    norm_item['pHsoil'] = (item['pHsoil'] - SOIL_PH_MIN_TRAINING) / (SOIL_PH_MAX_TRAINING - SOIL_PH_MIN_TRAINING)
    norm_item['organicmatter'] = (item['organicmatter'] - OM_MEAN_TRAINING) / OM_STD_TRAINING
    cdcontent = max(1e-9, item['Cdcontent'])
    log_cd = np.log(cdcontent)
    norm_item['log_Cdcontent'] = (log_cd - LOG_CD_MEAN_TRAINING) / LOG_CD_STD_TRAINING
    norm_item['CEC'] = (item['CEC'] - CEC_MEAN_TRAINING) / CEC_STD_TRAINING
    norm_item['variety_binary'] = 1.0
    return norm_item

def bilateral_minmax_denormalize(normalized_value):
    return normalized_value * 100.0

def normalize_fixed_biochar_features(material_code, temperature, biochar_ph, ratio):
    """
    归一化生物炭特征。生物炭类型直接使用编码值。
    其他生物炭特征使用训练时的参数进行Min-Max归一化。
    """
    # 生物炭类型 -直接使用编码值
    if material_code not in VALID_MATERIAL_CODES:
        raise ValueError(f"无效的生物炭类型编码: {material_code}. 允许的编码为: {VALID_MATERIAL_CODES}")
    norm_material = float(material_code)

    # 生物炭温度 (Min-Max)
    temp_clamped = max(TEMPERATURE_MIN_TRAINING, min(TEMPERATURE_MAX_TRAINING, temperature))
    norm_temp = (temp_clamped - TEMPERATURE_MIN_TRAINING) / (TEMPERATURE_MAX_TRAINING - TEMPERATURE_MIN_TRAINING)
    if temperature != temp_clamped:
         print(f"警告：固定生物炭温度 {temperature} 被限制在 [{TEMPERATURE_MIN_TRAINING}, {TEMPERATURE_MAX_TRAINING}] 范围内进行归一化。")

    # 生物炭 pH (Min-Max)
    biochar_ph_clamped = max(PH_BIOCHAR_MIN_TRAINING, min(PH_BIOCHAR_MAX_TRAINING, biochar_ph))
    norm_ph = (biochar_ph_clamped - PH_BIOCHAR_MIN_TRAINING) / (PH_BIOCHAR_MAX_TRAINING - PH_BIOCHAR_MIN_TRAINING)
    if biochar_ph != biochar_ph_clamped:
         print(f"警告：固定生物炭pH {biochar_ph} 被限制在 [{PH_BIOCHAR_MIN_TRAINING}, {PH_BIOCHAR_MAX_TRAINING}] 范围内进行归一化。")

    # 生物炭施炭比 (Ratio) (Min-Max)
    ratio_clamped = max(RATIO_MIN_TRAINING, min(RATIO_MAX_TRAINING, ratio))
    norm_ratio = (ratio_clamped - RATIO_MIN_TRAINING) / (RATIO_MAX_TRAINING - RATIO_MIN_TRAINING)
    if ratio != ratio_clamped:
         print(f"警告：固定施炭比 {ratio} 被限制在 [{RATIO_MIN_TRAINING}, {RATIO_MAX_TRAINING}] 范围内进行归一化。")

    return [norm_material, norm_temp, norm_ph, norm_ratio]

def prepare_model1_input(normalized_data, irrigation_type):
    irrigation_encoded = 1 if irrigation_type == "alternating" else 0
    X = np.zeros(6)
    X[0] = normalized_data['log_Cdcontent']; X[1] = normalized_data['organicmatter']
    X[2] = irrigation_encoded; X[3] = normalized_data['CEC']
    X[4] = normalized_data['pHsoil']; X[5] = normalized_data['variety_binary']
    return X.reshape(1, -1)

def prepare_model2_input(normalized_data, normalized_biochar_features, irrigation_type):
    irrigation_encoded = 1 if irrigation_type == "alternating" else 0
    X = np.zeros(10)
    X[0] = normalized_data['log_Cdcontent']; X[1] = normalized_data['organicmatter']
    X[2] = irrigation_encoded; X[3] = normalized_data['CEC']
    X[4] = normalized_data['pHsoil']; X[5] = normalized_data['variety_binary']
    X[6] = normalized_biochar_features[0]; X[7] = normalized_biochar_features[1]
    X[8] = normalized_biochar_features[2]; X[9] = normalized_biochar_features[3]
    return X.reshape(1, -1)

# --- 核心阈值查找逻辑 ---
def get_strategy_code_for_cd_detailed(ph_soil_val, initial_cd, fixed_soil_params_dict, normalized_fixed_biochar_list, models_tuple):
    model1, model2 = models_tuple
    input_data = {
        'pHsoil': ph_soil_val,
        'organicmatter': fixed_soil_params_dict['organic_matter'],
        'Cdcontent': initial_cd,
        'CEC': fixed_soil_params_dict['cec']
    }
    normalized_data = normalize_input_data(input_data)
    try:
        norm_pred1_cf = model1.predict(prepare_model1_input(normalized_data, irrigation_type="continuous"))[0]
        baseline_cd_cf = max(0, norm_pred1_cf * MODEL1_SCALE + MODEL1_OFFSET)
        if baseline_cd_cf <= TARGET_CD: return 1

        norm_pred1_awd = model1.predict(prepare_model1_input(normalized_data, irrigation_type="alternating"))[0]
        baseline_cd_awd = max(0, norm_pred1_awd * MODEL1_SCALE + MODEL1_OFFSET)
        
        model2_output_awd_norm = model2.predict(prepare_model2_input(normalized_data, normalized_fixed_biochar_list, irrigation_type="alternating"))[0]
        predicted_reduction_awd_pct = max(0, -bilateral_minmax_denormalize(model2_output_awd_norm))
        
        model2_output_cf_norm = model2.predict(prepare_model2_input(normalized_data, normalized_fixed_biochar_list, irrigation_type="continuous"))[0]
        predicted_reduction_cf_pct = max(0, -bilateral_minmax_denormalize(model2_output_cf_norm))

        cond2_met = False
        if baseline_cd_awd <= TARGET_CD:
            final_cd_awd_with_biochar = baseline_cd_awd * (1 - predicted_reduction_awd_pct / 100.0)
            if final_cd_awd_with_biochar <= TARGET_CD * (1 + TOLERANCE_PCT / 100.0):
                cond2_met = True
        else:
            final_cd_awd_with_biochar = baseline_cd_awd * (1 - predicted_reduction_awd_pct / 100.0)
            if final_cd_awd_with_biochar <= TARGET_CD * (1 + TOLERANCE_PCT / 100.0):
                cond2_met = True
        if cond2_met: return 2

        cond3_met = False
        final_cd_cf_with_biochar = baseline_cd_cf * (1 - predicted_reduction_cf_pct / 100.0)
        if final_cd_cf_with_biochar <= TARGET_CD * (1 + TOLERANCE_PCT / 100.0):
            cond3_met = True
        if cond3_met: return 3
        
        return 0
    except Exception as e:
        print(f"预测错误在 pH={ph_soil_val:.1f}, Cd={initial_cd:.3f}: {e}")
        return -1

# --- 绘图函数 ---
# --- 绘图函数 ---
def plot_segmented_bars(thresholds_df, cd_min_plot, cd_max_plot, title_suffix=""):
    ph_levels_plot = sorted(thresholds_df.index)
    n_ph = len(ph_levels_plot)
    colors = {1: 'green', 2: 'yellow', 3: 'orange', 0: 'red'}
    labels = {
        1: '策略1: CF无炭',
        2: '策略2: AWD+生物炭(固定)',
        3: '策略3: CF+生物炭(固定)',
        0: '策略0: 无效/不推荐' # 修改标签为策略0
    }

    fig, axes = plt.subplots(n_ph, 1, figsize=(10, n_ph * 1.0 + 0.8), sharex=True) # 略微增加整体高度
    if n_ph == 1: axes = [axes]

    for i, ph_val_plot in enumerate(ph_levels_plot):
        ax = axes[i]
        row = thresholds_df.loc[ph_val_plot]
        
        # --- 解析阈值 T1, T2, T3 ---
        # (这部分解析逻辑与你之前的代码一致，确保 t_values['T1'], t_values['T2'], t_values['T3'] 被正确赋值)
        t_values = {}
        for j_col_idx, col_name in enumerate(thresholds_df.columns): # 使用 j_col_idx 避免与外部 i 冲突
            val_str = row[col_name]
            # 确保只处理T1, T2, T3这三列，如果列名不是预期的，可能会出错
            # 最好是直接用列名索引 row['T1: ...'], row['T2: ...'], row['T3: ...']
            # 为了通用性，这里假设列的顺序是T1, T2, T3
            t_key = f'T{j_col_idx+1}' 
            if pd.isna(val_str) or val_str == '错误': t_values[t_key] = None
            elif isinstance(val_str, str):
                if '>' in val_str: t_values[t_key] = cd_max_plot + 1 
                elif '<' in val_str: t_values[t_key] = cd_min_plot -1
                else: 
                    try: t_values[t_key] = float(val_str)
                    except ValueError: t_values[t_key] = None
            else: t_values[t_key] = float(val_str)

        t1 = t_values.get('T1')
        t2 = t_values.get('T2')
        t3 = t_values.get('T3')

        # --- 确定不重叠的绘制分段 ---
        segments = [] # 存储 (start_cd, end_cd, strategy_code)
        current_plot_pos = cd_min_plot # 当前绘图起始点

        # 策略1的区间 (CF无炭)
        # 策略1的有效上限是 t1
        s1_end = cd_min_plot # 默认策略1无效或在绘图范围之外开始
        if t1 is not None and t1 > cd_min_plot: # 只有当T1有效且大于绘图下限时，策略1才可能在图中显示
            s1_end = min(t1, cd_max_plot)
            if s1_end > current_plot_pos: # 确保区间有宽度
                segments.append((current_plot_pos, s1_end, 1))
            current_plot_pos = max(current_plot_pos, s1_end) # 更新当前绘图位置

        # 策略2的区间 (AWD+生物炭)
        # 策略2在策略1失效后开始考虑，其上限是 t2
        s2_end = current_plot_pos # 默认策略2无效或从当前位置开始
        if current_plot_pos < cd_max_plot and t2 is not None and t2 > current_plot_pos:
            # 策略2的真实上限不能超过t2，也不能超过绘图上限
            s2_end = min(t2, cd_max_plot)
            if s2_end > current_plot_pos: # 确保区间有宽度
                segments.append((current_plot_pos, s2_end, 2))
            current_plot_pos = max(current_plot_pos, s2_end)

        # 策略3的区间 (CF+生物炭)
        # 策略3在策略2失效后开始考虑，其上限是 t3
        s3_end = current_plot_pos # 默认策略3无效或从当前位置开始
        if current_plot_pos < cd_max_plot and t3 is not None and t3 > current_plot_pos:
            s3_end = min(t3, cd_max_plot)
            if s3_end > current_plot_pos: # 确保区间有宽度
                segments.append((current_plot_pos, s3_end, 3))
            current_plot_pos = max(current_plot_pos, s3_end)
            
        # 策略0的区间 (无效)
        # 如果绘制完策略1, 2, 3后，还未到达绘图上限 cd_max_plot，则剩余部分为策略0
        if current_plot_pos < cd_max_plot:
            segments.append((current_plot_pos, cd_max_plot, 0))

        # --- 绘制分段颜色 ---
        for start, end, code in segments:
            if start < end: # 再次确保段有宽度
                 ax.axvspan(start, end, facecolor=colors[code], alpha=0.7, edgecolor='grey', linewidth=0.5) # 加一点边框

        # --- 添加实际的阈值转换点作为垂直线 ---
        # 我们只绘制实际发生策略转变的点
        # T1是策略1结束点（如果有效）
        # T2是策略2结束点（如果有效且在T1之后）
        # T3是策略3结束点（如果有效且在T2之后）
        
        # 提取 DataFrame 中的原始阈值用于标记，而不是用上面计算的 s1_end, s2_end, s3_end
        # 因为 sX_end 是绘图区间的末端，而我们想标记的是 DataFrame 中记录的阈值
        df_t1 = t_values.get('T1')
        df_t2 = t_values.get('T2')
        df_t3 = t_values.get('T3')

        plotted_vlines = set() # 防止重复绘制非常接近的阈值线

        if df_t1 is not None and cd_min_plot < df_t1 < cd_max_plot:
            ax.axvline(x=df_t1, color='black', linestyle='--', linewidth=1)
            ax.text(df_t1, 1.01, f'T1={df_t1:.2f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, rotation=0, backgroundcolor='white', alpha=0.7)
            plotted_vlines.add(round(df_t1,2))

        # 只有当T2与T1不同且在范围内时才绘制T2
        if df_t2 is not None and cd_min_plot < df_t2 < cd_max_plot and round(df_t2,2) not in plotted_vlines:
             # 并且，如果T1存在且是数值，T2应该大于T1才更有意义作为一个独立的阈值
            if not (isinstance(df_t1, (float,int)) and df_t2 <= df_t1):
                ax.axvline(x=df_t2, color='dimgray', linestyle=':', linewidth=1)
                ax.text(df_t2, 1.01, f'T2={df_t2:.2f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, rotation=0, backgroundcolor='white', alpha=0.7)
                plotted_vlines.add(round(df_t2,2))

        # 只有当T3与T1,T2不同且在范围内时才绘制T3
        if df_t3 is not None and cd_min_plot < df_t3 < cd_max_plot and round(df_t3,2) not in plotted_vlines:
            if not (isinstance(df_t2, (float,int)) and df_t3 <= df_t2) and \
               not (isinstance(df_t1, (float,int)) and df_t3 <= df_t1) : # 确保T3在T1和T2之后
                ax.axvline(x=df_t3, color='darkgray', linestyle='-.', linewidth=1)
                ax.text(df_t3, 1.01, f'T3={df_t3:.2f}', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, rotation=0, backgroundcolor='white', alpha=0.7)
        
        # --- 设置坐标轴和标题 ---
        ax.set_xlim(cd_min_plot, cd_max_plot)
        ax.set_yticks([]) # 隐藏Y轴刻度
        ax.set_ylabel(f"pH={ph_val_plot:.1f}", rotation=0, labelpad=30, va='center', ha='right', fontsize=10)
        if i == n_ph - 1:
            ax.set_xlabel("初始土壤镉浓度 (mg/kg)", fontsize=10)
        else:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    patches = [mpatches.Patch(color=colors[code], label=labels[code]) for code in sorted(colors.keys())]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.98 + 0.02 * n_ph), ncol=len(colors), fontsize=9)
    fig.suptitle(f"不同土壤pH下修复策略的有效初始Cd浓度区间{title_suffix}\n(OM={FIXED_ORGANIC_MATTER:.2f}, CEC={FIXED_CEC:.2f}, 固定生物炭, {TOLERANCE_PCT}%削减容差)", fontsize=12, y=1.03 + 0.02*n_ph) # 调整y值
    fig.tight_layout(rect=[0, 0.03, 1, 0.95 - 0.01*n_ph]) # 调整rect的top值给图例和标题留空间

    plot_output_dir = 'output_plots'
    os.makedirs(plot_output_dir, exist_ok=True)
    # 清理文件名中的非法字符，例如将点替换为p
    clean_suffix = title_suffix.replace(" ","_").replace("=","").replace(".","p").replace("(","").replace(")","").replace("[","").replace("]","")
    plot_filename = os.path.join(plot_output_dir, f'strategy_ph{clean_suffix}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n分段着色图已保存到: {plot_filename}")
    plt.close(fig)

# --- 主执行逻辑 ---
def main():
    print("精确阈值查找系统启动...")
    try:
        model1_path = r'C:\Users\AMDYE\OneDrive\python\forexample\soil\models\CD_brown_rice_XGBoost_model.pkl'
        model2_path = r'C:\Users\AMDYE\OneDrive\python\results\soil\models\CD_brown_rice_change_GBDT_model.pkl'
        with open(model1_path, 'rb') as f: model1_data = pickle.load(f)
        model1 = model1_data['model'] if isinstance(model1_data, dict) and 'model' in model1_data else model1_data
        with open(model2_path, 'rb') as f: model2_data = pickle.load(f)
        model2 = model2_data['model'] if isinstance(model2_data, dict) and 'model' in model2_data else model2_data
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}. 请检查路径。")
        return
    except Exception as e:
        print(f"模型加载失败: {str(e)}"); return
    print("模型加载完成。")

    fixed_soil_params_dict = { 'organic_matter': FIXED_ORGANIC_MATTER, 'cec': FIXED_CEC }
    fixed_biochar_params_dict = {
        'material_code': FIXED_BIOCHAR_MATERIAL_CODE,
        'temperature': FIXED_BIOCHAR_TEMP,
        'biochar_ph': FIXED_BIOCHAR_PH,
        'ratio': FIXED_BIOCHAR_RATE
    }
    print(f"固定土壤: OM={fixed_soil_params_dict['organic_matter']:.2f}, CEC={fixed_soil_params_dict['cec']:.2f}")
    print(f"固定生物炭: 类型编码={fixed_biochar_params_dict['material_code']}, 温度={fixed_biochar_params_dict['temperature']}C, pH={fixed_biochar_params_dict['biochar_ph']}, 施用率={fixed_biochar_params_dict['ratio']:.4f}")

    try:
        normalized_fixed_biochar_list = normalize_fixed_biochar_features(
            fixed_biochar_params_dict['material_code'], fixed_biochar_params_dict['temperature'],
            fixed_biochar_params_dict['biochar_ph'], fixed_biochar_params_dict['ratio']
        )
        print(f"归一化固定生物炭特征: {[round(f, 4) for f in normalized_fixed_biochar_list]}")
    except Exception as e:
        print(f"归一化固定生物炭特征失败: {e}"); return

    ph_levels_list = [5.0, 5.5, 6.0, 6.5, 7.0]
    cd_search_min = 0.1
    cd_search_max = 4.0
    cd_search_step = 0.01
    search_cd_concentrations = np.arange(cd_search_min, cd_search_max + cd_search_step, cd_search_step)
    print(f"将在Cd浓度范围 [{cd_search_min:.2f}, {cd_search_max:.2f}] 内搜索，步长为 {cd_search_step:.3f}")

    critical_thresholds_dict = {ph_val: {'T1': None, 'T2': None, 'T3': None} for ph_val in ph_levels_list}

    for ph_val_main in ph_levels_list:
        print(f"\n--- 处理 pH = {ph_val_main:.1f} ---")
        current_ph_thresholds = {'T1': None, 'T2': None, 'T3': None}
        found_T1, found_T2, found_T3 = False, False, False
        
        # 阶段1: 寻找 T1
        last_cd_val_t1 = cd_search_min - cd_search_step
        for current_cd_val in search_cd_concentrations:
            strategy_code = get_strategy_code_for_cd_detailed(ph_val_main, current_cd_val, fixed_soil_params_dict, normalized_fixed_biochar_list, (model1, model2))
            if strategy_code == -1:
                print(f"  错误：在 Cd={current_cd_val:.3f} 预测失败 (寻找T1时)。跳过此pH。")
                current_ph_thresholds = {k: '错误' for k in current_ph_thresholds}; found_T1 = found_T2 = found_T3 = True; break
            if strategy_code != 1:
                current_ph_thresholds['T1'] = round(last_cd_val_t1, 3) if last_cd_val_t1 >= cd_search_min else f"< {cd_search_min:.3f}"
                print(f"  找到 T1 ≈ {current_ph_thresholds['T1']} (策略1转为{strategy_code} @ Cd={current_cd_val:.3f})")
                found_T1 = True; break
            last_cd_val_t1 = current_cd_val
        if not found_T1: current_ph_thresholds['T1'] = f"> {cd_search_max:.3f}"; print(f"  T1 > {cd_search_max:.3f}"); found_T1 = True
        if current_ph_thresholds['T1'] == '错误': critical_thresholds_dict[ph_val_main] = current_ph_thresholds; continue

        # 阶段2: 寻找 T2
        start_cd_t2 = cd_search_min
        if isinstance(current_ph_thresholds['T1'], (float, int)): start_cd_t2 = current_ph_thresholds['T1'] + cd_search_step
        elif isinstance(current_ph_thresholds['T1'], str) and ">" in current_ph_thresholds['T1']: found_T2 = found_T3 = True; current_ph_thresholds['T2'] = None; current_ph_thresholds['T3'] = None
        
        if not found_T2:
            last_cd_val_t2 = start_cd_t2 - cd_search_step
            processed_any_for_t2 = False
            for current_cd_val in np.arange(start_cd_t2, cd_search_max + cd_search_step, cd_search_step):
                if current_cd_val < cd_search_min : continue #确保不从负数开始（如果T1是<min）
                processed_any_for_t2 = True
                strategy_code = get_strategy_code_for_cd_detailed(ph_val_main, current_cd_val, fixed_soil_params_dict, normalized_fixed_biochar_list, (model1, model2))
                if strategy_code == -1:
                    print(f"  错误：在 Cd={current_cd_val:.3f} 预测失败 (寻找T2时)。"); current_ph_thresholds['T2'] = '错误'; current_ph_thresholds['T3'] = '错误'; found_T2 = found_T3 = True; break
                
                # 如果在T1之后，第一个有效策略不是2，则T2的区间为0或不存在
                if last_cd_val_t2 < start_cd_t2 and strategy_code != 2 : # 检查T2搜索区间的第一个点
                     current_ph_thresholds['T2'] = current_ph_thresholds['T1'] if isinstance(current_ph_thresholds['T1'], (float,int)) else f"< {start_cd_t2:.3f}"
                     print(f"  T2在T1后未立即生效 (策略为{strategy_code} @ Cd={current_cd_val:.3f})。T2 ≈ {current_ph_thresholds['T2']}")
                     found_T2 = True; break
                
                if strategy_code != 2:
                    current_ph_thresholds['T2'] = round(last_cd_val_t2, 3)
                    print(f"  找到 T2 ≈ {current_ph_thresholds['T2']} (策略2转为{strategy_code} @ Cd={current_cd_val:.3f})")
                    found_T2 = True; break
                last_cd_val_t2 = current_cd_val
            if not found_T2 and processed_any_for_t2: current_ph_thresholds['T2'] = f"> {cd_search_max:.3f}"; print(f"  T2 > {cd_search_max:.3f}"); found_T2 = True
            elif not found_T2 and not processed_any_for_t2: # T2搜索区间无效 (例如T1 > max)
                 current_ph_thresholds['T2'] = None if not (isinstance(current_ph_thresholds['T1'], str) and ">" in current_ph_thresholds['T1']) else current_ph_thresholds['T1'] #保持T1的状态
                 found_T2 = True


        if current_ph_thresholds.get('T2') == '错误': critical_thresholds_dict[ph_val_main] = current_ph_thresholds; continue

        # 阶段3: 寻找 T3
        start_cd_t3 = cd_search_min
        numeric_t2_val = None
        if isinstance(current_ph_thresholds.get('T2'), (float, int)): numeric_t2_val = current_ph_thresholds['T2']; start_cd_t3 = numeric_t2_val + cd_search_step
        elif isinstance(current_ph_thresholds.get('T2'), str) and ">" in current_ph_thresholds['T2']: found_T3 = True; current_ph_thresholds['T3'] = None
        elif current_ph_thresholds.get('T2') is None or (isinstance(current_ph_thresholds.get('T2'), str) and "<" in current_ph_thresholds['T2']):
            if isinstance(current_ph_thresholds.get('T1'), (float, int)): start_cd_t3 = current_ph_thresholds['T1'] + cd_search_step
        
        if not found_T3:
            last_cd_val_t3 = start_cd_t3 - cd_search_step
            processed_any_for_t3 = False
            for current_cd_val in np.arange(start_cd_t3, cd_search_max + cd_search_step, cd_search_step):
                if current_cd_val < cd_search_min : continue
                processed_any_for_t3 = True
                strategy_code = get_strategy_code_for_cd_detailed(ph_val_main, current_cd_val, fixed_soil_params_dict, normalized_fixed_biochar_list, (model1, model2))
                if strategy_code == -1:
                    print(f"  错误：在 Cd={current_cd_val:.3f} 预测失败 (寻找T3时)。"); current_ph_thresholds['T3'] = '错误'; found_T3 = True; break

                if last_cd_val_t3 < start_cd_t3 and strategy_code != 3:
                     current_ph_thresholds['T3'] = current_ph_thresholds.get('T2') if numeric_t2_val is not None else (current_ph_thresholds.get('T1') if isinstance(current_ph_thresholds.get('T1'), (float,int)) else f"< {start_cd_t3:.3f}")
                     print(f"  T3在T2(或T1)后未立即生效 (策略为{strategy_code} @ Cd={current_cd_val:.3f})。T3 ≈ {current_ph_thresholds['T3']}")
                     found_T3 = True; break

                if strategy_code != 3:
                    current_ph_thresholds['T3'] = round(last_cd_val_t3, 3)
                    print(f"  找到 T3 ≈ {current_ph_thresholds['T3']} (策略3转为{strategy_code} @ Cd={current_cd_val:.3f})")
                    found_T3 = True; break
                last_cd_val_t3 = current_cd_val
            if not found_T3 and processed_any_for_t3: current_ph_thresholds['T3'] = f"> {cd_search_max:.3f}"; print(f"  T3 > {cd_search_max:.3f}"); found_T3 = True
            elif not found_T3 and not processed_any_for_t3:
                 current_ph_thresholds['T3'] = None if not (isinstance(current_ph_thresholds.get('T2'), str) and ">" in current_ph_thresholds.get('T2')) else current_ph_thresholds.get('T2')
                 found_T3 = True
        
        critical_thresholds_dict[ph_val_main] = current_ph_thresholds

    # --- 显示和保存阈值结果 ---
    print("\n" + "="*50)
    print(f"--- 精确临界初始Cd阈值 (mg/kg, 范围 {cd_search_min:.1f}-{cd_search_max:.1f}) ---")
    print(f"--- (固定 OM={FIXED_ORGANIC_MATTER:.2f}, CEC={FIXED_CEC:.2f}, 固定生物炭) ---")
    print(f"--- (基于 {TOLERANCE_PCT}% 削减率容差) ---")
    print("="*50)
    thresholds_df = pd.DataFrame(critical_thresholds_dict).T
    thresholds_df.index.name = '土壤pH'
    thresholds_df.columns = [
        'T1: 策略1(CF无炭)上限',
        'T2: 策略2(AWD有炭)上限',
        'T3: 策略3(CF有炭)上限'
    ]
    print(thresholds_df)
    print("\n说明:")
    print(" - T1: 'CF无炭'有效的最大初始Cd。")
    print(" - T2: 'AWD+固定生物炭'有效的最大初始Cd (假设此时策略1已失效)。")
    print(" - T3: 'CF+固定生物炭'有效的最大初始Cd (假设此时策略1和2已失效)。")
    print(f" - '< {cd_search_min:.3f}'表示在搜索起点时该策略已无效或被更优策略覆盖。")
    print(f" - '> {cd_search_max:.3f}'表示该策略在搜索上限时仍然有效。")
    print(" - None表示该策略在该pH下从未成为主导的有效策略，或其有效区间完全被其他策略的区间覆盖或在搜索范围之外。")
    print(" - '错误'表示在寻找该阈值过程中发生模型预测错误。")

    output_dir = 'output_plots'
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f'thresholds_OM{FIXED_ORGANIC_MATTER:.2f}_CEC{FIXED_CEC:.2f}.csv'.replace(".","p")
    threshold_output_path = os.path.join(output_dir, csv_filename)
    try:
        thresholds_df.to_csv(threshold_output_path, encoding='utf-8-sig')
        print(f"\n阈值数据已保存到: {threshold_output_path}")
    except Exception as e: print(f"保存阈值数据失败: {e}")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plot_title_suffix = f"_OM{FIXED_ORGANIC_MATTER:.2f}_CEC{FIXED_CEC:.2f}".replace(".","p")
    plot_segmented_bars(thresholds_df, cd_search_min, cd_search_max, title_suffix=plot_title_suffix)

    print("\n处理完成!")
    return thresholds_df

if __name__ == "__main__":
    calculated_thresholds = main()