import joblib
import pandas as pd
import numpy as np
import os
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import time

# 忽略警告
warnings.filterwarnings('ignore')

# 检查NumPy版本并设置兼容的随机数生成器
def get_random_state(seed=42):
    """获取兼容的随机数生成器"""
    try:
        # 尝试使用新版本的API
        return np.random.default_rng(seed)
    except AttributeError:
        # 回退到旧版本的API
        return np.random.RandomState(seed)

# --- 1. 定义文件路径和加载模型 ---
NATURAL_AGING_MODEL_PATH_PH = './trained_models/natural_aging_model_pH-S-CH.joblib'
NATURAL_AGING_MODEL_PATH_CD = './trained_models/natural_aging_model_Cd-CH.joblib'
ARTIFICIAL_AGING_MODEL_PATH_PH = './trained_models/artificial_aging_model_pH-S-CH.joblib'
ARTIFICIAL_AGING_MODEL_PATH_CD = './trained_models/artificial_aging_model_Cd-CH.joblib'

try:
    natural_aging_model_pH = joblib.load(NATURAL_AGING_MODEL_PATH_PH)
    natural_aging_model_Cd = joblib.load(NATURAL_AGING_MODEL_PATH_CD)
    print("自然老化模型 (pH 和 Cd) 加载成功！")
except FileNotFoundError as e:
    print(f"错误: 找不到自然老化模型文件。请检查路径。详细信息: {e}")
    exit()

try:
    artificial_aging_model_pH = joblib.load(ARTIFICIAL_AGING_MODEL_PATH_PH)
    artificial_aging_model_Cd = joblib.load(ARTIFICIAL_AGING_MODEL_PATH_CD)
    print("人工老化模型 (pH 和 Cd) 加载成功！")
except Exception as e:
    print(f"错误: 加载人工老化模型时出错: {e}")
    exit()

# --- 2. 定义固定生物炭特性和土壤特性 ---
fixed_biochar_props = {
    'RM': 2, 'T': 400, 'pH-B': 9.7, 'BET': 3.6, 'BR': 0.01
}

ph_s_values = [5, 5.5, 6, 6.5, 7]
cd_c_values = [0.4, 1, 2, 3, 4]
FIXED_CEC_VALUE = 15

# --- 3. 定义目标自然老化情景参数 ---
target_aging_scenario = {
    'T-A': 1, 'FT-D': 160, 'DW-D': 2, 'C-30%-40%-D': 60, 'C-100%-D':60
}

# --- 4. 定义误差阈值 ---
PH_ERROR_THRESHOLD_PERCENT = 0.20
CD_ERROR_THRESHOLD_PERCENT = 0.10
PH_ABS_ERROR_THRESHOLD = 0.05
CD_ABS_ERROR_THRESHOLD = 0.02
PENALTY_VALUE = 1e6

def get_error(pred_val, target_val, rel_thresh_perc, abs_thresh_val):
    abs_error = abs(pred_val - target_val)
    if abs(target_val) < 1e-6:
        return abs_error, abs_error <= abs_thresh_val
    else:
        rel_error = abs_error / abs(target_val)
        return abs_error, rel_error <= rel_thresh_perc

# --- 5. 定义 hyperopt 搜索空间 ---
parameter_space_rf = {
    'fr_days': hp.randint('fr_days', 0, 31,),  # 0-30 天
    'dw_days': hp.randint('dw_days', 0, 31),  # 0-30 天
    'c_days': hp.randint('c_days', 0, 181),   # 0-180 天
    'c_temp': hp.randint('c_temp', 20, 41),   # 20-40 ℃
    'c_humidity': hp.randint('c_humidity', 60, 101)  # 60-100 %
}

# --- 6. 定义目标函数工厂 (用于 hyperopt) ---
def create_objective_function_for_hyperopt(target_d_pH_val, target_d_Cd_val, current_soil_props, active_aging_modes):
    def objective_function(params):
        fr_days = int(params['fr_days'])
        dw_days = int(params['dw_days'])
        c_days = int(params['c_days'])
        c_temp = int(params['c_temp'])
        c_humidity = int(params['c_humidity'])
        
        cost = 0.0
        total_days = fr_days + dw_days + c_days
        
        # 【已移除】时间惩罚 - 不再鼓励更短的处理时间
        # duration_penalty_weight = 0.001
        # cost += total_days * duration_penalty_weight
        
        # 平衡惩罚 - 鼓励各处理方法的平衡使用
        balance_penalty_weight = 0.01
        cost += np.std([fr_days, dw_days, c_days]) * balance_penalty_weight
        
        # 模式约束惩罚
        if 'ft' in active_aging_modes and fr_days == 0: cost += PENALTY_VALUE * 10
        if 'dw' in active_aging_modes and dw_days == 0: cost += PENALTY_VALUE * 10
        if 'c' in active_aging_modes and c_days == 0: cost += PENALTY_VALUE * 10
        if fr_days < 0 or dw_days < 0 or c_days < 0: cost += PENALTY_VALUE * 50
        if total_days > 180: cost += (total_days - 180) * 1000.0
        
        # 只有当c_days > 0时才使用温度和湿度
        effective_c_temp = float(c_temp) if c_days > 0 else 0.0
        effective_c_humidity = float(c_humidity) if c_days > 0 else 0.0
        
        # 准备模型输入数据
        biochar_props_transformed = fixed_biochar_props.copy()
        soil_props_transformed = current_soil_props.copy()
        
        # 对数变换
        if 'Cd-C' in soil_props_transformed:
            soil_props_transformed['Cd-C'] = np.log1p(float(soil_props_transformed['Cd-C']))
        if 'BET' in biochar_props_transformed:
            biochar_props_transformed['BET'] = np.log1p(float(biochar_props_transformed['BET']))
        
        input_data = {
            **biochar_props_transformed, **soil_props_transformed,
            'FT-D': float(fr_days), 'DW-D': float(dw_days), 'C-D': float(c_days),
            'C-T': effective_c_temp, 'C-W': effective_c_humidity
        }
        artificial_model_input_df = pd.DataFrame([input_data])
        
        try:
            # 数据类型转换和验证
            for col in artificial_model_input_df.columns:
                artificial_model_input_df[col] = pd.to_numeric(artificial_model_input_df[col], errors='coerce')
            if artificial_model_input_df.isnull().values.any(): 
                return {'loss': PENALTY_VALUE * 20, 'status': STATUS_OK}
            
            # 模型预测
            pred_d_pH = float(artificial_aging_model_pH.predict(artificial_model_input_df)[0])
            pred_d_Cd = float(artificial_aging_model_Cd.predict(artificial_model_input_df)[0])
            
            if np.isnan(pred_d_pH) or np.isnan(pred_d_Cd): 
                return {'loss': PENALTY_VALUE * 10, 'status': STATUS_OK}
            
            # 预测误差成本
            cost += (pred_d_pH - target_d_pH_val)**2 * 0.5
            cost += (pred_d_Cd - target_d_Cd_val)**2 * 0.5
            
            # 阈值检查
            _, within_thresh_pH = get_error(pred_d_pH, target_d_pH_val, PH_ERROR_THRESHOLD_PERCENT, PH_ABS_ERROR_THRESHOLD)
            _, within_thresh_Cd = get_error(pred_d_Cd, target_d_Cd_val, CD_ERROR_THRESHOLD_PERCENT, CD_ABS_ERROR_THRESHOLD)
            
            if not within_thresh_pH: cost += PENALTY_VALUE
            if not within_thresh_Cd: cost += PENALTY_VALUE
            
            # 返回结果，包含额外信息用于调试
            return {
                'loss': cost,
                'status': STATUS_OK,
                'eval_time': time.time(),
                'pred_d_pH': pred_d_pH,
                'pred_d_Cd': pred_d_Cd,
                'within_thresh_pH': within_thresh_pH,
                'within_thresh_Cd': within_thresh_Cd
            }
            
        except Exception as e:
            return {'loss': PENALTY_VALUE * 10, 'status': STATUS_OK}
            
    return objective_function

# --- 7. hyperopt 优化运行函数 (简化版本) ---
def hyperopt_optimization_run(target_d_pH, target_d_Cd, current_soil_props, active_aging_modes):
    print("   启动 hyperopt 贝叶斯优化...")
    start_time = time.time()
    
    objective_func = create_objective_function_for_hyperopt(target_d_pH, target_d_Cd, current_soil_props, active_aging_modes)
    
    # 创建 Trials 对象来存储优化历史
    trials = Trials()
    
    try:
        # 运行优化 - 使用简化的可靠参数配置
        best_params = fmin(
            fn=objective_func,               # 目标函数
            space=parameter_space_rf,        # 搜索空间
            algo=tpe.suggest,                # 贝叶斯优化算法
            max_evals=50,
            trials=trials                    # 最大评估次数
        )
        
        end_time = time.time()
        print(f"   hyperopt 优化完成，用时: {end_time - start_time:.2f} 秒")
        
        # 获取最佳结果
        best_trial = trials.best_trial
        best_cost = best_trial['result']['loss']
        
        # 转换参数为整数
        fr_days = int(best_params['fr_days'])
        dw_days = int(best_params['dw_days'])
        c_days = int(best_params['c_days'])
        c_temp = int(best_params['c_temp'])
        c_humidity = int(best_params['c_humidity'])
        
        best_params_int = [fr_days, dw_days, c_days, c_temp, c_humidity]
        
        # 验证最佳参数
        effective_c_temp = float(c_temp) if c_days > 0 else 0.0
        effective_c_humidity = float(c_humidity) if c_days > 0 else 0.0
        
        biochar_props_verify = fixed_biochar_props.copy()
        soil_props_verify = current_soil_props.copy()
        if 'Cd-C' in soil_props_verify: soil_props_verify['Cd-C'] = np.log1p(float(soil_props_verify['Cd-C']))
        if 'BET' in biochar_props_verify: biochar_props_verify['BET'] = np.log1p(float(biochar_props_verify['BET']))
        
        verify_input_data = {
            **biochar_props_verify, **soil_props_verify,
            'FT-D': float(fr_days), 'DW-D': float(dw_days), 'C-D': float(c_days),
            'C-T': effective_c_temp, 'C-W': effective_c_humidity
        }
        verify_input_df = pd.DataFrame([verify_input_data])
        
        pred_d_pH = float(artificial_aging_model_pH.predict(verify_input_df)[0])
        pred_d_Cd = float(artificial_aging_model_Cd.predict(verify_input_df)[0])
        
        _, within_thresh_pH = get_error(pred_d_pH, target_d_pH, PH_ERROR_THRESHOLD_PERCENT, PH_ABS_ERROR_THRESHOLD)
        _, within_thresh_Cd = get_error(pred_d_Cd, target_d_Cd, CD_ERROR_THRESHOLD_PERCENT, CD_ABS_ERROR_THRESHOLD)
        
        # 打印优化统计信息
        print(f"   优化统计: 评估了 {len(trials.trials)} 个参数组合")
        print(f"   最佳损失: {best_cost:.6f}")
        
        return best_params_int, best_cost, (pred_d_pH, pred_d_Cd), (within_thresh_pH, within_thresh_Cd), trials

    except Exception as e:
        print(f"   优化过程中出现错误: {e}")
        return None, float('inf'), (None, None), (False, False), None

# --- 8. 主程序执行循环 ---
if __name__ == "__main__":
    all_results = []
    
    total_scenarios = len(ph_s_values) * len(cd_c_values)
    print(f"开始使用 hyperopt 贝叶斯优化算法处理 {total_scenarios} 种土壤情景 (CEC固定为{FIXED_CEC_VALUE})...")
    print(f"NumPy 版本: {np.__version__}")
    
    scenario_count = 0
    total_start_time = time.time()
    
    for ph_s in ph_s_values:
        for cd_c in cd_c_values:
            scenario_count += 1
            current_soil_props = {'pH-S': ph_s, 'Cd-C': cd_c, 'CEC': FIXED_CEC_VALUE}
            
            print(f"\n--- [{scenario_count}/{total_scenarios}] 优化土壤情景：pH-S={ph_s}, Cd-C={cd_c}, CEC={FIXED_CEC_VALUE} ---")
            
            # 计算目标自然老化效果
            biochar_props_nat = fixed_biochar_props.copy()
            soil_props_nat = current_soil_props.copy()
            if 'Cd-C' in soil_props_nat: soil_props_nat['Cd-C'] = np.log1p(float(soil_props_nat['Cd-C']))
            if 'BET' in biochar_props_nat: biochar_props_nat['BET'] = np.log1p(float(biochar_props_nat['BET']))
            
            natural_model_input_data = {**biochar_props_nat, **soil_props_nat, **target_aging_scenario}
            natural_model_input_df = pd.DataFrame([natural_model_input_data])
            
            try:
                target_d_pH = float(natural_aging_model_pH.predict(natural_model_input_df)[0])
                target_d_Cd = float(natural_aging_model_Cd.predict(natural_model_input_df)[0])
                print(f"   目标 ΔpH: {target_d_pH:.4f}, ΔCd: {target_d_Cd:.4f}")
            except Exception as e:
                print(f"   错误: 计算目标自然老化效果失败，跳过此情景。错误: {e}")
                continue
            
            # 确定活跃的老化模式
            active_modes = set()
            if target_aging_scenario['FT-D'] > 0: active_modes.add('ft')
            if target_aging_scenario['DW-D'] > 0: active_modes.add('dw')
            if target_aging_scenario['C-30%-40%-D'] > 0 or target_aging_scenario['C-100%-D'] > 0: active_modes.add('c')
            
            # 运行 hyperopt 优化
            best_params, best_cost, best_preds, best_threshs, trials = hyperopt_optimization_run(
                target_d_pH, target_d_Cd, current_soil_props, active_modes
            )
            
            if best_params is None:
                print("   警告: 优化失败，跳过此情景")
                continue
            
            fr_days_final, dw_days_final, c_days_final, c_temp_final, c_humidity_final = best_params
            pred_d_pH_opt, pred_d_Cd_opt = best_preds
            within_thresh_pH_opt, within_thresh_Cd_opt = best_threshs
            
            effective_c_temp_final = float(c_temp_final) if c_days_final > 0 else 0.0
            effective_c_humidity_final = float(c_humidity_final) if c_days_final > 0 else 0.0
            
            # 确定处理模式
            mode_desc = "混合模式"
            if c_days_final > 0 and fr_days_final == 0 and dw_days_final == 0: mode_desc = "纯恒温恒湿模式"
            elif (fr_days_final > 0 or dw_days_final > 0) and c_days_final == 0: mode_desc = "纯冻融干湿模式"
            elif fr_days_final == 0 and dw_days_final == 0 and c_days_final == 0: mode_desc = "无老化发生"
            
            print(f"   最佳参数 (成本: {best_cost:.4f}):")
            print(f"     冻融天数: {fr_days_final} 天, 干湿天数: {dw_days_final} 天, 恒温恒湿天数: {c_days_final} 天")
            if c_days_final > 0: print(f"     温度: {effective_c_temp_final:.0f} ℃, 湿度: {effective_c_humidity_final:.0f} %")
            print(f"     模式: {mode_desc}")
            
            abs_err_pH_opt, _ = get_error(pred_d_pH_opt, target_d_pH, PH_ERROR_THRESHOLD_PERCENT, PH_ABS_ERROR_THRESHOLD)
            abs_err_Cd_opt, _ = get_error(pred_d_Cd_opt, target_d_Cd, CD_ERROR_THRESHOLD_PERCENT, CD_ABS_ERROR_THRESHOLD)
            
            print(f"   预测效果:")
            print(f"     ΔpH: {pred_d_pH_opt:.4f} (误差: {abs_err_pH_opt:.4f}, 达标: {within_thresh_pH_opt})")
            print(f"     ΔCd: {pred_d_Cd_opt:.4f} (误差: {abs_err_Cd_opt:.4f}, 达标: {within_thresh_Cd_opt})")
            
            if within_thresh_pH_opt and within_thresh_Cd_opt: print("     🎉 方案达到所有指标要求！")
            else: print("     ⚠️ 方案未完全达到指标要求")
            
            # 保存结果
            result = {
                '模拟总年限': target_aging_scenario['T-A'], 'PH-S': ph_s, 'Cd-C': cd_c,
                'CEC': FIXED_CEC_VALUE,
                '自然累积FT-D': target_aging_scenario['FT-D'], '自然累积DW-D': target_aging_scenario['DW-D'],
                '自然累积C-30%-40%-D': target_aging_scenario['C-30%-40%-D'], '自然累积C-100%-D': target_aging_scenario['C-100%-D'],
                '目标ΔpH': target_d_pH, '目标ΔCd': target_d_Cd,
                '推荐人工FT-D': fr_days_final, '推荐人工DW-D': dw_days_final, '推荐人工C-D': c_days_final,
                '推荐人工C-T': effective_c_temp_final, '推荐人工C-W': effective_c_humidity_final,
                '推断人工模式': mode_desc, '模拟人工ΔpH': pred_d_pH_opt, '模拟人工ΔCd': pred_d_Cd_opt,
                'ΔpH_误差': abs_err_pH_opt, 'ΔCd_误差': abs_err_Cd_opt,
                'ΔpH_在阈值内': within_thresh_pH_opt, 'ΔCd_在阈值内': within_thresh_Cd_opt,
                '优化成本': best_cost,
                '优化评估次数': len(trials.trials)
            }
            all_results.append(result)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
   
    # 保存结果
    results_df = pd.DataFrame(all_results)
    output_csv_path = './optimization_results_hyperopt_final_modified(东北).csv'
    results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n\n=== hyperopt 贝叶斯优化完成 ===")
    print(f"总耗时: {total_time:.2f} 秒 (平均每个情景: {total_time/max(1, len(all_results)):.2f} 秒)")
    print(f"处理了 {len(all_results)} 种土壤情景")
    print(f"结果已保存至: {output_csv_path}")
    
    if all_results:
        success_count = sum(1 for result in all_results if result['ΔpH_在阈值内'] and result['ΔCd_在阈值内'])
        print(f"成功找到满足条件解的情景数: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)")
        
        avg_fr_days = np.mean([r['推荐人工FT-D'] for r in all_results])
        avg_dw_days = np.mean([r['推荐人工DW-D'] for r in all_results])
        avg_c_days = np.mean([r['推荐人工C-D'] for r in all_results])
        
        print(f"平均推荐参数: 冻融={avg_fr_days:.1f}天, 干湿={avg_dw_days:.1f}天, 恒温恒湿={avg_c_days:.1f}天")
        avg_evals = np.mean([r['优化评估次数'] for r in all_results])
        print(f"平均每个情景评估次数: {avg_evals:.1f}")