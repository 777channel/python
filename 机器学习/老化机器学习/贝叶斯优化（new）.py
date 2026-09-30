import joblib
import pandas as pd
import numpy as np
import warnings
from skopt import gp_minimize
from skopt.space import Integer, Real
warnings.filterwarnings('ignore')

# 禁用analytics追踪
import os
os.environ['ANALYTICS_WRITE_KEY'] = ''

# =============================================================================
# 配置信息
# =============================================================================
DATA_FILE_PATH_ARTIFICIAL = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
VERSION_SUFFIX_ARTIFICIAL = "v4_tabpfen_dynamic_device"
DATA_FILE_PATH_NATURAL = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\naturliche Aging（测试版本）.csv" 
VERSION_SUFFIX_NATURAL_PREFIX = "Natural_Aging_AutoTabPFN" 

OPTIMIZATION_STRATEGY = "CD_PRIORITY"
CD_WEIGHT = 0.7
PH_WEIGHT = 0.3
CD_TOLERANCE = 0.05
DAYS_MINIMIZE_WEIGHT = 0.1

# 地区参数配置
REGIONAL_PARAMS = [
    {'name': 'Northeast', 'DW-D': 2, 'FT-D': 150, 'C-30%-40%-D': 40, 'C-100%-D': 40, 'pH-S': 7, 'Cd-C_list': [1.0, 2.0, 3.0]},
    {'name': 'North China', 'DW-D': 1, 'FT-D': 109, 'C-30%-40%-D': 30, 'C-100%-D': 30, 'pH-S': 7, 'Cd-C_list': [1.0, 2.0, 3.0]},
    {'name': 'Central China', 'DW-D': 5, 'FT-D': 20, 'C-30%-40%-D': 80, 'C-100%-D': 80, 'pH-S': 5, 'Cd-C_list': [0.5, 1.0, 1.5]},
    {'name': 'South China', 'DW-D': 10, 'FT-D': 0, 'C-30%-40%-D': 100, 'C-100%-D': 100, 'pH-S': 5, 'Cd-C_list': [0.5, 1.0, 1.5]},
    {'name': 'East China', 'DW-D': 3, 'FT-D': 80, 'C-30%-40%-D': 60, 'C-100%-D': 60, 'pH-S': 6, 'Cd-C_list': [0.5, 1.0, 2.0]},
    {'name': 'Southwest', 'DW-D': 8, 'FT-D': 10, 'C-30%-40%-D': 90, 'C-100%-D': 90, 'pH-S': 6, 'Cd-C_list': [0.5, 1.0, 2.0]}
]

FIXED_PARAMS = {'CEC': 15, 'RM': 2, 'T': 400, 'pH-B': 9.7, 'BET': 3.6, 'BR': 0.01}
AGING_YEARS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 生成所有情景
BASE_CASES = []
for region in REGIONAL_PARAMS:
    for cd_value in region['Cd-C_list']:
        BASE_CASES.append({
            'name': f"{region['name']}_pH={region['pH-S']}_Cd={cd_value}",
            'region_data': {**region, 'Cd-C': cd_value}
        })

def calculate_decay_rate(values):
    if len(values) < 2:
        return []
    decay_rates = []
    for i in range(1, len(values)):
        rate = (values[i] - values[i-1]) / abs(values[i-1]) if values[i-1] != 0 else values[i] - values[i-1]
        decay_rates.append(rate)
    return decay_rates

def combined_dual_target_validation(cd_natural_values, cd_artificial_values, ph_natural_values, ph_artificial_values):
    cd_natural_decay = calculate_decay_rate(cd_natural_values)
    cd_artificial_decay = calculate_decay_rate(cd_artificial_values)
    ph_natural_decay = calculate_decay_rate(ph_natural_values)
    ph_artificial_decay = calculate_decay_rate(ph_artificial_values)
    
    if len(cd_natural_decay) >= 2:
        cd_correlation = np.corrcoef(cd_natural_decay, cd_artificial_decay)[0, 1]
        cd_correlation = 0 if np.isnan(cd_correlation) else cd_correlation
    else:
        cd_correlation = 1 if len(cd_natural_decay) == 1 and abs(cd_natural_decay[0] - cd_artificial_decay[0]) < 1e-6 else 0
    
    if len(ph_natural_decay) >= 2:
        ph_correlation = np.corrcoef(ph_natural_decay, ph_artificial_decay)[0, 1]
        ph_correlation = 0 if np.isnan(ph_correlation) else ph_correlation
    else:
        ph_correlation = 1 if len(ph_natural_decay) == 1 and abs(ph_natural_decay[0] - ph_artificial_decay[0]) < 1e-6 else 0
    
    combined_natural_decay = cd_natural_decay + ph_natural_decay
    combined_artificial_decay = cd_artificial_decay + ph_artificial_decay
    
    if len(combined_natural_decay) >= 2:
        combined_correlation = np.corrcoef(combined_natural_decay, combined_artificial_decay)[0, 1]
        combined_correlation = 0 if np.isnan(combined_correlation) else combined_correlation
    else:
        combined_correlation = (cd_correlation + ph_correlation) / 2
    
    weighted_decay_correlation = cd_correlation * CD_WEIGHT + ph_correlation * PH_WEIGHT
    
    def get_assessment(correlation):
        if correlation >= 0.8:
            return "Excellent"
        elif correlation >= 0.6:
            return "Good"
        elif correlation >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    cd_value_correlation = np.corrcoef(cd_natural_values, cd_artificial_values)[0, 1] if len(cd_natural_values) > 1 else 1
    ph_value_correlation = np.corrcoef(ph_natural_values, ph_artificial_values)[0, 1] if len(ph_natural_values) > 1 else 1
    cd_value_correlation = 0 if np.isnan(cd_value_correlation) else cd_value_correlation
    ph_value_correlation = 0 if np.isnan(ph_value_correlation) else ph_value_correlation
    combined_value_correlation = cd_value_correlation * CD_WEIGHT + ph_value_correlation * PH_WEIGHT
    
    return {
        'cd_decay_correlation': cd_correlation,
        'ph_decay_correlation': ph_correlation,
        'combined_decay_correlation': combined_correlation,
        'weighted_decay_correlation': weighted_decay_correlation,
        'cd_value_correlation': cd_value_correlation,
        'ph_value_correlation': ph_value_correlation,
        'combined_value_correlation': combined_value_correlation,
        'cd_natural_decay': cd_natural_decay,
        'cd_artificial_decay': cd_artificial_decay,
        'ph_natural_decay': ph_natural_decay,
        'ph_artificial_decay': ph_artificial_decay,
        'decay_assessment': get_assessment(weighted_decay_correlation),
        'value_assessment': get_assessment(combined_value_correlation),
        'overall_assessment': get_assessment((weighted_decay_correlation + combined_value_correlation) / 2)
    }

def create_objective_function(cd_target, ph_target, models, features, param_names_to_optimize, natural_aging_types):
    def predict_targets(params):
        params_dict = dict(zip(param_names_to_optimize, params))
        art_sample = {feat: 0 for feat in features['artificial']}
        art_sample.update(FIXED_PARAMS)
        art_sample.update({'pH-S': 5, 'Cd-C': 1, 'T_aging': 4})
        art_sample['DW-D'] = params_dict.get('DW-D', 0)
        art_sample['FT-D'] = params_dict.get('FT-D', 0)
        if 'C-D' in features['artificial']:
            art_sample['C-D'] = params_dict.get('C-D', 0)
        if 'C-T' in features['artificial']:
            art_sample['C-T'] = params_dict.get('C-T', 25)
        if 'C-W' in features['artificial']:
            art_sample['C-W'] = params_dict.get('C-W', 60) / 100.0
        
        df_art_virtual = pd.DataFrame([art_sample])[features['artificial']]
        cd_prediction = models['Cd-CH_artificial'].predict(df_art_virtual)[0]
        ph_prediction = models['pH-S-CH_artificial'].predict(df_art_virtual)[0]
        total_days = params_dict.get('DW-D', 0) + params_dict.get('FT-D', 0) + params_dict.get('C-D', 0)
        return cd_prediction, ph_prediction, total_days
    
    def apply_type_penalty(params):
        params_dict = dict(zip(param_names_to_optimize, params))
        penalty = 0
        for aging_type in ['DW-D', 'FT-D', 'C-D']:
            if aging_type in natural_aging_types and natural_aging_types[aging_type] > 0:
                if params_dict.get(aging_type, 0) == 0:
                    penalty += 1000
        return penalty
    
    def cd_priority_objective(params):
        cd_prediction, ph_prediction, total_days = predict_targets(params)
        cd_error = np.abs(cd_prediction - cd_target)
        ph_error = np.abs(ph_prediction - ph_target)
        normalized_days = total_days / 300.0
        type_penalty = apply_type_penalty(params)
        
        if cd_error <= CD_TOLERANCE:
            combined_error = cd_error * 0.3 + ph_error * 0.7
        else:
            combined_error = cd_error * CD_WEIGHT + ph_error * PH_WEIGHT
        
        return combined_error + DAYS_MINIMIZE_WEIGHT * normalized_days + type_penalty
    
    return cd_priority_objective

def find_equivalent_aging_days_yearly():
    all_final_results = []
    
    # 加载模型
    models = {}
    features = {}
    
    for target_column_name in ['Cd-CH', 'pH-S-CH']:
        version_suffix_art = f"{target_column_name}_{VERSION_SUFFIX_ARTIFICIAL}"
        model_filename_art = f'./saved_models/pipeline_{version_suffix_art}.joblib'
        models[f'{target_column_name}_artificial'] = joblib.load(model_filename_art)
        
        version_suffix_nat = f"{VERSION_SUFFIX_NATURAL_PREFIX}_{target_column_name}"
        model_filename_nat = f'./saved_models/pipeline_{version_suffix_nat}.joblib'
        models[f'{target_column_name}_natural'] = joblib.load(model_filename_nat)
    
    df_template_art = pd.read_csv(DATA_FILE_PATH_ARTIFICIAL, encoding='utf-8-sig')
    features['artificial'] = [c for c in df_template_art.columns if c not in ['pH-S-CH', 'Cd-CH', 'T_aging']]
    
    df_template_nat = pd.read_csv(DATA_FILE_PATH_NATURAL, encoding='utf-8-sig')
    features['natural'] = [c for c in df_template_nat.columns if c not in ['pH-S-CH', 'Cd-CH', 'T-Aging']]
    
    # 循环每个情景
    for case in BASE_CASES:
        region_data = case['region_data']
        region_name = region_data['name']
        case_name = case['name']
        current_cd = region_data['Cd-C']
        
        yearly_results = []
        cd_natural_predictions = []
        cd_artificial_predictions = []
        ph_natural_predictions = []
        ph_artificial_predictions = []
        
        for year in AGING_YEARS:
            natural_sample = {feat: 0 for feat in features['natural']}
            natural_sample.update(FIXED_PARAMS)
            natural_sample.update({'pH-S': region_data['pH-S'], 'Cd-C': current_cd, 'C-T': 25, 'C-W': 0.6})
            
            scaled_region_data = {}
            natural_aging_types = {}
            for key, value in region_data.items():
                if key.endswith('-D'):
                    scaled_region_data[key] = value * year
                    natural_aging_types[key] = value
                elif key not in ['name', 'pH-S', 'Cd-C', 'Cd-C_list']:
                    scaled_region_data[key] = value
            
            natural_sample.update(scaled_region_data)
            natural_sample['T-A'] = float(year)
            
            df_natural_virtual = pd.DataFrame([natural_sample])[features['natural']]
            cd_target = models['Cd-CH_natural'].predict(df_natural_virtual)[0]
            ph_target = models['pH-S-CH_natural'].predict(df_natural_virtual)[0]
            
            cd_natural_predictions.append(cd_target)
            ph_natural_predictions.append(ph_target)
            
            # 构建搜索空间
            search_space = []
            param_names_to_optimize = []
            
            dw_natural_value = region_data.get('DW-D', 0)
            ft_natural_value = region_data.get('FT-D', 0)
            
            search_space.append(Integer(1 if dw_natural_value > 0 else 0, 60, name='DW-D'))
            param_names_to_optimize.append('DW-D')
            search_space.append(Integer(1 if ft_natural_value > 0 else 0, 60, name='FT-D'))
            param_names_to_optimize.append('FT-D')
            search_space.append(Integer(1, 120, name='C-D'))
            param_names_to_optimize.append('C-D')
            search_space.append(Real(25, 40, name='C-T'))
            param_names_to_optimize.append('C-T')
            search_space.append(Real(40, 100, name='C-W'))
            param_names_to_optimize.append('C-W')
            
            if not search_space:
                continue
            
            objective_function = create_objective_function(
                cd_target, ph_target, models, features, param_names_to_optimize, natural_aging_types
            )
            
            result = gp_minimize(
                func=objective_function, dimensions=search_space,
                n_calls=100, random_state=42, n_initial_points=20
            )
            
            best_params_dict = dict(zip(param_names_to_optimize, result.x))
            
            final_art_sample = {feat: 0 for feat in features['artificial']}
            final_art_sample.update(FIXED_PARAMS)
            final_art_sample.update({'pH-S': region_data['pH-S'], 'Cd-C': current_cd, 'T_aging': 4})
            final_art_sample['DW-D'] = best_params_dict.get('DW-D', 0)
            final_art_sample['FT-D'] = best_params_dict.get('FT-D', 0)
            if 'C-D' in features['artificial']: 
                final_art_sample['C-D'] = best_params_dict.get('C-D', 0)
            if 'C-T' in features['artificial']: 
                final_art_sample['C-T'] = best_params_dict.get('C-T', 25)
            if 'C-W' in features['artificial']: 
                final_art_sample['C-W'] = best_params_dict.get('C-W', 60) / 100.0
            
            df_final_art_virtual = pd.DataFrame([final_art_sample])[features['artificial']]
            cd_final_prediction = models['Cd-CH_artificial'].predict(df_final_art_virtual)[0]
            ph_final_prediction = models['pH-S-CH_artificial'].predict(df_final_art_virtual)[0]
            
            cd_artificial_predictions.append(cd_final_prediction)
            ph_artificial_predictions.append(ph_final_prediction)
            
            total_days = best_params_dict.get('DW-D', 0) + best_params_dict.get('FT-D', 0) + best_params_dict.get('C-D', 0)
            theoretical_dw = dw_natural_value * year
            theoretical_ft = ft_natural_value * year
            
            yearly_result = {
                'Region': region_name,
                'Scenario': case_name,
                'Year': year,
                'Optimization_Strategy': OPTIMIZATION_STRATEGY,
                'pH_S': region_data['pH-S'],
                'Cd_C': current_cd,
                'Cd_Natural_Target': round(cd_target, 4),
                'Cd_Artificial_Prediction': round(cd_final_prediction, 4),
                'pH_Natural_Target': round(ph_target, 4),
                'pH_Artificial_Prediction': round(ph_final_prediction, 4),
                'Objective_Function_Value': round(result.fun, 4),
                'Cd_Individual_Error': round(abs(cd_final_prediction - cd_target), 4),
                'pH_Individual_Error': round(abs(ph_final_prediction - ph_target), 4),
                'Total_Days': total_days,
                'Equivalent_DW-D': best_params_dict.get('DW-D', 0),
                'Equivalent_FT-D': best_params_dict.get('FT-D', 0),
                'Equivalent_C-D': best_params_dict.get('C-D', 0),
                'Equivalent_C-T': round(best_params_dict.get('C-T', 0), 2),
                'Equivalent_C-W': round(best_params_dict.get('C-W', 0), 2),
                'Natural_DW-D': dw_natural_value * year,
                'Natural_FT-D': ft_natural_value * year,
                'Theoretical_DW-D': theoretical_dw,
                'Theoretical_FT-D': theoretical_ft,
                'DW-D_Ratio': round(best_params_dict.get('DW-D', 0) / theoretical_dw, 3) if theoretical_dw > 0 else 'N/A',
                'FT-D_Ratio': round(best_params_dict.get('FT-D', 0) / theoretical_ft, 3) if theoretical_ft > 0 else 'N/A',
                'Year_Multiplier': year,
            }
            yearly_results.append(yearly_result)
        
        # 验证分析
        if len(cd_natural_predictions) >= 3:
            validation_results = combined_dual_target_validation(
                cd_natural_predictions, cd_artificial_predictions,
                ph_natural_predictions, ph_artificial_predictions
            )
            
            for yearly_result in yearly_results:
                yearly_result.update({
                    'Cd_Decay_Correlation': round(validation_results['cd_decay_correlation'], 4),
                    'pH_Decay_Correlation': round(validation_results['ph_decay_correlation'], 4),
                    'Weighted_Decay_Correlation': round(validation_results['weighted_decay_correlation'], 4),
                    'Combined_Decay_Correlation': round(validation_results['combined_decay_correlation'], 4),
                    'Cd_Value_Correlation': round(validation_results['cd_value_correlation'], 4),
                    'pH_Value_Correlation': round(validation_results['ph_value_correlation'], 4),
                    'Combined_Value_Correlation': round(validation_results['combined_value_correlation'], 4),
                    'Decay_Assessment': validation_results['decay_assessment'],
                    'Value_Assessment': validation_results['value_assessment'],
                    'Overall_Assessment': validation_results['overall_assessment']
                })
            
            summary_result = {
                'Region': region_name,
                'Scenario': case_name,
                'Year': 'VALIDATION_SUMMARY',
                'Optimization_Strategy': OPTIMIZATION_STRATEGY,
                'pH_S': region_data['pH-S'],
                'Cd_C': current_cd,
                'Cd_Decay_Correlation': round(validation_results['cd_decay_correlation'], 4),
                'pH_Decay_Correlation': round(validation_results['ph_decay_correlation'], 4),
                'Weighted_Decay_Correlation': round(validation_results['weighted_decay_correlation'], 4),
                'Combined_Decay_Correlation': round(validation_results['combined_decay_correlation'], 4),
                'Cd_Value_Correlation': round(validation_results['cd_value_correlation'], 4),
                'pH_Value_Correlation': round(validation_results['ph_value_correlation'], 4),
                'Combined_Value_Correlation': round(validation_results['combined_value_correlation'], 4),
                'Decay_Assessment': validation_results['decay_assessment'],
                'Value_Assessment': validation_results['value_assessment'],
                'Overall_Assessment': validation_results['overall_assessment'],
                'Cd_Natural_Decay_Rates': str([round(x, 4) for x in validation_results['cd_natural_decay']]),
                'Cd_Artificial_Decay_Rates': str([round(x, 4) for x in validation_results['cd_artificial_decay']]),
                'pH_Natural_Decay_Rates': str([round(x, 4) for x in validation_results['ph_natural_decay']]),
                'pH_Artificial_Decay_Rates': str([round(x, 4) for x in validation_results['ph_artificial_decay']]),
                'Average_Cd_Error': round(np.mean([yr['Cd_Individual_Error'] for yr in yearly_results]), 4),
                'Average_pH_Error': round(np.mean([yr['pH_Individual_Error'] for yr in yearly_results]), 4),
                'Max_Cd_Error': round(np.max([yr['Cd_Individual_Error'] for yr in yearly_results]), 4),
                'Max_pH_Error': round(np.max([yr['pH_Individual_Error'] for yr in yearly_results]), 4),
                'Average_Total_Days': round(np.mean([yr['Total_Days'] for yr in yearly_results]), 2),
                'Min_Total_Days': min([yr['Total_Days'] for yr in yearly_results]),
                'Max_Total_Days': max([yr['Total_Days'] for yr in yearly_results])
            }
            yearly_results.append(summary_result)
        
        all_final_results.extend(yearly_results)
    
    # 保存结果
    if all_final_results:
        final_df = pd.DataFrame(all_final_results)
        output_filename = 'regional_aging_optimization_days_first_10years_multi_Cd(1%).csv'
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"完成! 结果已保存至: {output_filename}")

if __name__ == "__main__":
    find_equivalent_aging_days_yearly()