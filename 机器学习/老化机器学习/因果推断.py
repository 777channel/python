import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from tabpfn import TabPFNClassifier, TabPFNRegressor
from econml.dr import LinearDRLearner
import warnings

warnings.filterwarnings('ignore')

# --- 用户已确认的列名 (全局常量) ---
DW_BASE_NAME = 'DW'
FT_BASE_NAME = 'FT'
C_BASE_NAME = 'C'
# 原始条件列名，主要用于理解 _eff 列的来源 (因为这些原始列在最终数据中可能已被删除)
DW_DURATION_RAW_COL = 'DW-D'
FT_DURATION_RAW_COL = 'FT-D'
C_DAYS_RAW_COL = 'C-D'
C_TEMP_RAW_COL = 'C-T'
C_MOISTURE_RAW_COL = 'C-W'

OUTPUT_COL_NAME = 'Cd-CH' # 结果变量 Y
RAW_MATERIAL_COL = 'RM'   # 原料列
CEC_COL_NAME = 'CEC'      # CEC列名，用于专门的KNN插补
# --- 全局列名常量定义结束 ---


# --- 因果推断主函数 ---
def run_causal_inference_with_control_and_relative(
    df_final_processed: pd.DataFrame,
    y_col_name: str,
    x_cols: list, 
    w_cols: list, 
    t_col_name: str = 'T_aging_protocol'
):
    print("\nINFO: 开始因果推断流程 (包含与T=0的对比及处理组间相对比较)...")
    
    y_ph, t_ph = y_col_name, t_col_name 
    x_cols_str = ', '.join(x_cols) if x_cols else '无' 
    w_cols_str = ', '.join(w_cols) if w_cols else '无'
    print(f"""
    --- INFO: Variable Setup (基于用户提供的最终分类) ---
    Y (Outcome): '{y_ph}'
    T (Treatment): '{t_ph}' (编码0-4, 其中T=0为控制组)
    X (Heterogeneity Features): [{x_cols_str}]
    W (Confounders): [{w_cols_str}]
    """)

    # 验证关键列
    critical_cols_for_model = [y_col_name, t_col_name] + (x_cols or []) + (w_cols or [])
    critical_cols_for_model = list(dict.fromkeys(critical_cols_for_model)) 
    
    missing_critical_cols = [col for col in critical_cols_for_model if col not in df_final_processed.columns]
    if missing_critical_cols:
        print(f"错误: 以下模型所需的关键列在您的CSV文件中未找到: {missing_critical_cols}")
        return None, None

    # 对选定的Y,T,X,W列进行最终的 NaN 检查和填充
    df_subset_for_model = df_final_processed[critical_cols_for_model].copy()
    if df_subset_for_model.isnull().values.any():
        print(f"警告: 输入给模型的列中仍检测到NaN值！将用中位数/众数填充。NaN详情:")
        nan_summary = df_subset_for_model.isnull().sum()
        print(nan_summary[nan_summary > 0])
        for col in critical_cols_for_model: 
            if df_final_processed[col].isnull().any(): # 在原始传入的df上操作
                if pd.api.types.is_numeric_dtype(df_final_processed[col]):
                    fill_value = df_final_processed[col].median()
                else: 
                    fill_value = df_final_processed[col].mode()[0] if not df_final_processed[col].mode().empty else 0
                df_final_processed[col].fillna(fill_value, inplace=True)
        print("最终NaN填充尝试完成。")
    
    Y_series = df_final_processed[y_col_name].astype(np.float32)
    T_series = df_final_processed[t_col_name].astype(int)
    
    unique_t_values_in_data_fit = sorted(T_series.unique())
    print(f"INFO: 用于模型拟合的T值 (Treatment levels): {unique_t_values_in_data_fit}")
    if 0 not in unique_t_values_in_data_fit:
        print("警告: T=0 (控制组) 未在处理变量T中找到！ .summary() 的结果可能无法正确解释为与控制组的比较。")
    elif df_final_processed[T_series == 0].shape[0] < 10: 
        print(f"警告: T=0 (控制组) 的样本量较少 ({df_final_processed[T_series == 0].shape[0]}个)，模型结果可能不稳定。")

    X_df = pd.DataFrame()
    X_fit_arg = None
    if x_cols: 
        X_df = df_final_processed[x_cols].copy().astype(np.float32)
        X_fit_arg = X_df
    
    W_df = pd.DataFrame()
    W_fit_arg = None
    if w_cols: 
        W_df = df_final_processed[w_cols].copy().astype(np.float32)
        W_fit_arg = W_df
    
    final_check_passed = True
    if Y_series.isnull().any(): print(f"错误: Y ('{Y_series.name}') 最终检查发现NaN!"); final_check_passed = False
    if T_series.isnull().any(): print(f"错误: T ('{T_series.name}') 最终检查发现NaN!"); final_check_passed = False
    if X_fit_arg is not None and X_fit_arg.isnull().values.any(): print(f"错误: X ({X_fit_arg.columns.tolist()}) 最终检查发现NaN!"); final_check_passed = False
    if W_fit_arg is not None and W_fit_arg.isnull().values.any(): print(f"错误: W ({W_fit_arg.columns.tolist()}) 最终检查发现NaN!"); final_check_passed = False
    if not final_check_passed: return None, None

    print("\nINFO: 正在初始化和拟合 LinearDRLearner 模型 (使用 TabPFN)...")
    tabpfn_device = 'cuda' # 或 'cpu'
    model_reg = TabPFNRegressor(device=tabpfn_device) 
    model_prop = TabPFNClassifier(device=tabpfn_device) 

    econml_model = LinearDRLearner(
        model_regression=model_reg, 
        model_propensity=model_prop, 
        random_state=123, 
    )
    try:
        econml_model.fit(Y=Y_series, T=T_series, X=X_fit_arg, W=W_fit_arg, inference="statsmodels")
        print("INFO: 模型拟合完成。")
    except Exception as e:
        print(f"错误: 模型拟合时发生错误: {e}")
        return None, None
            
    print("\n--- 分析1: 各处理组相对于 T=0 (控制组) 的效应 ---")
    treatment_names_vs_control = { 
        1: f"仅 {DW_BASE_NAME} (T=1 vs T=0)",
        2: f"仅 {FT_BASE_NAME} (T=2 vs T=0)",
        3: f"仅 {C_BASE_NAME} (T=3 vs T=0)", 
        4: f"{DW_BASE_NAME} + {FT_BASE_NAME} (T=4 vs T=0)",
        5: f"{DW_BASE_NAME} + {C_BASE_NAME} (T=5 vs T=0)", # 如果有T=5
        6: f"{FT_BASE_NAME} + {C_BASE_NAME} (T=6 vs T=0)", # 如果有T=6
        7: f"{DW_BASE_NAME} + {FT_BASE_NAME} + {C_BASE_NAME} (T=7 vs T=0)" # 如果有T=7
    }
    
    active_treatment_values = sorted([t for t in T_series.unique() if t != 0]) 

    if 0 not in T_series.unique():
        print("警告: 数据中没有T=0的样本，以下.summary()结果的基准可能不明确或模型无法拟合。")
    
    for t_val in active_treatment_values: 
        if t_val in treatment_names_vs_control:
            treatment_name_display = treatment_names_vs_control[t_val]
            print(f"\nINFO: 处理效应分析: {treatment_name_display}")
            try:
                summary_output = econml_model.summary(T=int(t_val))
                print(summary_output)
            except Exception as e:
                print(f"错误: 为处理 {treatment_name_display} (T={t_val}) 生成摘要时出错: {e}")
        else:
            print(f"警告: T值 {t_val} 没有在 treatment_names_vs_control 中定义名称。")
            
    return econml_model, X_df


# --- 主程序逻辑 ---
if __name__ == "__main__":
    FULLY_PREPROCESSED_CSV_FILE = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\因果推断.csv"
    # CEC_COL_NAME 常量已在顶部定义
    K_NEIGHBORS = 5      

    try:
        df_final_for_inference = pd.read_csv(FULLY_PREPROCESSED_CSV_FILE)
        print(f"INFO: 已加载数据 '{FULLY_PREPROCESSED_CSV_FILE}'，共 {df_final_for_inference.shape[0]} 行, {df_final_for_inference.shape[1]} 列。")
    except FileNotFoundError:
        print(f"错误: 数据文件 '{FULLY_PREPROCESSED_CSV_FILE}' 未找到。")
        exit()
    except Exception as e:
        print(f"错误: 加载数据时出错: {e}")
        exit()

    # --- 步骤 1: 对所有列进行初步的数值转换和因子化 (如果需要) ---
    print("\nINFO: 进行初步的数值转换和因子化...")
    for col in df_final_for_inference.columns:
        if df_final_for_inference[col].dtype == 'object':
            df_final_for_inference[col] = pd.factorize(df_final_for_inference[col])[0]
            print(f"INFO: 列 '{col}' 已因子化。")
        else: 
            df_final_for_inference[col] = pd.to_numeric(df_final_for_inference[col], errors='coerce')

    # --- 步骤 2: 针对 CEC 列进行专门的 KNN 插补 ---
    if CEC_COL_NAME in df_final_for_inference.columns:
        if df_final_for_inference[CEC_COL_NAME].isnull().any():
            print(f"INFO: 列 '{CEC_COL_NAME}' 存在缺失值 ({df_final_for_inference[CEC_COL_NAME].isnull().sum()}个)，准备进行KNN插补...")
            
            user_specified_knn_features_for_cec = [ # 您指定的辅助特征
                'T', 'pH-B', 'BR', 'Cd-C', 'pH-S', 'BET', 'RM'
            ]
            knn_features_for_cec_imputation = []
            for col in user_specified_knn_features_for_cec:
                if col in df_final_for_inference.columns and pd.api.types.is_numeric_dtype(df_final_for_inference[col]):
                    knn_features_for_cec_imputation.append(col)
                elif col in df_final_for_inference.columns:
                    print(f"警告: 您为KNN指定的特征 '{col}' 非数值型，将不会用于插补CEC。")
                else:
                    print(f"警告: 您为KNN指定的特征 '{col}' 在DataFrame中未找到。")

            if CEC_COL_NAME in knn_features_for_cec_imputation: 
                knn_features_for_cec_imputation.remove(CEC_COL_NAME)
            
            if not knn_features_for_cec_imputation:
                print(f"警告: 未找到足够的数值特征为 '{CEC_COL_NAME}' 进行KNN。将用中位数填充。")
                if df_final_for_inference[CEC_COL_NAME].isnull().any():
                     df_final_for_inference[CEC_COL_NAME].fillna(df_final_for_inference[CEC_COL_NAME].median(), inplace=True)
            else:
                print(f"INFO: 使用以下有效特征为 '{CEC_COL_NAME}' 进行KNN插补: {knn_features_for_cec_imputation}")
                cols_for_knn_block_cec = [CEC_COL_NAME] + knn_features_for_cec_imputation
                df_knn_temp_cec = df_final_for_inference[cols_for_knn_block_cec].copy()
                
                for col_knn_feat in knn_features_for_cec_imputation:
                    if df_knn_temp_cec[col_knn_feat].isnull().any():
                        df_knn_temp_cec[col_knn_feat].fillna(df_knn_temp_cec[col_knn_feat].median(), inplace=True)
                
                cols_all_nan_in_knn_block_cec = df_knn_temp_cec[knn_features_for_cec_imputation].columns[df_knn_temp_cec[knn_features_for_cec_imputation].isnull().all()].tolist()
                if cols_all_nan_in_knn_block_cec:
                    df_knn_temp_cec.drop(columns=cols_all_nan_in_knn_block_cec, inplace=True)
                    cols_for_knn_block_cec = df_knn_temp_cec.columns.tolist()

                if CEC_COL_NAME not in df_knn_temp_cec.columns or df_knn_temp_cec.empty or (len(cols_for_knn_block_cec) <=1 and CEC_COL_NAME in cols_for_knn_block_cec):
                     print(f"警告: KNN插补准备阶段'{CEC_COL_NAME}'块出现问题。将使用中位数填充。")
                     if df_final_for_inference[CEC_COL_NAME].isnull().any():
                          df_final_for_inference[CEC_COL_NAME].fillna(df_final_for_inference[CEC_COL_NAME].median(), inplace=True)
                else:
                    try:
                        knn_imputer_cec_obj = KNNImputer(n_neighbors=K_NEIGHBORS)
                        imputed_block_values_cec = knn_imputer_cec_obj.fit_transform(df_knn_temp_cec)
                        
                        if imputed_block_values_cec.shape[1] != len(cols_for_knn_block_cec):
                            print(f"错误: KNNImputer为CEC返回的列数与输入不匹配！将对这些列尝试中位数填充。")
                            for col_knn_err_cec in cols_for_knn_block_cec:
                                if df_final_for_inference[col_knn_err_cec].isnull().any():
                                    df_final_for_inference[col_knn_err_cec].fillna(df_final_for_inference[col_knn_err_cec].median(), inplace=True)
                        else:
                            df_imputed_knn_block_cec = pd.DataFrame(imputed_block_values_cec, columns=cols_for_knn_block_cec, index=df_knn_temp_cec.index)
                            for col_knn_ok_cec in df_imputed_knn_block_cec.columns:
                                df_final_for_inference[col_knn_ok_cec] = df_imputed_knn_block_cec[col_knn_ok_cec]
                            print(f"INFO: 针对列 '{cols_for_knn_block_cec}' 的KNN插补完成。")
                    except Exception as e_knn_detail_cec:
                        print(f"错误: 对 '{CEC_COL_NAME}' KNN插补时发生错误: {e_knn_detail_cec}。用中位数填充。")
                        if df_final_for_inference[CEC_COL_NAME].isnull().any():
                            df_final_for_inference[CEC_COL_NAME].fillna(df_final_for_inference[CEC_COL_NAME].median(), inplace=True)
        else:
            print(f"INFO: 列 '{CEC_COL_NAME}' 没有缺失值。")
    else:
        print(f"警告: 列 '{CEC_COL_NAME}' 在DataFrame中未找到。")

    print("\nINFO: 进行最终的数值类型确认和NaN填充...")
    for col in df_final_for_inference.columns:
        df_final_for_inference[col] = pd.to_numeric(df_final_for_inference[col], errors='coerce')
        if df_final_for_inference[col].isnull().any():
            fill_value = df_final_for_inference[col].median() 
            df_final_for_inference[col].fillna(fill_value, inplace=True)
    
    T_AGING_PROTOCOL_COL = 'T_aging_protocol' # 处理变量的列名
    if T_AGING_PROTOCOL_COL in df_final_for_inference.columns:
         df_final_for_inference[T_AGING_PROTOCOL_COL] = df_final_for_inference[T_AGING_PROTOCOL_COL].astype(int)
    else:
        print(f"错误: 您的CSV文件中必须包含 '{T_AGING_PROTOCOL_COL}' 列！")
        exit()

    # --- 用户直接定义最终的 X 和 W 列名 ---
    USER_DEFINED_X_COLS = [
        'DW_D_eff', 'FT_D_eff', 'C_days_eff', 'C_temp_eff', 'C_moisture_eff'
    ]
    USER_DEFINED_W_COLS = [
        'T', 'pH-B', 'BR', CEC_COL_NAME, 'Cd-C', 'pH-S', 'BET', RAW_MATERIAL_COL
    ]
    
    final_x_cols_main = [col for col in USER_DEFINED_X_COLS if col in df_final_for_inference.columns]
    final_w_cols_main = [col for col in USER_DEFINED_W_COLS if col in df_final_for_inference.columns]

    missing_x_user_final = set(USER_DEFINED_X_COLS) - set(final_x_cols_main)
    if missing_x_user_final: print(f"警告: 您定义的X列中以下列在最终数据中未找到: {missing_x_user_final}")
    missing_w_user_final = set(USER_DEFINED_W_COLS) - set(final_w_cols_main)
    if missing_w_user_final: print(f"警告: 您定义的W列中以下列在最终数据中未找到: {missing_w_user_final}")
    
    common_xw_main_final = set(final_x_cols_main) & set(final_w_cols_main)
    if common_xw_main_final:
        print(f"错误: 您最终定义的X和W列表中有共同的列: {common_xw_main_final}。已从W中移除。")
        final_w_cols_main = [w for w in final_w_cols_main if w not in common_xw_main_final]

    for col_rm in [OUTPUT_COL_NAME, T_AGING_PROTOCOL_COL]:
        if col_rm in final_x_cols_main: final_x_cols_main.remove(col_rm)
        if col_rm in final_w_cols_main: final_w_cols_main.remove(col_rm)
            
    if not df_final_for_inference.empty:
        # 调用因果推断函数
        model_results = run_causal_inference_with_control_and_relative( # **确保函数名正确**
            df_final_processed=df_final_for_inference,
            y_col_name=OUTPUT_COL_NAME,
            x_cols=final_x_cols_main, 
            w_cols=final_w_cols_main, 
            t_col_name=T_AGING_PROTOCOL_COL
        )

        if model_results:
            fitted_model_main, X_dataframe_for_effects_main = model_results
            # --- 分析2: 指定的处理组间两两比较 ---
            print("\n--- 分析2: 指定的处理组间两两比较 ---")
            # 假设您的T值真实存在的有1,2,3,4 (不含0，或0样本极少)
            treatment_names_map_main_comp = {
                1: f"仅 {DW_BASE_NAME} (T=1)", 
                2: f"仅 {FT_BASE_NAME} (T=2)",
                3: f"仅 {C_BASE_NAME} (T=3)",    
                4: f"{DW_BASE_NAME}+{FT_BASE_NAME} (T=4)" 
                # 如果您还有T=5,6,7并且它们在数据中是1-4之外的有效编码，请在此添加
            }
            # 您指定的比较组合
            explicit_comparison_pairs = [
                (2, 1), (3, 1), (4, 1), # vs T=1
                (3, 2),                 # T=3 vs T=2
                (3, 4),                 # T=3 vs T=4
                (2, 4)                  # T=2 vs T=4
            ]
            
            valid_comparison_pairs = []
            data_t_values_main = df_final_for_inference[T_AGING_PROTOCOL_COL].unique()
            for t1_comp, t0_comp in explicit_comparison_pairs:
                # 确保比较的T值存在于数据中，并且不为0（因为0是summary的参照）
                if t0_comp in data_t_values_main and t1_comp in data_t_values_main and t0_comp !=0 and t1_comp !=0 : 
                    valid_comparison_pairs.append((t1_comp, t0_comp))
                else:
                    t1_n = treatment_names_map_main_comp.get(t1_comp, f"T={t1_comp}")
                    t0_n = treatment_names_map_main_comp.get(t0_comp, f"T={t0_comp}")
                    print(f"警告: 比较对 ({t1_n} vs {t0_n}) 中的一个或两个处理组不在数据中(或为T=0)，将跳过此比较。")

            X_input_for_effect_main = X_dataframe_for_effects_main if X_dataframe_for_effects_main is not None and not X_dataframe_for_effects_main.empty else None

            results_summary_main = []
            if not valid_comparison_pairs:
                print("错误：没有有效的处理对可供进行两两比较。")
            
            for t1, t0 in valid_comparison_pairs:
                t1_name = treatment_names_map_main_comp.get(t1, f"T={t1}")
                t0_name = treatment_names_map_main_comp.get(t0, f"T={t0}")
                print(f"\n比较: {t1_name} (T1={t1})  vs.  {t0_name} (T0={t0})")
                try:
                    relative_effects_array = fitted_model_main.effect(X_input_for_effect_main, T0=t0, T1=t1)
                    if relative_effects_array is not None:
                        avg_eff = np.mean(relative_effects_array)
                        print(f"  平均相对效应 ({t1_name} - {t0_name}): {avg_eff:.4f}"); note_val = ""
                    else:
                        avg_eff = np.nan; note_val = ".effect()返回None"; print(f"  错误: {note_val}")
                    results_summary_main.append({"Comparison": f"{t1_name} vs {t0_name}", "Avg_Relative_Effect": avg_eff, "Note": note_val})
                except Exception as e_eff_comp:
                    print(f"  计算效应 {t1_name} vs {t0_name} 时出错: {e_eff_comp}")
                    results_summary_main.append({"Comparison": f"{t1_name} vs {t0_name}", "Avg_Relative_Effect": np.nan, "Note": f"计算错误: {e_eff_comp}"})
            
            if results_summary_main:
                results_df_main = pd.DataFrame(results_summary_main)
                print("\n\n--- 处理组间相对效应总结 ---"); print(results_df_main.to_string())
                print("\n--- 初步结论 (基于处理组间比较) ---")
                for _, row in results_df_main.iterrows():
                    if pd.notna(row['Avg_Relative_Effect']):
                        comp, effect = row['Comparison'], row['Avg_Relative_Effect']
                        t1_curr, t0_curr = comp.split(' vs ')
                        # 使用一个小的阈值来判断效应是否“接近零”
                        if effect > 1e-5: print(f"- {t1_curr} 相对于 {t0_curr} 平均提高了 '{OUTPUT_COL_NAME}' {abs(effect):.4f} 个单位。")
                        elif effect < -1e-5: print(f"- {t1_curr} 相对于 {t0_curr} 平均降低了 '{OUTPUT_COL_NAME}' {abs(effect):.4f} 个单位。")
                        else: print(f"- {t1_curr} 相对于 {t0_curr} 平均影响非常接近零 (效应: {effect:.4f})。")
                    else: print(f"- 对于 {row['Comparison']}, 未能计算出明确的平均相对效应 ({row['Note']})。")
            else: print("未能计算任何处理组间的相对效应。")
        else:
            print("错误: 模型未能成功拟合或返回None，无法进行效应比较。")
    else:
        print("错误: 用于推断的DataFrame为空。")