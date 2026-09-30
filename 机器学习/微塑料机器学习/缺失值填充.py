import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe # 引入路径效果库，用于实现伪粗体
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore') # 忽略警告

# --- 全局绘图参数 (中文, 宋体, 大字体) ---
plt.rcParams['font.family'] = ['SimSun'] # 设置字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun'] # 备用字体设置
plt.rcParams['axes.unicode_minus'] = False # 保持负号正常显示
# 注意：虽然设置了 bold，但宋体通常需要下面的 PathEffects 才能真正显示粗体
plt.rcParams['font.weight'] = 'bold' 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titleweight'] = 'bold' 
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.size'] = 24 # 基础字体大小
plt.rcParams['axes.labelsize'] = 22 # 轴标签字体大小
plt.rcParams['axes.titlesize'] = 22 # 轴标题字体大小
plt.rcParams['xtick.labelsize'] = 22 # X轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 22 # Y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 20 # 图例字体大小

# --- 辅助函数：强制给宋体加粗 (描边) ---
def add_bold_effect(text_obj, linewidth=0.5):
    """
    给文字对象添加描边效果，使其在视觉上变粗。
    linewidth: 描边宽度，这里默认为用户设定的 0.5
    """
    if text_obj:
        # 获取当前文字颜色作为描边颜色
        color = text_obj.get_color()
        text_obj.set_path_effects([
            pe.withStroke(linewidth=linewidth, foreground=color)
        ])

# --- 数据加载函数 ---
def load_and_process_data(file_path):
    """Loads data, converts all columns to numeric (coercing errors to NaN), and removes BOM characters from column names."""
    try:
        read_successfully = False
        encodings_to_try = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030', 'latin1']
        df = None
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"Data loaded successfully from '{file_path}' using {enc} encoding.")
                read_successfully = True
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading file '{file_path}' with {enc} encoding: {e}")
                return None

        if not read_successfully:
            print(f"Error: Unable to read file '{file_path}' after trying common encodings. Check file encoding or if it's corrupted.")
            return None

    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

    # Remove BOM characters from column names
    df.columns = [col.lstrip('\ufeff') for col in df.columns]
    print("\nBOM characters removed from column names. Case and spacing of column names are preserved.")

    # Attempt to convert all columns to numeric, coercing errors to NaN
    print("\nAttempting to convert all columns to numeric...")
    for col in df.columns:
        df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors='coerce')
    print("Numeric conversion attempt on all columns completed.")

    print("\nData type information after conversion (df.info()):")
    df.info()
    return df

# --- 辅助函数：处理分层列 ---
def prepare_stratification_column(df, strata_col_name_original='T-Aging'):
    """
    Prepares the stratification column according to the original code's logic.
    """
    df_processed = df.copy()
    strata_col_name_final = strata_col_name_original # Final column name for stratification
    strata_series = None # Default to no stratification

    if strata_col_name_original in df_processed.columns:
        print(f"\nAttempting to process stratification information based on column '{strata_col_name_original}'...")

        # If it's numeric and not integer (i.e., has decimals), it needs binning
        if pd.api.types.is_numeric_dtype(df_processed.loc[:, strata_col_name_original]) and \
           not pd.api.types.is_integer_dtype(df_processed.loc[:, strata_col_name_original]):
            print(f"Warning: Stratification column '{strata_col_name_original}' is continuous. It will be binned for stratification.")
            try:
                # Bin into at least 2 and at most 10 quantiles, or no more than unique values
                q_bins = min(10, df_processed.loc[:, strata_col_name_original].nunique())
                if q_bins < 2:
                    print(f"Warning: Not enough unique values in stratification column '{strata_col_name_original}' ({df_processed.loc[:, strata_col_name_original].nunique()}) for effective binning. Reverting to regular random sampling.")
                    strata_series = None
                else:
                    strata_col_name_final = f'{strata_col_name_original}_binned_for_stratify'
                    # Dropna before qcut to avoid errors with NaN values
                    valid_strata_data = df_processed.loc[:, strata_col_name_original].dropna()
                    if not valid_strata_data.empty:
                        df_processed.loc[:, strata_col_name_final] = pd.qcut(
                            df_processed.loc[:, strata_col_name_original],
                            q=q_bins,
                            duplicates='drop', # Allow dropping duplicate bin edges
                            labels=False # Return integer labels instead of intervals
                        )
                        print(f"Created binned column '{strata_col_name_final}'. Value counts:\n{df_processed.loc[:, strata_col_name_final].value_counts()}")
                    else:
                        print(f"Warning: Stratification column '{strata_col_name_original}' contains all missing values, cannot bin. Reverting to regular random sampling.")
                        strata_series = None
            except Exception as e:
                print(f"Error: Auto-binning of '{strata_col_name_original}' failed: {e}. Reverting to regular random sampling.")
                strata_series = None

        # Check if the final stratification column has missing values and impute with most frequent (mode)
        if strata_col_name_final in df_processed.columns and df_processed.loc[:, strata_col_name_final].isnull().any():
            print(f"Warning: Stratification column '{strata_col_name_final}' has missing values, imputing with most frequent value for stratification.")
            imputer_strata = SimpleImputer(strategy='most_frequent')
            # Ensure type is maintained after imputation
            df_processed.loc[:, strata_col_name_final] = imputer_strata.fit_transform(
                df_processed[[strata_col_name_final]].astype(str) if df_processed.loc[:, strata_col_name_final].dtype == 'object' else df_processed[[strata_col_name_final]]
            ).flatten()
            if not pd.api.types.is_numeric_dtype(df_processed.loc[:, strata_col_name_final]):
                 df_processed.loc[:, strata_col_name_final] = pd.Categorical(df_processed.loc[:, strata_col_name_final])

        # Check if the final stratification column has enough categories for stratification (at least 2, with at least 2 samples each)
        if strata_col_name_final in df_processed.columns:
            value_counts = df_processed.loc[:, strata_col_name_final].value_counts()
            if len(value_counts) >= 2 and all(value_counts >= 2):
                strata_series = df_processed.loc[:, strata_col_name_final]
                print(f"Will use '{strata_col_name_final}' column for stratified sampling.")
            else:
                print(f"Warning: Not enough categories or samples per category in stratification column '{strata_col_name_final}' for effective stratification. Reverting to regular random sampling.")
                strata_series = None
        else:
             print(f"Warning: Stratification column '{strata_col_name_final}' not found or processing failed. Reverting to regular random sampling.")
    else:
        print(f"Warning: Stratification column '{strata_col_name_original}' not found. Using regular random sampling.")

    return df_processed, strata_series

# --- 缺失值填充和绘图函数 (仅训练集，每列一张大图，四子图，两颜色，中文，宋体人工加粗) ---
def plot_imputation_comparison_for_features_single_plot(original_series_dict, imputed_train_series_dict, feature_names_to_plot, output_dir='imputation_plots_features_cn_single'):
    """
    Plots before-and-after data distribution.
    Uses 'matplotlib.patheffects' to add a 0.5pt stroke to simulate bold Songti.
    """
    if not feature_names_to_plot:
        print("No feature columns specified for plotting. Skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 定义英文方法名到中文显示名的映射
    method_order = ['Mean Imputation', 'Median Imputation', 'KNN Imputation', 'Random Forest Imputation']
    method_name_cn_map = {
        'Mean Imputation': '均值插补',
        'Median Imputation': '中位数插补',
        'KNN Imputation': 'KNN插补',
        'Random Forest Imputation': '随机森林插补'
    }

    for feature_name in feature_names_to_plot:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12)) 
        axes = axes.flatten() 

        original_train_part = original_series_dict.get(feature_name).dropna()
        
        all_data_for_feature = [original_train_part]
        for method_name in method_order:
            if feature_name in imputed_train_series_dict.get(method_name, {}):
                all_data_for_feature.append(imputed_train_series_dict[method_name][feature_name].dropna())
        
        non_empty_data = [s for s in all_data_for_feature if not s.empty]

        if not non_empty_data:
            print(f"Feature '{feature_name}' has no valid data for plotting across all datasets. Skipping plot.")
            plt.close(fig) 
            continue

        min_val = min(s.min() for s in non_empty_data)
        max_val = max(s.max() for s in non_empty_data)
        x_range_buffer = (max_val - min_val) * 0.1
        xlim_min = min_val - x_range_buffer
        xlim_max = max_val + x_range_buffer

        for i, method_name in enumerate(method_order):
            ax = axes[i] 
            
            # Plot Original Data (原始数据)
            sns.histplot(original_train_part, kde=True, stat='density',
                         color='skyblue', label='原始数据', ax=ax,
                         edgecolor='black', line_kws={'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.7})
            
            # Plot Imputed Train Data (插补数据)
            if feature_name in imputed_train_series_dict.get(method_name, {}):
                sns.histplot(imputed_train_series_dict[method_name][feature_name].dropna(), kde=True, stat='density',
                             color='gold', label='插补数据', ax=ax,
                             edgecolor='black', line_kws={'linestyle': '-', 'linewidth': 1.5, 'alpha': 0.7})
            else:
                ax.text(0.5, 0.5, "无插补数据", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, color='red')
            
            # 使用中文标题并应用人工加粗 (0.5pt)
            cn_title = method_name_cn_map.get(method_name, method_name)
            
            # 设置标题
            title_obj = ax.set_title(f'{cn_title}')
            add_bold_effect(title_obj, linewidth=0.5)
            
            # 设置标签
            xlabel_obj = ax.set_xlabel('数值')
            ylabel_obj = ax.set_ylabel('密度')
            add_bold_effect(xlabel_obj, linewidth=0.5)
            add_bold_effect(ylabel_obj, linewidth=0.5)
            
            # 设置图例并加粗图例文字
            legend = ax.legend()
            if legend:
                for text in legend.get_texts():
                    add_bold_effect(text, linewidth=0.5)
            
            # 可选：给刻度标签也加一点描边，保持一致性
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                add_bold_effect(tick, linewidth=0.5)

            ax.set_xlim(xlim_min, xlim_max) 
            ax.grid(True, linestyle=':', alpha=0.6)

        # Main title for the entire figure (中文)
        suptitle_obj = fig.suptitle(f'插补方法对比', y=0.96)
        add_bold_effect(suptitle_obj, linewidth=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 

        # Save plot for each feature
        save_path = os.path.join(output_dir, f'{feature_name}_Imputation_Comparison.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"Comparison plot for {feature_name} (all methods) saved to: {save_path}")


def perform_and_plot_imputation_all_features(df_original, target_cols=['pH-S-CH', 'Cd-CH'], strata_col_name_original='T-Aging'):
    """
    Executes the entire missing value processing pipeline.
    """
    print("\n--- Starting Missing Value Imputation Process ---")

    # 1. Drop rows with missing values in target columns
    df_cleaned = df_original.copy()
    initial_rows = len(df_cleaned)
    for col in target_cols:
        if col in df_cleaned.columns:
            if df_cleaned.loc[:, col].isnull().any():
                print(f"Target column '{col}' has missing values. Rows with missing values will be dropped.")
                df_cleaned.dropna(subset=[col], inplace=True)
            else:
                print(f"Target column '{col}' has no missing values.")
        else:
            print(f"Warning: Target column '{col}' not found in data, skipping missing value removal for it.")
    
    rows_after_cleaning_targets = len(df_cleaned)
    if rows_after_cleaning_targets < initial_rows:
        print(f"Data reduced from {initial_rows} to {rows_after_cleaning_targets} rows after dropping missing values in target columns.")
    
    if df_cleaned.empty:
        print("Error: Data is empty after dropping missing values in target columns. Cannot proceed.")
        return

    # 2. Prepare stratification column
    df_processed_for_stratify, strata_series = prepare_stratification_column(df_cleaned.copy(), strata_col_name_original)

    # 3. Identify feature columns
    all_numeric_cols = df_processed_for_stratify.select_dtypes(include=np.number).columns.tolist()
    strata_col_actual = strata_series.name if strata_series is not None and strata_series.name in df_processed_for_stratify.columns else strata_col_name_original 
    feature_cols_for_imputation = [col for col in all_numeric_cols if col not in target_cols and col != strata_col_actual]
    feature_cols_with_nan = [col for col in feature_cols_for_imputation if df_processed_for_stratify.loc[:, col].isnull().any()]

    if not feature_cols_with_nan:
        print("\nNo missing values found in feature columns. Skipping imputation.")
        output_filename = os.path.join(output_dir, f'data_no_missing_values_after_cleaning_targets.csv')
        df_processed_for_stratify.to_csv(output_filename, index=False, encoding='utf-8')
        return

    print(f"\nFeature columns to be imputed: {feature_cols_with_nan}")

    # Prepare data subset for imputation
    X_to_impute = df_processed_for_stratify.loc[:, feature_cols_for_imputation].copy()

    # 4. Perform train-test split (with stratification)
    print("\nPerforming train/test split...")
    X_train_impute, X_test_impute = train_test_split(
        X_to_impute, test_size=0.25, random_state=42, stratify=strata_series if strata_series is not None else None
    )

    X_train_indexed = X_train_impute.copy()
    X_test_indexed = X_test_impute.copy()

    original_feature_series_for_plot = {col: df_original.loc[:, col] for col in feature_cols_with_nan}
    imputed_train_series_for_plot = {} 
    full_imputed_dfs = {} 

    # --- Imputation Methods ---
    def perform_imputation_and_update(imputer_display_name, imputer_instance):
        print(f"\nPerforming {imputer_display_name}...")
        try:
            X_train_imputed_array = imputer_instance.fit_transform(X_train_indexed)
            X_test_imputed_array = imputer_instance.transform(X_test_indexed)

            X_train_imputed_df = pd.DataFrame(X_train_imputed_array, columns=feature_cols_for_imputation, index=X_train_indexed.index)
            X_test_imputed_df = pd.DataFrame(X_test_imputed_array, columns=feature_cols_for_imputation, index=X_test_indexed.index)

            imputed_train_series_for_plot[imputer_display_name] = {col: X_train_imputed_df.loc[:, col] for col in feature_cols_with_nan}

            df_full_imputed_current_method = df_cleaned.copy()
            df_updated_features = pd.concat([X_train_imputed_df, X_test_imputed_df]).sort_index()
            for col in feature_cols_for_imputation:
                df_full_imputed_current_method.loc[:, col] = df_updated_features.loc[:, col]
            
            full_imputed_dfs[imputer_display_name.replace(' ', '')] = df_full_imputed_current_method
            
            output_filename = os.path.join(output_dir, f'imputed_data_{imputer_display_name.replace(" ", "_")}_all_features.csv')
            df_full_imputed_current_method.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"Complete data after {imputer_display_name} exported to: {output_filename}")
            return True
        except Exception as e:
            print(f"Error during {imputer_display_name}: {e}. Skipping this method.")
            return False

    perform_imputation_and_update('Mean Imputation', SimpleImputer(strategy='mean'))
    perform_imputation_and_update('Median Imputation', SimpleImputer(strategy='median'))

    n_neighbors_knn = min(5, X_train_indexed.shape[0] - 1)
    if n_neighbors_knn >= 1:
        perform_imputation_and_update('KNN Imputation', KNNImputer(n_neighbors=n_neighbors_knn, weights='uniform'))
    else:
        print(f"Skipping KNN due to insufficient samples.")

    perform_imputation_and_update('Random Forest Imputation', IterativeImputer(estimator=RandomForestRegressor(n_estimators=100, random_state=42), max_iter=10, random_state=42))

    if feature_cols_with_nan:
        print(f"\nGenerating imputation distribution plots for {len(feature_cols_with_nan)} feature columns...")
        plot_imputation_comparison_for_features_single_plot(
            original_feature_series_for_plot, 
            imputed_train_series_for_plot,    
            feature_cols_with_nan,            
            output_dir                        
        )
    else:
        print("\nNo feature columns found that require imputation.")

    print("\n--- Missing Value Imputation Process Completed ---")


# --- 主程序 ---
if __name__ == "__main__":
    # 请根据你的实际文件路径修改这里
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\微塑料机器学习\因果推断.csv"
    output_dir = '微塑料机器学习输出' 
    os.makedirs(output_dir, exist_ok=True)

    df_original = load_and_process_data(file_path)

    if df_original is not None:
        target_columns = ['pH-S-CH', 'Cd-CH']
        perform_and_plot_imputation_all_features(df_original.copy(), target_columns)
    else:
        print("Data loading and initial processing failed.")