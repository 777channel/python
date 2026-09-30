import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- 1. 全局绘图参数设置 (宋体 + 粗体) ---
plt.rcParams['font.family'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 20

def add_bold_effect(text_obj, linewidth=0.5):
    """给文字添加描边效果实现伪粗体"""
    if text_obj:
        color = text_obj.get_color()
        text_obj.set_path_effects([pe.withStroke(linewidth=linewidth, foreground=color)])

# --- 2. 数据转换逻辑 (处理中文和分类变量) ---

def preprocess_custom_data(df, numeric_cols, target_cols):
    """集成您的预处理逻辑：原料分类、品种分类、数值转换"""
    print("开始执行自定义转换逻辑...")
    
    # A. 品种分类
    if '品种' in df.columns:
        df['品种类型编码'] = df['品种'].apply(lambda x: 0 if '低累积' in str(x) else 1)
        print("已完成品种类型编码")

    # B. 原料分类映射
    material_categories = {
        '谷物秸秆类': ['小麦秸秆', '水稻秸秆', '玉米秸秆', '高硅水稻秸秆', '低硅水稻秸秆', '稻草'],
        '豆类油料秸秆类': ['花生秸秆', '油菜秸秆', '豆秸', '大蒜秸秆'],
        '果壳硬壳类': ['稻壳', '山胡桃果壳', '花生壳', '椰壳', '椰子壳'],
        '木质竹类': ['竹', '木屑', '硬木'],
        '其他植物副产品': ['玉米芯'],
        '非植物来源': ['蛋壳', '污泥', '酒糟'],
        '混合物': ['玉米秸秆、花生壳、稻壳混合']
    }
    mat_map = {m: k for k, v in material_categories.items() for m in v}
    if '原料' in df.columns:
        df['原料大类'] = df['原料'].map(mat_map).fillna('其他')
        # 独热编码 (One-Hot)
        df = pd.concat([df, pd.get_dummies(df['原料大类'], prefix='原料类')], axis=1)
        print("已完成原料大类映射与独热编码")

    # C. 灌溉方式编码
    if 'Irrigationmethod' in df.columns:
        le = LabelEncoder()
        df['灌溉方式编码'] = le.fit_transform(df['Irrigationmethod'].astype(str))
        print("已完成灌溉方式编码")

    # D. 数值强制转换 (处理空白字符等)
    all_potential_numeric = numeric_cols + target_cols
    for col in all_potential_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace(r'^\s*$', np.nan, regex=True), errors='coerce')

    return df

# --- 3. 核心填充与绘图逻辑 ---

def perform_imputation_and_plot(df, numeric_features, target_cols, output_dir):
    """隔离输出列，仅填充特征列，并绘制对比图"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 隔离输出列：只在特征列中寻找缺失值
    # 按照标准机器学习流程，通常先删除输出列（Target）为空的行
    df_clean = df.dropna(subset=target_cols).copy()
    
    # 自动识别需要填充的特征列 (数值型且有缺失)
    # 包含：原始数值特征 + 独热编码特征 + 品种/灌溉编码
    all_feature_cols = [c for c in df_clean.columns if c not in target_cols and 
                        (c in numeric_features or '原料类_' in c or '编码' in c)]
    
    # 筛选出确实有缺失值的特征列用于绘图对比
    cols_to_plot = [c for c in all_feature_cols if df_clean[c].isnull().any()]
    
    if not cols_to_plot:
        print("所有特征列均无缺失值，无需填充。")
        return

    # 准备特征矩阵 X (只包含特征)
    X = df_clean[all_feature_cols].copy()
    
    # 填充方法定义
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'Median': SimpleImputer(strategy='median'),
        'KNN': KNNImputer(n_neighbors=5),
        'RF': IterativeImputer(estimator=RandomForestRegressor(n_estimators=50, random_state=42), max_iter=10)
    }
    
    method_cn = {'Mean': '均值填充', 'Median': '中位数填充', 'KNN': 'KNN填充', 'RF': '随机森林填充'}
    
    imputed_results = {}
    print(f"\n开始对 {len(all_feature_cols)} 个特征进行填充...")

    for name, imp in imputers.items():
        print(f"正在执行: {method_cn[name]}...")
        X_imputed = pd.DataFrame(imp.fit_transform(X), columns=all_feature_cols, index=X.index)
        imputed_results[name] = X_imputed

    # 绘图对比
    for col in cols_to_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 13))
        axes = axes.flatten()
        
        orig_series = df_clean[col].dropna()
        
        for i, (m_name, m_cn) in enumerate(method_cn.items()):
            ax = axes[i]
            # 原始分布
            sns.histplot(orig_series, kde=True, stat='density', color='skyblue', 
                         label='原始数据', ax=ax, edgecolor='black', alpha=0.6)
            # 填充后分布
            sns.histplot(imputed_results[m_name][col], kde=True, stat='density', color='gold', 
                         label='插补数据', ax=ax, edgecolor='black', alpha=0.5)
            
            # 视觉修饰
            t = ax.set_title(m_cn); add_bold_effect(t)
            xl = ax.set_xlabel('数值'); add_bold_effect(xl)
            yl = ax.set_ylabel('密度'); add_bold_effect(yl)
            
            leg = ax.legend()
            if leg:
                for text in leg.get_texts(): add_bold_effect(text)
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                add_bold_effect(tick)
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f'特征 [{col}] 填充对比图', y=0.98, fontsize=26)
        add_bold_effect(plt.gcf().texts[0])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存为 PDF
        save_path = os.path.join(output_dir, f'{col}_comparison.pdf')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"-> 已保存 PDF: {save_path}")

    # 保存最终填充后的 CSV (特征已填，输出保持原始)
    for m_name, X_imp in imputed_results.items():
        # 合并填充后的特征和原始隔离的输出列
        final_df = pd.concat([X_imp, df_clean[target_cols]], axis=1)
        final_df.to_csv(os.path.join(output_dir, f'data_final_{m_name}.csv'), index=False, encoding='utf-8-sig')

# --- 4. 执行流程 ---

if __name__ == "__main__":
    # 配置区
    file_path = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\数据\finall_data.csv"
    output_directory = '填充结果与PDF图表'
    
    # 定义列
    numeric_features = ['pH', 'organicmatter', 'CEC', 'Cdcontent', 'temperature', 'pH_soil', 'Charcoal_application_ratio']
    target_columns = ['CD_soil_change', 'CD_brown_rice_change', 'CD_stem_change', 'CD_leaf_change', 'CD_shell_change', 'CD_root_change']

    # 加载
    try:
        raw_df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        raw_df = pd.read_csv(file_path, encoding='gbk')

    # 1. 数据转换逻辑
    processed_df = preprocess_custom_data(raw_df, numeric_features, target_columns)

    # 2. 填充与绘图 (内部已实现输出列隔离)
    perform_imputation_and_plot(processed_df, numeric_features, target_columns, output_directory)

    print("\n[完成] 填充后的特征已与原始输出列合并，PDF 矢量图已生成。")