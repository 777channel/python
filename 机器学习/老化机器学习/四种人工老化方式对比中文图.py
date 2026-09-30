import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import joblib
import traceback
import warnings
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --- 导入TabPFN ---
try:
    from tabpfn import TabPFNRegressor
    print("成功导入 TabPFNRegressor。")
    tabpfen_available = True
except ImportError:
    print("严重错误: 'tabpfen' 库未找到。模型将不可用。")
    tabpfen_available = False
    TabPFNRegressor = None

# =============================================================================
# --- 1. 智能字体检测与设置 ---
# =============================================================================

def find_chinese_font():
    """智能检测系统中可用的中文字体"""
    # 常见中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimSun',           # 宋体
        'NSimSun',          # 新宋体
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'KaiTi',            # 楷体
        'FangSong',         # 仿宋
        'STSong',           # 华文宋体
        'STHeiti',          # 华文黑体
        'Arial Unicode MS', # Arial Unicode (支持中文)
    ]
    
    # 获取系统所有可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    print("\n正在检测系统中的中文字体...")
    print(f"系统共有 {len(available_fonts)} 个字体")
    
    # 查找第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            print(f"✓ 找到中文字体: {font}")
            return font
    
    # 如果都没找到，显示所有包含"sim"或"song"或"hei"的字体
    print("\n未找到常见中文字体，显示可能的候选字体：")
    candidates = [f for f in available_fonts if any(keyword in f.lower() 
                  for keyword in ['sim', 'song', 'hei', 'kai', 'fang', 'chinese', 'cjk'])]
    for i, font in enumerate(candidates[:10], 1):
        print(f"  {i}. {font}")
    
    if candidates:
        print(f"\n建议使用: {candidates[0]}")
        return candidates[0]
    
    print("\n警告: 未找到任何中文字体，将使用系统默认字体")
    return None

# 检测并设置中文字体
CHINESE_FONT = find_chinese_font()

# 设置基础字体
if CHINESE_FONT:
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT]  # 中文字体
plt.rcParams['font.family'] = 'sans-serif'

# 数学公式和英文数字字体
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 基础样式设置
plt.rcParams['font.size'] = 30
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.major.width'] = 2.5
plt.rcParams['ytick.major.width'] = 2.5
plt.rcParams['xtick.major.size'] = 28
plt.rcParams['ytick.major.size'] = 28
plt.rcParams['xtick.labelsize'] = 28
plt.rcParams['ytick.labelsize'] = 28

# =============================================================================
# --- 2. 混合字体文本处理函数 ---
# =============================================================================

def create_mixed_text(ax, x, y, text, fontsize=30, ha='left', va='bottom', 
                     transform=None, chinese_stroke=0.5, english_bold=True):
    """
    创建混合字体的文本：中文用指定字体+描边，英文用Times New Roman粗体
    
    参数:
        ax: matplotlib axes对象
        x, y: 文本位置
        text: 要显示的文本
        fontsize: 字体大小
        ha, va: 对齐方式
        transform: 坐标变换
        chinese_stroke: 中文描边宽度(pt)
        english_bold: 英文是否加粗
    """
    import re
    
    if transform is None:
        transform = ax.transData
    
    # 分离中文和非中文字符
    pattern = r'([\u4e00-\u9fff]+|[^\u4e00-\u9fff]+)'
    parts = re.findall(pattern, text)
    
    current_x = x
    text_objects = []
    
    for part in parts:
        if not part.strip():
            continue
            
        # 判断是否为中文
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', part))
        
        if is_chinese:
            # 中文：使用检测到的中文字体 + 描边
            t = ax.text(current_x, y, part, 
                       fontsize=fontsize,
                       fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif',
                       fontweight='normal',
                       ha='left', va=va,
                       transform=transform)
            # 添加描边效果模拟加粗
            t.set_path_effects([pe.withStroke(linewidth=chinese_stroke, 
                                             foreground=t.get_color())])
        else:
            # 英文/数字：使用Times New Roman
            t = ax.text(current_x, y, part,
                       fontsize=fontsize,
                       fontfamily='Times New Roman',
                       fontweight='bold' if english_bold else 'normal',
                       ha='left', va=va,
                       transform=transform)
        
        text_objects.append(t)
        
        # 计算下一个文本片段的起始位置
        bbox = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        bbox_data = bbox.transformed(transform.inverted())
        current_x = bbox_data.x1
    
    # 调整整体对齐
    if text_objects and ha != 'left':
        total_bbox = text_objects[0].get_window_extent(
            renderer=ax.figure.canvas.get_renderer())
        for t in text_objects[1:]:
            bbox = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
            total_bbox = total_bbox.union([bbox])
        
        total_bbox_data = total_bbox.transformed(transform.inverted())
        
        if ha == 'center':
            shift = (total_bbox_data.x1 - total_bbox_data.x0) / 2
        elif ha == 'right':
            shift = total_bbox_data.x1 - total_bbox_data.x0
        else:
            shift = 0
            
        for t in text_objects:
            pos = t.get_position()
            t.set_position((pos[0] - shift, pos[1]))
    
    return text_objects

def apply_text_style(text_obj, is_chinese=None, chinese_stroke=0.5, english_bold=True):
    """
    为单个文本对象应用样式
    
    参数:
        text_obj: matplotlib text对象
        is_chinese: 是否为中文（None则自动判断）
        chinese_stroke: 中文描边宽度
        english_bold: 英文是否加粗
    """
    if text_obj is None:
        return
    
    text_content = text_obj.get_text()
    
    # 自动判断是否包含中文
    if is_chinese is None:
        import re
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text_content))
    
    if is_chinese:
        # 中文：使用中文字体 + 描边
        if CHINESE_FONT:
            text_obj.set_fontfamily(CHINESE_FONT)
        text_obj.set_fontweight('normal')
        text_obj.set_path_effects([
            pe.withStroke(linewidth=chinese_stroke, foreground=text_obj.get_color())
        ])
    else:
        # 英文：使用Times New Roman + 粗体
        text_obj.set_fontfamily('Times New Roman')
        text_obj.set_fontweight('bold' if english_bold else 'normal')

# =============================================================================
# [!!请在这里配置您的文件信息!!]
DATA_FILE_PATH = r"C:\Users\AMDYE\OneDrive\桌面\机器学习\机器学习-人工和自然老化对比\分列数据_特征工程后_v4.csv"
VERSION_SUFFIX = "v4_tabpfen_dynamic_device"
# =============================================================================

# --- 3. 数据加载与情景定义 ---
def load_and_process_data(file_path):
    """加载数据并对所有列进行初步数值转换。"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        print(f"数据从 '{file_path}' 加载成功。")
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}")
        return None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("所有列的数值转换尝试完成。")
    return df

FIXED_PARAMS = {
    'CEC': 15, 'RM': 2, 'T': 400, 'pH-B': 9.7, 'BET': 3.6, 'BR': 0.03
}

TARGET_CASE = {'name': 'pH-S=5, Cd-C=3', 'params': {'pH-S': 5, 'Cd-C': 3}}

AGING_METHODS = {
    1: '干湿循环',
    2: '冻融循环',
    3: '恒温恒湿(25°C,60%W)',
    4: '干湿+冻融循环'
}

# --- 4. 核心功能函数 ---
def prepare_data(df, target_column_name, all_potential_targets):
    """准备数据：清洗、定义特征、分层抽样并返回数据集。"""
    df_processed = df.copy()
    df_processed.dropna(subset=[target_column_name], inplace=True)
    if df_processed.empty:
        return None, None, None, None
    
    strata_col_name = 'T_aging'
    cols_to_exclude = list(all_potential_targets) + [strata_col_name]
    
    feature_columns = [col for col in df_processed.columns if col not in cols_to_exclude]
    X = df_processed[feature_columns]
    y = df_processed[target_column_name]
    
    strata = None
    if strata_col_name in df_processed.columns and all(df_processed[strata_col_name].value_counts() >= 2):
        strata = df_processed[strata_col_name]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strata
    )
    return X_train, X_test, y_train, y_test

def train_tabpfn_model(X_train, y_train, X_test, y_test, target_column_name):
    """使用 TabPFN 模型进行训练。"""
    version_suffix = f"{target_column_name}_{VERSION_SUFFIX}"
    print(f"\n--- 为目标 '{target_column_name}' 训练 TabPFN (人工老化) ---")
    
    n_samples = len(X_train)
    n_neighbors = min(5, n_samples - 1) if n_samples > 1 else 1
    preprocessor = KNNImputer(n_neighbors=n_neighbors)
    
    if not tabpfen_available:
        print("错误: TabPFNRegressor 不可用。")
        return False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_instance = TabPFNRegressor(device=device)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model_instance)
    ])
    
    try:
        print(f"开始在 {device} 上拟合...")
        full_pipeline.fit(X_train, y_train)
        print("模型拟合完成！")
        r2_test = full_pipeline.score(X_test, y_test)
        print(f"TabPFN - 测试集 R²: {r2_test:.4f}")
        
        model_save_dir = './saved_models'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f'pipeline_{version_suffix}.joblib')
        joblib.dump(full_pipeline, model_save_path)
        print(f"已保存模型至: {model_save_path}")
        return True
    except Exception as e:
        print(f"模型训练流程失败: {e}")
        traceback.print_exc()
        return False

def run_single_case_analysis_with_ci(target_column_name, X_train, y_train, n_bootstrap=500):
    version_suffix = f"{target_column_name}_{VERSION_SUFFIX}"
    print(f"\n--- 为 '{target_column_name}' 进行 pH=5, Cd=3 的四种老化方式分析 ---")
    
    if target_column_name == 'Cd-CH':
        display_name = r'Cd固定率变化 ($\Delta$Cd)'
        ylabel_display = r'$\Delta$Cd (%)'
    else:
        display_name = r'pH值变化 ($\Delta$pH)'
        ylabel_display = r'$\Delta$pH'

    model_filename = f'./saved_models/pipeline_{version_suffix}.joblib'
    if not os.path.exists(model_filename):
        print(f"ERROR: Model file not found at '{model_filename}'.")
        return
    
    full_pipeline = joblib.load(model_filename)
    
    print("正在计算模型残差...")
    train_predictions = full_pipeline.predict(X_train)
    residuals = y_train.values - train_predictions
    
    output_parent_dir = f'人工老化四种方法对比图：{target_column_name}'
    os.makedirs(output_parent_dir, exist_ok=True)

    aging_day_limits = {1: 60, 2: 60, 3: 365, 4: 120}
    
    # 生成预测数据
    all_results_list = []
    print(f"正在生成 {TARGET_CASE['name']} 的预测数据...")
    
    for t_aging_code, aging_name in AGING_METHODS.items():
        step = 10 if t_aging_code == 3 else 5
        day_limit = aging_day_limits[t_aging_code]
        days_range = np.arange(0, day_limit + 1, step)
        virtual_samples_features = []
        virtual_samples_info = []
        
        for day in days_range:
            sample = {feat: 0 for feat in X_train.columns}
            sample.update(FIXED_PARAMS)
            sample.update(TARGET_CASE['params'])
            sample['T_aging'] = t_aging_code
            
            if t_aging_code == 1:
                sample['DW-D'] = day
            elif t_aging_code == 2:
                sample['FT-D'] = day
            elif t_aging_code == 3:
                sample['C-D'], sample['C-T'], sample['C-W'] = day, 25, 0.6
            elif t_aging_code == 4:
                sample['DW-D'], sample['FT-D'] = day, day
            
            virtual_samples_features.append(sample)
            virtual_samples_info.append({
                'case': TARGET_CASE['name'],
                'aging_method': aging_name,
                'T_aging': t_aging_code,
                'days': day
            })
        
        df_virtual = pd.DataFrame(virtual_samples_features)[X_train.columns]
        df_info = pd.DataFrame(virtual_samples_info)
        point_predictions = full_pipeline.predict(df_virtual)
        bootstrap_residuals = np.random.choice(
            residuals, size=(len(df_virtual), n_bootstrap), replace=True
        )
        bootstrap_predictions = point_predictions[:, np.newaxis] + bootstrap_residuals
        
        df_preds_wide = pd.DataFrame(bootstrap_predictions, index=df_info.index)
        df_combined_wide = pd.concat([df_info, df_preds_wide], axis=1)
        results_df_long = df_combined_wide.melt(
            id_vars=['case', 'aging_method', 'T_aging', 'days'],
            var_name='bootstrap_run',
            value_name='prediction'
        )
        all_results_list.append(results_df_long)

    results_df = pd.concat(all_results_list, ignore_index=True)
    results_df = results_df.sort_values(by=['aging_method', 'bootstrap_run', 'days'])
    
    csv_filename = os.path.join(output_parent_dir, 'predicted_data_single_case.csv')
    results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    # --- 绘制图表 ---
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 18), 
                            sharex=False, sharey=True)
    
    # 总标题 - 使用混合字体
    suptitle_text = '人工老化四种方式 (95%置信区间)'
    fig.text(0.5, 0.95, suptitle_text, fontsize=32, ha='center', va='top',
            fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif',
            fontweight='normal')
    # 为总标题添加描边
    for text_obj in fig.texts:
        if text_obj.get_text() == suptitle_text:
            apply_text_style(text_obj, is_chinese=True, chinese_stroke=0.5)
    
    prediction_line_color = 'darkorange'
    prediction_ci_color = 'salmon'
    
    display_order = ['干湿循环', '冻融循环', '恒温恒湿(25°C,60%W)', '干湿+冻融循环']
    
    for idx, aging_name in enumerate(display_order):
        aging_group = results_df[results_df['aging_method'] == aging_name]
        ax = axes[idx // 2, idx % 2]
        
        if aging_group.empty:
            continue

        t_aging_code_current = aging_group['T_aging'].iloc[0]
        day_limit_current = aging_day_limits[t_aging_code_current]
        integer_ticks = np.linspace(0, day_limit_current, 12, dtype=int)
        
        # 绘制置信区间
        sns.lineplot(data=aging_group, x='days', y='prediction', ax=ax,
                    ci=95, err_style="band", n_boot=1000,
                    color=prediction_ci_color, linewidth=0, legend=False)
        
        # 绘制趋势线
        sns.lineplot(data=aging_group, x='days', y='prediction', ax=ax,
                    marker='o', markersize=8, color=prediction_line_color,
                    linewidth=3, ci=None)
        
        # Y轴格式化
        if target_column_name == 'Cd-CH':
            def percentage_formatter(x, pos):
                return f'{x*100:.1f}'
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        
        # 子图标题 - 中文用描边
        title_obj = ax.set_title(f'{aging_name}', fontsize=28,
                                fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif',
                                fontweight='normal')
        apply_text_style(title_obj, is_chinese=True, chinese_stroke=0.5)

        ax.grid(True, linestyle='--', linewidth=0.5)
        
        # 子图编号 - 英文用粗体
        label_char = chr(97 + idx)
        text_obj = ax.text(0.99, 0.98, f'({label_char})', 
                          transform=ax.transAxes,
                          fontsize=30, ha='right', va='top',
                          fontfamily='Times New Roman',
                          fontweight='bold', color='black')
        
        # X轴标签 - 中文用描边
        ax.set_xticks(integer_ticks)
        xlabel_obj = ax.set_xlabel('老化时间 (天)', fontsize=30,
                                  fontfamily=CHINESE_FONT if CHINESE_FONT else 'sans-serif',
                                  fontweight='normal')
        apply_text_style(xlabel_obj, is_chinese=True, chinese_stroke=0.5)
        
        # 刻度标签 - 数字用Times New Roman粗体
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily('Times New Roman')
            tick.set_fontweight('bold')

        # Y轴标签
        if idx % 2 == 0:
            ylabel_obj = ax.set_ylabel(ylabel_display, fontsize=30)
            # LaTeX符号保持原样，不需要特殊处理
        else:
            ax.tick_params(labelleft=False)
        
        # 图例 - 混合字体
        legend_elements = [
            Line2D([0], [0], color=prediction_line_color, lw=3,
                   marker='o', markersize=8, label='趋势线'),
            Patch(facecolor=prediction_ci_color, alpha=0.3,
                  label='95% 置信区间')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right',
                          fontsize=26, bbox_to_anchor=(1.01, 0.25),
                          frameon=False)
        
        if legend:
            for text in legend.get_texts():
                # 图例文本包含中英文，应用混合样式
                apply_text_style(text, chinese_stroke=0.5)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    safe_filename = f'{target_column_name}_MixedFonts_Plot_Fixed.pdf'
    output_filename = os.path.join(output_parent_dir, safe_filename)
    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存混合字体图表: {output_filename}")
    print(f"\n--- {display_name} 效果分析完成 ---")

# --- 5. 主程序执行 ---
if __name__ == "__main__":
    df_original = load_and_process_data(DATA_FILE_PATH)
    if df_original is not None:
        targets_to_analyze = ['Cd-CH', 'pH-S-CH']
        for target in targets_to_analyze:
            print(f"\n{'='*40}\n   开始处理: {target}\n{'='*40}")
            
            X_train, X_test, y_train, y_test = prepare_data(
                df_original.copy(), target, targets_to_analyze
            )
            if X_train is None:
                continue
            
            version_suffix = f"{target}_{VERSION_SUFFIX}"
            model_path = f'./saved_models/pipeline_{version_suffix}.joblib'
            
            if not os.path.exists(model_path):
                train_tabpfn_model(X_train, y_train, X_test, y_test, target)
            
            run_single_case_analysis_with_ci(target, X_train, y_train)
        
        print(f"\n{'='*10} 完成 {'='*10}")