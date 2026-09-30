# -*- coding: utf-8 -*-
"""
SEM 方程分析脚本：完整结果表版本

功能：
1. 读取 Excel 数据；
2. 自动识别并重命名变量：aging days、W、MPs、BCA、leaching、stable；
3. 拟合 3 类 SEM/路径模型：
   A. 无中介模型：结果变量 ~ aging_days + W + MPs
   B. 中介模型：BCA ~ aging_days + W + MPs；结果变量 ~ BCA
   C. 完整模型：BCA ~ aging_days + W + MPs；结果变量 ~ aging_days + W + MPs + BCA
4. 生成一个完整的大表，形式类似文献中的 SEM 路径结果表：
   Model / Outcome / R² / Pathway / Total Effect Coefficient / Std.Estimate / T-value / P-value / Significance
5. 额外导出：拟合指标、全部参数、R²、模型结构、SEM 路径图、结果表图片。

运行前安装依赖：
pip install pandas numpy openpyxl scikit-learn semopy matplotlib

作者：ChatGPT
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    from semopy import Model, calc_stats
except ImportError as e:
    raise ImportError("未检测到 semopy，请先运行：pip install semopy") from e


# =============================================================================
# 1. 文件路径与输出路径
# =============================================================================
file_path = r"C:\Users\AMDYE\OneDrive\桌面\新建 Microsoft Excel 工作表.xlsx"
output_dir = os.path.join(os.path.dirname(file_path), "SEM_Result_Complete_Table")
os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# 2. 变量设置
# =============================================================================
required_vars = ["aging_days", "W", "MPs", "BCA", "leaching", "stable"]

# Excel 列名别名。若你的表头略有不同，可以在这里继续添加。
column_aliases = {
    "aging_days": [
        "aging_days", "aging days", "Aging days", "Aging Days",
        "aging day", "Aging day", "aging", "Ageing days", "ageing days", "老化天数"
    ],
    "W": ["W", "w", "Water", "water", "Moisture", "moisture", "水分"],
    "MPs": ["MPs", "MPS", "mps", "MP", "mp", "微塑料"],
    "BCA": ["BCA", "bca"],
    "leaching": ["leaching", "Leaching", "LEACHING", "淋溶", "浸出"],
    "stable": ["stable", "Stable", "STABLE", "稳定态", "稳定"]
}

# 外生变量、中介变量、结果变量
x_vars = ["aging_days", "W", "MPs"]
mediator = "BCA"
y_vars = ["leaching", "stable"]

# 用于表格展示的变量名称
pretty_name = {
    "aging_days": "aging days",
    "W": "W",
    "MPs": "MPs",
    "BCA": "BCA",
    "leaching": "leaching",
    "stable": "stable"
}


# =============================================================================
# 3. 三类 SEM 模型
# =============================================================================
model_descs = {
    "Model 1: No mediator": """
leaching ~ aging_days + W + MPs
stable ~ aging_days + W + MPs

aging_days ~~ W
aging_days ~~ MPs
W ~~ MPs
leaching ~~ stable
""",

    "Model 2: Mediator only": """
BCA ~ aging_days + W + MPs
leaching ~ BCA
stable ~ BCA

aging_days ~~ W
aging_days ~~ MPs
W ~~ MPs
leaching ~~ stable
""",

    "Model 3: Full mediation model": """
BCA ~ aging_days + W + MPs
leaching ~ aging_days + W + MPs + BCA
stable ~ aging_days + W + MPs + BCA

aging_days ~~ W
aging_days ~~ MPs
W ~~ MPs
leaching ~~ stable
"""
}

# 每个模型需要在结果表中显示的路径顺序
model_paths = {
    "Model 1: No mediator": [
        ("leaching", "aging_days"), ("leaching", "W"), ("leaching", "MPs"),
        ("stable", "aging_days"), ("stable", "W"), ("stable", "MPs")
    ],
    "Model 2: Mediator only": [
        ("BCA", "aging_days"), ("BCA", "W"), ("BCA", "MPs"),
        ("leaching", "BCA"), ("stable", "BCA")
    ],
    "Model 3: Full mediation model": [
        ("BCA", "aging_days"), ("BCA", "W"), ("BCA", "MPs"),
        ("leaching", "aging_days"), ("leaching", "W"), ("leaching", "MPs"), ("leaching", "BCA"),
        ("stable", "aging_days"), ("stable", "W"), ("stable", "MPs"), ("stable", "BCA")
    ]
}

# 每个模型需要计算 R² 的变量
model_endogenous = {
    "Model 1: No mediator": ["leaching", "stable"],
    "Model 2: Mediator only": ["BCA", "leaching", "stable"],
    "Model 3: Full mediation model": ["BCA", "leaching", "stable"]
}


# =============================================================================
# 4. 工具函数
# =============================================================================
def find_column(df_columns, aliases):
    """根据别名寻找 Excel 中的真实列名。"""
    normalized = {str(c).strip(): c for c in df_columns}
    lower_map = {str(c).strip().lower(): c for c in df_columns}
    for name in aliases:
        if name in normalized:
            return normalized[name]
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def significance_symbol(p):
    """根据 P 值生成显著性符号。"""
    try:
        p = float(p)
    except Exception:
        return ""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_value(p):
    """P 值格式化。"""
    try:
        p = float(p)
    except Exception:
        return p
    if np.isnan(p):
        return np.nan
    if p < 0.001:
        return f"{p:.5E}"
    return f"{p:.6f}"


def safe_float(x):
    """安全转换为 float。"""
    try:
        return float(x)
    except Exception:
        return np.nan


def get_col(row, candidates, default=np.nan):
    """兼容 semopy 不同版本输出列名。"""
    for c in candidates:
        if c in row.index:
            return row[c]
    return default


def pathway_label(left, right):
    """生成路径名称。"""
    return f"{pretty_name.get(right, right)} → {pretty_name.get(left, left)}"


def clean_param_table(param_df):
    """统一 semopy 参数表中的列名。"""
    df = param_df.copy()
    # semopy 常见列名：lval, op, rval, Estimate, Est. Std, Std. Err, z-value, p-value
    # 部分版本可能大小写不同，这里做适度兼容。
    col_map = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["lval", "left"]:
            col_map[c] = "lval"
        elif cl in ["op", "operator"]:
            col_map[c] = "op"
        elif cl in ["rval", "right"]:
            col_map[c] = "rval"
    if col_map:
        df = df.rename(columns=col_map)
    return df


def extract_model_path_rows(model_name, param_df):
    """从一个 SEM 模型参数表中提取指定路径，生成完整结果行。"""
    param_df = clean_param_table(param_df)
    rows = []

    for left, right in model_paths[model_name]:
        if not {"lval", "op", "rval"}.issubset(set(param_df.columns)):
            hit = pd.DataFrame()
        else:
            hit = param_df[
                (param_df["lval"].astype(str) == left) &
                (param_df["op"].astype(str) == "~") &
                (param_df["rval"].astype(str) == right)
            ]

        if hit.empty:
            rows.append({
                "Model": model_name,
                "Outcome": pretty_name.get(left, left),
                "R2": np.nan,
                "Pathway": pathway_label(left, right),
                "Total Effect Coefficient": np.nan,
                "Std.Estimate": np.nan,
                "T-value": np.nan,
                "P-value": np.nan,
                "Significance": ""
            })
            continue

        row = hit.iloc[0]
        estimate = safe_float(get_col(row, ["Estimate", "estimate"]))
        std_estimate = safe_float(get_col(
            row,
            ["Est. Std", "Std. Estimate", "Std.Estimate", "Standardized Estimate", "Std.Est"],
            estimate
        ))
        t_value = safe_float(get_col(row, ["z-value", "Z-value", "T-value", "t-value", "z"] ))
        p_value = safe_float(get_col(row, ["p-value", "P-value", "P>|z|", "pvalue", "p"] ))

        rows.append({
            "Model": model_name,
            "Outcome": pretty_name.get(left, left),
            "R2": np.nan,
            "Pathway": pathway_label(left, right),
            "Total Effect Coefficient": round(estimate, 6),
            "Std.Estimate": round(std_estimate, 6),
            "T-value": round(t_value, 6),
            "P-value": format_p_value(p_value),
            "Significance": significance_symbol(p_value)
        })

    return pd.DataFrame(rows)


def fit_linear_r2(data_scaled, target, predictors):
    """用线性回归按给定路径结构计算 R²。"""
    y = data_scaled[target].values
    X = data_scaled[predictors].values
    pred = LinearRegression().fit(X, y).predict(X)
    return r2_score(y, pred)


def calculate_model_r2(model_name, data_scaled):
    """按不同模型结构计算 R²。"""
    rows = []
    if model_name == "Model 1: No mediator":
        structure = {
            "leaching": ["aging_days", "W", "MPs"],
            "stable": ["aging_days", "W", "MPs"]
        }
    elif model_name == "Model 2: Mediator only":
        structure = {
            "BCA": ["aging_days", "W", "MPs"],
            "leaching": ["BCA"],
            "stable": ["BCA"]
        }
    else:
        structure = {
            "BCA": ["aging_days", "W", "MPs"],
            "leaching": ["aging_days", "W", "MPs", "BCA"],
            "stable": ["aging_days", "W", "MPs", "BCA"]
        }

    for var, preds in structure.items():
        rows.append({
            "Model": model_name,
            "Variable": pretty_name.get(var, var),
            "R2": round(fit_linear_r2(data_scaled, var, preds), 6)
        })
    return pd.DataFrame(rows)


def extract_fit_indices(stats, model_name):
    """兼容不同 semopy 版本，提取拟合指标。"""
    if isinstance(stats, pd.DataFrame):
        if "Value" in stats.index:
            s = stats.loc["Value"]
        elif "Value" in stats.columns:
            s = stats["Value"]
        else:
            s = stats.iloc[0]
    elif isinstance(stats, pd.Series):
        s = stats
    else:
        s = pd.Series(stats)

    names = [
        "chi2", "DoF", "chi2 p-value",
        "CFI", "TLI", "GFI", "AGFI", "NFI", "RMSEA",
        "AIC", "BIC", "LogLik"
    ]
    rows = []
    for name in names:
        value = s[name] if name in s.index else np.nan
        rows.append({"Model": model_name, "Fit Index": name, "Value": value})
    return pd.DataFrame(rows)


def make_publication_table(path_rows, r2_rows):
    """
    生成类似文献的完整结果表。
    每个模型下面按 Outcome 分组，先放 R² 行，再放路径行。
    """
    all_rows = []
    for model_name in model_descs.keys():
        sub = path_rows[path_rows["Model"] == model_name].copy()
        r2_sub = r2_rows[r2_rows["Model"] == model_name].copy()

        # 模型标题行
        all_rows.append({
            "Model": model_name,
            "Outcome": "",
            "Pathway": "",
            "Total Effect Coefficient": "",
            "Std.Estimate": "",
            "T-value": "",
            "P-value": "",
            "Significance": ""
        })

        for outcome in model_endogenous[model_name]:
            outcome_pretty = pretty_name.get(outcome, outcome)
            r2_hit = r2_sub[r2_sub["Variable"] == outcome_pretty]
            r2_value = np.nan if r2_hit.empty else r2_hit["R2"].iloc[0]

            # R² 行，模仿你截图中的 “R²=...”
            all_rows.append({
                "Model": "",
                "Outcome": outcome_pretty,
                "Pathway": f"R²={r2_value:.4f}" if not pd.isna(r2_value) else "R²=",
                "Total Effect Coefficient": "",
                "Std.Estimate": "",
                "T-value": "",
                "P-value": "",
                "Significance": ""
            })

            # 当前 outcome 的路径行
            sub_outcome = sub[sub["Outcome"] == outcome_pretty].copy()
            for _, row in sub_outcome.iterrows():
                all_rows.append({
                    "Model": "",
                    "Outcome": "",
                    "Pathway": row["Pathway"],
                    "Total Effect Coefficient": row["Total Effect Coefficient"],
                    "Std.Estimate": row["Std.Estimate"],
                    "T-value": row["T-value"],
                    "P-value": row["P-value"],
                    "Significance": row["Significance"]
                })

    return pd.DataFrame(all_rows)


def style_excel(path):
    """美化 Excel。"""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

    wb = load_workbook(path)
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_fill = PatternFill("solid", fgColor="D9EAD3")
    section_fill = PatternFill("solid", fgColor="EADCF8")
    r2_fill = PatternFill("solid", fgColor="FCE5CD")

    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = border
                cell.font = Font(name="Times New Roman", size=11)

        # 表头
        for cell in ws[1]:
            cell.font = Font(name="Times New Roman", size=11, bold=True)
            cell.fill = header_fill

        # 分组行和 R² 行
        for r in range(2, ws.max_row + 1):
            model_cell = ws.cell(r, 1).value
            pathway_cell = ws.cell(r, 3).value if ws.max_column >= 3 else None
            if model_cell:
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).font = Font(name="Times New Roman", size=11, bold=True)
                    ws.cell(r, c).fill = section_fill
            if isinstance(pathway_cell, str) and pathway_cell.startswith("R²="):
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).font = Font(name="Times New Roman", size=11, bold=True)
                    ws.cell(r, c).fill = r2_fill

        # 冻结首行
        ws.freeze_panes = "A2"

        # 调整列宽
        for col in ws.columns:
            max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max(max_len + 3, 12), 40)

    wb.save(path)


def draw_publication_table(table_df, output_png, output_pdf):
    """将完整结果表绘制成图片，便于论文预览。"""
    # 表格行数可能较多，根据行数动态调节高度
    n_rows = len(table_df) + 1
    fig_h = max(6, n_rows * 0.32)
    fig_w = 13

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")

    display_cols = [
        "Model", "Outcome", "Pathway", "Total Effect Coefficient",
        "Std.Estimate", "T-value", "P-value", "Significance"
    ]
    data = table_df[display_cols].astype(str).values.tolist()

    col_labels = display_cols
    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#D9EAD3")
        else:
            model_value = table_df.iloc[row - 1]["Model"]
            pathway_value = table_df.iloc[row - 1]["Pathway"]
            if str(model_value).strip() != "":
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#EADCF8")
            if str(pathway_value).startswith("R²="):
                cell.set_text_props(weight="bold")
                cell.set_facecolor("#FCE5CD")

    plt.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def draw_sem_plot(full_model_path_df, r2_df, output_png, output_pdf):
    """绘制完整模型的 SEM 路径图。"""
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 13

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    pos = {
        "aging days": (1.5, 5.1),
        "W": (1.5, 3.25),
        "MPs": (1.5, 1.4),
        "BCA": (5.2, 3.25),
        "leaching": (9.3, 4.55),
        "stable": (9.3, 1.95)
    }

    r2_map = dict(zip(r2_df["Variable"], r2_df["R2"]))

    def node(name, xy):
        label = name
        if name in ["BCA", "leaching", "stable"]:
            label = f"{name}\n$R^2$={r2_map.get(name, np.nan):.3f}"
        ax.text(
            xy[0], xy[1], label,
            ha="center", va="center",
            fontsize=13.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.4)
        )

    for name, xy in pos.items():
        node(name, xy)

    coef_map = {}
    for _, row in full_model_path_df.iterrows():
        coef_map[row["Pathway"]] = (row["Std.Estimate"], row["Significance"])

    def arrow(start, end, pathway, rad=0.0, text_shift=(0, 0)):
        sx, sy = pos[start]
        ex, ey = pos[end]
        patch = FancyArrowPatch(
            (sx + 0.72, sy), (ex - 0.72, ey),
            arrowstyle="-|>", mutation_scale=15,
            linewidth=1.35, color="black",
            connectionstyle=f"arc3,rad={rad}"
        )
        ax.add_patch(patch)

        mx = (sx + ex) / 2 + text_shift[0]
        my = (sy + ey) / 2 + text_shift[1]
        val, sig = coef_map.get(pathway, (np.nan, ""))
        try:
            val = float(val)
        except Exception:
            val = np.nan
        label = "" if pd.isna(val) else f"{val:.3f}{sig if sig != 'ns' else ''}"
        ax.text(mx, my, label, ha="center", va="center", fontsize=11, fontweight="bold")

    # 自变量到 BCA
    arrow("aging days", "BCA", "aging days → BCA", text_shift=(0, 0.25))
    arrow("W", "BCA", "W → BCA", text_shift=(0, 0.25))
    arrow("MPs", "BCA", "MPs → BCA", text_shift=(0, -0.25))

    # BCA 到结果变量
    arrow("BCA", "leaching", "BCA → leaching", text_shift=(0, 0.25))
    arrow("BCA", "stable", "BCA → stable", text_shift=(0, -0.25))

    # 直接效应路径，使用弯曲箭头，避免重叠
    arrow("aging days", "leaching", "aging days → leaching", rad=-0.18, text_shift=(0.0, 0.55))
    arrow("W", "leaching", "W → leaching", rad=0.10, text_shift=(0.0, 0.20))
    arrow("MPs", "leaching", "MPs → leaching", rad=0.25, text_shift=(0.0, -0.10))

    arrow("aging days", "stable", "aging days → stable", rad=-0.25, text_shift=(0.0, 0.10))
    arrow("W", "stable", "W → stable", rad=-0.10, text_shift=(0.0, -0.20))
    arrow("MPs", "stable", "MPs → stable", rad=0.18, text_shift=(0.0, -0.55))

    ax.text(5.5, 0.35, "Significance: * p<0.05, ** p<0.01, *** p<0.001", ha="center", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 5. 主程序
# =============================================================================
def main():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到 Excel 文件，请检查路径：{file_path}")

    # 读取 Excel
    raw_df = pd.read_excel(file_path)
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    # 列名映射
    rename_dict = {}
    for target, aliases in column_aliases.items():
        col = find_column(raw_df.columns, aliases)
        if col is None:
            raise ValueError(
                f"Excel 中未找到变量 {target}，请检查列名。\n"
                f"当前列名为：{list(raw_df.columns)}"
            )
        rename_dict[col] = target

    df = raw_df.rename(columns=rename_dict)[required_vars].copy()

    # 转为数值型
    for col in required_vars:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 缺失值统计与删除
    missing_summary = df.isna().sum().reset_index()
    missing_summary.columns = ["Variable", "Missing_Count"]
    df_clean = df.dropna().reset_index(drop=True)

    if df_clean.shape[0] < 10:
        raise ValueError("删除缺失值后样本量过少，请检查数据。")

    # 标准化数据，用于输出标准化路径系数
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=required_vars)

    all_path_rows = []
    all_r2_rows = []
    all_fit_rows = []
    all_param_tables = {}

    # 逐个模型拟合
    for model_name, desc in model_descs.items():
        print(f"正在拟合：{model_name}")
        model = Model(desc)
        model.fit(data_scaled)

        try:
            param_df = model.inspect(std_est=True)
        except TypeError:
            param_df = model.inspect()

        path_df = extract_model_path_rows(model_name, param_df)
        r2_df = calculate_model_r2(model_name, data_scaled)
        fit_df = extract_fit_indices(calc_stats(model), model_name)

        all_path_rows.append(path_df)
        all_r2_rows.append(r2_df)
        all_fit_rows.append(fit_df)
        all_param_tables[model_name] = param_df

    path_all = pd.concat(all_path_rows, ignore_index=True)
    r2_all = pd.concat(all_r2_rows, ignore_index=True)
    fit_all = pd.concat(all_fit_rows, ignore_index=True)

    publication_table = make_publication_table(path_all, r2_all)

    # 输出 Excel
    result_xlsx = os.path.join(output_dir, "SEM_Complete_Result_Table.xlsx")
    with pd.ExcelWriter(result_xlsx, engine="openpyxl") as writer:
        publication_table.to_excel(writer, sheet_name="Publication_Table", index=False)
        path_all.to_excel(writer, sheet_name="Pathway_All", index=False)
        r2_all.to_excel(writer, sheet_name="R2_All", index=False)
        fit_all.to_excel(writer, sheet_name="Fit_Indices_All", index=False)
        missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)
        df_clean.to_excel(writer, sheet_name="Clean_Data", index=False)
        data_scaled.to_excel(writer, sheet_name="Standardized_Data", index=False)

        # 每个模型的原始 SEM 参数表
        for model_name, param_df in all_param_tables.items():
            sheet = model_name.replace(":", "").replace(" ", "_")[:31]
            param_df.to_excel(writer, sheet_name=sheet, index=False)

        # 模型结构说明
        model_text = []
        for model_name, desc in model_descs.items():
            model_text.append({"Model": model_name, "SEM_Structure": desc.strip()})
        pd.DataFrame(model_text).to_excel(writer, sheet_name="Model_Structure", index=False)

    style_excel(result_xlsx)

    # 单独输出拟合指标
    fit_xlsx = os.path.join(output_dir, "SEM_Fit_Indices_All.xlsx")
    with pd.ExcelWriter(fit_xlsx, engine="openpyxl") as writer:
        fit_all.to_excel(writer, sheet_name="Fit_Indices_All", index=False)
        r2_all.to_excel(writer, sheet_name="R2_All", index=False)
    style_excel(fit_xlsx)

    # 结果表图片
    table_png = os.path.join(output_dir, "SEM_Complete_Result_Table.png")
    table_pdf = os.path.join(output_dir, "SEM_Complete_Result_Table.pdf")
    draw_publication_table(publication_table, table_png, table_pdf)

    # 完整模型路径图
    full_model_name = "Model 3: Full mediation model"
    full_paths = path_all[path_all["Model"] == full_model_name].copy()
    full_r2 = r2_all[r2_all["Model"] == full_model_name].copy()
    sem_png = os.path.join(output_dir, "SEM_Full_Model_Path.png")
    sem_pdf = os.path.join(output_dir, "SEM_Full_Model_Path.pdf")
    draw_sem_plot(full_paths, full_r2, sem_png, sem_pdf)

    # 报告文件
    report_txt = os.path.join(output_dir, "SEM_Complete_Report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("SEM 分析完成\n")
        f.write(f"输入文件：{file_path}\n")
        f.write(f"有效样本量：{df_clean.shape[0]}\n")
        f.write(f"输出文件夹：{output_dir}\n\n")
        f.write("本脚本拟合了三类模型：\n")
        for model_name, desc in model_descs.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(model_name + "\n")
            f.write(desc.strip() + "\n")
        f.write("\n\n完整结果表：\n")
        f.write(publication_table.to_string(index=False))

    print("\nSEM 分析完成。")
    print(f"输出文件夹：{output_dir}")
    print(f"完整结果表：{result_xlsx}")
    print(f"结果表图片：{table_png}")
    print(f"完整模型路径图：{sem_png}")


if __name__ == "__main__":
    main()
