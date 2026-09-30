# -*- coding: utf-8 -*-
"""
S-Cd 固定最佳参数模型训练 + TOPSIS加权集成 + 模型保存 + SHAP分析

流程：
1. 读取数据
2. 按S-Cd输出列分层进行7:3划分
3. KNNImputer缺失值填充，仅训练集fit，再transform测试集
4. 对OM和Cd列进行log1p转换
5. 使用已优化得到的最佳参数直接训练6个模型
6. 根据测试集 MSE、RMSE、MAE、R2 进行TOPSIS排序
7. 取TOP3模型，按TOPSIS_Score归一化权重进行加权集成
8. 保存集成模型、KNN填充器、特征列名
9. 对集成模型进行Kernel SHAP分析
10. 保存SHAP值、原始特征值+SHAP值、特征重要性表和图
"""

import os

# 关闭部分库匿名统计上传，避免PostHog SSL报错影响运行显示
os.environ["POSTHOG_DISABLED"] = "true"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    shap = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None


# =========================
# 路径设置
# =========================
file_path = r"C:\Users\AMDYE\OneDrive\桌面\训练版本(1).xlsx"
output_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\S-cd结果\S-Cd SHAP"
os.makedirs(output_dir, exist_ok=True)

target_col = "S-Cd"
random_state = 42

model_save_path = os.path.join(output_dir, f"{target_col}_TOPSIS_Weighted_Ensemble.pkl")
imputer_save_path = os.path.join(output_dir, f"{target_col}_KNN_Imputer.pkl")
feature_cols_save_path = os.path.join(output_dir, f"{target_col}_Feature_Columns.pkl")


# =========================
# 字体设置
# =========================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 24,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold"
})


# =========================
# 数据读取
# =========================
def load_data(path):
    if path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================
# 特征列识别
# =========================
def get_feature_columns(df):
    expected_cols = ["MPs", "Size", "MPR", "pH", "OM", "CEC", "Cd", "Crop"]

    if all(col in df.columns for col in expected_cols):
        return expected_cols

    if target_col in df.columns:
        target_idx = list(df.columns).index(target_col)
        return df.columns[:target_idx].tolist()

    return df.columns[:8].tolist()


# =========================
# 按输出列分层抽样
# =========================
def make_y_strata(y, max_bins=5):
    for q in range(max_bins, 1, -1):
        try:
            strata = pd.qcut(
                y,
                q=q,
                labels=False,
                duplicates="drop"
            )

            counts = strata.value_counts().sort_index()

            if len(counts) >= 2 and counts.min() >= 2:
                return strata

        except Exception:
            continue

    return None


# =========================
# OM、Cd log1p转换
# =========================
def log_transform_om_cd(X_train, X_test):
    X_train_t = X_train.copy()
    X_test_t = X_test.copy()

    log_cols = [col for col in ["OM", "Cd"] if col in X_train_t.columns]

    for col in log_cols:
        if (X_train_t[col] <= -1).any() or (X_test_t[col] <= -1).any():
            raise ValueError(f"{col}列存在小于或等于-1的值，无法进行log1p转换。")

        X_train_t[col] = np.log1p(X_train_t[col])
        X_test_t[col] = np.log1p(X_test_t[col])

    return X_train_t, X_test_t, log_cols


# =========================
# 指标计算
# =========================
def calc_metrics(y_true, y_pred, prefix):
    mse = mean_squared_error(y_true, y_pred)

    return {
        f"{prefix}_MSE": mse,
        f"{prefix}_RMSE": np.sqrt(mse),
        f"{prefix}_MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix}_R2": r2_score(y_true, y_pred)
    }


# =========================
# 固定最佳参数模型
# =========================
def get_fixed_model(model_name):
    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed.")

        return LGBMRegressor(
            max_depth=6,
            n_estimators=508,
            learning_rate=0.13997677827043647,
            num_leaves=27,
            min_child_samples=11,
            subsample=0.9692785846509344,
            colsample_bytree=0.6180550011220218,
            reg_alpha=0.0004880965249931744,
            reg_lambda=0.5613864006908295,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )

    if model_name == "TabPFN":
        if TabPFNRegressor is None:
            raise ImportError("TabPFN is not installed.")

        return TabPFNRegressor()

    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed.")

        return CatBoostRegressor(
            iterations=661,
            depth=4,
            learning_rate=0.0899971126104831,
            l2_leaf_reg=75.53163504503563,
            random_strength=0.13646083003828985,
            bagging_temperature=9.98183521497046,
            rsm=0.5013625462932197,
            border_count=101,
            loss_function="RMSE",
            random_state=random_state,
            verbose=0
        )

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=611,
            max_depth=10,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

    if model_name == "SVR":
        svr = SVR(
            C=1.8582025568986882,
            epsilon=0.13279181416362879,
            gamma="scale",
            kernel="rbf",
            coef0=0.16033273553093502
        )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", svr)
        ])

    if model_name == "ExtraTrees":
        return ExtraTreesRegressor(
            n_estimators=373,
            max_depth=9,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model name: {model_name}")

# =========================
# TOPSIS排序
# =========================
def topsis_ranking(df):
    df = df.copy()

    metric_cols = ["Test_MSE", "Test_RMSE", "Test_MAE", "Test_R2"]
    metric_df = df[metric_cols].copy()

    norm_df = pd.DataFrame(index=metric_df.index)

    for col in ["Test_MSE", "Test_RMSE", "Test_MAE"]:
        max_v = metric_df[col].max()
        min_v = metric_df[col].min()

        if max_v == min_v:
            norm_df[col] = 1.0
        else:
            norm_df[col] = (max_v - metric_df[col]) / (max_v - min_v)

    max_r2 = metric_df["Test_R2"].max()
    min_r2 = metric_df["Test_R2"].min()

    if max_r2 == min_r2:
        norm_df["Test_R2"] = 1.0
    else:
        norm_df["Test_R2"] = (metric_df["Test_R2"] - min_r2) / (max_r2 - min_r2)

    mat = norm_df.values
    denom = np.sqrt((mat ** 2).sum(axis=0))
    denom[denom == 0] = 1
    mat = mat / denom

    best = mat.max(axis=0)
    worst = mat.min(axis=0)

    d_best = np.sqrt(((mat - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((mat - worst) ** 2).sum(axis=1))

    df["TOPSIS_Score"] = d_worst / (d_best + d_worst)

    df = df.sort_values("TOPSIS_Score", ascending=False)
    df["TOPSIS_Rank"] = range(1, len(df) + 1)
    df["TOPSIS_Weight_All"] = df["TOPSIS_Score"] / df["TOPSIS_Score"].sum()

    standardized_df = norm_df.copy()
    standardized_df.insert(0, "Model", df.loc[norm_df.index, "Model"].values)

    return df, standardized_df


# =========================
# TOPSIS加权集成模型
# =========================
class TOPSISWeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights, dtype=float)

        if len(self.models) != len(self.weights):
            raise ValueError("模型数量与权重数量不一致。")

        self.model_names = list(self.models.keys())
        self.weights_dict = dict(zip(self.model_names, self.weights))

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=float)

        for name, model in self.models.items():
            pred += model.predict(X) * self.weights_dict[name]

        return pred

    def get_model_weights(self):
        return self.weights_dict


# =========================
# 字体强制设置
# =========================
def force_font(fig, fontsize=24):
    for ax in fig.axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels() + [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontname("Times New Roman")
            item.set_fontweight("bold")
            item.set_fontsize(fontsize)

        legend = ax.get_legend()

        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname("Times New Roman")
                text.set_fontweight("bold")

    for text in fig.texts:
        text.set_fontname("Times New Roman")
        text.set_fontweight("bold")


# =========================
# SHAP组合图：蜂巢图 + 顶部条形重要性图
# =========================
def plot_shap_combined(target, shap_values_array, X_test):
    fig, ax1 = plt.subplots(figsize=(10, 8), dpi=120)

    shap.summary_plot(
        shap_values_array,
        X_test,
        feature_names=X_test.columns,
        plot_type="dot",
        show=False,
        color=plt.get_cmap("coolwarm"),
        color_bar=True
    )

    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])
    ax1 = plt.gca()

    ax2 = ax1.twiny()

    shap.summary_plot(
        shap_values_array,
        X_test,
        feature_names=X_test.columns,
        plot_type="bar",
        show=False
    )

    plt.gca().set_position([0.2, 0.2, 0.65, 0.65])

    ax2.axhline(
        y=len(X_test.columns) - 1,
        color="gray",
        linestyle="-",
        linewidth=1
    )

    for bar in ax2.patches:
        bar.set_alpha(0.20)

    ax1.set_xlabel(
        "Shapley Value Contribution (Bee Swarm)",
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman"
    )

    ax2.set_xlabel(
        "Mean Shapley Value (Feature Importance)",
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman"
    )

    ax2.xaxis.set_label_position("top")
    ax2.xaxis.tick_top()

    ax1.set_ylabel(
        "Features",
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman"
    )

    force_font(fig, fontsize=24)

    plt.tight_layout()

    for ext, dpi in [("png", 300), ("pdf", 1200), ("tif", 300)]:
        plt.savefig(
            os.path.join(output_dir, f"{target}_SHAP_Combined.{ext}"),
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.close(fig)


# =========================
# 特征重要性分类圆环图
# =========================
def plot_feature_importance_categories(target, shap_values_df):
    group_config = {
        "MP Features": ["MPs", "Size", "MPR"],
        "Soil Features": ["pH", "OM", "CEC", "Cd"],
        "Crop Feature": ["Crop"]
    }

    feature_importance = shap_values_df.abs().mean().reset_index()
    feature_importance.columns = ["Feature", "Importance"]
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    feature_to_group = {}
    for group, features in group_config.items():
        for feature in features:
            feature_to_group[feature] = group

    feature_importance["Group"] = feature_importance["Feature"].map(feature_to_group)
    feature_importance["Group"] = feature_importance["Group"].fillna("Other")

    feature_importance.to_excel(
        os.path.join(output_dir, f"{target}_SHAP_Feature_Importance_Grouped.xlsx"),
        index=False
    )

    group_importance = (
        feature_importance
        .groupby("Group", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )

    group_importance.to_excel(
        os.path.join(output_dir, f"{target}_SHAP_Group_Importance.xlsx"),
        index=False
    )

    group_color_map = {
        "MP Features": "Blues",
        "Soil Features": "Reds",
        "Crop Feature": "Greens",
        "Other": "Greys"
    }

    inner_sizes = []
    inner_labels = []
    inner_colors = []
    outer_sizes = []
    outer_labels = []
    outer_colors = []

    for _, row in group_importance.iterrows():
        group_name = row["Group"]
        group_val = row["Importance"]

        inner_sizes.append(group_val)
        inner_labels.append(group_name)

        cmap = plt.get_cmap(group_color_map.get(group_name, "Greys"))
        inner_colors.append(cmap(0.6))

        sub_df = (
            feature_importance[feature_importance["Group"] == group_name]
            .sort_values("Importance", ascending=False)
        )

        n_sub = len(sub_df)
        sub_colors = [cmap(i) for i in np.linspace(0.2, 0.5, n_sub)]

        outer_sizes.extend(sub_df["Importance"].tolist())
        outer_labels.extend(sub_df["Feature"].tolist())
        outer_colors.extend(sub_colors)

    fig, ax = plt.subplots(figsize=(20, 10))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.55, top=0.95)

    size = 0.3
    radius_outer = 1.0
    radius_inner = radius_outer - size

    patches_outer, _ = ax.pie(
        outer_sizes,
        radius=radius_outer,
        colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor="w", linewidth=1.2),
        startangle=90,
        counterclock=False
    )

    patches_inner, _ = ax.pie(
        inner_sizes,
        radius=radius_inner,
        colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor="w", linewidth=1.2),
        startangle=90,
        counterclock=False
    )

    ax.text(
        0,
        0,
        "Feature Importance\n(Weighted Ensemble)",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman",
        color="#333333"
    )

    for i, wedge in enumerate(patches_inner):
        ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        y_val = np.sin(np.deg2rad(ang))
        x_val = np.cos(np.deg2rad(ang))

        horizontalalignment = "center" if x_val == 0 else ("right" if x_val < 0 else "left")
        r_label = 1.15
        connection_color = inner_colors[i]

        ax.plot(
            [x_val * radius_outer, x_val * r_label * 0.92],
            [y_val * radius_outer, y_val * r_label * 0.92],
            color=connection_color,
            linestyle="-",
            linewidth=2
        )

        ax.text(
            x_val * r_label,
            y_val * r_label,
            inner_labels[i],
            ha=horizontalalignment,
            va="center",
            fontsize=18,
            fontweight="bold",
            fontname="Times New Roman",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=connection_color,
                edgecolor=connection_color,
                alpha=0.15
            )
        )

    legend_handles = []
    total_importance = sum(inner_sizes)
    idx_outer = 0

    for i, group_name in enumerate(inner_labels):
        pct = inner_sizes[i] / total_importance if total_importance != 0 else 0

        legend_handles.append(
            Patch(facecolor=inner_colors[i], label=f"{group_name} ({pct:.1%})")
        )

        n_sub = len(feature_importance[feature_importance["Group"] == group_name])

        for _ in range(n_sub):
            feat_name = outer_labels[idx_outer]
            feat_val = outer_sizes[idx_outer]
            feat_pct = feat_val / total_importance if total_importance != 0 else 0

            legend_handles.append(
                Patch(
                    facecolor=outer_colors[idx_outer],
                    label=f"    {feat_name} ({feat_pct:.1%})"
                )
            )

            idx_outer += 1

        legend_handles.append(Patch(facecolor="white", label="", alpha=0))

    legend = plt.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        frameon=False,
        fontsize=18,
        title="Feature Categories",
        title_fontsize=18
    )

    for text in legend.get_texts():
        text.set_fontname("Times New Roman")
        text.set_fontweight("bold")

    legend.get_title().set_fontname("Times New Roman")
    legend.get_title().set_fontweight("bold")

    ax.set_aspect("equal")
    ax.axis("off")

    for ext, dpi in [("png", 300), ("pdf", 1200), ("tif", 300)]:
        plt.savefig(
            os.path.join(output_dir, f"{target}_SHAP_Feature_Importance_Categories.{ext}"),
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.close(fig)


# =========================
# 保存特征原始值与SHAP值
# =========================
def save_feature_values_and_shap_values(target, X_test, shap_values_df):
    feature_value_df = X_test.copy()
    feature_value_df = feature_value_df.add_prefix("Value_")

    shap_value_df = shap_values_df.copy()
    shap_value_df = shap_value_df.add_prefix("SHAP_")

    wide_df = pd.concat([feature_value_df, shap_value_df], axis=1)
    wide_df.insert(0, "Sample_Index", X_test.index)

    long_rows = []

    for sample_idx in X_test.index:
        for feature in X_test.columns:
            long_rows.append({
                "Sample_Index": sample_idx,
                "Feature": feature,
                "Feature_Value": X_test.loc[sample_idx, feature],
                "SHAP_Value": shap_values_df.loc[sample_idx, feature]
            })

    long_df = pd.DataFrame(long_rows)

    save_path = os.path.join(output_dir, f"{target}_Feature_Values_and_SHAP_Values.xlsx")

    with pd.ExcelWriter(save_path) as writer:
        wide_df.to_excel(writer, sheet_name="Wide_Value_SHAP", index=False)
        long_df.to_excel(writer, sheet_name="Long_Value_SHAP", index=False)
        X_test.to_excel(writer, sheet_name="Feature_Values", index=True)
        shap_values_df.to_excel(writer, sheet_name="SHAP_Values", index=True)

    print("Feature values and SHAP values saved:")
    print(save_path)


# =========================
# SHAP分析入口
# =========================
def run_shap_analysis(target, ensemble_model, X_train, X_test):
    if shap is None:
        raise ImportError("shap is not installed. Please install it by: pip install shap")

    print("\nStarting SHAP analysis...")

    background_size = min(50, len(X_train))

    background = shap.sample(
        X_train,
        background_size,
        random_state=42
    )

    explainer = shap.KernelExplainer(
        ensemble_model.predict,
        background
    )

    shap_values = explainer.shap_values(
        X_test,
        nsamples=200
    )

    shap_values_array = np.array(shap_values)

    if shap_values_array.ndim == 3:
        shap_values_array = shap_values_array[:, :, 0]

    shap_values_df = pd.DataFrame(
        shap_values_array,
        columns=X_test.columns,
        index=X_test.index
    )

    shap_values_df.to_excel(
        os.path.join(output_dir, f"{target}_SHAP_Values.xlsx"),
        index=True
    )

    feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)

    feature_importance.to_excel(
        os.path.join(output_dir, f"{target}_SHAP_Feature_Importance.xlsx"),
        header=["Mean_Abs_SHAP"]
    )

    save_feature_values_and_shap_values(
        target=target,
        X_test=X_test,
        shap_values_df=shap_values_df
    )

    plot_shap_combined(
        target=target,
        shap_values_array=shap_values_array,
        X_test=X_test
    )

    plot_feature_importance_categories(
        target=target,
        shap_values_df=shap_values_df
    )

    print("SHAP analysis finished.")


# =========================
# 主程序
# =========================
def main():
    df = load_data(file_path)
    feature_cols = get_feature_columns(df)

    print("\nFeature columns:")
    for col in feature_cols:
        print(f"  {col}")

    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()

    X = data[feature_cols]
    y = data[target_col]

    strata = make_y_strata(y)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=strata
    )

    print("\nStratification method: Y quantile stratification")
    print("Target:", target_col)

    imputer = KNNImputer(n_neighbors=5, weights="uniform")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=feature_cols,
        index=X_train_raw.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=feature_cols,
        index=X_test_raw.index
    )

    X_train, X_test, log_cols = log_transform_om_cd(X_train, X_test)

    print("Log1p transformed columns:", log_cols)

    joblib.dump(imputer, imputer_save_path)
    joblib.dump(feature_cols, feature_cols_save_path)

    model_names = [
        "LightGBM",
        "TabPFN",
        "CatBoost",
        "SVR",
        "RandomForest",
        "ExtraTrees"
    ]

    rows = []
    pred_rows = []
    fitted_infos = []

    for name in model_names:
        print(f"Training fixed model: {name}")

        try:
            model = get_fixed_model(name)
            model.fit(X_train, y_train)

            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

            row = {
                "Target": target_col,
                "Model": name,
                "Status": "Success",
                "Error": ""
            }

            row.update(calc_metrics(y_train, pred_train, "Train"))
            row.update(calc_metrics(y_test, pred_test, "Test"))
            row["Overfit_Gap"] = row["Train_R2"] - row["Test_R2"]

            rows.append(row)

            fitted_infos.append({
                "Model": name,
                "ModelObj": model
            })

            pred_rows.append(pd.DataFrame({
                "Target": target_col,
                "Model": name,
                "Dataset": "Train",
                "Observed": y_train.values,
                "Predicted": pred_train
            }))

            pred_rows.append(pd.DataFrame({
                "Target": target_col,
                "Model": name,
                "Dataset": "Test",
                "Observed": y_test.values,
                "Predicted": pred_test
            }))

            print(
                f"{name}: "
                f"Train R2={row['Train_R2']:.4f}, "
                f"Test R2={row['Test_R2']:.4f}, "
                f"Gap={row['Overfit_Gap']:.4f}"
            )

        except Exception as e:
            rows.append({
                "Target": target_col,
                "Model": name,
                "Status": "Failed",
                "Error": str(e),
                "Train_MSE": np.nan,
                "Train_RMSE": np.nan,
                "Train_MAE": np.nan,
                "Train_R2": np.nan,
                "Test_MSE": np.nan,
                "Test_RMSE": np.nan,
                "Test_MAE": np.nan,
                "Test_R2": np.nan,
                "Overfit_Gap": np.nan
            })

            print(f"{name} failed: {e}")

    result_df = pd.DataFrame(rows)

    success_df = result_df[result_df["Status"] == "Success"].copy()

    ranked_df, standardized_df = topsis_ranking(success_df)

    top3_names = ranked_df.head(3)["Model"].tolist()

    top3_df = ranked_df[ranked_df["Model"].isin(top3_names)].copy()

    top3_df["Ensemble_Weight"] = top3_df["TOPSIS_Score"] / top3_df["TOPSIS_Score"].sum()

    weight_map = dict(zip(top3_df["Model"], top3_df["Ensemble_Weight"]))

    top3_infos = [info for info in fitted_infos if info["Model"] in top3_names]

    top3_infos = sorted(
        top3_infos,
        key=lambda x: top3_names.index(x["Model"])
    )

    print("\nTOP3 for TOPSIS weighted ensemble:")
    for i, name in enumerate(top3_names, start=1):
        print(f"{i}. {name}, weight = {weight_map[name]:.6f}")

    ensemble_models = {
        info["Model"]: info["ModelObj"]
        for info in top3_infos
    }

    ensemble_weights = [weight_map[name] for name in ensemble_models.keys()]

    ensemble_model = TOPSISWeightedEnsemble(
        models=ensemble_models,
        weights=ensemble_weights
    )

    ensemble_model.fit(X_train, y_train)

    ensemble_train = ensemble_model.predict(X_train)
    ensemble_test = ensemble_model.predict(X_test)

    ensemble_row = {
        "Target": target_col,
        "Model": "TOPSIS_Weighted_Ensemble",
        "Status": "Success",
        "Error": "",
        "Base_Models": " + ".join(top3_names),
        "Base_Model_Weights": str(ensemble_model.get_model_weights())
    }

    ensemble_row.update(calc_metrics(y_train, ensemble_train, "Train"))
    ensemble_row.update(calc_metrics(y_test, ensemble_test, "Test"))
    ensemble_row["Overfit_Gap"] = ensemble_row["Train_R2"] - ensemble_row["Test_R2"]

    ranked_with_ensemble = pd.concat(
        [ranked_df, pd.DataFrame([ensemble_row])],
        ignore_index=True
    )

    pred_rows.append(pd.DataFrame({
        "Target": target_col,
        "Model": "TOPSIS_Weighted_Ensemble",
        "Dataset": "Train",
        "Observed": y_train.values,
        "Predicted": ensemble_train
    }))

    pred_rows.append(pd.DataFrame({
        "Target": target_col,
        "Model": "TOPSIS_Weighted_Ensemble",
        "Dataset": "Test",
        "Observed": y_test.values,
        "Predicted": ensemble_test
    }))

    pred_df = pd.concat(pred_rows, ignore_index=True)

    joblib.dump(ensemble_model, model_save_path)

    print("\nModel saved:")
    print(model_save_path)

    save_path = os.path.join(
        output_dir,
        f"{target_col}_FixedParams_TOPSIS_Weighted_Ensemble_SHAP_Results.xlsx"
    )

    with pd.ExcelWriter(save_path) as writer:
        ranked_with_ensemble.to_excel(writer, sheet_name="Model_Performance", index=False)
        pred_df.to_excel(writer, sheet_name="Prediction_Result", index=False)
        top3_df.to_excel(writer, sheet_name="TOP3_Weights", index=False)
        standardized_df.to_excel(writer, sheet_name="Standardized_Metrics", index=False)

        pd.DataFrame({
            "Target": [target_col],
            "Stratification": ["Y quantile stratification"],
            "Imputation": ["KNNImputer(n_neighbors=5, weights='uniform')"],
            "Transform_Columns": [", ".join(log_cols)],
            "Transform_Method": ["np.log1p after KNN imputation"],
            "Train_Test_Split": ["7:3"],
            "Random_State": [random_state],
            "Model_Source": ["Fixed best parameters from Bayesian optimization"]
        }).to_excel(writer, sheet_name="Preprocessing_Info", index=False)

    print("\nResults saved:")
    print(save_path)

    run_shap_analysis(
        target=target_col,
        ensemble_model=ensemble_model,
        X_train=X_train,
        X_test=X_test
    )

    print("\nAll outputs saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()
