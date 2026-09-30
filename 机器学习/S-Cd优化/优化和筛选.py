# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    import optuna
except ImportError:
    optuna = None

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
output_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\S-cd结果"
os.makedirs(output_dir, exist_ok=True)

target_col = "S-Cd"
random_state = 42
n_trials = 300

model_names = [
    "SVR",
    "TabPFN",
    "RandomForest",
    "CatBoost",
    "ExtraTrees",
    "LightGBM"
]


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
    "legend.fontsize": 24,
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

    df.columns = [
        str(c).strip().lstrip("\ufeff")
        for c in df.columns
    ]

    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    return df


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
# SPXY划分
# =========================
def spxy_split(X, y, test_size=0.30):
    """
    SPXY（Sample set Partitioning based on joint X-Y distances）划分。
    思路：同时考虑特征空间X和输出变量y的距离，使训练集尽量覆盖X和y的整体分布。
    返回：X_train_raw, X_test_raw, y_train, y_test
    """
    X_for_split = X.copy()

    # SPXY需要完整数值矩阵；这里只用于划分距离计算，后续真正建模仍然只在训练集fit KNNImputer
    X_for_split = X_for_split.apply(
        lambda col: col.fillna(col.median()),
        axis=0
    )

    y_for_split = y.copy()
    y_for_split = y_for_split.fillna(y_for_split.median())

    X_values = X_for_split.values.astype(float)
    y_values = y_for_split.values.reshape(-1, 1).astype(float)

    # 标准化，避免量纲影响距离
    X_scaled = StandardScaler().fit_transform(X_values)
    y_scaled = StandardScaler().fit_transform(y_values)

    n_samples = X_scaled.shape[0]
    n_train = int(round(n_samples * (1 - test_size)))

    if n_train < 2:
        raise ValueError("训练集样本数过少，无法进行SPXY划分。")

    # 计算X距离矩阵
    diff_x = X_scaled[:, None, :] - X_scaled[None, :, :]
    dist_x = np.sqrt(np.sum(diff_x ** 2, axis=2))

    # 计算y距离矩阵
    diff_y = y_scaled[:, None, :] - y_scaled[None, :, :]
    dist_y = np.sqrt(np.sum(diff_y ** 2, axis=2))

    # 归一化后合并X和y距离
    if dist_x.max() > 0:
        dist_x = dist_x / dist_x.max()

    if dist_y.max() > 0:
        dist_y = dist_y / dist_y.max()

    dist_xy = dist_x + dist_y

    # 选择距离最远的两个样本作为初始训练集
    first_pair = np.unravel_index(
        np.argmax(dist_xy),
        dist_xy.shape
    )

    train_indices = [first_pair[0], first_pair[1]]
    remaining_indices = [i for i in range(n_samples) if i not in train_indices]

    # 逐步选择与当前训练集最近距离最大的样本
    while len(train_indices) < n_train:
        candidate_min_distances = []

        for idx in remaining_indices:
            min_dist = np.min(dist_xy[idx, train_indices])
            candidate_min_distances.append(min_dist)

        selected_pos = int(np.argmax(candidate_min_distances))
        selected_idx = remaining_indices[selected_pos]

        train_indices.append(selected_idx)
        remaining_indices.remove(selected_idx)

    test_indices = remaining_indices

    # 保留原始索引
    X_train_raw = X.iloc[train_indices].copy()
    X_test_raw = X.iloc[test_indices].copy()
    y_train = y.iloc[train_indices].copy()
    y_test = y.iloc[test_indices].copy()

    return X_train_raw, X_test_raw, y_train, y_test


# =========================
# OM、Cd log1p转换
# =========================
def log_transform_om_cd(X_train, X_test):
    X_train_t = X_train.copy()
    X_test_t = X_test.copy()

    log_cols = [
        col for col in ["OM", "Cd"]
        if col in X_train_t.columns
    ]

    for col in log_cols:
        if (X_train_t[col] <= -1).any() or (X_test_t[col] <= -1).any():
            raise ValueError(
                f"{col}列存在小于或等于-1的值，无法进行log1p转换。"
            )

        X_train_t[col] = np.log1p(X_train_t[col])
        X_test_t[col] = np.log1p(X_test_t[col])

    return X_train_t, X_test_t, log_cols


# =========================
# 模型包装
# =========================
def wrap_model_if_needed(model_name, model):
    if model_name == "SVR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])

    return model


# =========================
# 扩展贝叶斯优化范围
# =========================
def get_model(model_name, trial=None):
    if model_name == "TabPFN":
        if TabPFNRegressor is None:
            raise ImportError("TabPFN is not installed.")

        return TabPFNRegressor()

    if model_name == "SVR":
        if trial is None:
            return SVR()

        return SVR(
            C=trial.suggest_float("C", 0.01, 500.0, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
            gamma=trial.suggest_categorical(
                "gamma",
                ["scale", "auto"]
            ),
            kernel=trial.suggest_categorical(
                "kernel",
                ["rbf", "sigmoid"]
            ),
            coef0=trial.suggest_float("coef0", 0.0, 2.0)
        )

    if model_name == "RandomForest":
        if trial is None:
            return RandomForestRegressor(
                random_state=random_state,
                n_jobs=-1
            )

        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1200),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 6, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
            max_features=trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.4, 0.6, 0.8]
            ),
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

    if model_name == "ExtraTrees":
        if trial is None:
            return ExtraTreesRegressor(
                random_state=random_state,
                n_jobs=-1
            )

        return ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1200),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 6, 40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 3, 20),
            max_features=trial.suggest_categorical(
                "max_features",
                ["sqrt", "log2", 0.4, 0.6, 0.8]
            ),
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )

    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed.")

        if trial is None:
            return CatBoostRegressor(
                loss_function="RMSE",
                random_state=random_state,
                verbose=0
            )

        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 100, 1200),
            depth=trial.suggest_int("depth", 2, 6),
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.001,
                0.15,
                log=True
            ),
            l2_leaf_reg=trial.suggest_float(
                "l2_leaf_reg",
                3.0,
                300.0,
                log=True
            ),
            random_strength=trial.suggest_float(
                "random_strength",
                1e-4,
                20.0,
                log=True
            ),
            bagging_temperature=trial.suggest_float(
                "bagging_temperature",
                0.0,
                10.0
            ),
            rsm=trial.suggest_float("rsm", 0.5, 1.0),
            border_count=trial.suggest_int("border_count", 32, 128),
            loss_function="RMSE",
            random_state=random_state,
            verbose=0
        )

    if model_name == "LightGBM":
        if LGBMRegressor is None:
            raise ImportError("LightGBM is not installed.")

        if trial is None:
            return LGBMRegressor(
                random_state=random_state,
                verbose=-1,
                n_jobs=-1
            )

        max_depth = trial.suggest_int(
            "max_depth",
            2,
            8
        )

        max_leaves = min(
            2 ** max_depth,
            64
        )

        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1200),
            max_depth=max_depth,
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.001,
                0.15,
                log=True
            ),
            num_leaves=trial.suggest_int("num_leaves", 4, max_leaves),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 100.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 100.0, log=True),
            random_state=random_state,
            verbose=-1,
            n_jobs=-1
        )

    raise ValueError(f"Unknown model: {model_name}")


# =========================
# 贝叶斯优化
# =========================
def bayes_optimize(model_name, X_train, y_train):
    if model_name == "TabPFN":
        model = get_model("TabPFN")

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=1
        )

        return model, {"TabPFN": "Default parameters"}, result["test_score"].mean()

    if optuna is None:
        raise ImportError("Optuna is not installed.")

    def objective(trial):
        model = get_model(
            model_name,
            trial
        )

        model = wrap_model_if_needed(
            model_name,
            model
        )

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

        result = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        return result["test_score"].mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=random_state
        )
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False
    )

    best_model = get_model(
        model_name,
        study.best_trial
    )

    best_model = wrap_model_if_needed(
        model_name,
        best_model
    )

    return best_model, study.best_params, study.best_value


# =========================
# 指标计算
# =========================
def calc_metrics(y_true, y_pred, prefix):
    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        f"{prefix}_MSE": mse,
        f"{prefix}_RMSE": np.sqrt(mse),
        f"{prefix}_MAE": mean_absolute_error(y_true, y_pred),
        f"{prefix}_R2": r2_score(y_true, y_pred)
    }


# =========================
# TOPSIS排序
# =========================
def topsis_ranking(df):
    df = df.copy()

    metric_df = df[
        [
            "Test_MSE",
            "Test_RMSE",
            "Test_MAE",
            "Test_R2"
        ]
    ].copy()

    norm_df = pd.DataFrame(
        index=metric_df.index
    )

    for col in [
        "Test_MSE",
        "Test_RMSE",
        "Test_MAE"
    ]:
        max_v = metric_df[col].max()
        min_v = metric_df[col].min()

        if max_v == min_v:
            norm_df[col] = 1
        else:
            norm_df[col] = (max_v - metric_df[col]) / (max_v - min_v)

    max_r2 = metric_df["Test_R2"].max()
    min_r2 = metric_df["Test_R2"].min()

    if max_r2 == min_r2:
        norm_df["Test_R2"] = 1
    else:
        norm_df["Test_R2"] = (metric_df["Test_R2"] - min_r2) / (max_r2 - min_r2)

    mat = norm_df.values

    denom = np.sqrt(
        (mat ** 2).sum(axis=0)
    )

    denom[denom == 0] = 1

    mat = mat / denom

    best = mat.max(axis=0)
    worst = mat.min(axis=0)

    d_best = np.sqrt(
        ((mat - best) ** 2).sum(axis=1)
    )

    d_worst = np.sqrt(
        ((mat - worst) ** 2).sum(axis=1)
    )

    df["TOPSIS_Score"] = d_worst / (d_best + d_worst)

    df = df.sort_values(
        "TOPSIS_Score",
        ascending=False
    )

    df["TOPSIS_Rank"] = range(
        1,
        len(df) + 1
    )

    df["TOPSIS_Weight"] = (
        df["TOPSIS_Score"]
        /
        df["TOPSIS_Score"].sum()
    )

    return df


# =========================
# TOPSIS加权集成模型
# =========================
class TOPSISWeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(
            weights,
            dtype=float
        )

        if len(self.models) != len(self.weights):
            raise ValueError("模型数量与权重数量不一致。")

        self.model_names = list(self.models.keys())
        self.weights_dict = dict(
            zip(
                self.model_names,
                self.weights
            )
        )

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(
                X,
                y
            )

        return self

    def predict(self, X):
        pred = np.zeros(
            X.shape[0],
            dtype=float
        )

        for name, model in self.models.items():
            pred += model.predict(X) * self.weights_dict[name]

        return pred

    def get_model_weights(self):
        return self.weights_dict


# =========================
# 字体强制设置
# =========================
def force_font(fig):
    for ax in fig.axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels() + [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label
        ]:
            item.set_fontname("Times New Roman")
            item.set_fontweight("bold")
            item.set_fontsize(24)

        legend = ax.get_legend()

        if legend is not None:
            for text in legend.get_texts():
                text.set_fontname("Times New Roman")
                text.set_fontweight("bold")
                text.set_fontsize(24)


# =========================
# 集成模型图
# =========================
def draw_single_model(fig, outer_grid_spec,
                      y_train_true, y_train_pred,
                      y_test_true, y_test_pred,
                      model_name="EnsembleModel",
                      tag_letter="(a)"):

    y_train_true = np.asarray(y_train_true)
    y_train_pred = np.asarray(y_train_pred)
    y_test_true = np.asarray(y_test_true)
    y_test_pred = np.asarray(y_test_pred)

    resid_train = y_train_pred - y_train_true
    resid_test = y_test_pred - y_test_true

    train_r2 = r2_score(y_train_true, y_train_pred)
    test_r2 = r2_score(y_test_true, y_test_pred)

    c_train = "#d62728"
    c_test = "#1f77b4"

    inner_grid = gridspec.GridSpecFromSubplotSpec(
        3,
        2,
        subplot_spec=outer_grid_spec,
        width_ratios=[4, 1],
        height_ratios=[1, 4, 1.2],
        wspace=0.0,
        hspace=0.0
    )

    ax_main = fig.add_subplot(inner_grid[1, 0])
    ax_top = fig.add_subplot(inner_grid[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner_grid[1, 1], sharey=ax_main)
    ax_resid = fig.add_subplot(inner_grid[2, 0], sharex=ax_main)

    ax_main.scatter(
        y_train_true,
        y_train_pred,
        c=c_train,
        marker="o",
        alpha=0.45,
        label="Train",
        s=25,
        edgecolors="white",
        linewidth=0.3
    )

    ax_main.scatter(
        y_test_true,
        y_test_pred,
        c=c_test,
        marker="^",
        alpha=0.45,
        label="Test",
        s=25,
        edgecolors="white",
        linewidth=0.3
    )

    all_vals = np.concatenate([
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred
    ])

    min_v = all_vals.min()
    max_v = all_vals.max()

    margin = (max_v - min_v) * 0.05 if max_v > min_v else 1

    lims = [
        min_v - margin,
        max_v + margin
    ]

    ax_main.plot(
        lims,
        lims,
        "k--",
        alpha=0.75,
        linewidth=1.5
    )

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)

    try:
        line_x = np.linspace(lims[0], lims[1], 200)

        lr_train = LinearRegression().fit(
            y_train_true.reshape(-1, 1),
            y_train_pred
        )

        lr_test = LinearRegression().fit(
            y_test_true.reshape(-1, 1),
            y_test_pred
        )

        ax_main.plot(
            line_x,
            lr_train.predict(line_x.reshape(-1, 1)),
            c=c_train,
            lw=1.5
        )

        ax_main.plot(
            line_x,
            lr_test.predict(line_x.reshape(-1, 1)),
            c=c_test,
            lw=1.5
        )

    except Exception:
        pass

    ax_main.legend(
        loc="upper left",
        frameon=True,
        prop={
            "family": "Times New Roman",
            "weight": "bold",
            "size": 24
        }
    )

    ax_main.text(
        0.95,
        0.05,
        f"Train $\\mathbf{{R^2}}$ = {train_r2:.4f}\nTest  $\\mathbf{{R^2}}$ = {test_r2:.4f}",
        transform=ax_main.transAxes,
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman",
        ha="right",
        va="bottom"
    )

    ax_main.set_ylabel("Predicted Values")

    plt.setp(
        ax_main.get_xticklabels(),
        visible=False
    )

    ax_top.text(
        -0.10,
        1.20,
        tag_letter,
        transform=ax_top.transAxes,
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman",
        va="top",
        ha="right"
    )

    ax_top.set_title(
        model_name,
        fontsize=24,
        fontweight="bold",
        fontname="Times New Roman",
        y=1.25
    )

    sns.histplot(
        y_train_true,
        ax=ax_top,
        color=c_train,
        kde=True,
        stat="density",
        bins=20,
        element="step",
        alpha=0.35
    )

    sns.histplot(
        y_test_true,
        ax=ax_top,
        color=c_test,
        kde=True,
        stat="density",
        bins=20,
        element="step",
        alpha=0.35
    )

    ax_top.set_axis_off()

    sns.histplot(
        y=y_train_pred,
        ax=ax_right,
        color=c_train,
        kde=True,
        stat="density",
        bins=20,
        element="step",
        alpha=0.35
    )

    sns.histplot(
        y=y_test_pred,
        ax=ax_right,
        color=c_test,
        kde=True,
        stat="density",
        bins=20,
        element="step",
        alpha=0.35
    )

    ax_right.set_axis_off()

    ax_resid.scatter(
        y_train_true,
        resid_train,
        c=c_train,
        marker="o",
        alpha=0.45,
        s=20
    )

    ax_resid.scatter(
        y_test_true,
        resid_test,
        c=c_test,
        marker="^",
        alpha=0.45,
        s=20
    )

    ax_resid.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1
    )

    ax_resid.set_xlabel("True Values")
    ax_resid.set_ylabel("Residuals")

    max_resid = max(
        np.max(np.abs(resid_train)),
        np.max(np.abs(resid_test))
    )

    if max_resid == 0:
        max_resid = 1

    ax_resid.set_ylim(
        -max_resid * 1.1,
        max_resid * 1.1
    )

    force_font(fig)


def plot_ensemble(target, model, X_train, y_train, X_test, y_test):
    fig = plt.figure(
        figsize=(7, 8)
    )

    outer_grid = gridspec.GridSpec(
        1,
        1
    )

    draw_single_model(
        fig,
        outer_grid[0],
        y_train,
        model.predict(X_train),
        y_test,
        model.predict(X_test),
        model_name="EnsembleModel",
        tag_letter="(a)"
    )

    plt.tight_layout()

    for ext, dpi in [
        ("png", 300),
        ("pdf", 1200),
        ("tif", 300)
    ]:
        plt.savefig(
            os.path.join(
                output_dir,
                f"{target}_TOPSIS_Weighted_Ensemble.{ext}"
            ),
            dpi=dpi,
            bbox_inches="tight"
        )

    plt.close(fig)


# =========================
# 主程序
# =========================
def main():
    df = load_data(file_path)

    feature_cols = df.columns[:7].tolist()

    data = df[
        feature_cols + [target_col]
    ].dropna(
        subset=[target_col]
    ).copy()

    X = data[feature_cols]
    y = data[target_col]

    X_train_raw, X_test_raw, y_train, y_test = spxy_split(
        X,
        y,
        test_size=0.3
    )

    print("\nData splitting method: SPXY")
    print("Target:", target_col)
    print(f"Training samples: {len(X_train_raw)}")
    print(f"Testing samples : {len(X_test_raw)}")

    imputer = KNNImputer(
        n_neighbors=5,
        weights="uniform"
    )

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

    X_train, X_test, log_cols = log_transform_om_cd(
        X_train,
        X_test
    )

    print("Log1p transformed columns:", log_cols)

    rows = []
    pred_rows = []
    infos = []

    for name in model_names:
        print(f"\nOptimizing {target_col} - {name}")

        try:
            model, params, cv_r2 = bayes_optimize(
                name,
                X_train,
                y_train
            )

            model.fit(
                X_train,
                y_train
            )

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            row = {
                "Target": target_col,
                "Model": name,
                "Best_CV_R2": cv_r2,
                "Best_Params": str(params),
                "Status": "Success",
                "Error": ""
            }

            row.update(
                calc_metrics(
                    y_train,
                    train_pred,
                    "Train"
                )
            )

            row.update(
                calc_metrics(
                    y_test,
                    test_pred,
                    "Test"
                )
            )

            row["Overfit_Gap"] = row["Train_R2"] - row["Test_R2"]

            rows.append(row)

            infos.append({
                "Model": name,
                "ModelObj": model
            })

            pred_rows.append(pd.DataFrame({
                "Target": target_col,
                "Model": name,
                "Dataset": "Train",
                "Observed": y_train.values,
                "Predicted": train_pred
            }))

            pred_rows.append(pd.DataFrame({
                "Target": target_col,
                "Model": name,
                "Dataset": "Test",
                "Observed": y_test.values,
                "Predicted": test_pred
            }))

            print(
                f"{name}: "
                f"CV R2={cv_r2:.4f}, "
                f"Train R2={row['Train_R2']:.4f}, "
                f"Test R2={row['Test_R2']:.4f}, "
                f"Gap={row['Overfit_Gap']:.4f}"
            )

        except Exception as e:
            rows.append({
                "Target": target_col,
                "Model": name,
                "Best_CV_R2": np.nan,
                "Best_Params": "",
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

    success = result_df[
        result_df["Status"] == "Success"
    ].copy()

    ranked = topsis_ranking(success)

    top3_names = ranked.head(3)["Model"].tolist()

    top3_ranked = ranked[
        ranked["Model"].isin(top3_names)
    ].copy()

    top3_ranked["Ensemble_Weight"] = (
        top3_ranked["TOPSIS_Score"]
        /
        top3_ranked["TOPSIS_Score"].sum()
    )

    weight_map = dict(
        zip(
            top3_ranked["Model"],
            top3_ranked["Ensemble_Weight"]
        )
    )

    top3_infos = [
        info for info in infos
        if info["Model"] in top3_names
    ]

    top3_infos = sorted(
        top3_infos,
        key=lambda x: top3_names.index(x["Model"])
    )

    ensemble_models = {
        info["Model"]: info["ModelObj"]
        for info in top3_infos
    }

    ensemble_weights = [
        weight_map[name]
        for name in ensemble_models.keys()
    ]

    ensemble = TOPSISWeightedEnsemble(
        models=ensemble_models,
        weights=ensemble_weights
    )

    ensemble.fit(
        X_train,
        y_train
    )

    ensemble_train = ensemble.predict(X_train)
    ensemble_test = ensemble.predict(X_test)

    ensemble_row = {
        "Target": target_col,
        "Model": "TOPSIS_Weighted_Ensemble",
        "Best_CV_R2": np.nan,
        "Best_Params": "Weighted ensemble based on TOPSIS top 3",
        "Status": "Success",
        "Error": "",
        "Base_Models": " + ".join(top3_names),
        "Base_Model_Weights": str(ensemble.get_model_weights())
    }

    ensemble_row.update(
        calc_metrics(
            y_train,
            ensemble_train,
            "Train"
        )
    )

    ensemble_row.update(
        calc_metrics(
            y_test,
            ensemble_test,
            "Test"
        )
    )

    ensemble_row["Overfit_Gap"] = ensemble_row["Train_R2"] - ensemble_row["Test_R2"]

    final_df = pd.concat(
        [
            ranked,
            pd.DataFrame([ensemble_row])
        ],
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

    pred_df = pd.concat(
        pred_rows,
        ignore_index=True
    )

    plot_ensemble(
        target_col,
        ensemble,
        X_train,
        y_train.values,
        X_test,
        y_test.values
    )

    save_path = os.path.join(
        output_dir,
        f"{target_col}_SPXY_log1p_Bayes_TOPSIS_Weighted_Ensemble_Results.xlsx"
    )

    with pd.ExcelWriter(save_path) as writer:
        final_df.to_excel(
            writer,
            sheet_name="Model_Performance",
            index=False
        )

        pred_df.to_excel(
            writer,
            sheet_name="Prediction_Result",
            index=False
        )

        top3_ranked.to_excel(
            writer,
            sheet_name="TOP3_TOPSIS_Weights",
            index=False
        )

        pd.DataFrame({
            "Target": [target_col],
            "Splitting_Method": ["SPXY based on joint X-Y distances"],
            "Imputation": ["KNNImputer(n_neighbors=5, weights='uniform')"],
            "Transform_Columns": [", ".join(log_cols)],
            "Transform_Method": ["np.log1p after KNN imputation"],
            "Train_Test_Split": ["7:3"],
            "Random_State": [random_state],
            "n_trials": [n_trials]
        }).to_excel(
            writer,
            sheet_name="Preprocessing_Info",
            index=False
        )

    print("\nSaved:")
    print(save_path)

    print("\nFigures saved in:")
    print(output_dir)


if __name__ == "__main__":
    main()