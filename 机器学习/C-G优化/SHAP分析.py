# -*- coding: utf-8 -*-
import os

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

import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    shap = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None


# =========================
# 路径设置
# =========================
file_path = r"C:\Users\AMDYE\OneDrive\桌面\训练版本(1).xlsx"
output_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\C-G结果\SHAP"
os.makedirs(output_dir, exist_ok=True)

target_col = "C-G"
random_state = 42

model_save_path = os.path.join(output_dir, f"{target_col}_Entropy_TOPSIS_Weighted_Ensemble.pkl")
imputer_save_path = os.path.join(output_dir, f"{target_col}_KNN_Imputer.pkl")
feature_cols_save_path = os.path.join(output_dir, f"{target_col}_Feature_Columns.pkl")

model_names = [
    "RandomForest",
    "AdaBoost",
    "CatBoost",
    "XGBoost",
    "GBRT",
    "TabPFN"
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
# SPXY划分
# =========================
def spxy_split(X, y, test_size=0.30):
    X_for_split = X.copy()
    X_for_split = X_for_split.apply(lambda col: col.fillna(col.median()), axis=0)

    y_for_split = y.copy()
    y_for_split = y_for_split.fillna(y_for_split.median())

    X_values = X_for_split.values.astype(float)
    y_values = y_for_split.values.reshape(-1, 1).astype(float)

    X_scaled = StandardScaler().fit_transform(X_values)
    y_scaled = StandardScaler().fit_transform(y_values)

    n_samples = X_scaled.shape[0]
    n_train = int(round(n_samples * (1 - test_size)))

    if n_train < 2:
        raise ValueError("训练集样本数过少，无法进行SPXY划分。")

    diff_x = X_scaled[:, None, :] - X_scaled[None, :, :]
    dist_x = np.sqrt(np.sum(diff_x ** 2, axis=2))

    diff_y = y_scaled[:, None, :] - y_scaled[None, :, :]
    dist_y = np.sqrt(np.sum(diff_y ** 2, axis=2))

    if dist_x.max() > 0:
        dist_x = dist_x / dist_x.max()

    if dist_y.max() > 0:
        dist_y = dist_y / dist_y.max()

    dist_xy = dist_x + dist_y

    first_pair = np.unravel_index(np.argmax(dist_xy), dist_xy.shape)

    train_indices = [first_pair[0], first_pair[1]]
    remaining_indices = [i for i in range(n_samples) if i not in train_indices]

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

    log_cols = [col for col in ["OM", "Cd"] if col in X_train_t.columns]

    for col in log_cols:
        if (X_train_t[col] <= -1).any() or (X_test_t[col] <= -1).any():
            raise ValueError(f"{col}列存在小于或等于-1的值，无法进行log1p转换。")

        X_train_t[col] = np.log1p(X_train_t[col])
        X_test_t[col] = np.log1p(X_test_t[col])

    return X_train_t, X_test_t, log_cols


# =========================
# 按Crop列分层抽样
# =========================
def crop_stratified_split(X, y, crop_strata, test_size=0.30):
    crop_strata = pd.Series(
        crop_strata,
        index=X.index
    )

    valid_mask = crop_strata.notna()

    X_valid = X.loc[valid_mask].copy()
    y_valid = y.loc[valid_mask].copy()
    strata_valid = crop_strata.loc[valid_mask].copy()

    counts = strata_valid.value_counts()

    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            "Crop列无法用于分层抽样：至少需要两个类别，且每个类别样本数不少于2。"
        )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=test_size,
        random_state=random_state,
        stratify=strata_valid
    )

    return X_train_raw, X_test_raw, y_train, y_test


# =========================
# 固定最佳参数模型：C-G
# =========================
def get_fixed_model(model_name):
    if model_name == "TabPFN":
        if TabPFNRegressor is None:
            raise ImportError("TabPFN is not installed.")
        return TabPFNRegressor()

    if model_name == "CatBoost":
        if CatBoostRegressor is None:
            raise ImportError("CatBoost is not installed.")
        return CatBoostRegressor(
            iterations=1598,
            depth=11,
            learning_rate=0.012307305996828099,
            l2_leaf_reg=16.00142324467588,
            random_strength=0.00027827157766896796,
            bagging_temperature=3.3918175661229473,
            rsm=0.47811620309955505,
            border_count=121,
            loss_function="RMSE",
            random_state=random_state,
            verbose=0
        )

    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed.")
        return XGBRegressor(
            n_estimators=673,
            max_depth=11,
            learning_rate=0.004089507203857863,
            subsample=0.9661182544867442,
            colsample_bytree=0.9129907820163936,
            colsample_bylevel=0.4917215499913709,
            colsample_bynode=0.7374191901553558,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )

    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=1049,
            max_depth=59,
            min_samples_split=6,
            min_samples_leaf=1,
            max_features=0.2,
            bootstrap=False,
            random_state=random_state,
            n_jobs=-1
        )

    if model_name == "AdaBoost":
        return AdaBoostRegressor(
            n_estimators=354,
            learning_rate=0.0008094511788305424,
            loss="linear",
            random_state=random_state
        )

    if model_name == "GBRT":
        return GradientBoostingRegressor(
            n_estimators=1396,
            learning_rate=0.0030657504760013914,
            max_depth=7,
            min_samples_split=12,
            min_samples_leaf=1,
            subsample=0.7968312987673499,
            max_features=0.3,
            random_state=random_state
        )

    raise ValueError(f"Unknown model name: {model_name}")


# =========================
# 评价指标
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
# 熵权-TOPSIS
# =========================
def entropy_weight_topsis(df):
    df = df.copy()

    metric_cols = ["Test_MSE", "Test_RMSE", "Test_MAE", "Test_R2"]
    data = df[metric_cols].copy()
    data_bzh = pd.DataFrame(index=data.index)

    for col in ["Test_MSE", "Test_RMSE", "Test_MAE"]:
        max_v = data[col].max()
        min_v = data[col].min()
        data_bzh[col] = 1.0 if max_v == min_v else (max_v - data[col]) / (max_v - min_v)

    max_r2 = data["Test_R2"].max()
    min_r2 = data["Test_R2"].min()
    data_bzh["Test_R2"] = 1.0 if max_r2 == min_r2 else (data["Test_R2"] - min_r2) / (max_r2 - min_r2)

    data_bzh = data_bzh + 1e-12

    P = data_bzh / np.tile(data_bzh.sum(axis=0).values, (data_bzh.shape[0], 1))
    E = -1 / np.log(data_bzh.shape[0]) * (P * np.log(P)).sum(axis=0)
    D = 1 - E
    W = D / D.sum()

    Z = pd.DataFrame(
        [data_bzh.max(), data_bzh.min()],
        index=["Positive_Ideal_Solution", "Negative_Ideal_Solution"]
    )

    result = df.copy()
    result["Positive_Distance"] = np.sqrt(((data_bzh - Z.loc["Positive_Ideal_Solution"]) ** 2 * W).sum(axis=1))
    result["Negative_Distance"] = np.sqrt(((data_bzh - Z.loc["Negative_Ideal_Solution"]) ** 2 * W).sum(axis=1))
    result["TOPSIS_Score"] = result["Negative_Distance"] / (result["Negative_Distance"] + result["Positive_Distance"])
    result["TOPSIS_Percentage"] = result["TOPSIS_Score"] / result["TOPSIS_Score"].sum()

    result = result.sort_values(by="TOPSIS_Score", ascending=False)
    result["TOPSIS_Rank"] = range(1, len(result) + 1)

    weight_df = pd.DataFrame({
        "Metric": metric_cols,
        "Entropy": E.values,
        "Difference_Coefficient": D.values,
        "Entropy_Weight": W.values
    })

    ideal_solution_df = Z.copy()

    standardized_df = data_bzh.copy()
    standardized_df.insert(0, "Model", df["Model"].values)

    return result, weight_df, ideal_solution_df, standardized_df


# =========================
# 熵权-TOPSIS加权集成模型
# =========================
class EntropyTOPSISWeightedEnsemble(BaseEstimator, RegressorMixin):
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
# SHAP分析入口
# =========================
def run_shap_analysis(target, ensemble_model, X_train, X_test):
    if shap is None:
        raise ImportError("shap is not installed. Please install it by: pip install shap")

    print("\nStarting SHAP analysis...")

    background_size = min(50, len(X_train))
    background = shap.sample(X_train, background_size, random_state=random_state)

    explainer = shap.KernelExplainer(ensemble_model.predict, background)
    shap_values = explainer.shap_values(X_test, nsamples=200)

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

    feature_values_df = X_test.copy()
    feature_values_df.index = X_test.index

    feature_values_named = feature_values_df.copy()
    feature_values_named.columns = [f"{col}_Value" for col in feature_values_named.columns]

    shap_values_named = shap_values_df.copy()
    shap_values_named.columns = [f"{col}_SHAP" for col in shap_values_named.columns]

    prediction_df = pd.DataFrame(
        {
            "Sample_Index": X_test.index,
            "Predicted_Value": ensemble_model.predict(X_test)
        },
        index=X_test.index
    )

    paired_wide_df = pd.concat([prediction_df, feature_values_named, shap_values_named], axis=1)

    paired_long_rows = []

    for sample_index in X_test.index:
        for feature in X_test.columns:
            paired_long_rows.append(
                {
                    "Sample_Index": sample_index,
                    "Feature": feature,
                    "Feature_Value": feature_values_df.loc[sample_index, feature],
                    "SHAP_Value": shap_values_df.loc[sample_index, feature],
                    "Predicted_Value": prediction_df.loc[sample_index, "Predicted_Value"]
                }
            )

    paired_long_df = pd.DataFrame(paired_long_rows)

    value_shap_save_path = os.path.join(output_dir, f"{target}_Feature_Values_and_SHAP_Values.xlsx")

    with pd.ExcelWriter(value_shap_save_path) as writer:
        paired_wide_df.to_excel(writer, sheet_name="Wide_Format", index=False)
        paired_long_df.to_excel(writer, sheet_name="Long_Format", index=False)
        feature_values_df.to_excel(writer, sheet_name="Feature_Values", index=True)
        shap_values_df.to_excel(writer, sheet_name="SHAP_Values", index=True)

    print("Feature values and corresponding SHAP values saved:")
    print(value_shap_save_path)

    feature_importance = shap_values_df.abs().mean().sort_values(ascending=False)
    feature_importance.to_excel(
        os.path.join(output_dir, f"{target}_SHAP_Feature_Importance.xlsx"),
        header=["Mean_Abs_SHAP"]
    )

    plot_shap_combined(
        target=target,
        shap_values_array=shap_values_array,
        X_test=X_test
    )

    print("SHAP analysis finished.")


# =========================
# 主程序
# =========================
def main():
    df = load_data(file_path)

    # C-G新版：Crop既作为分层依据，也纳入模型输入特征
    expected_feature_cols = [
        "MPs",
        "Size",
        "MPR",
        "pH",
        "OM",
        "CEC",
        "Cd",
        "Crop"
    ]

    missing_cols = [
        col for col in expected_feature_cols
        if col not in df.columns
    ]

    if len(missing_cols) > 0:
        raise ValueError(
            "数据中缺少以下输入特征列：" + ", ".join(missing_cols)
        )

    feature_cols = expected_feature_cols

    print("\nFeature columns:")
    for col in feature_cols:
        print(f"  {col}")

    data = df[
        feature_cols + [target_col]
    ].dropna(
        subset=[target_col]
    ).copy()

    X = data[feature_cols]
    y = data[target_col]

    crop_strata = data["Crop"]

    X_train_raw, X_test_raw, y_train, y_test = crop_stratified_split(
        X,
        y,
        crop_strata=crop_strata,
        test_size=0.30
    )

    print("\nData splitting method: Crop-stratified train_test_split")
    print("Target:", target_col)
    print(f"Training samples: {len(X_train_raw)}")
    print(f"Testing samples : {len(X_test_raw)}")
    print("\nCrop distribution in train set:")
    print(df.loc[X_train_raw.index, "Crop"].value_counts().sort_index())
    print("\nCrop distribution in test set:")
    print(df.loc[X_test_raw.index, "Crop"].value_counts().sort_index())

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

    rows = []
    pred_rows = []
    fitted_infos = []

    for name in model_names:
        print(f"\nTraining fixed model: {target_col} - {name}")

        try:
            model = get_fixed_model(name)
            model.fit(X_train, y_train)

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            row = {
                "Target": target_col,
                "Model": name,
                "Status": "Success",
                "Error": ""
            }

            row.update(calc_metrics(y_train, train_pred, "Train"))
            row.update(calc_metrics(y_test, test_pred, "Test"))
            row["Overfit_Gap"] = row["Train_R2"] - row["Test_R2"]

            rows.append(row)

            fitted_infos.append({"Model": name, "ModelObj": model})

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

    ranked_df, entropy_weight_df, ideal_solution_df, standardized_df = entropy_weight_topsis(success_df)

    top3_names = ranked_df.head(3)["Model"].tolist()
    top3_df = ranked_df[ranked_df["Model"].isin(top3_names)].copy()

    top3_df["Ensemble_Weight"] = top3_df["TOPSIS_Percentage"] / top3_df["TOPSIS_Percentage"].sum()

    weight_map = dict(zip(top3_df["Model"], top3_df["Ensemble_Weight"]))

    top3_infos = [info for info in fitted_infos if info["Model"] in top3_names]
    top3_infos = sorted(top3_infos, key=lambda x: top3_names.index(x["Model"]))

    print("\nTOP3 for Entropy-TOPSIS weighted ensemble:")
    for i, name in enumerate(top3_names, start=1):
        print(f"{i}. {name}, weight = {weight_map[name]:.6f}")

    ensemble_models = {info["Model"]: info["ModelObj"] for info in top3_infos}
    ensemble_weights = [weight_map[name] for name in ensemble_models.keys()]

    ensemble_model = EntropyTOPSISWeightedEnsemble(
        models=ensemble_models,
        weights=ensemble_weights
    )

    ensemble_model.fit(X_train, y_train)

    ensemble_train = ensemble_model.predict(X_train)
    ensemble_test = ensemble_model.predict(X_test)

    ensemble_metrics = {}
    ensemble_metrics.update(calc_metrics(y_train, ensemble_train, "Train"))
    ensemble_metrics.update(calc_metrics(y_test, ensemble_test, "Test"))
    ensemble_metrics["Overfit_Gap"] = ensemble_metrics["Train_R2"] - ensemble_metrics["Test_R2"]

    print("\nEntropy-TOPSIS Weighted Ensemble Performance:")
    print(f"Base Models : {' + '.join(top3_names)}")
    print(f"Weights     : {ensemble_model.get_model_weights()}")
    print(f"Train R2    : {ensemble_metrics['Train_R2']:.4f}")
    print(f"Test R2     : {ensemble_metrics['Test_R2']:.4f}")
    print(f"Train RMSE  : {ensemble_metrics['Train_RMSE']:.4f}")
    print(f"Test RMSE   : {ensemble_metrics['Test_RMSE']:.4f}")
    print(f"Train MAE   : {ensemble_metrics['Train_MAE']:.4f}")
    print(f"Test MAE    : {ensemble_metrics['Test_MAE']:.4f}")
    print(f"Overfit Gap : {ensemble_metrics['Overfit_Gap']:.4f}")

    ensemble_row = {
        "Target": target_col,
        "Model": "Entropy_TOPSIS_Weighted_Ensemble",
        "Status": "Success",
        "Error": "",
        "Base_Models": " + ".join(top3_names),
        "Base_Model_Weights": str(ensemble_model.get_model_weights())
    }

    ensemble_row.update(ensemble_metrics)

    ranked_with_ensemble = pd.concat([ranked_df, pd.DataFrame([ensemble_row])], ignore_index=True)

    pred_rows.append(pd.DataFrame({
        "Target": target_col,
        "Model": "Entropy_TOPSIS_Weighted_Ensemble",
        "Dataset": "Train",
        "Observed": y_train.values,
        "Predicted": ensemble_train
    }))

    pred_rows.append(pd.DataFrame({
        "Target": target_col,
        "Model": "Entropy_TOPSIS_Weighted_Ensemble",
        "Dataset": "Test",
        "Observed": y_test.values,
        "Predicted": ensemble_test
    }))

    pred_df = pd.concat(pred_rows, ignore_index=True)

    joblib.dump(ensemble_model, model_save_path)

    print("\nModel saved:")
    print(model_save_path)

    save_path = os.path.join(output_dir, f"{target_col}_CropStratified_log1p_Fixed_Entropy_TOPSIS_Weighted_Ensemble_SHAP_Results.xlsx")

    with pd.ExcelWriter(save_path) as writer:
        ranked_with_ensemble.to_excel(writer, sheet_name="Model_Performance", index=False)
        pred_df.to_excel(writer, sheet_name="Prediction_Result", index=False)
        top3_df.to_excel(writer, sheet_name="TOP3_Weights", index=False)
        entropy_weight_df.to_excel(writer, sheet_name="Entropy_Weight", index=False)
        ideal_solution_df.to_excel(writer, sheet_name="Ideal_Solution", index=True)
        standardized_df.to_excel(writer, sheet_name="Standardized_Metrics", index=False)

        pd.DataFrame({
            "Target": [target_col],
            "Feature_Columns": [", ".join(feature_cols)],
            "Splitting_Method": ["Crop-stratified train_test_split"],
            "Imputation": ["KNNImputer(n_neighbors=5, weights='uniform')"],
            "Transform_Columns": [", ".join(log_cols)],
            "Transform_Method": ["np.log1p after KNN imputation"],
            "Train_Test_Split": ["7:3"],
            "Random_State": [random_state],
            "SHAP_Method": ["KernelExplainer for ensemble model"],
            "SHAP_Background_Size": [min(50, len(X_train))],
            "SHAP_NSamples": [200]
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
