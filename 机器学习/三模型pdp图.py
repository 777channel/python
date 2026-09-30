# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import joblib

from sklearn.base import BaseEstimator, RegressorMixin


# =====================================================
# 集成模型类：用于 joblib 读取 pkl
# 注意：类名需要和保存 pkl 时的类名一致或兼容
# =====================================================
class EntropyTOPSISWeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights, dtype=float)
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


class TOPSISWeightedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights, dtype=float)
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


class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = np.array(weights, dtype=float)
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


# =====================================================
# 路径设置
# =====================================================
output_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\PDP热力图 三个集成模型"
os.makedirs(output_dir, exist_ok=True)

model_paths = {
    "C-G": r"C:\Users\AMDYE\OneDrive\python\机器学习\C-G结果\SHAP\C-G_Entropy_TOPSIS_Weighted_Ensemble.pkl",
    "C-Cd": r"C:\Users\AMDYE\OneDrive\python\机器学习\C-Cd_SPXY_log1p_SHAP_Ensemble结果\C-Cd_Entropy_TOPSIS_Weighted_Ensemble.pkl",
    "S-Cd": r"C:\Users\AMDYE\OneDrive\python\机器学习\S-cd结果\S-Cd SHAP\S-Cd_TOPSIS_Weighted_Ensemble.pkl"
}

crop_feature = "Crop"

base_feature_cols = [
    "MPs",
    "Size",
    "MPR",
    "pH",
    "OM",
    "CEC",
    "Cd",
    crop_feature
]


# =====================================================
# 字体设置：Times New Roman
# =====================================================
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 20,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold"
})


# =====================================================
# 固定输入与网格范围
# =====================================================
size_fixed = 150
om_fixed_raw = 15
cec_fixed = 15

# 绘图网格：MPR 0-5%（模型输入为0-0.05）取50个点；Cd 0-3取30个点
mpr_values_plot = np.linspace(0, 0.05, 500)
cd_values_plot_raw = np.linspace(0, 3.0, 300)

MPR_mesh_plot, Cd_mesh_plot_raw = np.meshgrid(
    mpr_values_plot,
    cd_values_plot_raw
)

# Excel分析网格：保存关键分析点
# MPR按0.01保存：0、0.01、0.02、0.03、0.04、0.05
# Cd按0.01保存：0、0.01、0.02 ... 3.00
mpr_values_excel = np.round(np.arange(0, 0.0501, 0.01), 4)
cd_values_excel_raw = np.round(np.arange(0, 3.0001, 0.01), 4)

MPR_mesh_excel, Cd_mesh_excel_raw = np.meshgrid(
    mpr_values_excel,
    cd_values_excel_raw
)

ph_list = [5, 6, 7]
crop_list = [0, 1]
mps_list = [0, 1]

crop_label_map = {
    0: "Leafy vegetables",
    1: "Grain crops"
}


# =====================================================
# 模型特征自动识别与对齐
# =====================================================
def get_model_feature_cols(model):
    """
    从集成模型的基学习器中读取训练时的特征名。
    如果读取不到，则使用base_feature_cols。
    """
    if hasattr(model, "models"):
        for base_model in model.models.values():
            if hasattr(base_model, "feature_names_in_"):
                return list(base_model.feature_names_in_)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return base_feature_cols


# =====================================================
# 构建预测网格
# =====================================================
def build_prediction_grid(mps_value, ph_value, crop_value, mpr_mesh, cd_mesh_raw, feature_cols):
    X_pred_all = pd.DataFrame({
        "MPs": np.full(mpr_mesh.size, mps_value),
        "Size": np.full(mpr_mesh.size, size_fixed),
        "MPR": mpr_mesh.ravel(),
        "pH": np.full(mpr_mesh.size, ph_value),
        "OM": np.full(mpr_mesh.size, om_fixed_raw),
        "CEC": np.full(mpr_mesh.size, cec_fixed),
        "Cd": cd_mesh_raw.ravel(),
        crop_feature: np.full(mpr_mesh.size, crop_value)
    })

    # 与训练阶段保持一致：OM 和 Cd 输入模型前 log1p
    if "OM" in X_pred_all.columns:
        if (X_pred_all["OM"] <= -1).any():
            raise ValueError("OM contains values <= -1, cannot apply log1p.")
        X_pred_all["OM"] = np.log1p(X_pred_all["OM"])

    if "Cd" in X_pred_all.columns:
        if (X_pred_all["Cd"] <= -1).any():
            raise ValueError("Cd contains values <= -1, cannot apply log1p.")
        X_pred_all["Cd"] = np.log1p(X_pred_all["Cd"])

    # 按模型训练时的特征列和顺序输入
    missing_cols = [col for col in feature_cols if col not in X_pred_all.columns]
    if missing_cols:
        raise ValueError(f"Prediction grid is missing columns required by model: {missing_cols}")

    return X_pred_all[feature_cols]


# =====================================================
# 字体强制设置
# =====================================================
def force_font(fig, fontsize=18):
    for ax in fig.axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels() + [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label
        ]:
            item.set_fontname("Times New Roman")
            item.set_fontweight("bold")
            item.set_fontsize(fontsize)

    for text in fig.texts:
        text.set_fontname("Times New Roman")
        text.set_fontweight("bold")


# =====================================================
# 保存Excel分析数据
# =====================================================
def save_pdp_excel_data(target_name, model, feature_cols, target_dir):
    rows = []

    for mps_value in mps_list:
        for crop_value in crop_list:
            for ph_value in ph_list:
                X_pred = build_prediction_grid(
                    mps_value=mps_value,
                    ph_value=ph_value,
                    crop_value=crop_value,
                    mpr_mesh=MPR_mesh_excel,
                    cd_mesh_raw=Cd_mesh_excel_raw,
                    feature_cols=feature_cols
                )

                y_pred = model.predict(X_pred)

                temp_df = pd.DataFrame({
                    "Target": target_name,
                    "MPs": mps_value,
                    "Crop": crop_value,
                    "Crop_Label": crop_label_map[crop_value],
                    "pH": ph_value,
                    "MPR": MPR_mesh_excel.ravel(),
                    "Cd_raw": Cd_mesh_excel_raw.ravel(),
                    "OM_raw_fixed": om_fixed_raw,
                    "OM_log1p_input": np.log1p(om_fixed_raw),
                    "CEC_fixed": cec_fixed,
                    "Size_fixed": size_fixed,
                    "Cd_log1p_input": np.log1p(Cd_mesh_excel_raw.ravel()),
                    "Predicted_Value": y_pred
                })

                rows.append(temp_df)

    pdp_df = pd.concat(rows, ignore_index=True)

    excel_path = os.path.join(
        target_dir,
        f"{target_name}_PDP_Analysis_Data_MPR_step_0.01_Cd_step_0.01.xlsx"
    )

    with pd.ExcelWriter(excel_path) as writer:
        pdp_df.to_excel(
            writer,
            sheet_name="Long_Format",
            index=False
        )

        pd.DataFrame({
            "Item": [
                "Plot_MPR_Points",
                "Plot_Cd_Points",
                "Excel_MPR_Values",
                "Excel_Cd_Values",
                "OM_fixed_raw",
                "OM_input_log1p",
                "CEC_fixed",
                "Size_fixed",
                "Cd_input_transform",
                "Model_Features"
            ],
            "Value": [
                len(mpr_values_plot),
                len(cd_values_plot_raw),
                ", ".join(map(str, mpr_values_excel)),
                "0 to 3 with step 0.01",
                om_fixed_raw,
                np.log1p(om_fixed_raw),
                cec_fixed,
                size_fixed,
                "np.log1p(Cd_raw)",
                ", ".join(feature_cols)
            ]
        }).to_excel(
            writer,
            sheet_name="Info",
            index=False
        )

    print(f"Saved PDP analysis Excel: {excel_path}")


# =====================================================
# 绘制单个模型的 PDP 热力图
# 每个模型单独文件夹；每个模型输出 MPs=0 和 MPs=1 两张图
# 子图布局：2 行 × 3 列 + 单独色标列
# 第一行显示 pH=5/6/7
# 第一列显示作物类型
# =====================================================
def plot_pdp_heatmap(target_name, model):
    target_dir = os.path.join(output_dir, target_name)
    os.makedirs(target_dir, exist_ok=True)

    feature_cols = get_model_feature_cols(model)
    print(f"Model features for {target_name}: {feature_cols}")

    # 每个模型单独保存一份用于分析的Excel数据
    save_pdp_excel_data(
        target_name=target_name,
        model=model,
        feature_cols=feature_cols,
        target_dir=target_dir
    )

    for mps_value in mps_list:
        fig = plt.figure(
            figsize=(18, 10),
            dpi=120,
            constrained_layout=False
        )

        grid_spec = fig.add_gridspec(
            2,
            4,
            width_ratios=[1.0, 1.0, 1.0, 0.06],
            wspace=0.18,
            hspace=0.18
        )

        axes = np.empty((2, 3), dtype=object)
        for i in range(2):
            for j in range(3):
                axes[i, j] = fig.add_subplot(grid_spec[i, j])

        cbar_ax = fig.add_subplot(grid_spec[:, 3])

        pred_maps = {}
        all_preds = []

        for i, crop_value in enumerate(crop_list):
            for j, ph_value in enumerate(ph_list):
                X_pred = build_prediction_grid(
                    mps_value=mps_value,
                    ph_value=ph_value,
                    crop_value=crop_value,
                    mpr_mesh=MPR_mesh_plot,
                    cd_mesh_raw=Cd_mesh_plot_raw,
                    feature_cols=feature_cols
                )

                y_pred = model.predict(X_pred)
                Z = y_pred.reshape(Cd_mesh_plot_raw.shape)

                pred_maps[(crop_value, ph_value)] = Z
                all_preds.append(Z.ravel())

        all_preds = np.concatenate(all_preds)
        vmin = np.nanmin(all_preds)
        vmax = np.nanmax(all_preds)

        for i, crop_value in enumerate(crop_list):
            for j, ph_value in enumerate(ph_list):
                ax = axes[i, j]
                Z = pred_maps[(crop_value, ph_value)]

                im = ax.contourf(
                    MPR_mesh_plot,
                    Cd_mesh_plot_raw,
                    Z,
                    levels=80,
                    cmap="coolwarm",
                    vmin=vmin,
                    vmax=vmax,
                    antialiased=True
                )

                # 仅第一行显示 pH=5/6/7
                if i == 0:
                    ax.set_title(
                        f"pH = {ph_value}",
                        fontsize=24,
                        fontweight="bold",
                        fontname="Times New Roman"
                    )
                else:
                    ax.set_title("")

                # 仅第一列显示 Crop 类型，英文 Times New Roman
                if j == 0:
                    ax.set_ylabel(
                    crop_label_map[crop_value] + "\n\nCd (mg·kg$^{-1}$)",
                    fontsize=24,
                    fontweight="bold",
                    fontname="Times New Roman",
                    labelpad=35
                )
                else:
                    ax.set_ylabel("")

                # 仅底行显示 x 轴名称
                if i == 1:
                    ax.set_xlabel(
                        "MPR",
                        fontsize=22,
                        fontweight="bold",
                        fontname="Times New Roman"
                    )
                else:
                    ax.set_xlabel("")

                ax.set_xlim(0, 0.05)
                ax.set_ylim(0, 3)

                # 隐藏 y 轴 0.0 刻度标签，避免与 x 轴 0.00 重叠；保留 x 轴 0.00
                y_ticks = ax.get_yticks()
                y_tick_labels = []
                for tick in y_ticks:
                    if abs(tick) < 1e-8:
                        y_tick_labels.append("")
                    else:
                        y_tick_labels.append(f"{tick:.1f}")

                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels)

                ax.tick_params(
                    axis="both",
                    labelsize=18,
                    width=1.5
                )

                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontname("Times New Roman")
                    label.set_fontweight("bold")

        cbar = fig.colorbar(
            im,
            cax=cbar_ax
        )

        cbar.set_label(
            f"Predicted {target_name}",
            fontsize=22,
            fontweight="bold",
            fontname="Times New Roman",
            labelpad=35
        )

        for label in cbar.ax.get_yticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontweight("bold")
            label.set_fontsize(18)

        fig.suptitle(
            f"{target_name} Ensemble PDP Heatmap | MPs = {mps_value}",
            fontsize=24,
            fontweight="bold",
            fontname="Times New Roman",
            y=0.97
        )

        force_font(fig, fontsize=18)

        fig.subplots_adjust(
            left=0.08,
            right=0.92,
            bottom=0.08,
            top=0.90
        )

        save_path = os.path.join(
            target_dir,
            f"{target_name}_MPs_{mps_value}_PDP_Heatmap.pdf"
        )

        plt.savefig(
            save_path,
            dpi=1200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {target_name}, MPs={mps_value}")


# =====================================================
# 主程序
# =====================================================
def main():
    for target_name, model_path in model_paths.items():
        print("\nLoading model:")
        print(target_name)
        print(model_path)

        if not os.path.exists(model_path):
            print(f"Model file does not exist, skipped: {model_path}")
            continue

        model = joblib.load(model_path)

        plot_pdp_heatmap(
            target_name=target_name,
            model=model
        )

    print("\nAll PDP heatmaps finished.")
    print(output_dir)


if __name__ == "__main__":
    main()
