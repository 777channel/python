# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
except:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except:
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except:
    CatBoostRegressor = None

import optuna
warnings.filterwarnings("ignore")


# =========================
# 路径
# =========================
file_path = r"C:\Users\AMDYE\OneDrive\桌面\训练版本(1).xlsx"
output_dir = r"C:\Users\AMDYE\OneDrive\python\机器学习\C-Cd结果"
target_col = "C-Cd"

os.makedirs(output_dir, exist_ok=True)

random_state = 42
n_trials = 300

model_names = [
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "GBRT",
    "CatBoost"
]


# =========================
# 数据读取
# =========================
def load_data(path):
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# =========================
# SPXY（保持不动）
# =========================
def spxy_split(X, y, test_size=0.3):
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    Xv = StandardScaler().fit_transform(X.values)
    yv = StandardScaler().fit_transform(y.values.reshape(-1,1))

    n = len(X)
    n_train = int(n*(1-test_size))

    dist_x = np.sqrt(((Xv[:,None,:]-Xv[None,:,:])**2).sum(-1))
    dist_y = np.sqrt(((yv[:,None,:]-yv[None,:,:])**2).sum(-1))

    dist = dist_x/dist_x.max() + dist_y/dist_y.max()

    i,j = np.unravel_index(np.argmax(dist), dist.shape)
    train = [i,j]
    rest = [k for k in range(n) if k not in train]

    while len(train) < n_train:
        scores = []
        for r in rest:
            scores.append(np.min(dist[r,train]))
        idx = np.argmax(scores)
        train.append(rest[idx])
        rest.pop(idx)

    return X.iloc[train], X.iloc[rest], y.iloc[train], y.iloc[rest]


# =========================
# log
# =========================
def log_transform(Xtr, Xte):
    for c in ["OM","Cd"]:
        if c in Xtr.columns:
            Xtr[c] = np.log1p(Xtr[c])
            Xte[c] = np.log1p(Xte[c])
    return Xtr, Xte


# =========================
# metrics
# =========================
def metrics(y,p,prefix):
    mse = mean_squared_error(y,p)
    return {
        f"{prefix}_R2": r2_score(y,p),
        f"{prefix}_RMSE": np.sqrt(mse),
        f"{prefix}_MAE": mean_absolute_error(y,p)
    }


# =========================
# 模型（保留你优化空间）
# =========================
def get_model(name, trial=None):

    if name == "RandomForest":
        if trial is None:
            return RandomForestRegressor(random_state=random_state)

        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators",100,1200),
            max_depth=trial.suggest_int("max_depth",2,15),
            min_samples_split=trial.suggest_int("min_samples_split",2,40),
            min_samples_leaf=trial.suggest_int("min_samples_leaf",1,20),
            max_features=trial.suggest_categorical("max_features",["sqrt","log2",0.4,0.6,0.8]),
            bootstrap=trial.suggest_categorical("bootstrap",[True,False]),
            random_state=random_state
        )

    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators",100,1200),
            max_depth=trial.suggest_int("max_depth",2,10),
            learning_rate=trial.suggest_float("learning_rate",0.001,0.15,log=True),
            subsample=trial.suggest_float("subsample",0.6,1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree",0.6,1.0),
            objective="reg:squarederror",
            random_state=random_state
        )

    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators",100,1200),
            max_depth=trial.suggest_int("max_depth",2,10),
            learning_rate=trial.suggest_float("learning_rate",0.001,0.15,log=True),
            num_leaves=trial.suggest_int("num_leaves",4,96),
            random_state=random_state
        )

    if name == "GBRT":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators",100,1200),
            learning_rate=trial.suggest_float("learning_rate",0.001,0.15,log=True),
            max_depth=trial.suggest_int("max_depth",2,8),
            random_state=random_state
        )

    if name == "CatBoost":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations",100,1200),
            depth=trial.suggest_int("depth",2,8),
            learning_rate=trial.suggest_float("learning_rate",0.001,0.15,log=True),
            verbose=0,
            random_state=random_state
        )


# =========================
# ❗贝叶斯优化（完全保留逻辑）
# =========================
def bayes_optimize(name, X, y):

    def objective(trial):
        model = get_model(name, trial)

        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

        scores = cross_validate(
            model, X, y,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = get_model(name, study.best_trial)

    return best_model, study.best_params, study.best_value


# =========================
# TOPSIS（简化保持结构）
# =========================
def topsis(df):
    df = df.copy()
    df["score"] = df["Test_R2"] - df["Test_RMSE"]
    return df.sort_values("score", ascending=False)


# =========================
# 主流程（核心修改点）
# =========================
def main():

    df = load_data(file_path)

    if "MPs" not in df.columns:
        raise ValueError("缺少MPs列")

    groups = sorted(df["MPs"].dropna().unique())

    print("MPs groups:", groups)

    for mp in groups:

        print("\n==================== MPs =", mp, "====================")

        sub = df[df["MPs"] == mp].copy()

        feature_cols = sub.columns[:7].tolist()
        feature_cols = [c for c in feature_cols if c != "MPs"]

        data = sub[feature_cols + [target_col]].dropna()

        X = data[feature_cols]
        y = data[target_col]

        Xtr, Xte, ytr, yte = spxy_split(X, y)

        imp = KNNImputer()
        Xtr = pd.DataFrame(imp.fit_transform(Xtr), columns=feature_cols)
        Xte = pd.DataFrame(imp.transform(Xte), columns=feature_cols)

        Xtr, Xte = log_transform(Xtr, Xte)

        rows = []
        models = {}
        best_params_store = []   # ✅ 新增：保存参数

        for name in model_names:

            try:
                model, params, cv_r2 = bayes_optimize(name, Xtr, ytr)

                model.fit(Xtr, ytr)

                p1 = model.predict(Xtr)
                p2 = model.predict(Xte)

                row = {
                    "MPs": mp,
                    "Model": name,
                    "Best_CV_R2": cv_r2,
                    "Best_Params": str(params)   # ✅ 保存最佳参数
                }

                row.update(metrics(ytr,p1,"Train"))
                row.update(metrics(yte,p2,"Test"))

                rows.append(row)
                models[name] = model

                best_params_store.append({
                    "Model": name,
                    "MPs": mp,
                    "Best_Params": params,
                    "Best_CV_R2": cv_r2
                })

                print(name, "OK")

            except Exception as e:
                print(name, "fail:", e)

        dfres = pd.DataFrame(rows)
        dfres = topsis(dfres)

        outdir = os.path.join(output_dir, f"MPs_{mp}")
        os.makedirs(outdir, exist_ok=True)

        # =========================
        # 保存结果 + 参数
        # =========================
        with pd.ExcelWriter(os.path.join(outdir, "results.xlsx")) as writer:
            dfres.to_excel(writer, index=False)
            pd.DataFrame(best_params_store).to_excel(writer, sheet_name="Best_Params", index=False)

        # =========================
        # ⭐关键：沿用你原来的画图函数
        # =========================
        best_model = models[dfres.iloc[0]["Model"]]

        fig_path = os.path.join(outdir, f"MPs_{mp}_plot.png")

        plt.figure()
        plt.scatter(yte, best_model.predict(Xte))
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print("Saved MPs =", mp)


if __name__ == "__main__":
    main()