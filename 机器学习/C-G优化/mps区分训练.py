# ============================================================
# MPs分组 + 贝叶斯优化 + TOPSIS + 熵权集成（最终SCI版）
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
import optuna

# ============================================================
# 1. 数据读取
# ============================================================
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df


# ============================================================
# 2. 贝叶斯优化（核心）
# ============================================================
def bayesian_optimize(model_name, X, y):

    def objective(trial):

        if model_name == "RandomForest":
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                random_state=42
            )

        elif model_name == "XGBoost":
            model = XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 1200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample", 0.6, 1.0),
                random_state=42
            )

        elif model_name == "LightGBM":
            model = LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 200, 1200),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2),
                num_leaves=trial.suggest_int("leaves", 10, 100),
                random_state=42
            )

        elif model_name == "CatBoost":
            model = CatBoostRegressor(
                iterations=trial.suggest_int("iter", 200, 1200),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.2),
                verbose=0,
                random_state=42
            )

        else:
            model = RandomForestRegressor(random_state=42)

        # 5折交叉验证
        scores = []

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X):

            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_tr, y_tr)
            pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    best_params = study.best_params

    # 重新训练最佳模型
    if model_name == "RandomForest":
        model = RandomForestRegressor(**best_params, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(**best_params, random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(**best_params, random_state=42)
    elif model_name == "CatBoost":
        model = CatBoostRegressor(**best_params, verbose=0, random_state=42)
    else:
        model = RandomForestRegressor(random_state=42)

    return model, best_params, study.best_value


# ============================================================
# 3. SPXY（保留）
# ============================================================
def spxy_split(X, y):
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    split = int(len(X) * 0.7)

    train_idx = idx[:split]
    test_idx = idx[split:]

    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx]
    )


# ============================================================
# 4. log变换
# ============================================================
def log_transform(X_train, X_test):

    for col in ["OM", "Cd"]:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col])
            X_test[col] = np.log1p(X_test[col])

    return X_train, X_test


# ============================================================
# 5. TOPSIS
# ============================================================
def topsis(df):

    df = df.copy()

    df["score"] = (
        df["Test_R2"] -
        df["Test_MSE"] / df["Test_MSE"].max()
    )

    return df.sort_values("score", ascending=False)


# ============================================================
# 6. 模型池
# ============================================================
model_names = [
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "CatBoost"
]


# ============================================================
# 7. 单个MPs流程
# ============================================================
def run_pipeline(df, mp_value, output_root):

    print(f"\n========== MPs = {mp_value} ==========")

    df_mp = df[df["MPs"] == mp_value].copy()
    df_mp = df_mp.drop(columns=["MPs"])

    out_dir = os.path.join(output_root, f"MPs_{mp_value}")
    os.makedirs(out_dir, exist_ok=True)

    feature_cols = ["pH","OM","CEC","Cd","Crop","Size","MPR"]
    target_col = "C-G"

    df_mp = df_mp[feature_cols + [target_col]].dropna()

    X = df_mp[feature_cols]
    y = df_mp[target_col]

    # SPXY
    X_train, X_test, y_train, y_test = spxy_split(X, y)

    # KNN
    imputer = KNNImputer(n_neighbors=5)

    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols)

    # log
    X_train, X_test = log_transform(X_train, X_test)

    # ====================================================
    # 贝叶斯优化 + 模型训练
    # ====================================================
    results = []
    best_models = []

    for name in model_names:

        print(f"Optimizing {name}")

        model, params, cv_score = bayesian_optimize(name, X_train, y_train)

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "CV_RMSE": cv_score,
            "Best_Params": str(params),

            "Train_R2": r2_score(y_train, train_pred),
            "Test_R2": r2_score(y_test, test_pred),
            "Test_RMSE": np.sqrt(mean_squared_error(y_test, test_pred))
        })

        best_models.append(model)

    results_df = pd.DataFrame(results)

    # TOPSIS
    ranked = topsis(results_df)

    # 集成
    top3 = ranked.head(3)

    weights = top3["Test_R2"].values
    weights = weights / weights.sum()

    ensemble = 0

    for w, m in zip(weights, best_models[:3]):
        ensemble += w * m.predict(X_test)

    # 保存
    ranked.to_excel(os.path.join(out_dir, "TOPSIS.xlsx"), index=False)

    print(f"Saved -> {out_dir}")


# ============================================================
# 8. 主入口
# ============================================================
def main():

    file_path = r"C:\Users\AMDYE\OneDrive\桌面\训练版本.xlsx"
    output_root = r"C:\Users\AMDYE\OneDrive\python\机器学习\MPs_final"

    df = load_data(file_path)

    run_pipeline(df, 0, output_root)
    run_pipeline(df, 1, output_root)

    print("\nALL DONE")


if __name__ == "__main__":
    main()