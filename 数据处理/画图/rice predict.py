import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

plt.ioff()

df = pd.read_excel(r"C:\Users\AMDYE\OneDrive\桌面\Sheet5(1).xlsx", sheet_name='Sheet5')

label_encoder = LabelEncoder()
df['灌溉条件_编码'] = label_encoder.fit_transform(df['灌溉条件'])
print(f"灌溉条件编码映射: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

X = df[['水分处理', 'SOC', 'pH', 'Cd','As','灌溉条件_编码']].copy()
y_cd = df['cd吸附'].copy()
y_as = df['as吸附'].copy()

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train_cd, X_test_cd, y_cd_train, y_cd_test = train_test_split(
   X_imputed, y_cd, test_size=0.2, random_state=42
)

X_train_as, X_test_as, y_as_train, y_as_test = train_test_split(
   X_imputed, y_as, test_size=0.2, random_state=42
)

print(f"Cd split - Training set size: {len(X_train_cd)}, Test set size: {len(X_test_cd)}")
print(f"As split - Training set size: {len(X_train_as)}, Test set size: {len(X_test_as)}")

print("Training AutoTabPFN ensemble for Cd...")
auto_tabpfn_cd = AutoTabPFNRegressor(max_time=160, device='cpu')
auto_tabpfn_cd.fit(X_train_cd, y_cd_train)

print("Training AutoTabPFN ensemble for As...")
auto_tabpfn_as = AutoTabPFNRegressor(max_time=180, device='cpu')
auto_tabpfn_as.fit(X_train_as, y_as_train)

cd_pred_train = auto_tabpfn_cd.predict(X_train_cd)
cd_pred_test = auto_tabpfn_cd.predict(X_test_cd)
as_pred_train = auto_tabpfn_as.predict(X_train_as)
as_pred_test = auto_tabpfn_as.predict(X_test_as)

cd_train_rmse = np.sqrt(mean_squared_error(y_cd_train, cd_pred_train))
cd_train_r2 = r2_score(y_cd_train, cd_pred_train)
cd_test_rmse = np.sqrt(mean_squared_error(y_cd_test, cd_pred_test))
cd_test_r2 = r2_score(y_cd_test, cd_pred_test)
cd_test_mae = mean_absolute_error(y_cd_test, cd_pred_test)

as_train_rmse = np.sqrt(mean_squared_error(y_as_train, as_pred_train))
as_train_r2 = r2_score(y_as_train, as_pred_train)
as_test_rmse = np.sqrt(mean_squared_error(y_as_test, as_pred_test))
as_test_r2 = r2_score(y_as_test, as_pred_test)
as_test_mae = mean_absolute_error(y_as_test, as_pred_test)

print("\nCd ensemble model info:")
if hasattr(auto_tabpfn_cd, 'weights_'):
    for i, weight in enumerate(auto_tabpfn_cd.weights_):
        print(f"  Model {i+1}: Weight = {weight:.4f}")
else:
    print("Weights attribute not available")

print("\nAs ensemble model info:")
if hasattr(auto_tabpfn_as, 'weights_'):
    for i, weight in enumerate(auto_tabpfn_as.weights_):
        print(f"  Model {i+1}: Weight = {weight:.4f}")
else:
    print("Weights attribute not available")

print("\nAutoTabPFN Ensemble Results:")
print("===========================\n")

print("Model Performance for Cadmium (Cd) Absorption")
print("---------------------------------------------")
print(f"Training Set:")
print(f"  R² Score: {cd_train_r2:.3f}")
print(f"  RMSE: {cd_train_rmse:.3f}")
print(f"  \nTest Set:")
print(f"  R² Score: {cd_test_r2:.3f}")
print(f"  RMSE: {cd_test_rmse:.3f}")
print(f"  MAE: {cd_test_mae:.3f}\n\n")

print("Model Performance for Arsenic (As) Absorption")
print("---------------------------------------------")
print(f"Training Set:")
print(f"  R² Score: {as_train_r2:.3f}")
print(f"  RMSE: {as_train_rmse:.3f}")
print(f"  \nTest Set:")
print(f"  R² Score: {as_test_r2:.3f}")
print(f"  RMSE: {as_test_rmse:.3f}")
print(f"  MAE: {as_test_mae:.3f}")

print("\nComputing feature importance using permutation method...")

cd_perm_importance = permutation_importance(auto_tabpfn_cd, X_test_cd, y_cd_test, 
                                          n_repeats=10, random_state=42)
cd_importance = cd_perm_importance.importances_mean

as_perm_importance = permutation_importance(auto_tabpfn_as, X_test_as, y_as_test, 
                                          n_repeats=10, random_state=42)
as_importance = as_perm_importance.importances_mean

cd_importance = cd_importance / np.sum(cd_importance) if np.sum(cd_importance) > 0 else cd_importance
as_importance = as_importance / np.sum(as_importance) if np.sum(as_importance) > 0 else as_importance

feature_names_english = ['Water Treatment', 'SOC', 'pH', 'Cd', 'Irrigation']

plt.figure(figsize=(10, 6))
plt.bar(range(len(cd_importance)), cd_importance)
plt.title('Feature Importance for Cd Absorption')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(range(len(cd_importance)), feature_names_english, rotation=45)

for i, v in enumerate(cd_importance):
   plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('cd_feature_importance_auto_ensemble.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
importance_df = pd.DataFrame({
   'Feature': feature_names_english,
   'Importance': cd_importance
})
importance_df = importance_df.sort_values('Importance', ascending=True)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance Ranking for Cd Absorption')
plt.xlabel('Importance')

for i, v in enumerate(importance_df['Importance']):
   plt.text(v + 0.01, i, f"{v*100:.1f}%", va='center')

plt.tight_layout()
plt.savefig('cd_feature_importance_ranked_auto_ensemble.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(range(len(as_importance)), as_importance)
plt.title('Feature Importance for As Absorption')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(range(len(as_importance)), feature_names_english, rotation=45)

for i, v in enumerate(as_importance):
   plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('as_feature_importance_auto_ensemble.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 8))
importance_df = pd.DataFrame({
   'Feature': feature_names_english,
   'Importance': as_importance
})
importance_df = importance_df.sort_values('Importance', ascending=True)

plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.title('Feature Importance Ranking for As Absorption')
plt.xlabel('Importance')

for i, v in enumerate(importance_df['Importance']):
   plt.text(v + 0.01, i, f"{v*100:.1f}%", va='center')

plt.tight_layout()
plt.savefig('as_feature_importance_ranked_auto_ensemble.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_cd_train, cd_pred_train, color='blue', alpha=0.6, label='Training Set')
plt.scatter(y_cd_test, cd_pred_test, color='red', alpha=0.6, label='Test Set')

all_cd_actual = np.concatenate([y_cd_train, y_cd_test])
all_cd_pred = np.concatenate([cd_pred_train, cd_pred_test])
min_val = min(min(all_cd_actual), min(all_cd_pred))
max_val = max(max(all_cd_actual), max(all_cd_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction Line')

plt.title('Cd Absorption: Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cd_actual_vs_predicted_auto_ensemble.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_as_train, as_pred_train, color='green', alpha=0.6, label='Training Set')
plt.scatter(y_as_test, as_pred_test, color='purple', alpha=0.6, label='Test Set')

all_as_actual = np.concatenate([y_as_train, y_as_test])
all_as_pred = np.concatenate([as_pred_train, as_pred_test])
min_val = min(min(all_as_actual), min(all_as_pred))
max_val = max(max(all_as_actual), max(all_as_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction Line')

plt.title('As Absorption: Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('as_actual_vs_predicted_auto_ensemble.png', dpi=300)
plt.close()