import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

plt.ioff()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def detect_and_clean_outliers(df, columns, method='zscore', threshold=3.0):
    df_clean = df.copy()
    outliers_count = {}
    
    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' does not exist or is not numeric, skipping outlier detection")
            continue
        
        x = df[col].dropna()
        mask = pd.Series(True, index=df.index)
        
        if method.lower() == 'zscore':
            z_scores = stats.zscore(x)
            mask[x.index] = abs(z_scores) < threshold
            
        elif method.lower() == 'iqr':
            q1 = x.quantile(0.25)
            q3 = x.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            mask[x.index] = (x >= lower_bound) & (x <= upper_bound)
        
        outliers = df.loc[~mask, col]
        outliers_count[col] = len(outliers)
        
        df_clean.loc[~mask, col] = np.nan
    
    total_outliers = sum(outliers_count.values())
    print(f"\nOutlier Cleaning Statistics (Method: {method}, Threshold: {threshold}):")
    print(f"Total outliers removed: {total_outliers}")
    for col, count in outliers_count.items():
        print(f"  {col}: {count} outliers")
    
    return df_clean, outliers_count

print("Reading and processing data...")
df = pd.read_excel(r"C:\Users\AMDYE\OneDrive\桌面\Sheet5(1) (2).xlsx", sheet_name='Sheet5')

print(f"Original data rows: {len(df)}")
print(f"Original data columns: {df.shape[1]}")

print("\nOriginal data statistics:")
print(df.describe())

numeric_cols = ['水分处理', 'SOC', 'pH', 'Cd', 'As', 'as吸附']
df_cleaned, outliers_stats = detect_and_clean_outliers(
    df, 
    columns=numeric_cols, 
    method='iqr',
    threshold=4.5
)

df_cleaned_no_na = df_cleaned.dropna()
print(f"\nCleaned data rows: {len(df_cleaned_no_na)} (removed {len(df) - len(df_cleaned_no_na)} rows)")

plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(len(numeric_cols), 2, i*2-1)
    sns.boxplot(x=df[col])
    plt.title(f'{col} (Before Cleaning)')
    
    plt.subplot(len(numeric_cols), 2, i*2)
    sns.boxplot(x=df_cleaned_no_na[col])
    plt.title(f'{col} (After Cleaning)')

plt.tight_layout()
plt.savefig('outliers_cleaning_comparison.png', dpi=300)
plt.close()

df = df_cleaned_no_na

feature_mapping = {
    '水分处理': 'Water Treatment',
    'SOC': 'SOC',
    'pH': 'pH',
    'Cd': 'Cd',
    'As': 'As',
    '灌溉条件': 'Irrigation Type',
    'as吸附': 'As Absorption'
}

renamed_cols = {col: feature_mapping.get(col, col) for col in df.columns if col in feature_mapping}
df = df.rename(columns=renamed_cols)

label_encoder = LabelEncoder()
df['Irrigation Method'] = label_encoder.fit_transform(df['Irrigation Type'])
irrigation_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"\nIrrigation Type Encoding: {irrigation_mapping}")

X = df[['Water Treatment', 'SOC', 'pH', 'Cd', 'As', 'Irrigation Method']].copy()
y_as = df['As Absorption'].copy()

correlation_matrix = df[['Water Treatment', 'SOC', 'pH', 'Cd', 'As', 'Irrigation Method', 'As Absorption']].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
plt.title('Feature and Target Variable Correlation Heatmap (After Cleaning)')
plt.tight_layout()
plt.savefig('correlation_heatmap_cleaned.png', dpi=300)
plt.close()

numeric_features = X.columns
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

scaler_as = StandardScaler()
y_as_scaled = scaler_as.fit_transform(y_as.values.reshape(-1, 1)).flatten()

random_state = 42
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_as_scaled, test_size=test_size, random_state=random_state, shuffle=True
)

print(f"\nData Split - Training Set Size: {len(X_train)}, Test Set Size: {len(X_test)}")
print(f"Number of Features: {X_train.shape[1]}")
print(f"Feature List: {list(numeric_features)}")

models = {
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=random_state),
        'params': {
            'n_estimators': [100, 300, 500, 600],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05],
            'min_child_weight': [1, 3, 5],
            'subsample': [1],
            'colsample_bytree': [0.7, 0.9]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=random_state, bootstrap=True),
        'params': {
            'n_estimators': [300],
            'max_depth': [3],
            'min_samples_split': [3],
            'min_samples_leaf': [15],
            'max_features': [0.6],
            'max_leaf_nodes': [20],
            'min_impurity_decrease': [0.01, 0.05, 0.1] 
        }
    },
    'GBDT': {
        'model': GradientBoostingRegressor(random_state=random_state),
        'params': {
            'n_estimators': [300],
            'max_depth': [4],
            'learning_rate': [0.01],
            'min_samples_split': [4, 5, 6],
            'min_samples_leaf': [1, 2, 3],
            'subsample': [0.9],
            'max_features': [0.9],
            'max_leaf_nodes': [25, 30, 35],
            'min_weight_fraction_leaf': [0.03]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesRegressor(random_state=random_state),
        'params': {
            'n_estimators': [80],
            'max_depth': [None, 5, 10],
            'min_samples_split': [5],
            'min_samples_leaf': [2]
        }
    },
    'NeuralNetwork': {
        'model': MLPRegressor(random_state=random_state, max_iter=5000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'kernel': ['linear', 'poly', 'rbf'],
            'C': [10],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }
}

best_models = {}
model_results = {}
cv_results = {}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"Training and optimizing {model_name} model...")
    start_time = time.time()
    
    print(f"Performing {n_folds}-fold cross-validation...")
    model = model_info['model']
    cv_scores_r2 = cross_val_score(model, X_imputed, y_as_scaled, 
                                  cv=kf, scoring='r2')
    cv_scores_rmse = np.sqrt(-cross_val_score(model, X_imputed, y_as_scaled, 
                                             cv=kf, scoring='neg_mean_squared_error'))
    
    print(f"Cross-validation R² (normalized scale): {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
    print(f"Cross-validation RMSE (normalized scale): {cv_scores_rmse.mean():.4f} ± {cv_scores_rmse.std():.4f}")
    
    cv_results[model_name] = {
        'r2_mean': cv_scores_r2.mean(),
        'r2_std': cv_scores_r2.std(),
        'rmse_mean': cv_scores_rmse.mean(),
        'rmse_std': cv_scores_rmse.std()
    }
    
    print("Performing grid search for hyperparameter optimization...")
    grid_search = GridSearchCV(
        estimator=model_info['model'],
        param_grid=model_info['params'],
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE (normalized scale): {np.sqrt(-grid_search.best_score_):.4f}")
    
    y_pred_train_scaled = best_model.predict(X_train)
    y_pred_test_scaled = best_model.predict(X_test)
    
    y_pred_train = scaler_as.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).flatten()
    y_pred_test = scaler_as.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).flatten()
    y_train_orig = scaler_as.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler_as.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    train_rmse = np.sqrt(mean_squared_error(y_train_orig, y_pred_train))
    train_r2 = r2_score(y_train_orig, y_pred_train)
    train_mae = mean_absolute_error(y_train_orig, y_pred_train)
    
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test))
    test_r2 = r2_score(y_test_orig, y_pred_test)
    test_mae = mean_absolute_error(y_test_orig, y_pred_test)
    
    train_rmse_scaled = np.sqrt(mean_squared_error(y_train, y_pred_train_scaled))
    train_r2_scaled = r2_score(y_train, y_pred_train_scaled)
    
    test_rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_test_scaled))
    test_r2_scaled = r2_score(y_test, y_pred_test_scaled)
    
    model_results[model_name] = {
        'training_time': time.time() - start_time,
        'best_params': grid_search.best_params_,
        'cv_performance': cv_results[model_name],
        'train': {
            'r2': train_r2,
            'rmse': train_rmse,
            'mae': train_mae,
            'r2_scaled': train_r2_scaled,
            'rmse_scaled': train_rmse_scaled
        },
        'test': {
            'r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'r2_scaled': test_r2_scaled,
            'rmse_scaled': test_rmse_scaled
        },
        'predictions': {
            'train': y_pred_train,
            'test': y_pred_test,
            'train_scaled': y_pred_train_scaled,
            'test_scaled': y_pred_test_scaled
        }
    }
    
    print(f"\n{model_name} model performance (training time: {model_results[model_name]['training_time']:.2f} seconds)")
    print("=" * 50)
    print("Training set (original scale):")
    print(f"  R² score: {train_r2:.3f}")
    print(f"  RMSE: {train_rmse:.3f}")
    print(f"  MAE: {train_mae:.3f}")
    print("Training set (normalized scale):")
    print(f"  R² score: {train_r2_scaled:.3f}")
    print(f"  RMSE: {train_rmse_scaled:.3f}")
    print("\nTest set (original scale):")
    print(f"  R² score: {test_r2:.3f}")
    print(f"  RMSE: {test_rmse:.3f}")
    print(f"  MAE: {test_mae:.3f}")
    print("Test set (normalized scale):")
    print(f"  R² score: {test_r2_scaled:.3f}")
    print(f"  RMSE: {test_rmse_scaled:.3f}")

print("\n\nCross-validation Results Comparison")
print("=" * 50)
print(f"{'Model Name':<15} {'CV R² Mean':<12} {'CV R² Std':<12} {'CV RMSE Mean':<12} {'CV RMSE Std':<12}")
print("-" * 70)
for model_name, result in cv_results.items():
    print(f"{model_name:<15} {result['r2_mean']:<12.3f} {result['r2_std']:<12.3f} {result['rmse_mean']:<12.3f} {result['rmse_std']:<12.3f}")

print("\n\nTest Set Performance Comparison")
print("=" * 50)
print(f"{'Model Name':<15} {'Test R²':<10} {'Test RMSE':<10} {'Test MAE':<10}")
print("-" * 50)
for model_name, result in model_results.items():
    print(f"{model_name:<15} {result['test']['r2']:<10.3f} {result['test']['rmse']:<10.3f} {result['test']['mae']:<10.3f}")

valid_models = {name: results for name, results in model_results.items() 
               if 'test' in results and 'r2' in results['test'] and not np.isnan(results['test']['r2'])}
if valid_models:
    best_model_name = max(valid_models, key=lambda x: valid_models[x]['test']['r2'])
    print(f"\nBased on test set R² score, the best model is: {best_model_name}")
    print(f"Test set R² = {valid_models[best_model_name]['test']['r2']:.3f}")
    print(f"Test set RMSE = {valid_models[best_model_name]['test']['rmse']:.3f}")
    print(f"Test set MAE = {valid_models[best_model_name]['test']['mae']:.3f}")
    print(f"Cross-validation R² = {cv_results[best_model_name]['r2_mean']:.3f} ± {cv_results[best_model_name]['r2_std']:.3f}")
    print(f"Outlier detection method: IQR (threshold = 4.5)")
    print(f"Data split method: Random split (test_size={test_size}, random_state={random_state})")
else:
    print("\nNo valid model found!")
    best_model_name = None

plt.figure(figsize=(12, 8))
model_names = list(cv_results.keys())
r2_means = [cv_results[model]['r2_mean'] for model in model_names]
r2_stds = [cv_results[model]['r2_std'] for model in model_names]

sorted_indices = np.argsort(r2_means)[::-1]
sorted_names = [model_names[i] for i in sorted_indices]
sorted_means = [r2_means[i] for i in sorted_indices]
sorted_stds = [r2_stds[i] for i in sorted_indices]

bars = plt.bar(sorted_names, sorted_means, yerr=sorted_stds, 
              capsize=5, color='skyblue', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Cross-Validation R² Score Comparison (After Data Cleaning)', fontsize=15)
plt.ylabel('R² Score (Mean ± Std Dev)', fontsize=12)
plt.ylim(min(min(r2_means)-max(r2_stds)-0.1, -0.2), 1.0)

for i, (v, std) in enumerate(zip(sorted_means, sorted_stds)):
    plt.text(i, v + std + 0.02, f"{v:.3f}±{std:.3f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('as_model_comparison_cv_r2_cleaned.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
model_names = list(model_results.keys())
test_r2_values = [model_results[model]['test']['r2'] for model in model_names]

sorted_indices = np.argsort(test_r2_values)[::-1]
sorted_names = [model_names[i] for i in sorted_indices]
sorted_values = [test_r2_values[i] for i in sorted_indices]

bars = plt.bar(sorted_names, sorted_values, 
              capsize=5, color='lightgreen', edgecolor='black')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Test Set R² Score Comparison (After Data Cleaning)', fontsize=15)
plt.ylabel('Test Set R² Score', fontsize=12)
plt.ylim(min(min(test_r2_values)-0.1, -0.2), 1.0)

for i, v in enumerate(sorted_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('as_model_comparison_test_r2_cleaned.png', dpi=300)
plt.close()

if best_model_name and best_model_name in best_models:
    best_model = best_models[best_model_name]
    
    feature_names = list(numeric_features)
    
    if best_model_name in ['XGBoost', 'RandomForest', 'GBDT', 'ExtraTrees']:
        importances = best_model.feature_importances_
        importances = importances / np.sum(importances)
        
        plt.figure(figsize=(12, 8))
        
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        plt.bar(range(len(sorted_importances)), sorted_importances, color='lightgreen')
        plt.title(f'As Absorption Feature Importance ({best_model_name} Model, After Cleaning)', fontsize=15)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=0, fontsize=10)
        
        for i, v in enumerate(sorted_importances):
            plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'as_feature_importance_{best_model_name}_cleaned.png', dpi=300)
        plt.close()
    elif best_model_name == 'SVR' and hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_[0]
        abs_coefficients = np.abs(coefficients)
        
        abs_coefficients = abs_coefficients / np.sum(abs_coefficients)
        
        plt.figure(figsize=(12, 8))
        
        sorted_indices = np.argsort(abs_coefficients)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_importances = [abs_coefficients[i] for i in sorted_indices]
        
        plt.bar(range(len(sorted_importances)), sorted_importances, color='lightgreen')
        plt.title(f'As Absorption Feature Coefficient Absolute Values (Linear SVR Model, After Cleaning)', fontsize=15)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Normalized Coefficient Absolute Value', fontsize=12)
        plt.xticks(range(len(sorted_importances)), sorted_features, rotation=45, fontsize=10)
        
        for i, v in enumerate(sorted_importances):
            plt.text(i, v + 0.01, f"{v*100:.1f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('as_feature_importance_SVR_cleaned.png', dpi=300)
        plt.close()
    
    if 'predictions' in model_results[best_model_name]:
        y_train_orig = scaler_as.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_orig = scaler_as.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_train = model_results[best_model_name]['predictions']['train']
        y_pred_test = model_results[best_model_name]['predictions']['test']
        
        plt.figure(figsize=(10, 8))
        
        plt.scatter(y_train_orig, y_pred_train, color='blue', alpha=0.6, label='Training Set')
        plt.scatter(y_test_orig, y_pred_test, color='red', alpha=0.6, label='Test Set')
        
        all_y_actual = np.concatenate([y_train_orig, y_test_orig])
        all_y_pred = np.concatenate([y_pred_train, y_pred_test])
        min_val = min(min(all_y_actual), min(all_y_pred))
        max_val = max(max(all_y_actual), max(all_y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal Prediction Line')
        
        test_r2 = model_results[best_model_name]['test']['r2']
        test_rmse = model_results[best_model_name]['test']['rmse']
        plt.title(f'Best Model ({best_model_name}) As Absorption Prediction Performance\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}\n(After Data Cleaning)', fontsize=14)
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'as_actual_vs_predicted_{best_model_name}_cleaned.png', dpi=300)
        plt.close()

if best_model_name and best_model_name in best_models:
    y_train_orig = scaler_as.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_test_orig = scaler_as.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_train = model_results[best_model_name]['predictions']['train']
    y_pred_test = model_results[best_model_name]['predictions']['test']
    
    train_residuals = y_train_orig - y_pred_train
    test_residuals = y_test_orig - y_pred_test
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.scatter(y_pred_train, train_residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'Training Set Residual Analysis ({best_model_name} Model)', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.scatter(y_pred_test, test_residuals, alpha=0.6, color='red')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title(f'Test Set Residual Analysis ({best_model_name} Model)', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'as_residual_analysis_{best_model_name}_cleaned.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    stats.probplot(train_residuals, dist="norm", plot=plt)
    plt.title(f'Training Set Residual Q-Q Plot ({best_model_name} Model)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    stats.probplot(test_residuals, dist="norm", plot=plt)
    plt.title(f'Test Set Residual Q-Q Plot ({best_model_name} Model)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'as_residual_qq_{best_model_name}_cleaned.png', dpi=300)
    plt.close()

print("\n" + "="*60)
print("Model Training and Evaluation Summary")
print("="*60)

print("\nData Processing Information:")
print(f"- Original Sample Count: {len(df) + (len(df) - len(df_cleaned_no_na))}")
print(f"- Cleaned Sample Count: {len(df)}")
print(f"- Removed Outlier Samples: {len(df) - len(df_cleaned_no_na)}")
print(f"- Outlier Detection Method: IQR (Threshold = 4.5)")
print(f"- Data Split Method: Random Split (test_size={test_size}, random_state={random_state})")
print(f"- Training Set Size: {len(X_train)}, Test Set Size: {len(X_test)}")

print("\nBest Model Information:")
if best_model_name:
    print(f"- Best Model Type: {best_model_name}")
    print(f"- Test Set R²: {valid_models[best_model_name]['test']['r2']:.3f}")
    print(f"- Test Set RMSE: {valid_models[best_model_name]['test']['rmse']:.3f}")
    print(f"- Test Set MAE: {valid_models[best_model_name]['test']['mae']:.3f}")
    print(f"- Cross-Validation R²: {cv_results[best_model_name]['r2_mean']:.3f} ± {cv_results[best_model_name]['r2_std']:.3f}")
    print(f"- Best Hyperparameters: {model_results[best_model_name]['best_params']}")
else:
    print("- No valid model found!")

print("\nAll Model Test Set R² Rankings:")
sorted_models = sorted([(name, result['test']['r2']) for name, result in model_results.items()], 
                      key=lambda x: x[1], reverse=True)
for i, (model_name, r2) in enumerate(sorted_models, 1):
    print(f"{i}. {model_name}: R² = {r2:.3f}")

print("\nAll charts saved. Model training and evaluation completed.")