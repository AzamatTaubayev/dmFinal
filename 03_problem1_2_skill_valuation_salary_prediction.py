import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("="*70)
print("  PROBLEM 1: SKILL VALUATION")
print("  PROBLEM 2: MISSING SALARY PREDICTION")
print("="*70)

# Load processed data
df = pd.read_csv('data_processed.csv')
print(f"\n📊 Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# PART 1: FEATURE PREPARATION

print("\n" + "="*70)
print("PART 1: FEATURE PREPARATION")
print("="*70)

# Get skill columns
skill_cols = [col for col in df.columns if col.startswith('skill_')]
print(f"\n✅ Found {len(skill_cols)} skill features")

# Prepare feature matrix X and target y
feature_cols = skill_cols + ['experience_level', 'area_normalized']

# Create dummy variables for area
df_encoded = pd.get_dummies(df[feature_cols], columns=['area_normalized'], drop_first=True)

# Split data into WITH salary and WITHOUT salary
df_with_salary = df[df['salary_monthly_kzt'].notna()].copy()
df_without_salary = df[df['salary_monthly_kzt'].isna()].copy()

print(f"\n📊 Data Split:")
print(f"   Jobs WITH salary: {len(df_with_salary)} ({len(df_with_salary)/len(df)*100:.1f}%)")
print(f"   Jobs WITHOUT salary: {len(df_without_salary)} ({len(df_without_salary)/len(df)*100:.1f}%)")

# Prepare X and y for jobs WITH salary
X_known = pd.get_dummies(df_with_salary[feature_cols], 
                         columns=['area_normalized'], drop_first=True)
y_known = df_with_salary['salary_monthly_kzt']

# Prepare X for jobs WITHOUT salary (for later prediction)
X_unknown = pd.get_dummies(df_without_salary[feature_cols], 
                          columns=['area_normalized'], drop_first=True)

# Ensuring both have same columns
missing_cols = set(X_known.columns) - set(X_unknown.columns)
for col in missing_cols:
    X_unknown[col] = 0
X_unknown = X_unknown[X_known.columns]

print(f"\n✅ Feature matrix prepared:")
print(f"   Features (X): {X_known.shape[1]} columns")
print(f"   Target (y): {len(y_known)} samples")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_known, y_known, test_size=0.2, random_state=42
)

print(f"\n✅ Train/Test split (80/20):")
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")


# PART 2: MODEL TRAINING & COMPARISON

print("\n" + "="*70)
print("PART 2: MODEL TRAINING & EVALUATION")
print("="*70)

models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_known, y_known, 
                                cv=5, scoring='r2', n_jobs=-1)
    
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred_test': y_pred_test
    }
    
    print(f"   ✅ {name} trained")
    print(f"      Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
    print(f"      Train RMSE: {train_rmse:,.0f} | Test RMSE: {test_rmse:,.0f}")
    print(f"      CV R² (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Print comparison table
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test R²': [r['test_r2'] for r in results.values()],
    'Test RMSE': [r['test_rmse'] for r in results.values()],
    'Test MAE': [r['test_mae'] for r in results.values()],
    'CV R² Mean': [r['cv_mean'] for r in results.values()],
    'CV R² Std': [r['cv_std'] for r in results.values()],
    'Overfit Gap': [r['train_r2'] - r['test_r2'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('Test R²', ascending=False)
print(comparison_df.to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {results[best_model_name]['test_r2']:.3f}")
print(f"   Test RMSE: {results[best_model_name]['test_rmse']:,.0f} KZT")


# PART 3: PROBLEM 1 - SKILL VALUATION


print("\n" + "="*70)
print("PROBLEM 1: SKILL PREMIUM QUANTIFICATION")
print("="*70)

# Use Ridge regression for interpretability
ridge_model = results['Ridge Regression']['model']

# Get feature coefficients
feature_importance = pd.DataFrame({
    'Feature': X_known.columns,
    'Coefficient': ridge_model.coef_
})

# Extract skill features only
skill_importance = feature_importance[feature_importance['Feature'].str.startswith('skill_')].copy()
skill_importance['Skill'] = skill_importance['Feature'].str.replace('skill_', '').str.replace('_', ' ').str.title()
skill_importance = skill_importance[['Skill', 'Coefficient']].sort_values('Coefficient', ascending=False)

print("\n💎 TOP 20 SKILLS BY SALARY PREMIUM (Ridge Regression):")
print("-" * 70)
print(f"{'Rank':>4s} {'Skill':25s} {'Monthly Premium (KZT)':>25s}")
print("-" * 70)

for idx, (_, row) in enumerate(skill_importance.head(20).iterrows(), 1):
    print(f"{idx:4d}. {row['Skill']:25s} {row['Coefficient']:>+24,.0f}")

print("\n💸 BOTTOM 10 SKILLS (Negative or Low Premium):")
print("-" * 70)
for idx, (_, row) in enumerate(skill_importance.tail(10).iterrows(), 1):
    print(f"{idx:4d}. {row['Skill']:25s} {row['Coefficient']:>+24,.0f}")

# Visualization: Skill Premium
plt.figure(figsize=(12, 8))
top_20 = skill_importance.head(20)
colors = ['green' if x > 0 else 'red' for x in top_20['Coefficient']]
plt.barh(range(len(top_20)), top_20['Coefficient'], color=colors, edgecolor='black')
plt.yticks(range(len(top_20)), top_20['Skill'])
plt.xlabel('Salary Premium (KZT/month)')
plt.title('Top 20 Skills by Salary Premium\n(Controlling for Experience & Location)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('07_problem1_skill_premiums.png', dpi=300, bbox_inches='tight')
print("\n   💾 Visualization saved: 07_problem1_skill_premiums.png")
plt.close()

# Also show Random Forest feature importance
print("\n🌲 RANDOM FOREST FEATURE IMPORTANCE (Top 20):")
print("-" * 70)

rf_model = results['Random Forest']['model']
rf_importance = pd.DataFrame({
    'Feature': X_known.columns,
    'Importance': rf_model.feature_importances_
})

rf_skill_importance = rf_importance[rf_importance['Feature'].str.startswith('skill_')].copy()
rf_skill_importance['Skill'] = rf_skill_importance['Feature'].str.replace('skill_', '').str.replace('_', ' ').str.title()
rf_skill_importance = rf_skill_importance[['Skill', 'Importance']].sort_values('Importance', ascending=False)

for idx, (_, row) in enumerate(rf_skill_importance.head(20).iterrows(), 1):
    print(f"{idx:4d}. {row['Skill']:25s} {row['Importance']:>12.4f}")

# Save skill premiums to CSV
skill_importance.to_csv('skill_premiums.csv', index=False)
print("\n   💾 Skill premiums saved to: skill_premiums.csv")


# PART 4: PROBLEM 2 - MISSING SALARY PREDICTION

print("\n" + "="*70)
print("PROBLEM 2: MISSING SALARY PREDICTION")
print("="*70)

# Use ensemble of best models for prediction
print("\n🔮 Creating ensemble predictor...")
print(f"   Using: Random Forest + Gradient Boosting")

# Get predictions from both models
rf_predictions = results['Random Forest']['model'].predict(X_unknown)
gb_predictions = results['Gradient Boosting']['model'].predict(X_unknown)

# Ensemble: Weighted average (RF gets 60%, GB gets 40%)
ensemble_predictions = 0.6 * rf_predictions + 0.4 * gb_predictions

# Calculate prediction intervals (approximate using model std)
# Use training set residuals to estimate prediction uncertainty
rf_residuals = y_train - results['Random Forest']['model'].predict(X_train)
prediction_std = np.std(rf_residuals)

# 95% confidence interval
lower_bound = ensemble_predictions - 1.96 * prediction_std
upper_bound = ensemble_predictions + 1.96 * prediction_std

# Add predictions to dataframe
df_without_salary['salary_predicted'] = ensemble_predictions
df_without_salary['salary_lower_95'] = lower_bound
df_without_salary['salary_upper_95'] = upper_bound

print(f"\n✅ Predicted salaries for {len(df_without_salary)} jobs")
print(f"\n📊 PREDICTION STATISTICS:")
print(f"   Mean predicted salary: {ensemble_predictions.mean():,.0f} KZT")
print(f"   Median predicted salary: {np.median(ensemble_predictions):,.0f} KZT")
print(f"   Min predicted salary: {ensemble_predictions.min():,.0f} KZT")
print(f"   Max predicted salary: {ensemble_predictions.max():,.0f} KZT")
print(f"   Prediction std dev: {prediction_std:,.0f} KZT")

# Show examples
print("\n📋 SAMPLE PREDICTIONS (First 10 jobs):")
print("-" * 70)
print(f"{'Job Title':40s} {'Predicted Salary':>20s} {'95% CI':>25s}")
print("-" * 70)

for idx in range(min(10, len(df_without_salary))):
    row = df_without_salary.iloc[idx]
    title = str(row['name'])[:37] + '...' if len(str(row['name'])) > 40 else str(row['name'])
    pred = row['salary_predicted']
    ci_lower = row['salary_lower_95']
    ci_upper = row['salary_upper_95']
    print(f"{title:40s} {pred:>19,.0f} ({ci_lower:>10,.0f} - {ci_upper:>10,.0f})")

# Save predictions
df_without_salary[['vacancy_id', 'name', 'salary_predicted', 
                  'salary_lower_95', 'salary_upper_95']].to_csv(
    'salary_predictions.csv', index=False
)
print("\n   💾 Predictions saved to: salary_predictions.csv")

# Create complete dataset
df_complete = df.copy()
df_complete.loc[df_complete['salary_monthly_kzt'].isna(), 'salary_monthly_kzt'] = ensemble_predictions
df_complete.to_csv('data_with_imputed_salaries.csv', index=False)
print("   💾 Complete dataset saved to: data_with_imputed_salaries.csv")

# PART 5: MODEL VALIDATION & DIAGNOSTICS

print("\n" + "="*70)
print("PART 5: MODEL VALIDATION & DIAGNOSTICS")
print("="*70)

# Residual analysis for best model
y_pred_best = results[best_model_name]['y_pred_test']
residuals = y_test - y_pred_best

# Statistics
print(f"\n📊 RESIDUAL ANALYSIS:")
print(f"   Mean residual: {residuals.mean():,.0f} KZT")
print(f"   Std residual: {residuals.std():,.0f} KZT")
print(f"   Max over-prediction: {residuals.max():,.0f} KZT")
print(f"   Max under-prediction: {residuals.min():,.0f} KZT")

# Visualization: Residual plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle(f'Model Diagnostics: {best_model_name}', fontsize=16, fontweight='bold')

# 1. Predicted vs Actual
axes[0, 0].scatter(y_test, y_pred_best, alpha=0.5, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Salary (KZT)')
axes[0, 0].set_ylabel('Predicted Salary (KZT)')
axes[0, 0].set_title('Predicted vs Actual Salary')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residual plot
axes[0, 1].scatter(y_pred_best, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Salary (KZT)')
axes[0, 1].set_ylabel('Residuals (KZT)')
axes[0, 1].set_title('Residual Plot')
axes[0, 1].grid(True, alpha=0.3)

# 3. Residual distribution
axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Residuals (KZT)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4. Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality of Residuals)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_problem2_model_diagnostics.png', dpi=300, bbox_inches='tight')
print("\n   💾 Visualization saved: 08_problem2_model_diagnostics.png")
plt.close()

# Model comparison visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# R² scores
models_list = list(results.keys())
r2_scores = [results[m]['test_r2'] for m in models_list]
axes[0].bar(range(len(models_list)), r2_scores, color='skyblue', edgecolor='black')
axes[0].set_xticks(range(len(models_list)))
axes[0].set_xticklabels(models_list, rotation=45, ha='right')
axes[0].set_ylabel('R² Score')
axes[0].set_title('Model Performance Comparison (Test R²)')
axes[0].axhline(y=0.7, color='r', linestyle='--', label='Target: R² > 0.7')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# RMSE
rmse_scores = [results[m]['test_rmse'] for m in models_list]
axes[1].bar(range(len(models_list)), rmse_scores, color='lightcoral', edgecolor='black')
axes[1].set_xticks(range(len(models_list)))
axes[1].set_xticklabels(models_list, rotation=45, ha='right')
axes[1].set_ylabel('RMSE (KZT)')
axes[1].set_title('Model Performance Comparison (Test RMSE)')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('09_problem2_model_comparison.png', dpi=300, bbox_inches='tight')
print("   💾 Visualization saved: 09_problem2_model_comparison.png")
plt.close()

print("\n" + "="*70)
print("  PROBLEMS 1 & 2 COMPLETED SUCCESSFULLY!")
print("  ✅ Skill premiums quantified")
print("  ✅ Missing salaries predicted")
print("  ✅ 4 visualizations saved")
print("  ✅ 3 output files created")
print("="*70)
