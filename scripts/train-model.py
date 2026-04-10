import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# -------------------------------
# PATH SETUP
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "expanded_bmtc_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "delay_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# LOAD DATASET
# -------------------------------
print("🔄 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------
# SORT DATA (CRITICAL)
# -------------------------------
df = df.sort_values(by=["bus_id", "arrival_time"])

# -------------------------------
# ENCODE CATEGORICAL FEATURE
# -------------------------------
demand_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["demand_level_encoded"] = df["demand_level"].map(demand_mapping)

# -------------------------------
# CREATE TEMPORAL FEATURES 🔥
# -------------------------------

# Group by bus_id for temporal features
df = df.sort_values(by=["bus_id", "arrival_time"]).reset_index(drop=True)

# Previous delays (lag features)
df["prev_delay_1"] = df.groupby("bus_id")["total_delay"].shift(1)
df["prev_delay_2"] = df.groupby("bus_id")["total_delay"].shift(2)
df["prev_delay_3"] = df.groupby("bus_id")["total_delay"].shift(3)

# Rolling average of delays (trend)
df["delay_rolling_avg_3"] = df.groupby("bus_id")["total_delay"].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
).shift(1)

# Delay momentum (difference from previous)
df["delay_momentum"] = df.groupby("bus_id")["total_delay"].diff()

# Target: next delay (what we want to predict)
df["next_delay"] = df.groupby("bus_id")["total_delay"].shift(-1)

# Encode categorical variables
if "congestion_level" in df.columns:
    congestion_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df["congestion_encoded"] = df["congestion_level"].map(congestion_mapping)

# -------------------------------
# FEATURE LIST (OPTIMIZED FOR ACCURACY 🔥)
# -------------------------------
features = [
    # Temporal features (MOST IMPORTANT)
    "prev_delay_1",           # 🔥 Immediate previous delay
    "prev_delay_2",           # 🔥 2-step previous delay
    "delay_momentum",         # 🔥 Delay trend direction
    "delay_rolling_avg_3",    # 🔥 Moving average (smoothed trend)
    
    # Current state
    "total_delay",            # Current actual delay
    "traffic_delay",          # Current traffic delay
    "signal_delay",           # Current signal delay
    
    # Passenger & Operational
    "passenger_load",
    "waiting_passengers",
    "dwell_time",
    
    # Route & Schedule
    "distance_to_next_stop",
    "bus_frequency_per_hour",
    "headway",
    "headway_change_rate",
    
    # Environmental
    "demand_level_encoded",
    "congestion_encoded" if "congestion_encoded" in df.columns else None,
    "traffic_factor",
    "num_signals"
]

# Remote None values from feature list
features = [f for f in features if f is not None and f in df.columns]

# -------------------------------
# DROP MISSING VALUES
# -------------------------------
df = df.dropna(subset=features + ["next_delay"])

# -------------------------------
# DEFINE INPUT & OUTPUT
# -------------------------------
X = df[features]
y = df["next_delay"]

print(f"\n📊 Features: {features}")
print("📊 Target: next_delay")
print(f"📊 Dataset size for training: {X.shape[0]} samples")

# -------------------------------
# FEATURE SCALING (IMPORTANT FOR ACCURACY)
# -------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Save scaler for later use
joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\n📈 Training set: {X_train.shape[0]} samples")
print(f"📈 Test set: {X_test.shape[0]} samples")

# ================================
# TRAIN GRADIENT BOOSTING (BETTER ACCURACY)
# ================================
print("\n🚀 Training Gradient Boosting Regressor for Next Delay Prediction...")

model = GradientBoostingRegressor(
    n_estimators=500,                    # More boosting rounds
    learning_rate=0.05,                  # Slower learning for better generalization
    max_depth=7,                         # Moderate depth for complex relationships
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,                       # Stochastic boosting
    random_state=42,
    verbose=0
)

print(f"   Using {len(features)} features with scaling")
print(f"   Training on {X_train.shape[0]} samples")

# Cross-validation for better evaluation
cv_scores = cross_val_score(
    model, X_train, y_train,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
print(f"   Cross-validation R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train model
model.fit(X_train, y_train)

# Save model metadata
metadata = {
    'features': features,
    'model_type': 'GradientBoostingRegressor',
    'n_features': len(features),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}
joblib.dump(metadata, os.path.join(MODEL_DIR, "model_metadata.pkl"))

# -------------------------------
# EVALUATE MODEL
# -------------------------------
print("\n📊 Evaluating Model...")

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Training metrics
train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = (mean_squared_error(y_train, y_pred_train)) ** 0.5

# Test metrics
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = (mean_squared_error(y_test, y_pred_test)) ** 0.5
test_mape = (abs(y_test - y_pred_test) / (abs(y_test) + 1e-6)).mean() * 100

residuals = y_test - y_pred_test
residual_std = residuals.std()

print(f"\n📊 Model Performance Metrics:")
print(f"\n   🏋️  TRAINING SET:")
print(f"      R² Score: {train_r2:.4f}")
print(f"      MAE: {train_mae:.4f} minutes")
print(f"      RMSE: {train_rmse:.4f} minutes")

print(f"\n   🧪 TEST SET (Generalization):")
print(f"      R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"      MAE: {test_mae:.4f} minutes (avg prediction error)")
print(f"      RMSE: {test_rmse:.4f} minutes")
print(f"      MAPE: {test_mape:.2f}% (relative error %)")
print(f"      Residual Std Dev: {residual_std:.4f}")

# Model quality assessment
if test_r2 >= 0.90:
    quality = "✅ EXCELLENT - Outstanding prediction accuracy!"
elif test_r2 >= 0.80:
    quality = "✅ VERY GOOD - High prediction accuracy"
elif test_r2 >= 0.70:
    quality = "✅ GOOD - Reliable predictions"
elif test_r2 >= 0.60:
    quality = "⚠️  FAIR - Model needs improvement"
else:
    quality = "❌ POOR - Consider different features/algorithms"

print(f"\n   ⭐ Model Quality: {quality}")

# Check for overfitting
overfitting_gap = train_r2 - test_r2
if overfitting_gap > 0.1:
    print(f"   ⚠️  Potential overfitting detected (gap: {overfitting_gap:.4f})")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top 12 Most Important Features for Next Delay Prediction:")
print(f"   {'Feature':<32} {'Importance':<12} {'Visual':<30}")
print(f"   {'-'*74}")

cumulative_importance = 0
for idx, (_, row) in enumerate(feature_importance.head(12).iterrows()):
    cumulative_importance += row['importance']
    bar_length = int(row['importance'] * 500)
    bar = "█" * bar_length
    print(f"   {row['feature']:<32} {row['importance']:>10.4f}  {bar}")

print(f"\n   📊 Top 12 features explain {cumulative_importance*100:.1f}% of predictions")

# -------------------------------
# SAVE MODEL & ARTIFACTS
# -------------------------------
print(f"\n💾 Saving model and artifacts...")

# Save the trained model
joblib.dump(model, MODEL_PATH)
print(f"   ✅ Model saved: {MODEL_PATH}")

# Scaler already saved above
print(f"   ✅ Scaler saved: {os.path.join(MODEL_DIR, 'feature_scaler.pkl')}")

# Save metadata
print(f"   ✅ Metadata saved: {os.path.join(MODEL_DIR, 'model_metadata.pkl')}")

# Save feature importance chart
feature_importance.to_csv(
    os.path.join(MODEL_DIR, "feature_importance.csv"),
    index=False
)
print(f"   ✅ Feature importance saved: {os.path.join(MODEL_DIR, 'feature_importance.csv')}")

# -------------------------------
# SAMPLE PREDICTIONS
# -------------------------------
print(f"\n📋 Sample Next Delay Predictions on Test Set:")
print(f"   {'Actual':<12} {'Predicted':<12} {'Error (min)':<15} {'Error %':<12}")
print(f"   {'-'*51}")

sample_indices = np.random.choice(len(y_test), min(15, len(y_test)), replace=False)
for i in sorted(sample_indices):
    actual = y_test.iloc[i]
    pred = y_pred_test[i]
    error = abs(actual - pred)
    error_pct = (error / (abs(actual) + 1e-6)) * 100
    indicator = "✓" if error_pct < 10 else "~" if error_pct < 20 else "✗"
    print(f"   {actual:>11.2f} {pred:>11.2f} {error:>14.2f} {error_pct:>10.1f}% {indicator}")

# ================================
# TRAINING SUMMARY
# ================================
print(f"\n" + "="*74)
print(f"{'TRAINING SUMMARY - NEXT DELAY PREDICTION MODEL':^74}")
print(f"="*74)

print(f"\n📈 Model Configuration:")
print(f"   Algorithm: Gradient Boosting Regressor")
print(f"   Estimators: 500 | Learning Rate: 0.05 | Max Depth: 7")
print(f"   Input Features: {len(features)}")
print(f"   Feature Scaling: Enabled (StandardScaler)")
print(f"   Cross-Validation: 5-fold (R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")

print(f"\n📊 Final Results:")
print(f"   Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"   Test MAE: {test_mae:.4f} minutes")
print(f"   Test RMSE: {test_rmse:.4f} minutes") 
print(f"   Test MAPE: {test_mape:.2f}%")

print(f"\n💾 Saved Files:")
print(f"   Model: {MODEL_PATH}")
print(f"   Scaler: {os.path.join(MODEL_DIR, 'feature_scaler.pkl')}")
print(f"   Metadata: {os.path.join(MODEL_DIR, 'model_metadata.pkl')}")
print(f"   Feature Importance: {os.path.join(MODEL_DIR, 'feature_importance.csv')}")

print(f"\n✨ Training Complete! Model ready for predictions.")
print(f"="*74)