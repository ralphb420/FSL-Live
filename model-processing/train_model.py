import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time

print("Loading data...")
df = pd.read_csv("landmarks.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features (important for many ML models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded  # Ensure balanced splits
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Classes: {label_encoder.classes_}\n")

# ============================================
# Model 1: XGBoost (Best for tabular data)
# ============================================
try:
    from xgboost import XGBClassifier
    
    print("Training XGBoost (recommended)...")
    start_time = time.time()
    
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    print(f"Training time: {time.time() - start_time:.2f}s\n")
    
    best_model = xgb_model
    best_model_name = "XGBoost"
    best_accuracy = xgb_accuracy

except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost\n")
    best_model = None
    best_accuracy = 0


# ============================================
# Model 2: Random Forest (optimized)
# ============================================
print("Training Random Forest...")
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"Training time: {time.time() - start_time:.2f}s\n")

if rf_accuracy > best_accuracy:
    best_model = rf_model
    best_model_name = "RandomForest"
    best_accuracy = rf_accuracy

# ============================================
# Select best model and show detailed metrics
# ============================================
print("="*60)
print(f"BEST MODEL: {best_model_name} (Accuracy: {best_accuracy:.4f})")
print("="*60)

# Get predictions from best model
if best_model_name == "XGBoost":
    y_pred = xgb_pred
else:
    y_pred = rf_pred

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=label_encoder.classes_
))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Cross-validation score
print("\n5-Fold Cross-Validation Score:")
cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5)
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================
# Save the best model
# ============================================
print("\nSaving models...")
joblib.dump(best_model, "model.pkl")
joblib.dump(label_encoder, "labels.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save model info
with open("model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_accuracy:.4f}\n")
    f.write(f"Classes: {', '.join(label_encoder.classes_)}\n")

print("✓ Model training complete!")
print(f"✓ Saved: model.pkl (best model: {best_model_name})")
print("✓ Saved: labels.pkl (label encoder)")
print("✓ Saved: scaler.pkl (feature scaler)")
print("✓ Saved: model_info.txt (model information)")