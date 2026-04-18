import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time

# ============================================
# CONFIGURATION
# ============================================
# List letters that need improvement (from confusion analysis)
PROBLEMATIC_LETTERS = ['A', 'C', 'M', 'N', 'O', 'Q', 'S']  # Update based on your analysis

# Load original data
print("Loading original dataset...")
df = pd.read_csv("landmarks.csv")

# ============================================
# OPTION 1: CLASS WEIGHTING
# ============================================
print("\n" + "="*60)
print("METHOD 1: CLASS WEIGHTING (Give more importance to confused letters)")
print("="*60)

X = df.drop("label", axis=1)
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Calculate class weights (give more weight to problematic classes)
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Create weight dictionary
class_weight_dict = dict(enumerate(class_weights))

# Train with class weights
from xgboost import XGBClassifier

print("Training with class weights...")
start_time = time.time()

model_weighted = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss',
    scale_pos_weight=1  # XGBoost handles class weights differently
)

# For scikit-learn models (RandomForest, GradientBoosting), use class_weight parameter
# model_weighted.fit(X_train, y_train, sample_weight=class_weights[y_train])

model_weighted.fit(X_train, y_train)

y_pred_weighted = model_weighted.predict(X_test)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)

print(f"Accuracy with class weighting: {accuracy_weighted:.4f}")
print(f"Training time: {time.time() - start_time:.2f}s")

# ============================================
# OPTION 2: OVERSAMPLING PROBLEMATIC CLASSES
# ============================================
print("\n" + "="*60)
print("METHOD 2: OVERSAMPLING (Add more samples of confused letters)")
print("="*60)

# Get original data
df_train = pd.concat([
    pd.DataFrame(X_train, columns=df.columns[:-1]),
    pd.DataFrame({'label': label_encoder.inverse_transform(y_train)})
], axis=1)

# Oversample problematic letters
oversampled_data = []

for letter in PROBLEMATIC_LETTERS:
    letter_data = df_train[df_train['label'] == letter]
    # Duplicate problematic class samples
    oversampled_data.append(letter_data)
    oversampled_data.append(letter_data)  # 2x oversampling

df_augmented = pd.concat([df_train] + oversampled_data, ignore_index=True)

# Shuffle
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

X_aug = df_augmented.drop("label", axis=1)
y_aug = df_augmented["label"]
y_aug_encoded = label_encoder.transform(y_aug)
X_aug_scaled = scaler.transform(X_aug)

print(f"Original training size: {len(X_train)}")
print(f"Augmented training size: {len(X_aug_scaled)}")

print("Training with oversampling...")
start_time = time.time()

model_oversampled = XGBClassifier(
    n_estimators=400,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

model_oversampled.fit(X_aug_scaled, y_aug_encoded)

y_pred_oversample = model_oversampled.predict(X_test)
accuracy_oversample = accuracy_score(y_test, y_pred_oversample)

print(f"Accuracy with oversampling: {accuracy_oversample:.4f}")
print(f"Training time: {time.time() - start_time:.2f}s")

# ============================================
# OPTION 3: HYPERPARAMETER TUNING FOR PROBLEMATIC CLASSES
# ============================================
print("\n" + "="*60)
print("METHOD 3: FOCUSED HYPERPARAMETER TUNING")
print("="*60)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [300, 400, 500],
    'max_depth': [8, 10, 12],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

print("Running grid search (this may take 5-10 minutes)...")
start_time = time.time()

grid_search = GridSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss'),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

model_tuned = grid_search.best_estimator_
y_pred_tuned = model_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print(f"Accuracy with tuned parameters: {accuracy_tuned:.4f}")
print(f"Training time: {time.time() - start_time:.2f}s")

# ============================================
# COMPARE ALL METHODS
# ============================================
print("\n" + "="*60)
print("COMPARISON OF ALL METHODS")
print("="*60)

# Load original model
model_original = joblib.load("model.pkl")
y_pred_original = model_original.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred_original)

results = pd.DataFrame({
    'Method': ['Original', 'Class Weighting', 'Oversampling', 'Hyperparameter Tuning'],
    'Accuracy': [accuracy_original, accuracy_weighted, accuracy_oversample, accuracy_tuned]
})

print(results.to_string(index=False))

# Find best model
best_idx = results['Accuracy'].idxmax()
best_method = results.iloc[best_idx]['Method']
best_accuracy = results.iloc[best_idx]['Accuracy']

print(f"\n🏆 Best Method: {best_method} ({best_accuracy:.4f})")

# Save best model
if best_method == 'Class Weighting':
    best_model = model_weighted
elif best_method == 'Oversampling':
    best_model = model_oversampled
elif best_method == 'Hyperparameter Tuning':
    best_model = model_tuned
else:
    best_model = model_original

# ============================================
# SAVE IMPROVED MODEL
# ============================================
print("\nDo you want to save the improved model? (y/n): ", end='')
response = input().strip().lower()

if response == 'y':
    joblib.dump(best_model, "model_improved.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(label_encoder, "labels.pkl")
    
    print(f"\n✓ Saved improved model as model_improved.pkl")
    print(f"✓ Improvement: {(best_accuracy - accuracy_original) * 100:.2f}% accuracy gain")
    
    # Show per-class improvement for problematic letters
    y_pred_best = best_model.predict(X_test)
    
    print("\nPer-class accuracy for problematic letters:")
    for letter in PROBLEMATIC_LETTERS:
        letter_idx = label_encoder.transform([letter])[0]
        letter_mask = y_test == letter_idx
        
        if letter_mask.sum() > 0:
            original_acc = (y_pred_original[letter_mask] == y_test[letter_mask]).mean() * 100
            improved_acc = (y_pred_best[letter_mask] == y_test[letter_mask]).mean() * 100
            
            print(f"  {letter}: {original_acc:.1f}% → {improved_acc:.1f}% ({improved_acc - original_acc:+.1f}%)")
else:
    print("\nModel not saved. Original model unchanged.")