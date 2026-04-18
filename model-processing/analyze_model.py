import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Load data
df = pd.read_csv("landmarks.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Encode and scale
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Load trained model
model = joblib.load("model.pkl")

# Get predictions
y_pred = model.predict(X_test)

# ============================================
# 1. CONFUSION MATRIX VISUALIZATION
# ============================================
print("Generating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 12))

# Create heatmap
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Number of predictions'}
)

plt.title('Confusion Matrix - FSL Letter Recognition', fontsize=16, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")
plt.show()

# ============================================
# 2. FIND MOST CONFUSED LETTER PAIRS
# ============================================
print("\n" + "="*60)
print("MOST CONFUSED LETTER PAIRS")
print("="*60)

confusion_pairs = []

for i in range(len(cm)):
    for j in range(len(cm)):
        if i != j and cm[i][j] > 0:  # Misclassifications
            confusion_pairs.append({
                'True Letter': label_encoder.classes_[i],
                'Predicted As': label_encoder.classes_[j],
                'Count': cm[i][j],
                'Percentage': (cm[i][j] / cm[i].sum()) * 100
            })

# Sort by count
confusion_df = pd.DataFrame(confusion_pairs)
confusion_df = confusion_df.sort_values('Count', ascending=False)

print("\nTop 10 Most Confused Pairs:")
print(confusion_df.head(10).to_string(index=False))

# Save to CSV
confusion_df.to_csv('confused_pairs.csv', index=False)
print("\n✓ Saved confused_pairs.csv")

# ============================================
# 3. PER-CLASS ACCURACY
# ============================================
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

class_accuracy = []

for i, letter in enumerate(label_encoder.classes_):
    correct = cm[i][i]
    total = cm[i].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    class_accuracy.append({
        'Letter': letter,
        'Correct': correct,
        'Total': total,
        'Accuracy': accuracy
    })

accuracy_df = pd.DataFrame(class_accuracy)
accuracy_df = accuracy_df.sort_values('Accuracy')

print("\nLowest Accuracy Letters (need improvement):")
print(accuracy_df.head(10).to_string(index=False))

print("\nHighest Accuracy Letters:")
print(accuracy_df.tail(5).to_string(index=False))

# Save to CSV
accuracy_df.to_csv('class_accuracy.csv', index=False)
print("\n✓ Saved class_accuracy.csv")

# ============================================
# 4. VISUALIZATION: Per-Class Accuracy
# ============================================
plt.figure(figsize=(14, 6))
colors = ['red' if acc < 80 else 'orange' if acc < 90 else 'green' 
          for acc in accuracy_df['Accuracy']]

plt.bar(accuracy_df['Letter'], accuracy_df['Accuracy'], color=colors)
plt.axhline(y=90, color='r', linestyle='--', label='90% threshold')
plt.title('Per-Class Accuracy - FSL Letters', fontsize=16)
plt.xlabel('Letter', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.ylim(0, 105)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved class_accuracy.png")
plt.show()

# ============================================
# 5. DETAILED CLASSIFICATION REPORT
# ============================================
print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Overall Accuracy: {(y_pred == y_test).mean() * 100:.2f}%")
print(f"Letters with <90% accuracy: {len(accuracy_df[accuracy_df['Accuracy'] < 90])}")
print(f"Most confused pair: {confusion_df.iloc[0]['True Letter']} → {confusion_df.iloc[0]['Predicted As']} ({confusion_df.iloc[0]['Count']} times)")