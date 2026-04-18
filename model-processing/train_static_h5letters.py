##### Training Script for Static FSL Letters (A-Z) using TensorFlow
##### Uses JPEG images with TensorFlow/Keras CNN

import os
import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tqdm import tqdm
import time

# ============================================
# 1. Configuration
# ============================================
# Path for dataset
DATASET_PATH = "dataset"  # Folder containing A-Z subfolders with JPEGs

# Actions (A-Z letters)
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

print("="*60)
print("FSL STATIC LETTERS (A-Z) TRAINING - TensorFlow")
print("="*60)
print(f"Actions to train: {actions}")
print(f"Number of classes: {len(actions)}")

# ============================================
# 2. MediaPipe Setup for Landmark Extraction
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    model_complexity=1
)

def extract_landmarks(image_path):
    """Extract hand landmarks from a single image"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmarks = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    
    # Pad to fixed size for 2 hands = 126 values
    while len(landmarks) < 126:
        landmarks.append(0)
    
    return np.array(landmarks[:126])

# ============================================
# 3. Setup Folders and Load Dataset
# ============================================
print("\n" + "="*60)
print("LOADING DATASET")
print("="*60)

# Create label mapping
label_map = {label: num for num, label in enumerate(actions)}
print(f"\nLabel mapping:")
print(label_map)

# Collect all image paths
print("\nCounting images per class:")
total_images = 0
for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    if os.path.exists(action_path):
        num_images = len([f for f in os.listdir(action_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f"  {action}: {num_images} images")
        total_images += num_images
    else:
        print(f"  ⚠ {action}: folder not found!")

print(f"\nTotal images: {total_images}")

# ============================================
# 4. Preprocess Data and Create Labels and Features
# ============================================
print("\n" + "="*60)
print("EXTRACTING LANDMARKS")
print("="*60)

sequences = []
labels = []

start_time = time.time()

for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    
    if not os.path.exists(action_path):
        print(f"⚠ Skipping {action} - folder not found")
        continue
    
    image_files = [f for f in os.listdir(action_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {action} ({len(image_files)} images)...")
    
    for image_name in tqdm(image_files, desc=f"  {action}"):
        image_path = os.path.join(action_path, image_name)
        
        # Extract landmarks
        landmarks = extract_landmarks(image_path)
        
        if landmarks is not None:
            sequences.append(landmarks)
            labels.append(label_map[action])

elapsed = time.time() - start_time

print(f"\n✓ Extracted landmarks from {len(sequences)}/{total_images} images")
print(f"✓ Time taken: {elapsed/60:.2f} minutes")
print(f"✓ Speed: {len(sequences)/elapsed:.2f} images/second")

# Convert to numpy arrays
X = np.array(sequences)
print(f'\nX shape (features): {X.shape}')

y = to_categorical(labels, num_classes=len(actions)).astype(int)
print(f'y shape (labels): {y.shape}')

# ============================================
# 5. Split Data
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=labels
)

print(f'\nTraining samples: {len(X_train)}')
print(f'Test samples: {len(X_test)}')
print(f'y_test shape: {y_test.shape}')

# ============================================
# 6. Build and Train Neural Network
# ============================================
print("\n" + "="*60)
print("BUILDING MODEL")
print("="*60)

# Setup TensorBoard
log_dir = os.path.join('Logs')
os.makedirs(log_dir, exist_ok=True)
tb_callback = TensorBoard(log_dir=log_dir)

# Early stopping to prevent overfitting
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint to save best model
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_categorical_accuracy',
    save_best_only=True,
    verbose=1
)

# Model design - Deep Neural Network for static hand poses
model = Sequential()
model.add(Input(shape=(126,)))  # 126 features (hand landmarks)
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(actions.shape[0], activation='softmax'))  # 26 classes (A-Z)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# ============================================
# 7. Train Model
# ============================================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=500,  # Will stop early if no improvement
    batch_size=32,
    callbacks=[tb_callback, early_stop, checkpoint],
    verbose=1
)

# ============================================
# 8. Save Model
# ============================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

modelName = 'fsl_az_static.h5'
model.save(modelName)
print(f"✓ Saved final model: {modelName}")
print(f"✓ Saved best model: best_model.h5")

# ============================================
# 9. Evaluate Model using Confusion Matrix and Accuracy
# ============================================
print("\n" + "="*60)
print("EVALUATING MODEL")
print("="*60)

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, confusion_matrix

# Predictions
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat_classes = np.argmax(yhat, axis=1).tolist()

# Accuracy
accuracy = accuracy_score(ytrue, yhat_classes)
print(f"\n🎯 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\nClassification Report:")
print(classification_report(ytrue, yhat_classes, target_names=actions))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(ytrue, yhat_classes)
print(cm)

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, letter in enumerate(actions):
    if cm[i].sum() > 0:
        acc = (cm[i][i] / cm[i].sum()) * 100
        status = "✓" if acc >= 90 else "⚠" if acc >= 80 else "❌"
        print(f"  {status} {letter}: {acc:.2f}%")

# Multilabel confusion matrix
print("\nMultilabel Confusion Matrix:")
print(multilabel_confusion_matrix(ytrue, yhat_classes))

# ============================================
# 10. Save Model Info
# ============================================
with open('model_info.txt', 'w') as f:
    f.write("FSL A-Z Static Letters Model\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: Deep Neural Network (DNN)\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Classes: {len(actions)}\n")
    f.write(f"Actions: {', '.join(actions)}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Input shape: (126,) - hand landmarks\n")

print("\n✓ Saved model info: model_info.txt")

# ============================================
# 11. Optional: Convert to TensorFlow.js
# ============================================
print("\n" + "="*60)
print("TENSORFLOWJS CONVERSION")
print("="*60)

convert_to_tfjs = input("\nConvert model to TensorFlow.js? (y/n): ").strip().lower()

if convert_to_tfjs == 'y':
    try:
        import tensorflowjs as tfjs
        
        output_dir = 'tfjs_model'
        os.makedirs(output_dir, exist_ok=True)
        
        tfjs.converters.save_keras_model(model, output_dir)
        print(f"✓ Converted model to TensorFlow.js: {output_dir}/")
        
    except ImportError:
        print("❌ tensorflowjs not installed. Install with: pip install tensorflowjs")
    except Exception as e:
        print(f"❌ Error converting: {e}")

# ============================================
# 12. Training Complete
# ============================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nSaved files:")
print(f"  - {modelName} (final model)")
print(f"  - best_model.h5 (best weights)")
print(f"  - model_info.txt (model information)")
print(f"  - Logs/ (TensorBoard logs)")
print(f"\nTo view training progress:")
print(f"  tensorboard --logdir=Logs")
print("="*60)

# Close MediaPipe
hands.close()