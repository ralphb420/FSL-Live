##### Live Testing for FSL Gesture Recognition (Dynamic Signs)
##### Uses TensorFlow LSTM Model

import os
import numpy as np
import mediapipe as mp
import cv2
import tensorflow as tf
from tensorflow import keras
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Conv1D


# ============================================
# 1. MediaPipe Setup
# ============================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Process image with MediaPipe"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    """Draw landmarks with custom styling"""
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# ============================================
# 2. Define Actions and Sequence Length
# ============================================
actions = np.array(['ako',  'bakit', 'F', 'hi', 'hindi', 'ikaw',  'kamusta', 'L', 'maganda', 'magandang umaga', 'N', 'O', 'oo', 'P', 'salamat'])
sequence_length = 30

print(f"Actions to detect: {actions}")
print(f"Sequence length: {sequence_length} frames")

# ============================================
# 3. Load the Model
# ============================================

MODEL_PATH = 'models/NF1.h5'

print(f"\nLoading model from: {MODEL_PATH}")

try:
    model = keras.Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(30,258)))
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    model.load_weights(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nPlease update MODEL_PATH to the correct location of your .h5 file")
    exit()

# ============================================
# 4. Probability Visualization Function
# ============================================
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def prob_viz(res, actions, input_frame, colors):
    """Visualize prediction probabilities"""
    output_frame = input_frame.copy()
    
    # Get top 3 predictions
    top_3_idx = np.argsort(res)[-3:][::-1]
    
    for num, idx in enumerate(top_3_idx):
        prob = res[idx]
        action = actions[idx]
        
        # Draw probability bar
        cv2.rectangle(
            output_frame, 
            (0, 60 + num * 40), 
            (int(prob * 300), 90 + num * 40), 
            colors[num], 
            -1
        )
        
        # Draw text
        cv2.putText(
            output_frame, 
            f"{action}: {prob*100:.1f}%", 
            (10, 85 + num * 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
    
    return output_frame

# ============================================
# 5. Live Testing
# ============================================
# Detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5  # Confidence threshold

print("\n" + "="*60)
print("FSL GESTURE RECOGNITION - LIVE")
print("="*60)
print("Instructions:")
print("  - Perform gestures slowly and clearly")
print("  - Each gesture needs 30 frames (about 1 second)")
print("  - Press 'q' to quit")
print("  - Press 'c' to clear sentence")
print("="*60 + "\n")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Keep only last 30 frames
        
        if len(sequence) == 30:
            # Make prediction
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            print(f"Prediction: {predicted_action} (Confidence: {confidence*100:.1f}%)")
            predictions.append(np.argmax(res))
            
            # Visualization logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if confidence > threshold:
                    if len(sentence) > 0:
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)

            # Keep last 5 predictions in sentence
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities
            image = prob_viz(res, actions, image, colors)
        
        # Draw sentence display
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image, 
            ' '.join(sentence), 
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # Draw frame counter
        cv2.putText(
            image,
            f"Frames: {len(sequence)}/30",
            (image.shape[1] - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Show to screen
        cv2.imshow('FSL Gesture Recognition', image)

        # Keyboard controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence = []
            predictions = []
            sequence = []
            print("Sentence cleared")

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("FINAL SENTENCE:")
print("="*60)
print(' '.join(sentence))
print("="*60)