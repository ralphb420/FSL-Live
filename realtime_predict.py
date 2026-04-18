import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# Load model and preprocessing

MODEL_PATH = 'models/model.pkl'
LABEL_PATH = 'models/labels.pkl"'
SCALER_PATH = 'models/scaler.pkl'

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_PATH)

# Load scaler if it exists (from the improved training script)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    use_scaler = True
    print("✓ Loaded scaler for feature normalization")
else:
    use_scaler = False
    print("⚠ No scaler found - predictions may be less accurate")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1  # 0=lite, 1=full (better accuracy)
)

cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting FSL Recognition...")
print("Press ESC to quit")

# For smoothing predictions
prediction_history = []
history_length = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Improve performance
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    landmarks = []
    predicted_label = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks with better styling
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # Pad to fixed size for 2 hands = 126 values
        while len(landmarks) < 126:
            landmarks.append(0)

        # Apply scaler if available
        if use_scaler:
            landmarks_scaled = scaler.transform([landmarks])
        else:
            landmarks_scaled = [landmarks]

        # Get prediction and confidence
        prediction = model.predict(landmarks_scaled)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(landmarks_scaled)[0]
            confidence = np.max(probabilities) * 100
        
        # Smooth predictions using history
        prediction_history.append(predicted_label)
        if len(prediction_history) > history_length:
            prediction_history.pop(0)
        
        # Use most common prediction in history
        from collections import Counter
        predicted_label = Counter(prediction_history).most_common(1)[0][0]

    # Create info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Display prediction
    if predicted_label:
        cv2.putText(
            frame,
            f"Sign: {predicted_label}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )
        
        if confidence > 0:
            cv2.putText(
                frame,
                f"Confidence: {confidence:.1f}%",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
    else:
        cv2.putText(
            frame,
            "No hand detected",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # Display FPS
    cv2.putText(
        frame,
        "Press ESC to quit",
        (frame.shape[1] - 250, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.imshow("FSL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):  # Clear prediction history
        prediction_history = []

cap.release()
cv2.destroyAllWindows()
hands.close()

print("FSL Recognition stopped")