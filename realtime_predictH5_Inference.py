##### Enhanced Inference with Auto Hand Zoom
import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras

# Load model

MODEL_PATH = 'models/fsl_az_static.h5'

model = keras.models.load_model(MODEL_PATH)

# Actions A-Z
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,  # Lower threshold to detect hands farther away
    min_tracking_confidence=0.5
)

def get_hand_bbox(hand_landmarks, image_shape):
    """Get bounding box around hand with padding"""
    h, w = image_shape[:2]
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Get min/max coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding (30% on each side)
    padding_x = (x_max - x_min) * 0.3
    padding_y = (y_max - y_min) * 0.3
    
    x_min = max(0, x_min - padding_x)
    x_max = min(1, x_max + padding_x)
    y_min = max(0, y_min - padding_y)
    y_max = min(1, y_max + padding_y)
    
    # Convert to pixel coordinates
    x_min_px = int(x_min * w)
    x_max_px = int(x_max * w)
    y_min_px = int(y_min * h)
    y_max_px = int(y_max * h)
    
    return x_min_px, y_min_px, x_max_px, y_max_px

def crop_and_resize_hand(frame, bbox, target_size=(400, 400)):
    """Crop hand region and resize to standard size"""
    x_min, y_min, x_max, y_max = bbox
    
    # Crop
    cropped = frame[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return None
    
    # Resize to standard size
    resized = cv2.resize(cropped, target_size)
    
    return resized

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sentence = []
prediction_history = []
HISTORY_LENGTH = 5

print("="*60)
print("FSL A-Z Recognition with Auto Hand Zoom")
print("="*60)
print("Controls:")
print("  ESC - Quit")
print("  C - Clear sentence")
print("  SPACE - Add current letter to sentence")
print("="*60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Process original frame for hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    predicted_letter = None
    confidence = 0
    zoomed_frame = None
    
    if results.multi_hand_landmarks:
        # Use first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Get bounding box
        bbox = get_hand_bbox(hand_landmarks, frame.shape)
        x_min, y_min, x_max, y_max = bbox
        
        # Crop and resize hand region
        hand_crop = crop_and_resize_hand(frame, bbox)
        
        if hand_crop is not None:
            # Process cropped hand region
            crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            crop_results = hands.process(crop_rgb)
            
            if crop_results.multi_hand_landmarks:
                # Draw landmarks on zoomed view
                zoomed_frame = hand_crop.copy()
                for crop_hand in crop_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        zoomed_frame, crop_hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Extract landmarks from cropped image
                landmarks = []
                for crop_hand in crop_results.multi_hand_landmarks:
                    for lm in crop_hand.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                
                while len(landmarks) < 126:
                    landmarks.append(0)
                
                # Predict
                prediction = model.predict(np.array([landmarks[:126]]), verbose=0)[0]
                predicted_idx = np.argmax(prediction)
                predicted_letter = actions[predicted_idx]
                confidence = prediction[predicted_idx] * 100
                
                # Add to history for smoothing
                if confidence > 60:
                    prediction_history.append(predicted_letter)
                    if len(prediction_history) > HISTORY_LENGTH:
                        prediction_history.pop(0)
        
        # Draw bounding box on original frame
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, "Hand Region", (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw landmarks on original frame
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    # Get stable prediction from history
    stable_letter = None
    if len(prediction_history) >= 3:
        from collections import Counter
        most_common = Counter(prediction_history).most_common(1)[0]
        stable_letter = most_common[0]
    
    # Display prediction on main frame
    if predicted_letter:
        cv2.putText(frame, f"{predicted_letter} ({confidence:.1f}%)", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    if stable_letter:
        cv2.putText(frame, f"Stable: {stable_letter}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    # Display sentence
    cv2.rectangle(frame, (0, h-60), (w, h), (50, 50, 50), -1)
    cv2.putText(frame, ''.join(sentence), (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Show main frame
    cv2.imshow('FSL A-Z Recognition', frame)
    
    # Show zoomed hand view (if available)
    if zoomed_frame is not None:
        cv2.imshow('Zoomed Hand', zoomed_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):  # Clear
        sentence = []
        prediction_history = []
    elif key == ord(' '):  # Add letter
        if stable_letter and confidence > 70:
            if not sentence or sentence[-1] != stable_letter:
                sentence.append(stable_letter)
                print(f"Added: {stable_letter}")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\nFinal sentence:", ''.join(sentence))