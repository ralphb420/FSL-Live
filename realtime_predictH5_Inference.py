import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
from collections import deque

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
    max_num_hands=1,  # Change to 2 if both hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ============================================
# Hand Tracking & Locking
# ============================================
class HandTracker:
    def __init__(self):
        self.locked_hand_id = None
        self.hand_history = deque(maxlen=10)
        self.lock_threshold = 5  # Frames to confirm lock
        
    def get_hand_center(self, hand_landmarks):
        """Get center point of hand"""
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        return (np.mean(x_coords), np.mean(y_coords))
    
    def is_same_hand(self, center1, center2, threshold=0.1):
        """Check if two centers belong to same hand"""
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return distance < threshold
    
    def select_hand(self, multi_hand_landmarks, multi_handedness):
        """Select and lock onto a hand"""
        if not multi_hand_landmarks:
            return None, None
        
        # If no hand is locked yet, lock to the first detected hand
        if self.locked_hand_id is None:
            selected_hand = multi_hand_landmarks[0]
            hand_center = self.get_hand_center(selected_hand)
            self.hand_history.append(hand_center)
            
            # Lock after seeing hand for enough frames
            if len(self.hand_history) >= self.lock_threshold:
                self.locked_hand_id = 0
                return selected_hand, multi_handedness[0] if multi_handedness else None
            
            return selected_hand, multi_handedness[0] if multi_handedness else None
        
        # Hand is locked - track it across frames
        current_centers = [self.get_hand_center(hand) for hand in multi_hand_landmarks]
        last_center = self.hand_history[-1] if self.hand_history else None
        
        if last_center:
            # Find hand closest to last known position
            distances = [np.sqrt((c[0] - last_center[0])**2 + (c[1] - last_center[1])**2) 
                        for c in current_centers]
            closest_idx = np.argmin(distances)
            
            # If closest hand is within threshold, it's the same hand
            if distances[closest_idx] < 0.15:
                selected_hand = multi_hand_landmarks[closest_idx]
                hand_center = self.get_hand_center(selected_hand)
                self.hand_history.append(hand_center)
                return selected_hand, multi_handedness[closest_idx] if multi_handedness else None
        
        # Lost track - reset lock
        self.locked_hand_id = None
        self.hand_history.clear()
        return None, None
    
    def reset(self):
        """Reset hand lock"""
        self.locked_hand_id = None
        self.hand_history.clear()

hand_tracker = HandTracker()

# ============================================
# Hand Region Processing
# ============================================
def get_hand_bbox(hand_landmarks, image_shape, padding=0.4):
    """Get bounding box around hand with padding"""
    h, w = image_shape[:2]
    
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    # Get min/max coordinates
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    
    x_min = max(0, x_min - width * padding)
    x_max = min(1, x_max + width * padding)
    y_min = max(0, y_min - height * padding)
    y_max = min(1, y_max + height * padding)
    
    # Convert to pixel coordinates
    x_min_px = int(x_min * w)
    x_max_px = int(x_max * w)
    y_min_px = int(y_min * h)
    y_max_px = int(y_max * h)
    
    return x_min_px, y_min_px, x_max_px, y_max_px

def crop_and_resize_hand(frame, bbox, target_size=640):
    """
    Crop hand region and resize to optimal size
    
    Resolution matters! Higher = better accuracy but slower
    Optimal sizes tested:
    - 224x224: Fast but lower accuracy at distance
    - 320x320: Good balance
    - 480x480: Better accuracy
    - 640x640: Best accuracy (recommended)
    - 1024x1024: Diminishing returns, slower
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Crop
    cropped = frame[y_min:y_max, x_min:x_max]
    
    if cropped.size == 0:
        return None
    
    # Make square (maintain aspect ratio), Landmarks might bug due to aspect ratio, DUE to changes
    h, w = cropped.shape[:2]
    max_dim = max(h, w)
    
    # Create square canvas
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    
    # Center the crop
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped
    
    # Resize to target size
    resized = cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    
    return resized

# ============================================
# Main Loop
# ============================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

sentence = []
prediction_history = deque(maxlen=7)
CONFIDENCE_THRESHOLD = 65
ZOOM_RESOLUTION = 640  # Optimal resolution for zoom (can adjust: 320, 480, 640, 1024)

print("="*60)
print("FSL A-Z Recognition with Hand Locking & Optimized Zoom")
print("="*60)
print(f"Zoom resolution: {ZOOM_RESOLUTION}x{ZOOM_RESOLUTION}")
print("\nControls:")
print("  ESC - Quit")
print("  C - Clear sentence")
print("  R - Reset hand lock")
print("  SPACE - Add current letter")
print("  + - Increase zoom resolution")
print("  - - Decrease zoom resolution")
print("="*60)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Process full frame for hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    
    predicted_letter = None
    confidence = 0
    zoomed_frame = None
    hand_locked = False
    
    if results.multi_hand_landmarks:
        # Select and lock hand
        selected_hand, handedness = hand_tracker.select_hand(
            results.multi_hand_landmarks,
            results.multi_handedness
        )
        
        if selected_hand:
            hand_locked = hand_tracker.locked_hand_id is not None
            
            # Get bounding box
            bbox = get_hand_bbox(selected_hand, frame.shape, padding=0.4)
            x_min, y_min, x_max, y_max = bbox
            
            # Crop and resize hand region
            hand_crop = crop_and_resize_hand(frame, bbox, target_size=ZOOM_RESOLUTION)
            
            if hand_crop is not None:
                # Process cropped hand region
                crop_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                crop_rgb.flags.writeable = False
                crop_results = hands.process(crop_rgb)
                crop_rgb.flags.writeable = True
                
                if crop_results.multi_hand_landmarks:
                    # Create zoomed display
                    zoomed_frame = hand_crop.copy()
                    
                    # Draw landmarks on zoomed view
                    for crop_hand in crop_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            zoomed_frame, crop_hand, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extract landmarks from ZOOMED view
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
                    
                    # Add to history for smoothing (only if confident and hand is locked)
                    if confidence > CONFIDENCE_THRESHOLD and hand_locked:
                        prediction_history.append(predicted_letter)
            
            # Draw bounding box on original frame
            box_color = (0, 255, 0) if hand_locked else (0, 255, 255)
            box_thickness = 3 if hand_locked else 2
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
            
            # Draw lock status
            lock_text = "LOCKED" if hand_locked else "Locking..."
            cv2.putText(frame, lock_text, (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            # Draw landmarks on original frame
            mp_drawing.draw_landmarks(
                frame, selected_hand, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Draw handedness (Left/Right)
            if handedness:
                hand_label = handedness.classification[0].label
                cv2.putText(frame, hand_label, (x_min, y_max + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Get stable prediction from history
    stable_letter = None
    stability = 0
    if len(prediction_history) >= 4:
        from collections import Counter
        most_common = Counter(prediction_history).most_common(1)[0]
        stable_letter = most_common[0]
        stability = (most_common[1] / len(prediction_history)) * 100
    
    # Display info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 180), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    
    # Display current prediction
    if predicted_letter:
        color = (0, 255, 0) if hand_locked else (0, 255, 255)
        cv2.putText(frame, f"Current: {predicted_letter} ({confidence:.1f}%)", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Display stable prediction
    if stable_letter:
        cv2.putText(frame, f"Stable: {stable_letter} ({stability:.0f}%)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    predicted_letter = stable_letter
    
    # Display hand lock status
    lock_status = "Hand Locked" if hand_locked else "Searching for hand..."
    lock_color = (0, 255, 0) if hand_locked else (0, 255, 255)
    cv2.putText(frame, lock_status, (10, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, lock_color, 2)
    
    # Display zoom resolution
    cv2.putText(frame, f"Zoom: {ZOOM_RESOLUTION}x{ZOOM_RESOLUTION}", (10, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Display sentence
    cv2.rectangle(frame, (0, h-60), (w, h), (50, 50, 50), -1)
    sentence_text = ''.join(sentence) if sentence else "Press SPACE to add letters"
    cv2.putText(frame, sentence_text, (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    # Show main frame
    cv2.imshow('FSL A-Z Recognition', frame)
    
    # Show zoomed hand view (if available)
    if zoomed_frame is not None:
        # Add prediction overlay on zoom window
        zoom_overlay = zoomed_frame.copy()
        cv2.rectangle(zoom_overlay, (0, 0), (ZOOM_RESOLUTION, 80), (0, 0, 0), -1)
        zoomed_frame = cv2.addWeighted(zoomed_frame, 0.7, zoom_overlay, 0.3, 0)
        
        if predicted_letter:
            cv2.putText(zoomed_frame, f"{predicted_letter}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow('Zoomed Hand (for prediction)', zoomed_frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c'):  # Clear
        sentence = []
        prediction_history.clear()
        print("✓ Sentence cleared")
    elif key == ord('r'):  # Reset hand lock
        hand_tracker.reset()
        prediction_history.clear()
        print("✓ Hand lock reset")
    elif key == ord(' '):  # Add letter
        if stable_letter and hand_locked and stability > 60:
            if not sentence or sentence[-1] != stable_letter:
                sentence.append(stable_letter)
                print(f"Added: {stable_letter} (Stability: {stability:.0f}%)")
    elif key == ord('+') or key == ord('='):  # Increase resolution
        ZOOM_RESOLUTION = min(1024, ZOOM_RESOLUTION + 160)
        print(f"Zoom resolution: {ZOOM_RESOLUTION}x{ZOOM_RESOLUTION}")
    elif key == ord('-') or key == ord('_'):  # Decrease resolution
        ZOOM_RESOLUTION = max(224, ZOOM_RESOLUTION - 160)
        print(f"Zoom resolution: {ZOOM_RESOLUTION}x{ZOOM_RESOLUTION}")

cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*60)
print("FINAL SENTENCE:")
print("="*60)
print(''.join(sentence))
print("="*60)