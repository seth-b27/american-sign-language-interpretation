
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque

print("ASL REAL-TIME RECOGNITION - STAGE 2")
print("\nInitializing...")

# Model configuration
MODEL_PATH = '../STAGE1/asl_final_cnn_model.h5'
IMG_SIZE = 64
CLASS_NAMES = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# Webcam configuration
CAMERA_INDEX = 0  # 0 = default webcam, 1 = external webcam
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Prediction configuration
CONFIDENCE_THRESHOLD = 0.70  # Only show predictions above 70% confidence
STABILITY_FRAMES = 10        # Number of consistent frames needed for stable prediction
MAX_PREDICTION_HISTORY = 15  # Keep last 15 predictions for smoothing

# Performance optimization
SKIP_FRAMES = 2  # Process every Nth frame (1=every frame, 2=every other frame)

# Visual configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)      # Green
BOX_COLOR = (0, 255, 0)       # Green
WARNING_COLOR = (0, 165, 255) # Orange

print(f"Configuration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  Stability frames: {STABILITY_FRAMES}")
print(f"  Frame skip: {SKIP_FRAMES}")

# load the model
print("\nLoading trained model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure the dir is correct.")
    exit()

# Initialize mediapipe
print("\nInitializing MediaPipe Hand Detection...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configure MediaPipe for optimal performance
hands = mp_hands.Hands(
    static_image_mode=False,        # Video mode (faster)
    max_num_hands=1,                # Only detect one hand 
    min_detection_confidence=0.5,   # Balance between speed and accuracy
    min_tracking_confidence=0.5     # Tracking confidence
)

print("MEDIAPIPE INITIALIZED !")


def extract_hand_region(frame, hand_landmarks, padding=20):
    """
    Extract hand region from frame using bounding box
    
    Args:
        frame: Input video frame
        hand_landmarks: MediaPipe hand landmarks
        padding: Extra pixels around hand (default: 20)
    
    Returns:
        hand_img: Cropped hand image (or None if extraction fails)
        bbox: Bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    h, w, _ = frame.shape
    
    # Get all landmark coordinates
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    # Calculate bounding box with padding
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Validate bounding box
    if x_max <= x_min or y_max <= y_min:
        return None, None
    
    # Crop hand region
    hand_img = frame[y_min:y_max, x_min:x_max]
    
    return hand_img, (x_min, y_min, x_max, y_max)


def preprocess_hand_image(hand_img):
    """
    Preprocess hand image for model prediction
    (Must match training preprocessing!)
    
    Args:
        hand_img: Cropped hand image (RGB)
    
    Returns:
        preprocessed: Image ready for model (1, 64, 64, 3)
    """
    # Resize to model input size
    resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    # Add batch dimension
    batched = np.expand_dims(normalized, axis=0)
    
    return batched


def get_stable_prediction(prediction_history, stability_frames):
    """
    Get stable prediction using voting mechanism
    
    Args:
        prediction_history: List of recent predictions
        stability_frames: Minimum number of consistent predictions needed
    
    Returns:
        stable_label: Most common prediction (or None if not stable)
        confidence: Average confidence of stable prediction
    """
    if len(prediction_history) < stability_frames:
        return None, 0.0
    
    # Get last N predictions
    recent_predictions = list(prediction_history)[-stability_frames:]
    
    # Extract labels and confidences
    labels = [pred[0] for pred in recent_predictions]
    confidences = [pred[1] for pred in recent_predictions]
    
    # Check if predictions are consistent (at least 70% same)
    from collections import Counter
    label_counts = Counter(labels)
    most_common_label, count = label_counts.most_common(1)[0]
    
    if count >= int(stability_frames * 0.7):  # 70% agreement
        # Calculate average confidence for this label
        label_confidences = [conf for lbl, conf in recent_predictions if lbl == most_common_label]
        avg_confidence = np.mean(label_confidences)
        return most_common_label, avg_confidence
    
    return None, 0.0


def draw_info_panel(frame, fps, hand_detected, prediction, confidence, stable):
    """
    Draw information panel on frame
    
    Args:
        frame: Video frame to draw on
        fps: Current FPS
        hand_detected: Whether hand is detected
        prediction: Current prediction
        confidence: Prediction confidence
        stable: Whether prediction is stable
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 150), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Hand detection status
    if hand_detected:
        status_text = "Hand: DETECTED"
        status_color = (0, 255, 0)
    else:
        status_text = "Hand: NOT DETECTED"
        status_color = (0, 0, 255)
    
    cv2.putText(frame, status_text, (20, 70), 
                FONT, 0.6, status_color, 1, cv2.LINE_AA)
    
    # Prediction
    if prediction and confidence > 0:
        if stable:
            pred_text = f"Sign: {prediction}"
            conf_text = f"Confidence: {confidence*100:.1f}%"
            pred_color = TEXT_COLOR
        else:
            pred_text = f"Detecting: {prediction}..."
            conf_text = f"Confidence: {confidence*100:.1f}%"
            pred_color = WARNING_COLOR
        
        cv2.putText(frame, pred_text, (20, 100), 
                    FONT, 0.8, pred_color, 2, cv2.LINE_AA)
        cv2.putText(frame, conf_text, (20, 130), 
                    FONT, 0.6, pred_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Sign: ---", (20, 100), 
                    FONT, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
    
    return frame


def calculate_fps(prev_time):
    """Calculate FPS"""
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

# ============================================================================
# SECTION 6: MAIN WEBCAM LOOP
# ============================================================================

def main():
    """
    Main function for real-time webcam ASL recognition
    """
    print("\n" + "="*70)
    print("STARTING WEBCAM RECOGNITION")
    print("="*70)
    print("\nControls:")
    print("  'q' or 'ESC' - Quit")
    print("  SPACE - Pause/Resume")
    print("\nInitializing webcam...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        print("Make sure your webcam is connected and not used by another application.")
        return
    
    print("✓ Webcam initialized")
    print("\n" + "="*70)
    print("WEBCAM ACTIVE - Show your ASL signs!")
    print("="*70 + "\n")
    
    # Initialize variables
    prediction_history = deque(maxlen=MAX_PREDICTION_HISTORY)
    frame_count = 0
    prev_time = time.time()
    paused = False
    
    # Current stable prediction
    current_prediction = None
    current_confidence = 0.0
    is_stable = False
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Failed to read frame from webcam!")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)
        
        hand_detected = False
        prediction = None
        confidence = 0.0
        
        # Process every Nth frame for performance
        if not paused and frame_count % SKIP_FRAMES == 0:
            
            # STEP 1: Detect hand with MediaPipe
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # STEP 2: Extract hand region (bounding box)
                hand_img, bbox = extract_hand_region(frame_rgb, hand_landmarks)
                
                if hand_img is not None and bbox is not None:
                    # Draw bounding box on frame
                    x_min, y_min, x_max, y_max = bbox
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 
                                BOX_COLOR, 2)
                    
                    # STEP 3: Preprocess hand image
                    preprocessed = preprocess_hand_image(hand_img)
                    
                    # STEP 4: Predict with CNN (low latency!)
                    pred_probs = model.predict(preprocessed, verbose=0)[0]
                    pred_index = np.argmax(pred_probs)
                    prediction = CLASS_NAMES[pred_index]
                    confidence = pred_probs[pred_index]
                    
                    # STEP 5: Add to history if confidence is high enough
                    if confidence >= CONFIDENCE_THRESHOLD:
                        prediction_history.append((prediction, confidence))
                    
                    # STEP 6: Get stable prediction
                    stable_pred, stable_conf = get_stable_prediction(
                        prediction_history, STABILITY_FRAMES
                    )
                    
                    if stable_pred:
                        current_prediction = stable_pred
                        current_confidence = stable_conf
                        is_stable = True
                    else:
                        # Show current prediction but mark as unstable
                        if confidence >= CONFIDENCE_THRESHOLD:
                            current_prediction = prediction
                            current_confidence = confidence
                            is_stable = False
            else:
                # No hand detected - clear history
                prediction_history.clear()
                current_prediction = None
                current_confidence = 0.0
                is_stable = False
        
        # STEP 7: Draw info panel
        frame = draw_info_panel(
            frame, fps, hand_detected, 
            current_prediction, current_confidence, is_stable
        )
        
        # Show pause indicator
        if paused:
            h, w = frame.shape[:2]
            cv2.putText(frame, "PAUSED", (w//2 - 80, h//2), 
                       FONT, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        
        # Display frame
        cv2.imshow('ASL Real-time Recognition - Stage 2', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("\nExiting...")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        
        frame_count += 1
    
    # Cleanup
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("\n" + "="*70)
    print("STAGE 2 COMPLETED!")
    print("="*70)
    print(f"\nTotal frames processed: {frame_count}")
    print("Ready for Stage 3: Buffer integration!")

# ============================================================================
# SECTION 7: RUN
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("\nProgram terminated.")