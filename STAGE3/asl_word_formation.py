# the code here is mostly identical to stage2's ; we only add the text buffer logic and its UI
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
import time
from collections import deque
from datetime import datetime

print("\n📸 A S L    R E C O G N I T I O N    A P P L I C A T I O N")

# C O N F I G U R A T I O N

# model config
MODEL_PATH = '../STAGE1/cnn_mediapipe/asl_final_cnn_mediapipe_model.keras'
IMG_SIZE = 64
CLASS_NAMES = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

# webcam resolution config
CAMERA_INDEX = 0
CAMERA_WIDTH = 640  
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# mediapipe config
MIN_DETECTION_CONFIDENCE = 0.4 # mp will ignore hands detected with confidence below this 
MIN_TRACKING_CONFIDENCE = 0.4 
#mp uses temporal info (landmarks from previous frames) and is faster. but if confidence drops below this, it will run detection again.
BBOX_PADDING = 0.15  

# pediction stability config
CONFIDENCE_THRESHOLD = 0.5  
STABILITY_FRAMES = 3         
MAX_PREDICTION_HISTORY = 5   

SKIP_FRAMES = 1  # process EVERY frame for responsiveness           

# UI Configuration
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_PRED = 1.0
FONT_SCALE_BUFFER = 0.9
FONT_SCALE_STATS = 0.5
FONT_SCALE_CONTROLS = 0.45
FONT_THICKNESS_BOLD = 2
FONT_THICKNESS_NORMAL = 2
FONT_THICKNESS_THIN = 1

# colors
COLOR_BBOX = (0, 255, 0)
COLOR_PRED_STABLE = (0, 255, 0)
COLOR_PRED_DETECTING = (0, 165, 255)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_BUFFER_BG = (25, 25, 25)
COLOR_BUFFER_TEXT = (255, 255, 255)
COLOR_STATS = (100, 200, 255)
COLOR_CONTROLS = (200, 200, 200)
COLOR_CURSOR = (0, 255, 0)

INFO_PANEL_HEIGHT = 90  
BBOX_THICKNESS = 2

# buffer config
BUFFER_HEIGHT = 140          
BUFFER_FONT_SIZE = 0.9       
BUFFER_MAX_CHARS = 60        
AUTO_ADD_DELAY = 0.5         
SHOW_STATS = True 

# pre-typed words (just to make it look a bit cool, no?)
EXAMPLE_WORDS = ["HELLO", "WORLD", "ASL", "THANK YOU", "GOOD LUCK", "I LIKE YOU", "GOOD BYE"]

# L O A D   M O D E L
print("\nLOADING THE MODEL ...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"\n🐥 MODEL LOADED !")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Warm up model (which is important for first prediction speed!)
print("\nWARMING UP THE MODEL ...")
dummy_input = np.random.rand(1, 64, 64, 3).astype('float32')
_ = model.predict(dummy_input, verbose=0)
print("\n 🐥 MODEL IS READY !")

# I N I T I A L I Z E   M E D I A P I P E
print("\nINITIALIZING MEDIAPIPE ...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands= 1,
    min_detection_confidence= MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    model_complexity= 0  # use lighter model (0=lite, 1=full)
)

print("\n🐥 MEDIAPIPE INITIALIZED !")

# T E X T   B U F F E R   C L A S S
class TextBuffer:
    def __init__(self):
        self.text = ""
        self.last_added_char = None
        self.last_added_time = 0
        self.total_letters = 0
        self.total_words = 0
        
    def add_letter(self, letter, current_time):
        # prevent adding same letter too quickly
        if (self.last_added_char == letter and current_time - self.last_added_time < AUTO_ADD_DELAY):
            return False
        
        self.text += letter
        self.last_added_char = letter
        self.last_added_time = current_time
        self.total_letters += 1
        return True
    
    def add_space(self):
        if self.text and self.text[-1] != ' ':
            self.text += ' '
            self.last_added_char = ' '
            self.update_word_count()
            return True
        return False
    
    def delete_last(self):
        if self.text:
            self.text = self.text[:-1]
            self.total_letters = max(0, self.total_letters - 1)
            self.update_word_count()
            return True
        return False
    
    def clear(self):
        self.text = ""
        self.last_added_char = None
        self.total_letters = 0
        self.total_words = 0
    
    def load_example(self, example_text):
        self.text = example_text
        self.update_word_count()
        self.total_letters = sum(1 for c in self.text if c.isalnum())
    
    def update_word_count(self):
        words = self.text.strip().split()
        self.total_words = len(words) if words else 0
    
    def get_display_text(self, max_chars=60):
        if len(self.text) <= max_chars:
            return self.text
        else:
            return "..." + self.text[-(max_chars-3):]
    
    def save_to_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asl_text_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(self.text)
        return filename

# H E L P E R   F U N C T I O N S
def get_bbox_from_landmarks(landmarks, img_width, img_height):

    try:
        # get coordinates directly in pixel space
        x_coords = [lm.x * img_width for lm in landmarks.landmark]
        y_coords = [lm.y * img_height for lm in landmarks.landmark]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        # padding definition
        width = x_max - x_min
        height = y_max - y_min
        
        padding_x = int(width * BBOX_PADDING)
        padding_y = int(height * BBOX_PADDING)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(img_width, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(img_height, y_max + padding_y)
        
        # validate bbox
        if x_max <= x_min or y_max <= y_min:
            return None
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        print(f"Bbox error: {e}")
        return None


# this one is basically the same as in training
def preprocess_hand_crop(hand_crop):
    try:
        # the input must be RGB
        if hand_crop.shape[2] == 3:
            # and its already RGB from frame_rgb
            pass
        # resize to 64x64
        resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype('float32') / 255.0
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
        
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None


def get_stable_prediction(prediction_history):
    if len(prediction_history) == 0:
        return None, 0.0, False
    
    # if we have enough history
    if len(prediction_history) >= STABILITY_FRAMES:
        # get recent predictions
        recent = list(prediction_history)[-STABILITY_FRAMES:]
        labels = [p[0] for p in recent]

        # check if last prediction appears multiple times
        last_label = labels[-1]
        count = labels.count(last_label)
        
        if count >= 2:  # but if appears at least twice in recent history
            # get the confidence average 
            confs = [p[1] for p in recent if p[0] == last_label]
            avg_conf = np.mean(confs)
            return last_label, avg_conf, True
    
    # here it returns most recent prediction (not stable yet)
    return prediction_history[-1][0], prediction_history[-1][1], False


def draw_text_with_bg(frame, text, pos, font_scale, color, thickness, bg_color=COLOR_TEXT_BG):
    x, y = pos
    (w, h), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    
    cv2.rectangle(frame, (x-5, y-h-5), (x+w+5, y+baseline+5), bg_color, -1)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
    
    return h


def draw_prediction_on_bbox(frame, bbox, prediction, confidence, is_stable):
    x_min, y_min, x_max, y_max = bbox
    
    color = COLOR_PRED_STABLE if is_stable else COLOR_PRED_DETECTING
    status = "" if is_stable else " ..."
    
    pred_text = f"{prediction}{status}"
    combined_text = f"{pred_text} ({confidence*100:.1f}%)"

    text_x = x_min
    text_y = max(50, y_min - 10)

    cv2.putText(frame, combined_text, (text_x, text_y), FONT, FONT_SCALE_PRED, color, FONT_THICKNESS_BOLD, cv2.LINE_AA)

def draw_top_panel(frame, fps, hand_detected):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, INFO_PANEL_HEIGHT), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # hand status
    status = "Hand: YES" if hand_detected else "Hand: NO"
    status_color = COLOR_PRED_STABLE if hand_detected else (0, 0, 255)
    cv2.putText(frame, status, (10, 50), 
                FONT, 0.5, status_color, 1, cv2.LINE_AA)
    
    # controls
    controls_y = 70
    cv2.putText(frame, "Controls:", (10, controls_y), FONT, FONT_SCALE_CONTROLS, 
                COLOR_CONTROLS, 1, cv2.LINE_AA)
    
    # Control hints
    controls_text = "SPACE=Space | BS=Del | C=Clear | S=Save | 1-7=Examples | Q=Quit"
    cv2.putText(frame, controls_text, (90, controls_y), FONT, FONT_SCALE_CONTROLS, 
                COLOR_CONTROLS, 1, cv2.LINE_AA) 

def draw_buffer_panel(frame, text_buffer, cursor_visible=True):
    h, w = frame.shape[:2]
    buffer_y = h - BUFFER_HEIGHT
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, buffer_y), (w, h), COLOR_BUFFER_BG, -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    title_y = buffer_y + 25
    cv2.putText(frame, "Text Buffer:", (15, title_y), FONT, 0.6, COLOR_STATS, 1, cv2.LINE_AA)
    
    # for predicted text 
    display_text = text_buffer.get_display_text(BUFFER_MAX_CHARS)
    
    if cursor_visible:
        display_text += "_"
    
    text_y = buffer_y + 60
    cv2.putText(frame, display_text, (15, text_y), FONT, BUFFER_FONT_SIZE, COLOR_BUFFER_TEXT, 
                FONT_THICKNESS_BOLD, cv2.LINE_AA)
    
    if SHOW_STATS:
        stats_y = buffer_y + 95
        stats_text = f"Words: {text_buffer.total_words}  |  Letters: {text_buffer.total_letters}"
        
        if text_buffer.last_added_char and text_buffer.last_added_char != ' ':
            stats_text += f"  |  Last: '{text_buffer.last_added_char}'"
        
        cv2.putText(frame, stats_text, (15, stats_y), FONT, FONT_SCALE_STATS, COLOR_STATS, 
                    FONT_THICKNESS_THIN, cv2.LINE_AA)
    
    hint_y = buffer_y + 125
    hint_text = "Quick: 1=HELLO | 2=WORLD | 3=ASL | 4=THANK YOU | 5=GOOD LUCK | 6=I LIKE YOU | 7=GOOD BYE"
    cv2.putText(frame, hint_text, (15, hint_y), FONT, FONT_SCALE_CONTROLS-0.1, (150, 150, 150), 
                FONT_THICKNESS_THIN, cv2.LINE_AA)

# M A I N 
def main():
    print("\n🎥 S T A R T I N G   W E B C A M ")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT + BUFFER_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    text_buffer = TextBuffer()
    prediction_history = deque(maxlen=MAX_PREDICTION_HISTORY)
    prev_time = time.time()
    cursor_blink_time = time.time()
    cursor_visible = True
    
    current_prediction = None
    current_confidence = 0.0
    is_stable = False
    current_bbox = None
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # create a larger frame for buffer
        full_frame = np.zeros((CAMERA_HEIGHT + BUFFER_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 0.0001)
        prev_time = current_time
        
        # cursor blink 
        if current_time - cursor_blink_time > 0.5:
            cursor_visible = not cursor_visible
            cursor_blink_time = current_time
        
        hand_detected = False
        
        if frame_count % SKIP_FRAMES == 0:
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                h, w, _ = frame.shape
                bbox = get_bbox_from_landmarks(hand_landmarks, w, h)
                
                if bbox is not None:
                    x_min, y_min, x_max, y_max = bbox
                    current_bbox = bbox
                    
                    # crop and predict
                    hand_crop = frame_rgb[y_min:y_max, x_min:x_max]
                    
                    if hand_crop.size > 0:
                        preprocessed = preprocess_hand_crop(hand_crop)
                        
                        if preprocessed is not None:
                            pred_probs = model.predict(preprocessed, verbose=0)[0]
                            pred_idx = np.argmax(pred_probs)
                            prediction = CLASS_NAMES[pred_idx]
                            confidence = pred_probs[pred_idx]
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                prediction_history.append((prediction, confidence))
                                
                                stable_pred, stable_conf, stable = get_stable_prediction(prediction_history)
                                
                                current_prediction = stable_pred
                                current_confidence = stable_conf
                                is_stable = stable
                                
                                if is_stable and stable_pred:
                                    text_buffer.add_letter(stable_pred, current_time)
            else:
                prediction_history.clear()
                current_prediction = None
                current_bbox = None
        
        # draw on webcam section
        if current_bbox is not None and hand_detected:
            x_min, y_min, x_max, y_max = current_bbox
            bbox_color = COLOR_PRED_STABLE if is_stable else COLOR_PRED_DETECTING
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), bbox_color, BBOX_THICKNESS)
            
            if current_prediction:
                draw_prediction_on_bbox(frame, current_bbox, current_prediction, current_confidence, is_stable)
        
        draw_top_panel(frame, fps, hand_detected)
        
        # combine webcam + buffer
        full_frame[:CAMERA_HEIGHT, :] = frame
        draw_buffer_panel(full_frame, text_buffer, cursor_visible)
        
        cv2.imshow('ASL Recognition - Stage 3', full_frame)
        
        # keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q or esc
            break
        elif key == ord(' '): 
            if text_buffer.add_space():
                print(f"Added space")
        elif key == 8 or key == 127:  # backspace
            if text_buffer.delete_last():
                print(f"Deleted last character")
        elif key == ord('c') or key == ord('C'):  # c - clear
            text_buffer.clear()
            print("Cleared buffer")
        elif key == ord('s') or key == ord('S'):  # s - save
            filename = text_buffer.save_to_file()
            print(f"✓ Saved to: {filename}")
        elif key == ord('1'): 
            text_buffer.load_example(EXAMPLE_WORDS[0])
            print(f"Loaded: {EXAMPLE_WORDS[0]}")
        elif key == ord('2'): 
            text_buffer.load_example(EXAMPLE_WORDS[1])
            print(f"Loaded: {EXAMPLE_WORDS[1]}")
        elif key == ord('3'):  
            text_buffer.load_example(EXAMPLE_WORDS[2])
            print(f"Loaded: {EXAMPLE_WORDS[2]}")
        elif key == ord('4'):  
            text_buffer.load_example(EXAMPLE_WORDS[3])
            print(f"Loaded: {EXAMPLE_WORDS[3]}")
        elif key == ord('5'): 
            text_buffer.load_example(EXAMPLE_WORDS[4])
            print(f"Loaded: {EXAMPLE_WORDS[4]}")
        elif key == ord('6'):
            text_buffer.load_example(EXAMPLE_WORDS[5])
            print(f"Loaded: {EXAMPLE_WORDS[5]}")
        elif key == ord('7'):
            text_buffer.load_example(EXAMPLE_WORDS[6])
            print(f"Loaded: {EXAMPLE_WORDS[6]}")
        
        frame_count += 1
    
    print("\n🧹 CLEANING UP...")
    
    if text_buffer.text:
        print(f"\nFinal text: '{text_buffer.text}'")
        save_final = input("Would you like to save final text? (y/n): ")
        if save_final.lower() == 'y':
            filename = text_buffer.save_to_file()
            print(f"🐥 Saved to: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print("🐥 ASL COMPLETE SYSTEM - SESSION ENDED")
    print(f"\nSession stats:")
    print(f"a. Total frames      : {frame_count}")
    print(f"b. Words formed      : {text_buffer.total_words}")
    print(f"c. Letters recognized: {text_buffer.total_letters}")
    print(f"d. Final text        : '{text_buffer.text}'")
    print("\n🐔 S H U T T I N G   D O W N . . . ")
    print("\n📕 P R O J E C T   C O M P L E T E D ! 📕")

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