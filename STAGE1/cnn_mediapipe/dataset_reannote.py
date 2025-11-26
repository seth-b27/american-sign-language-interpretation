"""
Here, we use mediapipe landmarks-based bounding box to reannotate the dataset (crop hand region more precisely).
Process:
a. load images from original dataset
b. detect hand landmarks with MediaPipe
c. calculate bounding box from landmarks
d. then, crop hand region with some padding
e. saved to a new dataset folder (asl_dataset_mediapipe)
"""
import cv2
import numpy as np
import mediapipe as mp
import os
from tqdm import tqdm
import shutil

print("ASL DATASET RE-ANNOTATION WITH MEDIAPIPE")

ORIGINAL_DATASET_PATH = '../asl_dataset'  
NEW_DATASET_PATH = '../asl_dataset_mediapipe' 

# preprocess config
PADDING_PERCENT = 0.15 # extra space around hand
MIN_DETECTION_CONFIDENCE = 0.3  
TARGET_SIZE = 64  

# class names
def get_class_names_from_dataset(dataset_path):

    class_names = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    digits = sorted([c for c in class_names if c.isdigit()])
    letters = sorted([c for c in class_names if c.isalpha()])
    
    return digits + letters
CLASS_NAMES = get_class_names_from_dataset(ORIGINAL_DATASET_PATH)
print(f"Classes: {CLASS_NAMES}")


print("\nINITIALIZING MEDIAPIPE HAND DETECTION . . .")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,  
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE
)

print("🐥 MEDIAPIPE INITIALIZED !")

def get_bbox_from_landmarks(landmarks, img_width, img_height, padding_percent=0.15):
    # extract all landmark coordinates (that is already normalized to [0,1])
    x_coords = [lm.x for lm in landmarks.landmark]
    y_coords = [lm.y for lm in landmarks.landmark]
    
    # get bbox in normalized coordinates
    x_min_norm = min(x_coords)
    x_max_norm = max(x_coords)
    y_min_norm = min(y_coords)
    y_max_norm = max(y_coords)
    
    width_norm = x_max_norm - x_min_norm
    height_norm = y_max_norm - y_min_norm
    
    # add padding (stay within [0, 1])
    x_min_norm = max(0, x_min_norm - width_norm * padding_percent)
    x_max_norm = min(1, x_max_norm + width_norm * padding_percent)
    y_min_norm = max(0, y_min_norm - height_norm * padding_percent)
    y_max_norm = min(1, y_max_norm + height_norm * padding_percent)
    
    # converting to pixel coordinates
    x_min_px = int(x_min_norm * img_width)
    x_max_px = int(x_max_norm * img_width)
    y_min_px = int(y_min_norm * img_height)
    y_max_px = int(y_max_norm * img_height)
    
    # ensure valid bbox
    x_min_px = max(0, x_min_px)
    x_max_px = min(img_width, x_max_px)
    y_min_px = max(0, y_min_px)
    y_max_px = min(img_height, y_max_px)
    
    bbox_pixels = (x_min_px, y_min_px, x_max_px, y_max_px)
    
    # calculate normalized center and dimensions for reference
    x_center_norm = (x_min_norm + x_max_norm) / 2
    y_center_norm = (y_min_norm + y_max_norm) / 2
    width_final = x_max_norm - x_min_norm
    height_final = y_max_norm - y_min_norm
    
    bbox_normalized = (x_center_norm, y_center_norm, width_final, height_final)
    
    return bbox_pixels, bbox_normalized


def extract_and_crop_hand(image, landmarks):
    h, w, _ = image.shape
    
    # get bounding box from landmarks
    bbox_px, bbox_norm = get_bbox_from_landmarks(landmarks, w, h, PADDING_PERCENT)
    x_min, y_min, x_max, y_max = bbox_px
    
    # validate bounding box
    if x_max <= x_min or y_max <= y_min:
        return None, None
    
    # crop hand region
    hand_crop = image[y_min:y_max, x_min:x_max]
    
    # validate crop
    if hand_crop.size == 0:
        return None, None
    
    return hand_crop, bbox_px

"""
b. each image processed by: detect hand, extract region, resize
a. use_fallback: If True, use original image when MediaPipe fails
"""
def process_image(image_path, use_fallback=True):
    # read image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Failed to read image"
    
    # convert BGR to RGB 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect hand landmarks
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        if use_fallback:
            h, w = image_rgb.shape[:2]
            # center crop to make it square
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            cropped = image_rgb[start_y:start_y+size, start_x:start_x+size]
            
            # resize to target size
            resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
            resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
            return resized_bgr, "Fallback (no MediaPipe detection)"
        else:
            return None, "No hand detected"
    
    # get first hand (we use max_num_hands=1)
    hand_landmarks = results.multi_hand_landmarks[0]
    
    # extract hand region
    hand_crop, bbox = extract_and_crop_hand(image_rgb, hand_landmarks)
    
    if hand_crop is None:
        if use_fallback:
            h, w = image_rgb.shape[:2]
            size = min(h, w)
            start_y = (h - size) // 2
            start_x = (w - size) // 2
            cropped = image_rgb[start_y:start_y+size, start_x:start_x+size]
            
            resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
            resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            
            return resized_bgr, "Fallback (crop failed)"
        else:
            return None, "Failed to crop hand"
    
    # target resize to 64x64
    hand_resized = cv2.resize(hand_crop, (TARGET_SIZE, TARGET_SIZE))
    
    # convert back to BGR for saving
    hand_bgr = cv2.cvtColor(hand_resized, cv2.COLOR_RGB2BGR)
    
    return hand_bgr, "Success (MediaPipe)"

print("CREATING NEW DATASET STRUCTURE . . .")

os.makedirs(NEW_DATASET_PATH, exist_ok=True)
for class_name in CLASS_NAMES:
    class_folder = os.path.join(NEW_DATASET_PATH, class_name)
    os.makedirs(class_folder, exist_ok=True)

print(f"🐥 Created new dataset structure at: {NEW_DATASET_PATH}")


print("PROCESSING ALL IMAGES . . .")

total_images = 0
successful_images = 0
mediapipe_success = 0
fallback_used = 0
failed_images = 0
failed_details = {'read_failed': 0}

# process each class folder
for class_idx, class_name in enumerate(CLASS_NAMES):
    print(f"\n[{class_idx+1}/{len(CLASS_NAMES)}] Processing class: {class_name}")
    
    src_folder = os.path.join(ORIGINAL_DATASET_PATH, class_name)
    dst_folder = os.path.join(NEW_DATASET_PATH, class_name)
    
    if not os.path.exists(src_folder):
        print(f"⚠️ Warning: Folder not found: {src_folder}")
        continue
    
    # get all image types
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"⚠️ Warning: No images found in {src_folder}")
        continue
    
    print(f"Found {len(image_files)} images")
    
    # process each image with progress bar
    class_success = 0
    class_failed = 0
    
    for img_file in tqdm(image_files, desc=f"  Processing {class_name}", ncols=80):
        src_path = os.path.join(src_folder, img_file)
        dst_path = os.path.join(dst_folder, img_file)
        
        processed_img, status = process_image(src_path, use_fallback=True)
        
        total_images += 1
        if processed_img is not None:
            cv2.imwrite(dst_path, processed_img)
            successful_images += 1
            class_success += 1
            
            if "MediaPipe" in status:
                mediapipe_success += 1
            elif "Fallback" in status:
                fallback_used += 1
        else:
            failed_images += 1
            class_failed += 1
            failed_details['read_failed'] += 1
    
    print(f"  ✅ Success: {class_success}/{len(image_files)} ({class_success/len(image_files)*100:.1f}%)")
    if class_failed > 0:
        print(f"  ❌ Failed: {class_failed} images")


print("COMPLETE PROCESSING\n📊 Summary Statistics:")
print(f"Total images processed: {total_images}")
print(f"a. Successful: {successful_images} ({successful_images/total_images*100:.1f}%)")
print(f"    1. MediaPipe extracted: {mediapipe_success} ({mediapipe_success/total_images*100:.1f}%)")
print(f"    2. Fallback used: {fallback_used} ({fallback_used/total_images*100:.1f}%)")
print(f"b. Failed: {failed_images} ({failed_images/total_images*100:.1f}%)")

if failed_images > 0:
    print(f"c. Read failed: {failed_details['read_failed']}")

print(f"\n🐥 New annotated dataset saved to: {NEW_DATASET_PATH}")

# cleanup
hands.close()