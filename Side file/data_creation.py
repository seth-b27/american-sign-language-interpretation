import cv2
import os

sign = "0"  # we can change this for a specific sign
save_dir = f"webcam_data/{sign}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting data for sign: {sign}")
print("Space to capture, and 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    cv2.imshow('Collect Data', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE to capture
        cv2.imwrite(f"{save_dir}/img_{count:04d}.jpg", frame)
        count += 1
        print(f"Captured {count} images")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Collected {count} images for {sign}")