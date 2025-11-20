# this project runs on .venv38 (python 3.8.8) 

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipeq1
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('../STAGE1/asl_final_cnn_model.h5')

# Load class names
classNames = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print ("Error: Could not open webcam.")
    exit()

while True:
    # Read each frame from the webcam
    # _, frame = cap.read()
    ret, frame = cap.read()
    print ("frame read:", ret)
    if not ret:
        print ("could not read from the webcam. Exiting...")
        break

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break
# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()

'''
This program uses
- OpenCV to capture video from the webcam
- MediaPipe to detect and track hand landmarks
- TensorFlow to load a pre-trained gesture recognition model and make predictions based on the detected hand landmarks.
- keras to load the model
- packages and their versions used: run pip list
- all .venv38 packages stored in requirements.txt: pip freeze > requirements.txt
- to use those packages, create a virtual environment with python 3.8 and install the packages with pip install -r requirements.txt
'''