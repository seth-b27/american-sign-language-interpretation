## ASL Recognition with Deep Learning Model

This project presents a real-time ASL alphabet and numeral recognition system that combines both deep learning model (CNN) with MediaPipe hand landmark-based bounding box preprocessing pipeline, capable of recognizing 36 distinct ASL gestures (0-9 and A-Z). While this does not cover the full richness and complexity of ASL itself, it was built with the sincere intention of assissting and supporting people who use sign language as their mean to communicate. This work is not a final solution, but a small stepping stone toward a more inclusive and accessible communication technologies.

The project operates through a three-stage pipeline designed for both training (stage 1), and real-time inference (stage 2,3). 

## STAGE 1
This phase explores three different models (CNN alone, SVM with HOG feature, and CNN + MediaPipe). During this stage, a dataset `asl_dataset` of 7,200 RGB images (200 per class) is used to train with our first two experimental models (CNN and SVM). As a result, both models showed signs of overfitting despite their high test accuracy. To deal with this issue, the dataset is then re-annotated with MediaPipe framwork that would detect 21 hand anatomical keypoints in normalized [0,1] coordinate space, calculate axis-aligned bbox around these 21 landmarks with 15% padding area on each side (context for CNN), convert to pixel-coordinates, crop and extract the hand using computed pixel bbox, resize to 64×64 pixels, convert each image back to BGR for OpenCV file saving, and save them to a new dataset. The same CNN model is then re-trained with the newly re-annotated dataset `asl_dataset_mediapipe`, and performs decently achieving test accuracy of 97.92%. This model is then used in the next two stages.

## STAGE 2
The process starts by capturing video from a webcam at 640×480 resolution and 30 FPS using OpenCV's VideoCapture interface, with BGR-to-RGB colorspace conversion for MediaPipe compatibility. Each frame undergoes MediaPipe hand detection (`MIN_DETECTION_CONFIDENCE=0.4`), landmark-based bbox computation (identical to stege 1), and preprocessing before being fed to the trained CNN model. Raw CNN outputs (softmax probabilities) undergo a two-layer stability filtering: (1) confidence threshold of 0.5, and (2) a voting system (that requires the same prediction to appear ≥2 times within the last 3 frames). Results are then visualized only when predictions are stable, displaying 21 landmarks, bbox, predicted class, and confidence score.

## STAGE 3
This stage is largely identical to the previous phase, only with some additional steps to accumulate stable predictions into a sequential text buffer representing the users’ intended messages.

## CNN Arch.
CNN model used in this project consists of 3 convolutional blocks with progressively increased feature count (32->64->128), each followed by max pooling and dropout layers for regularization. After flatening, 2 fully connected layers (256 & 128 neurons) with heavy dropout (0.5) refine the learned features before the final 36-class softmax output layer.

## Usage
Please install all dependencies as stated in `requirements.txt` file stored in the Side file folder.
