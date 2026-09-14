# Experiment: Drowsiness Detection Using Deep Learning

## Aim
To design and implement a deep-learning-based drowsiness detection system using a Convolutional Neural Network (CNN) to classify eye images as open or closed and generate an alert when closed eyes persist for several consecutive frames.

## Software / Libraries
Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, scikit-learn, KaggleHub.

## Dataset
MRL Eye Dataset subset containing two classes:
1. Open Eye
2. Closed Eye

## Theory
Driver drowsiness can be estimated from visual cues such as prolonged eye closure. A CNN learns spatial features directly from eye images. Convolution layers learn edges, shapes, eyelid patterns, and higher-level visual features. Pooling reduces feature-map size, dropout reduces overfitting, and a sigmoid output gives the probability of the eye being open.

## CNN Architecture
- Input: 64 x 64 grayscale eye image
- Conv2D(32) + ReLU + MaxPooling
- Conv2D(64) + ReLU + MaxPooling
- Conv2D(128) + ReLU + MaxPooling
- GlobalAveragePooling
- Dense(64) + ReLU
- Dropout(0.4)
- Dense(1) + Sigmoid

## Algorithm
1. Load open-eye and closed-eye images.
2. Resize all images to 64 x 64.
3. Normalize pixels to [0, 1].
4. Split data into training and validation sets.
5. Apply light image augmentation to training samples.
6. Train the CNN using binary cross-entropy and Adam optimizer.
7. Evaluate using accuracy, classification report, and confusion matrix.
8. Save the trained model.
9. Open webcam using OpenCV.
10. Detect face and eye regions.
11. Classify each detected eye using the trained CNN.
12. Maintain a consecutive-closed-eye score.
13. If the score exceeds a threshold, show and sound a drowsiness alert.

## Expected Result
The CNN should learn to distinguish open and closed eyes with high validation accuracy. During real-time testing, normal blinks are tolerated, while prolonged eye closure raises the drowsiness score and activates an alert.

## Conclusion
A CNN-based drowsiness detector was implemented successfully. Deep learning was used for eye-state classification, while OpenCV handled real-time image capture and eye localization. Temporal scoring prevented a single blink from being treated as drowsiness. The system demonstrates how computer vision and deep learning can be combined for real-time safety-oriented applications.

## Limitations
Performance may reduce with poor lighting, extreme head pose, sunglasses, very low-quality webcams, or failed eye detection. It is an academic prototype and not a certified safety system.
