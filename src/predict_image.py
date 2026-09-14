from pathlib import Path
import sys
import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (64, 64)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "drowsiness_cnn.keras"

def preprocess(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/predict_image.py <image_path>")
        return
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Train the model using Drowsiness_Detector_Colab.ipynb first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    image = preprocess(Path(sys.argv[1]))
    p_open = float(model.predict(image, verbose=0)[0][0])
    label = "OPEN" if p_open >= 0.5 else "CLOSED"
    confidence = p_open if label == "OPEN" else 1 - p_open
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
