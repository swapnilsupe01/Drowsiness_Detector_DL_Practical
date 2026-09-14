from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import time

IMG_SIZE = (64, 64)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "drowsiness_cnn.keras"
DROWSY_SCORE_THRESHOLD = 18

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Train it using Drowsiness_Detector_Colab.ipynb first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

score = 0
last_beep = 0

print("Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
    state_text = "NO FACE"
    frame_closed = False

    for (x, y, w, h) in faces[:1]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        # upper half of face is enough for eye search
        roi_gray = gray[y:y + h//2, x:x+w]
        roi_color = frame[y:y + h//2, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))

        predictions = []
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            if eye_img.size == 0:
                continue

            eye_img = cv2.resize(eye_img, IMG_SIZE).astype("float32") / 255.0
            eye_img = np.expand_dims(eye_img, axis=(0, -1))
            p_open = float(model.predict(eye_img, verbose=0)[0][0])
            predictions.append(p_open)

            label = "OPEN" if p_open >= 0.5 else "CLOSED"
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (200, 200, 200), 1)
            cv2.putText(
                roi_color, f"{label} {p_open:.2f}",
                (ex, max(15, ey-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1
            )

        if predictions:
            mean_open = float(np.mean(predictions))
            frame_closed = mean_open < 0.5
            state_text = "EYES CLOSED" if frame_closed else "EYES OPEN"
        else:
            state_text = "EYES NOT DETECTED"

    if frame_closed:
        score = min(score + 1, 100)
    else:
        score = max(score - 1, 0)

    drowsy = score >= DROWSY_SCORE_THRESHOLD
    if drowsy:
        status = "DROWSINESS ALERT!"
        thickness = 3
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), thickness)
        # Windows audible alert (safe fallback if winsound unavailable)
        if time.time() - last_beep > 1.0:
            try:
                import winsound
                winsound.Beep(1800, 250)
            except Exception:
                print("\a", end="", flush=True)
            last_beep = time.time()
    else:
        status = "NORMAL"

    cv2.putText(frame, state_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Score: {score}/{DROWSY_SCORE_THRESHOLD}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, status, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,0,255) if drowsy else (0,255,0), 2)

    cv2.imshow("Deep Learning Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
