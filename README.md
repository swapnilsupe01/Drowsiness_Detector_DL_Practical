# Deep Learning Practical — Drowsiness Detector

## Objective
Build a CNN-based drowsiness detector that classifies an eye image as **Open** or **Closed** and uses consecutive closed-eye predictions to raise a drowsiness alert.

## Included
- `Drowsiness_Detector_Colab.ipynb` — complete training + evaluation notebook
- `src/webcam_detector.py` — real-time webcam detection
- `src/predict_image.py` — predict one image
- `requirements.txt` — Python dependencies
- `run_webcam.bat` — Windows shortcut
- `PRACTICAL_WRITEUP.md` — ready theory/algorithm/result/conclusion
- `VIVA_QUESTIONS.md` — common viva questions with answers
- `models/` — trained model is saved here as `drowsiness_cnn.keras`
- `outputs/` — graphs/confusion matrix generated here
- `dataset/` — optional local dataset folder

## Recommended: Google Colab
1. Upload this ZIP to Google Drive or extract it locally.
2. Open `Drowsiness_Detector_Colab.ipynb` in Colab.
3. Runtime -> Change runtime type -> GPU (optional but recommended).
4. Run all cells.
5. The notebook downloads the public Kaggle MRL-based eye dataset automatically with `kagglehub`.
6. After training, download `drowsiness_cnn.keras`.
7. Put it inside the `models` folder on your PC.
8. Install dependencies:
   `pip install -r requirements.txt`
9. Run:
   `python src/webcam_detector.py`

## Dataset
Default Kaggle handle used in the notebook:
`kutaykutlu/drowsiness-detection`

The dataset contains MRL eye images split into `closed_eye` and `open_eye`.

## Drowsiness Logic
A single blink should NOT trigger an alarm. The webcam program keeps a score:
- closed eye -> score increases
- open eye -> score decreases
- score >= threshold -> DROWSINESS ALERT

## Notes
- Webcam inference uses OpenCV Haar cascades only to locate eyes.
- CNN performs the actual Open/Closed classification.
- This is an academic prototype, not a medical or road-safety certified system.
