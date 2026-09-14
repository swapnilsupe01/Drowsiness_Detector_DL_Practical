# Viva Questions — Drowsiness Detector

1. **Why is CNN used?**  
   CNNs automatically learn spatial features such as edges, eyelid shapes and eye-state patterns from images.

2. **Why use grayscale images?**  
   Eye openness mostly depends on shape/texture, so grayscale reduces computation without needing color information.

3. **Why normalize pixel values?**  
   Scaling 0–255 values to 0–1 makes neural-network optimization more stable.

4. **What is ReLU?**  
   ReLU is an activation function `max(0, x)` that introduces non-linearity.

5. **Why MaxPooling?**  
   It reduces spatial dimensions and computation while keeping important features.

6. **Why Dropout?**  
   Dropout randomly disables some neurons during training to reduce overfitting.

7. **Why sigmoid in the output layer?**  
   This is binary classification, so sigmoid produces a probability between 0 and 1.

8. **What loss function is used?**  
   Binary cross-entropy.

9. **What optimizer is used?**  
   Adam.

10. **Why not trigger drowsiness on one closed-eye frame?**  
    A normal blink is very short. Consecutive closed-eye frames are a better indication of prolonged closure.

11. **Difference between OpenCV and CNN in this project?**  
    OpenCV captures video and locates eye regions; the CNN classifies the eye state.

12. **What is overfitting?**  
    When a model performs very well on training data but poorly on unseen data.

13. **How is overfitting reduced here?**  
    Data augmentation, dropout and early stopping.

14. **What is a confusion matrix?**  
    A table showing true/false predictions for each class.

15. **Can accuracy alone be misleading?**  
    Yes, especially on imbalanced datasets; precision, recall and F1-score should also be checked.
