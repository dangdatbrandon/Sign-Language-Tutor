import cv2
import numpy as np
import tensorflow as tf
from feedback import check_feedback, get_target_letter
import os

# Load model
MODEL_PATH = os.path.join("model", "best_model_CNN_Final.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# ASL classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    'space', 'delete', 'nothing'
]

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Set the initial target letter and confidence threshold
target_letter = get_target_letter()
confidence_threshold = 0.6

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Region of interest
    x0, y0, width, height = 300, 100, 200, 200
    roi = frame[y0:y0+height, x0:x0+width]

    # Preprocess image
    img = cv2.resize(roi, (58, 58))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, 58, 58, 1).astype("float32") / 255.0

    # Predict
    preds = model.predict(img, verbose=0)
    pred_index = np.argmax(preds)
    pred_letter = ASL_CLASSES[pred_index]
    confidence = preds[0][pred_index]

    # Check feedback
    feedback = check_feedback(pred_letter, confidence)

    # ✅ Move on to new target letter only when confident & correct
    if pred_letter == target_letter and confidence >= confidence_threshold:
        target_letter = get_target_letter()

    # Draw UI
    cv2.rectangle(frame, (x0, y0), (x0+width, y0+height), (255, 0, 0), 2)
    cv2.putText(frame, f"Prediction: {pred_letter} ({confidence*100:.1f}%)", (x0, y0-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Target: {target_letter}", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, f"Feedback: {feedback}", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("ASL Tutor", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
