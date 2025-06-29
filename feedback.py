import random

ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def check_feedback(predicted_letter: str, confidence: float) -> str:
    if predicted_letter in ASL_CLASSES:
        return f"Attempting: {predicted_letter} ({confidence*100:.1f}%)"
    else:
        return f"Unrecognized ({confidence*100:.1f}%)"

def get_target_letter():
    return random.choice(ASL_CLASSES)
