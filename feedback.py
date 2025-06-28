# feedback.py
'''
TARGET_LETTER = 'A'  # Change this letter as needed

def check_feedback(predicted_letter: str) -> str:
    if predicted_letter == TARGET_LETTER:
        return "Correct"
    else:
        return "Incorrect"
'''
import random

# Generate a random target letter from the ASL alphabet
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Select a random target letter for the user to sign
TARGET_LETTER = random.choice(ASL_CLASSES)  # Change each time the program runs

def check_feedback(predicted_letter: str, confidence: float) -> str:
    """
    Function to provide feedback based on the predicted letter and its confidence.

    Args:
    - predicted_letter (str): The letter predicted by the model.
    - confidence (float): The confidence score (between 0 and 1) for the prediction.

    Returns:
    - feedback (str): A feedback message indicating whether the prediction is correct and the confidence level.
    """
    
    if predicted_letter == TARGET_LETTER:
        return f"Correct! Confidence: {confidence*100:.1f}%"
    else:
        return f"Incorrect. The target letter was {TARGET_LETTER}. Confidence: {confidence*100:.1f}%"

# Optional: If you want to display the target letter (useful for debugging)
def get_target_letter():
    return TARGET_LETTER
