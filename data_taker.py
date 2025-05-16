# File path: handwritten_alphabet_recognizer.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply thresholding to convert to binary image
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def find_letter_boxes(thresh_image):
    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_boxes = [cv2.boundingRect(c) for c in contours]
    # Sort boxes from left to right
    letter_boxes = sorted(letter_boxes, key=lambda x: x[0])
    return letter_boxes


def extract_and_resize_letters(thresh_image, letter_boxes):
    letters = []
    for (x, y, w, h) in letter_boxes:
        # Extract the letter from the threshold image
        letter_image = thresh_image[y:y+h, x:x+w]
        # Resize to 28x28 pixels (standard size for most models)
        letter_image = cv2.resize(letter_image, (28, 28), interpolation=cv2.INTER_AREA)
        # Normalize pixel values
        letter_image = letter_image.astype('float32') / 255.0
        # Expand dimensions to fit the model input shape (1, 28, 28, 1)
        letter_image = np.expand_dims(letter_image, axis=-1)
        letters.append(letter_image)
    return np.array(letters)


def load_recognition_model(model_path):
    # Load a pre-trained model (e.g., a CNN trained on EMNIST)
    model = load_model(model_path)
    return model


def recognize_letters(model, letters):
    predictions = model.predict(letters)
    # Map predictions to corresponding alphabet letters
    recognized_text = ''.join([chr(np.argmax(pred) + ord('A')) for pred in predictions])
    return recognized_text


def main(image_path, model_path):
    # Preprocess the image
    thresh_image = preprocess_image(image_path)
    # Find bounding boxes for each letter
    letter_boxes = find_letter_boxes(thresh_image)
    # Extract and resize letter images
    letters = extract_and_resize_letters(thresh_image, letter_boxes)
    # Load the pre-trained recognition model
    model = load_recognition_model(model_path)
    # Recognize the letters
    recognized_text = recognize_letters(model, letters)
    # Output the recognized text
    print("Recognized Text:", recognized_text)


if __name__ == '__main__':
    # Example usage
    image_path = 'path/to/your/image.jpg'  # Provide the image file path
    model_path = 'path/to/your/model.h5'   # Provide the model file path
    main(image_path, model_path)

