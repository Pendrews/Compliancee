import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pandas as pd
from spellchecker import SpellChecker
from pdf2image import convert_from_path

# Function to preprocess the image for OCR
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding
    _, thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optionally, deskew the image if it's skewed
    coords = np.column_stack(np.where(thresholded > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresholded.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresholded, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save the processed image
    processed_image_path = "processed_image.png"
    cv2.imwrite(processed_image_path, deskewed)

    return processed_image_path

# Function to extract text using OCR on the preprocessed image
def extract_text_from_image(image_path):
    processed_image_path = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine with page segmentation mode 6
    text = pytesseract.image_to_string(Image.open(processed_image_path), config=custom_config)
    return text

# Function to correct text using spell checker
def correct_text(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = [spell.correction(word) if word.isalpha() else word for word in words]  # Only correct alphabetical words
    return ' '.join(corrected_words)

# Function to extract fields from a form using predefined regions
def extract_fields_from_form(image_path, field_regions):
    extracted_data = {}
    for field_name, region in field_regions.items():
        cropped_image = crop_region(image_path, region)
        extracted_text = extract_text_from_image(cropped_image)
        corrected_text = correct_text(extracted_text)
        extracted_data[field_name] = corrected_text
    return extracted_data

# Function to crop specific regions for a given field
def crop_region(image_path, region):
    image = Image.open(image_path)
    cropped_image = image.crop(region)  # Crop the image to the specific region
    cropped_image_path = "cropped_image.png"
    cropped_image.save(cropped_image_path)
    return cropped_image_path

# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f'page_{i+1}.png')
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

# Function to write the extracted data to an Excel file
def write_to_excel(data, output_file):
    df = pd.DataFrame([data])  # Wrap the dict in a list for a single row
    if not os.path.exists(output_file):
        df.to_excel(output_file, index=False)
    else:
        existing_df = pd.read_excel(output_file)
        new_df = pd.concat([existing_df, df], ignore_index=True)
        new_df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

# Main function to process a PDF form and extract specific fields
def extract_data_from_pdf_form(pdf_path, output_excel_file, field_regions, output_image_folder='images'):
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    # Convert PDF to images
    image_paths = convert_pdf_to_images(pdf_path, output_image_folder)

    # For each image (page), extract fields
    for image_path in image_paths:
        extracted_data = extract_fields_from_form(image_path, field_regions)
        write_to_excel(extracted_data, output_excel_file)

# Define the regions where each field is located (top-left x, top-left y, bottom-right x, bottom-right y)
field_regions = {
    "Name": (50, 100, 400, 150),  # Example coordinates for Name field
    "Date of Birth": (50, 200, 400, 250),  # Example coordinates for Date of Birth field
    "Address": (50, 300, 600, 400),  # Example coordinates for Address field
}

# Run the script
if __name__ == "__main__":
    pdf_file_path = 'hand_filled_form.pdf'  # Path to the input PDF file
    output_excel_file = 'extracted_data.xlsx'  # Path to save the extracted Excel file

    extract_data_from_pdf_form(pdf_file_path, output_excel_file, field_regions)
