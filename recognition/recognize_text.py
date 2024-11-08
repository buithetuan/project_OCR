from segmentation.line_segment import segment_lines
from segmentation.word_segment import segment_words
from segmentation.char_segment import segment_characters
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.preprocess_image import preprocess_image
import os
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    return load_model(model_path)

def preprocess_input_image(image_path):
    """
    Tiền xử lý ảnh đầu vào để phù hợp với mô hình đã huấn luyện.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at path {image_path}")
        return None 
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to read image at path {image_path}")
        return None 
    
    img = cv2.resize(img, (28, 28))
    img = preprocess_image(img)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

def predict_character(char_img, model):
    """
    Hàm dự đoán ký tự từ ảnh đầu vào.
    """
    predictions = model.predict(char_img)
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted character: {predicted_class[0]}")    
    return predicted_class[0]

def recognize_text_from_image(image_path, model):
    """
    Hàm nhận diện văn bản từ một ảnh chứa nhiều ký tự.
    """
    img = preprocess_input_image(image_path)

    if img is None:
        return " "
    
    img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1] 

    if len(img.shape) == 3:
        img = img.squeeze()

    plt.imshow(img, cmap='gray')
    plt.title("Processed Image")
    plt.show()

    img = img.astype(np.uint8)

    characters = []

    lines = segment_lines(img)
    print(f"Detected lines: {len(lines)}")
    for line in lines:
        words = segment_words(line)
        print(f"Detected words in line: {len(words)}") 
        for word in words:
            chars = segment_characters(word)
            print(f"Detected characters in word: {len(chars)}")
            for char_img in chars:
                char = predict_character(char_img, model)
                characters.append(str(char))

    return ''.join(characters)
