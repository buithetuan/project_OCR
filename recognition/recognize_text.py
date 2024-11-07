# recognition/recognize_text.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from segmentation.line_segment import segment_lines
from segmentation.word_segment import segment_words
from segmentation.char_segment import segment_characters
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

def predict_text_from_image(image_path, model):
    # Đọc ảnh và chuyển sang grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # Thay đổi kích thước ảnh phù hợp với mô hình
    image = np.reshape(image, (1, 28, 28, 1))  # Định dạng lại ảnh cho mô hình
    image = image.astype('float32') / 255.0  # Chuẩn hóa ảnh

    # Dự đoán ký tự từ ảnh
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)  # Lấy chỉ số lớp với xác suất cao nhất

    # Đổi chỉ số lớp thành nhãn (ví dụ 36 ký tự)
    label_map = '0123456789abcdefghijklmnopqrstuvwxyz'
    return label_map[predicted_class[0]]

def extract_text_from_image(image_path, model):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Phân đoạn thành các dòng
    line_images = segment_lines(image)

    # Dự đoán văn bản từ các dòng
    full_text = ''
    for line_image in line_images:
        # Phân đoạn mỗi dòng thành các từ
        word_images = segment_words(line_image)

        # Dự đoán văn bản từ các từ
        for word_image in word_images:
            # Phân đoạn mỗi từ thành các ký tự
            character_images = segment_characters(word_image)

            # Dự đoán văn bản từ các ký tự
            word = ''
            for char_img in character_images:
                char_img_resized = cv2.resize(char_img, (28, 28))  # Điều chỉnh kích thước
                char_img_resized = np.reshape(char_img_resized, (1, 28, 28, 1))  # Định dạng lại ảnh
                char_img_resized = char_img_resized.astype('float32') / 255.0  # Chuẩn hóa ảnh

                # Dự đoán ký tự
                prediction = model.predict(char_img_resized)
                predicted_class = np.argmax(prediction, axis=1)
                label_map = '0123456789abcdefghijklmnopqrstuvwxyz'
                word += label_map[predicted_class[0]]
            
            full_text += word + ' '

    return full_text
