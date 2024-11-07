# recognition/predict.py

import tensorflow as tf
import cv2
import numpy as np

class OCRPredictor:
    def __init__(self, model_path):
        # Tải mô hình đã huấn luyện
        self.model = tf.keras.models.load_model(model_path)
        self.char_list = "0123456789abcdefghijklmnopqrstuvwxyz"  # Danh sách các ký tự trong mô hình

    def preprocess_image(self, image):
        # Tiền xử lý ảnh cho phù hợp với mô hình (chuyển sang grayscale, resize và chuẩn hóa)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Chuyển sang grayscale
        image = cv2.resize(image, (32, 32))  # Resize ảnh thành 32x32
        image = image / 255.0  # Chuẩn hóa ảnh
        image = np.expand_dims(image, axis=-1)  # Thêm chiều kênh cho mô hình (32, 32, 1)
        image = np.expand_dims(image, axis=0)  # Thêm chiều batch (1, 32, 32, 1)
        return image

    def predict(self, image):
        # Tiền xử lý ảnh trước khi đưa vào mô hình
        processed_image = self.preprocess_image(image)
        
        # Dự đoán ký tự
        prediction = self.model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Chọn lớp có xác suất cao nhất

        # Chuyển đổi dự đoán thành ký tự
        predicted_char = self.char_list[predicted_class]
        return predicted_char
