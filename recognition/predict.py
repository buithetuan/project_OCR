import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.binarize import binarize_image
from preprocessing.deskew import deskew_image
from preprocessing.denoise import denoise_image

class OCRModelInference:
    def __init__(self, model_path, label_map_file):
        # Tải mô hình đã huấn luyện từ file
        self.model = load_model(model_path)
        
        # Tải label map để giải mã kết quả dự đoán
        self.label_map = self.load_label_map(label_map_file)
        self.index_to_label = {v: k for k, v in self.label_map.items()}
    
    def load_label_map(self, label_map_file):
        import pandas as pd
        df = pd.read_csv(label_map_file)
        label_to_index = {label: index for index, label in enumerate(df['label'].unique())}
        return label_to_index

    def preprocess_image(self, image_path):
        # Đọc ảnh và tiền xử lý (Resize, Binarize, Deskew, Denoise)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("Không thể đọc ảnh từ đường dẫn này.")
        
        # Resize về kích thước mô hình yêu cầu (ví dụ: 32x32)
        img = cv2.resize(img, (32, 32))

        # Áp dụng tiền xử lý: Binarize, Deskew và Denoise
        img = binarize_image(img)
        img = deskew_image(img)
        img = denoise_image(img)

        # Thêm chiều cho ảnh: (batch_size, height, width, channels)
        img = np.expand_dims(img, axis=-1)  # Thêm chiều màu (grayscale)
        img = np.expand_dims(img, axis=0)  # Thêm chiều batch_size

        # Chuẩn hóa ảnh
        img = img.astype('float32') / 255.0
        
        return img

    def predict_text(self, image_path):
        # Tiền xử lý ảnh đầu vào
        preprocessed_img = self.preprocess_image(image_path)

        # Dự đoán từ mô hình
        predictions = self.model.predict(preprocessed_img)

        # Chuyển đổi kết quả dự đoán thành nhãn
        predicted_class = np.argmax(predictions, axis=-1)[0]

        # Giải mã nhãn thành văn bản
        text = self.index_to_label[predicted_class]
        return text

# Sử dụng mô hình để nhận diện văn bản từ ảnh
ocr_inference = OCRModelInference(model_path="recognition/ocr_model.h5", label_map_file="labels/training_labels.csv")

image_path = 'data_test/0Jl54.png'
predicted_text = ocr_inference.predict_text(image_path)

print(f"Predicted text: {predicted_text}")
