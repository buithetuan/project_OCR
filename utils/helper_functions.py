import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, labels_file, max_images_per_class=100):
    """
    Tải dữ liệu từ thư mục, chỉ lấy số lượng ảnh tối đa cho mỗi lớp ký tự.
    
    :param data_dir: Thư mục chứa dữ liệu
    :param labels_file: File chứa nhãn cho các ảnh
    :param max_images_per_class: Số ảnh tối đa cho mỗi ký tự (dùng cho train và test)
    :return: Dữ liệu và nhãn đã được phân loại
    """
    # Đọc nhãn từ file CSV
    labels_df = pd.read_csv(labels_file)
    
    # Tạo dictionary để lưu ảnh cho từng ký tự
    data = []
    labels = []
    
    # Lặp qua tất cả các nhãn (tương ứng với các ký tự)
    for label in labels_df['label'].unique():
        # Lấy tất cả các ảnh có nhãn này
        label_images = labels_df[labels_df['label'] == label]
        
        # Chỉ lấy số lượng ảnh tối đa cho mỗi ký tự
        selected_images = label_images.sample(n=min(max_images_per_class, len(label_images)))
        
        # Duyệt qua các ảnh đã chọn
        for _, row in selected_images.iterrows():
            image_path = os.path.join(data_dir, row['filename'])
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dưới dạng ảnh xám
            
            # Tiền xử lý ảnh nếu cần (ví dụ: xoay thẳng, khử nhiễu, nhị phân...)
            image = preprocess_image(image)
            
            # Thêm ảnh và nhãn vào dữ liệu
            data.append(image)
            labels.append(row['label'])
    
    # Chuyển danh sách dữ liệu thành mảng numpy
    data = np.array(data)
    labels = np.array(labels)
    
    # Tiến hành mã hóa nhãn (Label Encoding)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    # Chuyển nhãn thành dạng one-hot encoding
    labels = to_categorical(labels)
    
    # Chuẩn hóa dữ liệu (có thể thay đổi tùy theo yêu cầu của mô hình)
    data = data.astype('float32') / 255.0
    
    # Thêm chiều cho dữ liệu ảnh (bắt buộc đối với CNN)
    data = np.expand_dims(data, axis=-1)
    
    return data, labels

def preprocess_image(image):
    """
    Tiền xử lý ảnh: chuyển ảnh thành nhị phân và chuẩn bị cho phân đoạn.
    
    :param image: ảnh đầu vào
    :return: ảnh đã tiền xử lý
    """
    # Chuyển ảnh thành ảnh nhị phân (có thể thay đổi tùy theo yêu cầu)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    return binary
