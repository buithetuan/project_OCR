import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data():
    # Đường dẫn đến thư mục dữ liệu
    training_data_dir = 'data_set/training_data'
    testing_data_dir = 'data_set/testing_data'
    
    # Đọc nhãn từ các file CSV
    train_labels_df = pd.read_csv('labels/training_labels.csv')
    test_labels_df = pd.read_csv('labels/testing_labels.csv')
    
    # Dữ liệu huấn luyện
    X_train = []
    y_train = []
    
    for _, row in train_labels_df.iterrows():
        image_path = os.path.join(training_data_dir, str(row['label']), row['image_name'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Thay đổi kích thước ảnh nếu cần
        X_train.append(img)
        y_train.append(row['label'])
    
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 28, 28, 1)  # Định dạng lại ảnh cho phù hợp với Keras
    X_train = X_train.astype('float32') / 255.0  # Chuẩn hóa ảnh
    
    # Mã hóa nhãn (chuyển nhãn thành dạng one-hot)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, num_classes=36)  # Nếu bạn có 36 ký tự
    
    # Dữ liệu kiểm tra
    X_test = []
    y_test = []
    
    for _, row in test_labels_df.iterrows():
        image_path = os.path.join(testing_data_dir, str(row['label']), row['image_name'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Thay đổi kích thước ảnh nếu cần
        X_test.append(img)
        y_test.append(row['label'])
    
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 28, 28, 1)  # Định dạng lại ảnh cho phù hợp với Keras
    X_test = X_test.astype('float32') / 255.0  # Chuẩn hóa ảnh
    
    # Mã hóa nhãn cho dữ liệu kiểm tra
    y_test = label_encoder.transform(y_test)
    y_test = to_categorical(y_test, num_classes=36)
    
    return X_train, y_train, X_test, y_test
