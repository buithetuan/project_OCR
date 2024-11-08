import os
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from preprocessing.preprocess_image import preprocess_image  # Import hàm preprocess_image

def load_data(training_data_dir='data_set/training_data', testing_data_dir='data_set/testing_data'):
    """
    Hàm này tải dữ liệu huấn luyện và kiểm tra từ các thư mục đã định sẵn.
    
    :param training_data_dir: Thư mục chứa dữ liệu huấn luyện
    :param testing_data_dir: Thư mục chứa dữ liệu kiểm tra
    :return: X_train, y_train, X_test, y_test
    """
    train_labels_df = pd.read_csv('labels/training_labels.csv')
    test_labels_df = pd.read_csv('labels/testing_labels.csv')
    
    # Xử lý dữ liệu huấn luyện
    X_train = []
    y_train = []
    
    for _, row in train_labels_df.iterrows():
        image_path = os.path.join(training_data_dir, str(row['label']), row['image_name'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (28, 28))
        
        # Tiến hành tiền xử lý ảnh
        img = preprocess_image(img) 
        
        X_train.append(img)
        y_train.append(row['label'])
    
    X_train = np.array(X_train)
    X_train = X_train.reshape(-1, 28, 28, 1)  
    X_train = X_train.astype('float32') / 255.0  
    
    # Mã hóa nhãn
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_train = to_categorical(y_train, num_classes=36)  

    # Xử lý dữ liệu kiểm tra
    X_test = []
    y_test = []
    
    for _, row in test_labels_df.iterrows():
        image_path = os.path.join(testing_data_dir, str(row['label']), row['image_name'])
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        img = cv2.resize(img, (28, 28))
        
        # Tiến hành tiền xử lý ảnh
        img = preprocess_image(img)  
        
        X_test.append(img)
        y_test.append(row['label'])
    
    X_test = np.array(X_test)
    X_test = X_test.reshape(-1, 28, 28, 1)  
    X_test = X_test.astype('float32') / 255.0  
    
    y_test = label_encoder.transform(y_test)
    y_test = to_categorical(y_test, num_classes=36)  

    return X_train, y_train, X_test, y_test
