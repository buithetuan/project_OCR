import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()

    # Thêm các lớp vào mô hình (CNN)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Chuyển ảnh 2D thành 1D
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    
    # Lớp Output: số lớp bằng với số ký tự
    model.add(layers.Dense(36, activation='softmax'))

    # Tóm tắt mô hình
    model.summary()
    
    return model
