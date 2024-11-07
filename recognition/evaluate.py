import tensorflow as tf
from utils.helper_functions import load_data

def evaluate_model():
    # Tải dữ liệu kiểm tra với giới hạn 10 ảnh mỗi ký tự cho test
    x_test, y_test = load_data('data_set/testing_data', 'labels/testing_labels.csv', max_images_per_class=10)
    
    model = tf.keras.models.load_model('ocr_model.h5')
    
    # Đánh giá mô hình trên dữ liệu kiểm tra
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print(f"Loss: {loss}, Accuracy: {accuracy}")
