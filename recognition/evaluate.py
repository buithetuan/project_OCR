import tensorflow as tf
from utils.helper_functions import load_data

def evaluate_model():
    x_test, y_test = load_data('data_set/testing_data', 'labels/testing_labels.csv', max_images_per_class=10)
    
    model = tf.keras.models.load_model('ocr_model.h5')
    
    loss, accuracy = model.evaluate(x_test, y_test)
    
    print(f"Loss: {loss}, Accuracy: {accuracy}")
