import cv2
import numpy as np
from recognition.recognize_text import load_trained_model, extract_text_from_image
from preprocessing.preprocess_image import preprocess_image
from segmentation.line_segment import segment_lines
from segmentation.char_segment import segment_characters

def main():
    image_path = 'data_test/0Jl54.png'
    model_path = 'recognition/ocr_model.h5'
    
    # Tải mô hình đã huấn luyện
    model = load_trained_model(model_path)

    # Tiền xử lý ảnh đầu vào (giữ ở dạng nhị phân)
    processed_image = preprocess_image(image_path, target_size=(64, 64))

    # Phân đoạn các dòng trong ảnh
    line_images = segment_lines(processed_image)

    for line_img in line_images:
        # Phân đoạn ký tự trong từng dòng
        char_images = segment_characters(line_img)
        
        text = ""
        for char_img in char_images:
            # Tiền xử lý từng ký tự, dùng `is_character=True`
            processed_char = preprocess_image(char_img, target_size=(64, 64), is_character=True)
            
            # Dự đoán văn bản từ ký tự
            char_text = extract_text_from_image(processed_char, model)
            text += char_text

        print(f'Detected text: {text}')

if __name__ == '__main__':
    main()
