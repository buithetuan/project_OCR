# main.py

from preprocessing import deskew, denoise, binarize
from segmentation import line_segment, word_segment, char_segment
from recognition.predict import OCRPredictor
from postprocessing import spell_check, grammar_check

def main(image_path):
    # Tiền xử lý ảnh đầu vào
    print("Running preprocessing...")
    image = deskew.deskew(image_path)
    image = denoise.denoise(image)
    image = binarize.binarize(image)
    
    # Phân đoạn ảnh thành các ký tự
    print("Running segmentation...")
    lines = line_segment.segment_lines(image)
    words = [word_segment.segment_words(line) for line in lines]
    chars = [char_segment.segment_chars(word) for word in words]
    
    # Khởi tạo predictor để nhận dạng ký tự
    ocr_predictor = OCRPredictor(model_path="recognition/ocr_model.h5")
    
    # Nhận dạng từng ký tự và ghép thành văn bản
    print("Running recognition...")
    recognized_text = ""
    for line in chars:
        for word in line:
            for char_image in word:
                predicted_char = ocr_predictor.predict(char_image)
                recognized_text += predicted_char
            recognized_text += " "  # Khoảng trống giữa các từ
        recognized_text += "\n"  # Xuống dòng giữa các dòng

    # Hậu xử lý văn bản nhận dạng
    print("Running postprocessing...")
    corrected_text = spell_check.spell_check(recognized_text)
    final_text = grammar_check.grammar_check(corrected_text)
    
    print("Recognized Text:")
    print(final_text)

if __name__ == "__main__":
    # Đường dẫn đến ảnh đầu vào cần OCR
    input_image_path = "path/to/your/image.png"
    main(input_image_path)
