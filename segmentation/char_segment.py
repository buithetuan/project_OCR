# segmentation/char_segment.py

import cv2
import numpy as np

def segment_chars(word_image):
    # Chuyển ảnh từ thành ảnh nhị phân
    _, binary_image = cv2.threshold(word_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các contours của các ký tự trong từ
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = word_image[y:y + h, x:x + w]
        chars.append(char_image)

    return chars
