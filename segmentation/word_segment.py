# segmentation/word_segment.py

import cv2
import numpy as np

def segment_words(line_image):
    # Chuyển ảnh dòng văn bản thành ảnh nhị phân
    _, binary_image = cv2.threshold(line_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các contours của các từ trong dòng
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        word_image = line_image[y:y + h, x:x + w]
        words.append(word_image)

    return words
