# segmentation/word_segment.py
import cv2
import numpy as np

def segment_words(line_image):
    # Chuyển ảnh dòng sang nhị phân
    _, binary_image = cv2.threshold(line_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các contours trong ảnh nhị phân
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contours theo trục hoành (theo chiều từ trái sang phải)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Cắt các từ trong dòng
    word_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        word_image = line_image[y:y+h, x:x+w]
        word_images.append(word_image)
    
    return word_images
