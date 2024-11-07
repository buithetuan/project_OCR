# segmentation/char_segment.py
import cv2
import numpy as np

def segment_characters(word_image):
    # Chuyển ảnh từ sang nhị phân
    _, binary_image = cv2.threshold(word_image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các contours trong ảnh nhị phân
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contours theo trục hoành (theo chiều từ trái sang phải)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    # Cắt các ký tự trong từ
    character_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        char_image = word_image[y:y+h, x:x+w]
        character_images.append(char_image)
    
    return character_images
