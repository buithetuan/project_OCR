# segmentation/line_segment.py
import cv2
import numpy as np

def segment_lines(image):
    # Chuyển ảnh sang ảnh nhị phân
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các contours trong ảnh nhị phân
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contours theo trục tung (theo chiều từ trên xuống dưới)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    # Cắt các dòng văn bản
    line_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        line_image = image[y:y+h, x:x+w]
        line_images.append(line_image)
    
    return line_images
