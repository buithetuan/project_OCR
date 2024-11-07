# segmentation/line_segment.py

import cv2
import numpy as np

def segment_lines(image):
    # Chuyển ảnh sang ảnh nhị phân (nếu chưa làm)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Tìm các đường viền (contours) của các dòng văn bản
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các contours theo tọa độ Y để tách các dòng văn bản
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        line_image = image[y:y + h, x:x + w]
        lines.append(line_image)

    return lines