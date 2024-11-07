# preprocessing/deskew.py

import cv2
import numpy as np
import os
def deskew_image(image):
    # Check if the input is a valid image array
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Invalid image array: {image}")

    print(f"Deskwing image")
    # Tính toán góc xoay cần thiết
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated