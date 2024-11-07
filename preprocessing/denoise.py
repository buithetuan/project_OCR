# preprocessing/denoise.py

import cv2

def denoise_image(image):
    # Khử nhiễu bằng phương pháp lọc Gaussian
    return cv2.GaussianBlur(image, (5, 5), 0)
