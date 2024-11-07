# preprocessing/binarize.py
import cv2
def binarize_image(image):
    # Chuyển thành ảnh nhị phân bằng phương pháp Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image
