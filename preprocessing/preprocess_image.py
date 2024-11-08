import cv2
import numpy as np

def preprocess_image(image, denoise=True, deskew=True, binarize=True):
    """
    Tiền xử lý ảnh: Khử nhiễu, xoay thẳng, và chuyển thành ảnh nhị phân.
    
    :param image: ảnh đầu vào
    :param denoise: Có thực hiện khử nhiễu không
    :param deskew: Có thực hiện xoay thẳng không
    :param binarize: Có chuyển ảnh thành nhị phân không
    :return: ảnh đã được tiền xử lý
    """
    if denoise:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    if deskew:
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if binarize:
        _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV) 

    return image
