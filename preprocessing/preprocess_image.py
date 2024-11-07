import cv2
import numpy as np

def preprocess_image(image, target_size=(64, 64), is_character=False):
    """
    Tiền xử lý ảnh: Đọc ảnh, chuyển đổi sang grayscale, thay đổi kích thước và chuẩn hóa.
    
    Args:
    - image (str hoặc numpy.ndarray): Đường dẫn tới ảnh hoặc ảnh đã tải.
    - target_size (tuple): Kích thước đích của ảnh.
    - is_character (bool): Nếu là ký tự đơn lẻ, thêm chiều batch.

    Returns:
    - processed_image (numpy.ndarray): Ảnh đã qua tiền xử lý.
    """
    # Nếu đầu vào là đường dẫn, đọc ảnh; nếu không, giả định là numpy array
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh tại {image}")
    
    # Chuyển sang ảnh nhị phân
    _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Thay đổi kích thước ảnh
    resized_image = cv2.resize(thresh_image, target_size, interpolation=cv2.INTER_AREA)
    
    # Chuẩn hóa
    processed_image = resized_image.astype('float32') / 255.0
    
    # Thêm chiều nếu là ký tự đơn
    if is_character:
        processed_image = np.expand_dims(processed_image, axis=-1)  # Thêm chiều kênh màu
        processed_image = np.expand_dims(processed_image, axis=0)  # Thêm chiều batch

    return processed_image
