import cv2

def segment_lines(image):
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])

    line_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        line_image = image[y:y+h, x:x+w]
        line_images.append(line_image)
    
    return line_images
