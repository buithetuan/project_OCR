import cv2

def segment_words(line_image):
    _, binary_image = cv2.threshold(line_image, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    word_images = []
    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        word_image = line_image[y:y+h, x:x+w]
        word_images.append(word_image)
    
    return word_images
