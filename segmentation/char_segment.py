import cv2

def segment_characters(word_img):
    """
    Tách ảnh của một từ thành các ký tự riêng biệt.
    """
    contours, _ = cv2.findContours(word_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    characters = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w * h > 100: 
            char_img = word_img[y:y+h, x:x+w]
            characters.append(char_img)
    
    return characters
