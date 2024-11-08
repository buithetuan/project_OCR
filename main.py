from recognition.recognize_text import load_trained_model, recognize_text_from_image

def main():
    image_path = 'data_test/0Jl54.png'
    
    model_path = 'recognition/ocr_model.h5'

    model = load_trained_model(model_path)

    try:
        text = recognize_text_from_image(image_path, model)
        print(f"Detected text: {text}")
    except ValueError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
