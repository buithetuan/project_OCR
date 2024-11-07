from recognition.train import train_model
from recognition.evaluate import evaluate_model

def main():
    # Huấn luyện mô hình
    train_model()
    
    # Đánh giá mô hình
    evaluate_model()

if __name__ == "__main__":
    main()
