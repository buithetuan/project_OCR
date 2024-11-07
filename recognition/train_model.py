# recognition/train_model.py

from .dataset import DataLoader
from .model import OCRModel

def train_model():
    train_data_dir = 'data_set/training_data'
    test_data_dir = 'data_set/testing_data'
    train_labels_file = 'labels/training_labels.csv'
    test_labels_file = 'labels/testing_labels.csv'

    print("Starting the training model process.")

    train_loader = DataLoader(train_data_dir, train_labels_file)
    test_loader = DataLoader(test_data_dir, test_labels_file)

    print("Loading training images and labels.")
    train_images, train_labels = train_loader.load_images_from_each_class()
    print("Loading testing images and labels.")
    test_images, test_labels = test_loader.load_all_test_images()

    num_classes = len(train_loader.create_label_map_from_file())

    ocr_model = OCRModel(num_classes=num_classes)

    print("Training the model.")
    ocr_model.model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

    loss, accuracy = ocr_model.model.evaluate(test_images, test_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    print("Saving the model.")
    ocr_model.model.save("recognition/ocr_model.h5")
    print("Model saved to recognition/ocr_model.h5")

if __name__ == "__main__":
    train_model()