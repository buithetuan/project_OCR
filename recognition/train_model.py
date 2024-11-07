import numpy as np
from PIL import Image
from .dataset import DataLoader
from .model import OCRModel

def train_model():
    # Đường dẫn đến dữ liệu và nhãn
    train_data_dir = 'data_set/training_data'
    test_data_dir = 'data_set/testing_data'
    train_labels_file = 'labels/training_labels.csv'
    test_labels_file = 'labels/testing_labels.csv'

    print("Starting the training model process.")

    # Tải dữ liệu huấn luyện và kiểm tra
    train_loader = DataLoader(train_data_dir, train_labels_file)
    test_loader = DataLoader(test_data_dir, test_labels_file)

    # Kiểm tra unique labels trong train_labels và test_labels
    print("Unique labels in the CSV file:", np.unique(train_loader.label_map))
    print("Unique labels in the CSV file:", np.unique(test_loader.label_map))

    print("Generated label_map (first 10 items):", list(train_loader.label_map.items())[:10])

    print("Loading training images and labels.")
    train_images, train_labels = train_loader.load_images_from_each_class()

    print("Loading testing images and labels.")
    test_images, test_labels = test_loader.load_all_test_images()

    # Kiểm tra nhãn trong train_labels và test_labels
    print("Train labels:", train_labels[:10])
    print("Test labels:", test_labels[:10])

    # Chuyển nhãn thành kiểu số nguyên và gán -1 cho nhãn không hợp lệ
    train_labels = np.array([train_loader.label_map.get(label, -1) for label in train_labels])
    test_labels = np.array([test_loader.label_map.get(label, -1) for label in test_labels])

    # Loại bỏ các nhãn -1
    valid_train_indices = [i for i, label in enumerate(train_labels) if label != -1]
    valid_test_indices = [i for i, label in enumerate(test_labels) if label != -1]

    train_images = [train_images[i] for i in valid_train_indices]
    train_labels = [train_labels[i] for i in valid_train_indices]

    test_images = [test_images[i] for i in valid_test_indices]
    test_labels = [test_labels[i] for i in valid_test_indices]

    print("Updated train_labels:", train_labels[:10])
    print("Updated test_labels:", test_labels[:10])

    # Tiền xử lý ảnh: chuyển đổi sang grayscale, thay đổi kích thước và chuẩn hóa
    train_images_resized = []
    for img in train_images:
        img = Image.open(img).convert('L').resize((32, 32))  # Chuyển thành grayscale và resize
        train_images_resized.append(np.array(img))

    test_images_resized = []
    for img in test_images:
        img = Image.open(img).convert('L').resize((32, 32))  # Chuyển thành grayscale và resize
        test_images_resized.append(np.array(img))

    # Chuyển thành numpy array và chuẩn hóa
    train_images_resized = np.array(train_images_resized).astype('float32') / 255.0
    test_images_resized = np.array(test_images_resized).astype('float32') / 255.0

    # Reshape thành định dạng (batch_size, 32, 32, 1)
    train_images_resized = np.reshape(train_images_resized, (-1, 32, 32, 1))
    test_images_resized = np.reshape(test_images_resized, (-1, 32, 32, 1))

    # Số lớp phân loại
    num_classes = len(train_loader.label_map)
    print(f"Number of classes: {num_classes}")

    # Tạo và huấn luyện mô hình
    ocr_model = OCRModel(num_classes=num_classes)

    # Biên dịch mô hình
    ocr_model.model.compile(
        optimizer='adam',  # Trình tối ưu Adam
        loss='sparse_categorical_crossentropy',  # Hàm mất mát cho phân loại nhiều lớp (dùng khi nhãn là số nguyên)
        metrics=['accuracy']  # Chỉ số accuracy để đánh giá mô hình
    )

    print("Training the model.")
    history = ocr_model.model.fit(
        train_images_resized,
        train_labels,
        epochs=10,
        batch_size=32,
        validation_data=(test_images_resized, test_labels),
        verbose=1,
    )

    # Đánh giá mô hình
    loss, accuracy = ocr_model.model.evaluate(test_images_resized, test_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Lưu mô hình
    print("Saving the model.")
    ocr_model.model.save("recognition/ocr_model.h5")
    print("Model saved to recognition/ocr_model.h5")


if __name__ == "__main__":
    train_model()
