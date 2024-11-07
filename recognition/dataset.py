import cv2
import numpy as np
from preprocessing.binarize import binarize_image
from preprocessing.deskew import deskew_image
from preprocessing.denoise import denoise_image
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.label_map = self.create_label_map_from_file()

    def create_label_map_from_file(self):
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"The labels file {self.labels_file} does not exist.")
        
        df = pd.read_csv(self.labels_file)
        
        # Loại bỏ khoảng trắng thừa trong cột 'label'
        df['label'] = df['label'].str.strip()

        # Kiểm tra các giá trị trong cột 'label' để đảm bảo chúng hợp lệ
        print("Unique labels in the CSV file:", df['label'].unique())
        
        # Tạo ánh xạ từ nhãn thành chỉ số
        unique_labels = df['label'].unique()
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        
        # Tạo label_map và kiểm tra sự tồn tại của mỗi tệp ảnh trong thư mục con tương ứng với label
        label_map = {}
        for _, row in df.iterrows():
            img_name = row['image_name']
            label = row['label']
            
            # Tạo đường dẫn đầy đủ tới thư mục con tương ứng với nhãn (label)
            label_dir = os.path.join(self.data_dir, label)  # Ví dụ: 'testing_data/Z'
            
            if os.path.exists(os.path.join(label_dir, img_name)):
                label_map[img_name] = label_to_index[label]
            else:
                print(f"Warning: {img_name} not found in {label_dir}")
        
        # In label_map để kiểm tra
        print("Generated label_map (first 10 items):", {k: label_map[k] for k in list(label_map)[:10]})
        
        return label_map

    def load_images_from_each_class(self):
        train_images = []
        train_labels = []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                count = 0
                for img_name in os.listdir(class_path):
                    if count >= 1000:
                        break
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize ảnh về kích thước đồng nhất (ví dụ: 32x32)
                        img = cv2.resize(img, (32, 32))
                        # Tiền xử lý ảnh: Binarize, Deskew và Denoise
                        img = binarize_image(img)  # Chuyển ảnh thành ảnh nhị phân
                        img = deskew_image(img)    # Deskew ảnh
                        img = denoise_image(img)   # Denoise ảnh

                        train_images.append(img)
                        # Lấy nhãn từ label_map, nếu không có nhãn thì gán nhãn mặc định (ví dụ: 0)
                        label = self.label_map.get(img_name, -1)  # Sử dụng -1 nếu không tìm thấy nhãn
                        train_labels.append(label)
                        count += 1
        return train_images, train_labels

    def load_all_test_images(self):
        test_images = []
        test_labels = []
        
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Resize ảnh về kích thước đồng nhất (ví dụ: 32x32)
                        img = cv2.resize(img, (32, 32))
                        # Tiền xử lý ảnh: Binarize, Deskew và Denoise
                        img = binarize_image(img)
                        img = deskew_image(img)
                        img = denoise_image(img)

                        test_images.append(img)
                        # Lấy nhãn từ label_map cho bộ kiểm thử
                        label = self.label_map.get(img_name, -1)  # Sử dụng -1 nếu không tìm thấy nhãn
                        
                        if label == -1:
                            print(f"Warning: {img_name} not found in label_map for test data!")
                        
                        test_labels.append(label)
        return test_images, test_labels
