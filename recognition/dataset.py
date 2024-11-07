import os
import cv2
import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, data_dir, labels_file):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.label_map = self.create_label_map_from_file()

    def create_label_map_from_file(self):
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"The labels file {self.labels_file} does not exist.")
        
        df = pd.read_csv(self.labels_file)
        unique_labels = df['label'].unique()
        label_to_index = {label: index for index, label in enumerate(unique_labels)}
        label_map = {row['image_name']: label_to_index[row['label']] for _, row in df.iterrows()}
        return label_map

    def load_images_from_each_class(self):
        train_images = []
        train_labels = []
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        train_images.append(img)
                        train_labels.append(self.label_map.get(img_name, 'unknown'))
        return train_images, train_labels

    def load_all_test_images(self):
        test_images = []
        test_labels = []
        
        # Duyệt qua từng thư mục con trong thư mục testing_data
        for class_dir in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_dir)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        test_images.append(img)
                        test_labels.append(class_dir)  # Sử dụng tên thư mục làm nhãn
        return test_images, test_labels