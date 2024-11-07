# Main Flow
1. Input → Ảnh chứa text
2. Preprocessing → Chuẩn hóa ảnh (resize, normalize, threshold)
3. Feature Extraction → CNN trích xuất đặc trưng từ ảnh
4. Sequence Processing → LSTM xử lý sequence của features
5. Text Recognition → Decode sequence thành text
6. Output → Text được nhận dạng


# OCR System Implementation Guide
## Overview
Hệ thống OCR (Optical Character Recognition) đơn giản để nhận dạng văn bản tiếng Việt từ ảnh, sử dụng CNN + LSTM.

## Prepare data
### Format data

- **Image**: Image file (.png) chứa văn bản
- **Label**: Text file (.txt) chứa nội dung text tương ứng

### Create dictionary
1. Thu thập tất cả ký tự unique từ labels
2. Tạo mapping giữa ký tự và index
3. Lưu thành file JSON
#### Ví dụ vocabulary.json
```json
{
    "char_to_idx": {
        "a": 0,
        "b": 1,
        ...
    },
    "idx_to_char": {
        "0": "a",
        "1": "b",
        ...
    }
}```

## Model Architecture

CNN -> BatchNorm -> MaxPool -> LSTM -> Dense

## Training Parameters

PARAMS = {
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 5e-3,
    'image_size': (128, 512)
}
