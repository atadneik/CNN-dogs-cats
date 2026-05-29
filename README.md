# CNN Cats vs Dogs - Phân loại ảnh Chó và Mèo

Dự án sử dụng **Convolutional Neural Network (CNN)** kết hợp **Transfer Learning** với model **ResNet-50** để phân loại ảnh chó và mèo. Đây là bài tập nhóm (Group 5) thuộc Đại học Giao thông Vận tải TP.HCM.

---

## Mục tiêu

Xây dựng hệ thống phân loại ảnh nhị phân (Cat vs Dog) đạt độ chính xác >= 95% trên tập test, sử dụng kỹ thuật Transfer Learning để tận dụng model đã được huấn luyện sẵn trên tập ImageNet (1.2 triệu ảnh, 1000 classes).

---

## Lý thuyết nền tảng

### CNN là gì?

Convolutional Neural Network là kiến trúc mạng neural chuyên xử lý dữ liệu dạng lưới (grid-like data), đặc biệt là ảnh. Máy tính nhìn ảnh dưới dạng ma trận số (0-255 cho mỗi pixel), CNN giúp máy trích xuất đặc trưng và "hiểu" nội dung ảnh.

### Tại sao dùng CNN thay vì MLP (Multi-Layer Perceptron)?

| Vấn đề của MLP | CNN giải quyết bằng cách |
|----------------|--------------------------|
| Quá nhiều parameters khi ảnh lớn → dễ overfitting | Parameter sharing: cùng 1 filter dùng cho toàn bộ ảnh |
| Mất thông tin không gian giữa các pixel | Giữ nguyên cấu trúc 2D của ảnh |
| Không nhận diện được vật thể khi dịch chuyển vị trí | Translation invariance nhờ convolution + pooling |

### Các layer chính trong CNN

1. **Convolutional Layer**: Dùng kernel/filter (ma trận nhỏ, ví dụ 3x3) trượt qua ảnh để trích xuất đặc trưng (cạnh, góc, texture). Mỗi filter học một loại feature khác nhau.

2. **Activation - ReLU**: Hàm kích hoạt `f(x) = max(0, x)`. Loại bỏ giá trị âm, thêm tính phi tuyến (non-linearity) cho model, giúp học được các pattern phức tạp.

3. **Pooling Layer (Max Pooling)**: Giảm kích thước spatial bằng cách giữ giá trị lớn nhất trong mỗi vùng (thường 2x2). Giúp giảm computation và tạo tính bất biến với dịch chuyển nhỏ.

4. **Batch Normalization**: Chuẩn hóa input của mỗi layer về mean=0, std=1. Giúp training nhanh hơn, ổn định hơn, và cho phép dùng learning rate lớn hơn.

5. **Dropout**: Tắt ngẫu nhiên một tỷ lệ neuron trong quá trình training (ví dụ 50%). Buộc model không phụ thuộc vào bất kỳ neuron cụ thể nào → chống overfitting hiệu quả.

6. **Fully Connected (Dense) Layer**: Kết nối tất cả neurons từ layer trước, tổng hợp các features đã trích xuất để đưa ra quyết định phân loại.

7. **Softmax**: Chuyển output thành xác suất cho mỗi class (tổng = 1). Ví dụ: Cat: 0.97, Dog: 0.03.

### Luồng xử lý tổng quát

```
Input Image (224x224x3)
    ↓
[Convolution + ReLU → Pooling] × N lần
    ↓
Flatten (chuyển từ 2D → 1D vector)
    ↓
Fully Connected Layers
    ↓
Softmax → Output (Cat hoặc Dog)
```

### Transfer Learning

Thay vì train CNN từ đầu (cần hàng triệu ảnh + GPU mạnh + nhiều ngày), ta dùng model đã được train sẵn trên ImageNet và chỉ thay đổi phần classifier cuối cùng cho bài toán của mình.

**Lợi ích:**
- Tiết kiệm thời gian và tài nguyên tính toán
- Đạt accuracy cao ngay cả với dataset nhỏ
- Các layer đầu đã học được features tổng quát (cạnh, texture, hình dạng) có thể tái sử dụng

**ResNet-50** được chọn vì:
- 50 layers sâu nhưng vẫn train được nhờ skip connections (residual connections)
- Skip connections giải quyết vấn đề vanishing gradient khi mạng quá sâu
- Đạt top-5 error rate 3.57% trên ImageNet

---

## Dataset

- **Nguồn**: Kaggle Dogs vs Cats Competition
- **Tổng**: 25,000 ảnh (12,500 cat + 12,500 dog)
- **Phân chia** (tỷ lệ 70/15/15):

| Tập | Cats | Dogs | Tổng |
|-----|------|------|------|
| Train | 8,750 | 8,750 | 17,500 |
| Validation | 1,875 | 1,875 | 3,750 |
| Test | 1,875 | 1,875 | 3,750 |

- **Kích thước ảnh**: Resize về 224x224x3 (RGB)
- **Balanced**: 50% cat, 50% dog → không cần xử lý class imbalance

---

## Kiến trúc Model

```
ResNet50 (pre-trained trên ImageNet)
│   - Freeze các layer đầu (giữ nguyên features đã học)
│   - Fine-tune 20 layers cuối (cho phép điều chỉnh theo dataset mới)
│
↓ Output shape: (7, 7, 2048)
│
GlobalAveragePooling2D
│   → Giảm từ (7,7,2048) xuống vector (2048,)
│   → Thay thế Flatten, giảm parameters đáng kể
│
↓ (2048,)
│
BatchNormalization
│   → Chuẩn hóa, giúp training ổn định
│
↓ (2048,)
│
Dense(512, activation='relu')
│   → Layer fully connected, học tổ hợp features
│
↓ (512,)
│
Dropout(0.5)
│   → Tắt 50% neurons ngẫu nhiên, chống overfitting
│
↓ (512,)
│
Dense(128, activation='relu')
│   → Layer fully connected thứ 2, giảm chiều dần
│
↓ (128,)
│
Dropout(0.3)
│   → Tắt 30% neurons
│
↓ (128,)
│
Dense(2, activation='softmax')
│   → Output: xác suất [Cat, Dog]
```

---

## Data Preprocessing & Augmentation

### Tập Train (có augmentation)

Data augmentation tạo thêm biến thể của ảnh gốc trong quá trình training, giúp model học được nhiều góc nhìn khác nhau và giảm overfitting:

| Kỹ thuật | Giá trị | Mục đích |
|----------|---------|----------|
| Rotation | ±20° | Nhận diện vật thể khi xoay |
| Width/Height Shift | ±20% | Nhận diện khi vật thể không ở giữa |
| Shear | 20% | Nhận diện khi ảnh bị biến dạng nhẹ |
| Zoom | ±20% | Nhận diện ở các khoảng cách khác nhau |
| Horizontal Flip | Có | Nhận diện khi lật trái/phải |
| Preprocessing | ResNet50 preprocess_input | Chuẩn hóa pixel theo chuẩn ImageNet |

### Tập Validation & Test (không augmentation)

Chỉ áp dụng preprocessing (chuẩn hóa pixel) để đánh giá chính xác khả năng thực tế của model.

---

## Cấu hình Training

| Tham số | Giá trị | Giải thích |
|---------|---------|------------|
| Optimizer | Adam | Kết hợp Momentum + RMSProp, tự điều chỉnh learning rate cho từng parameter |
| Learning Rate | 0.0001 | Nhỏ vì đang fine-tune model đã train, tránh phá hỏng features đã học |
| Loss Function | Categorical Crossentropy | Hàm mất mát chuẩn cho bài toán phân loại nhiều class |
| Batch Size | 32 | Số ảnh xử lý mỗi lần cập nhật weights |
| Max Epochs | 30 | Giới hạn trên, thực tế sẽ dừng sớm nhờ EarlyStopping |

### Callbacks (cơ chế tự động điều chỉnh)

1. **EarlyStopping** (patience=5): Nếu validation loss không giảm sau 5 epochs liên tiếp → dừng training, khôi phục weights tốt nhất. Tránh overfitting do train quá lâu.

2. **ModelCheckpoint**: Tự động lưu model mỗi khi validation accuracy đạt giá trị cao nhất mới. Đảm bảo luôn giữ được phiên bản tốt nhất.

3. **ReduceLROnPlateau** (patience=2, factor=0.5): Nếu validation loss không giảm sau 2 epochs → giảm learning rate xuống 1/2. Giúp model "tinh chỉnh" khi đã gần hội tụ.

---

## Cấu trúc thư mục

```
CNN/
├── README.md               ← File này
├── requirements.txt        ← Danh sách packages cần cài
├── train.py                ← Script huấn luyện model
├── evaluate.py             ← Script đánh giá trên tập test
├── predict.py              ← Script dự đoán ảnh mới
├── data/
│   ├── train/
│   │   ├── cats/           ← Ảnh mèo dùng để train
│   │   └── dogs/           ← Ảnh chó dùng để train
│   ├── validation/
│   │   ├── cats/           ← Ảnh mèo dùng để validation
│   │   └── dogs/           ← Ảnh chó dùng để validation
│   ├── test/
│   │   ├── cats/           ← Ảnh mèo dùng để test
│   │   └── dogs/           ← Ảnh chó dùng để test
│   └── predict/            ← Ảnh mới để dự đoán thử
├── model/
│   └── cats_dogs_resnet50.h5       ← Model đã train xong
└── results/
    ├── training_history.png        ← Biểu đồ accuracy/loss qua các epoch
    ├── confusion_matrix.png        ← Ma trận nhầm lẫn trên tập test
    ├── classification_report.txt   ← Báo cáo precision/recall/F1
    └── class_distribution.png      ← Biểu đồ phân bố dữ liệu
```

---

## Hướng dẫn chạy

### 1. Cài đặt môi trường

```bash
pip install -r requirements.txt
```

### 2. Huấn luyện model

```bash
python train.py
```

Quá trình training sẽ:
- Load ResNet-50 pre-trained, thêm các layer classifier mới
- Train với data augmentation trên tập train
- Đánh giá trên tập validation sau mỗi epoch
- Tự động lưu model tốt nhất vào `model/cats_dogs_resnet50.h5`
- Xuất biểu đồ accuracy/loss vào `results/training_history.png`

### 3. Đánh giá model trên tập test

```bash
python evaluate.py
```

Xuất ra:
- Test accuracy và test loss
- Classification report (precision, recall, F1-score cho mỗi class)
- Confusion matrix (lưu ảnh tại `results/confusion_matrix.png`)

### 4. Dự đoán ảnh mới

```bash
python predict.py <đường_dẫn_ảnh>
python predict.py data/test/cats/cat.100.jpg
python predict.py anh1.jpg anh2.jpg anh3.jpg
```

Hiển thị kết quả phân loại (Cat/Dog) kèm độ tin cậy (%) và hiển thị ảnh.

---

## Kết quả mong đợi

| Metric | Mục tiêu |
|--------|-----------|
| Validation Accuracy | >= 95% |
| Test Accuracy | >= 95% |
| Overfitting gap (train - val) | < 5% |
| Precision / Recall / F1 | >= 0.95 cho cả 2 class |

---

## Giải thích chi tiết từng file

### `train.py`

- **create_data_generators()**: Tạo ImageDataGenerator cho train (có augmentation) và validation (chỉ preprocessing). Dùng `flow_from_directory` để tự động đọc ảnh theo cấu trúc thư mục và gán label.
- **build_model()**: Load ResNet-50 pre-trained, freeze các layer đầu, fine-tune 20 layers cuối, thêm head classifier mới (GAP → BN → Dense → Dropout → Dense → Dropout → Softmax).
- **plot_history()**: Vẽ biểu đồ accuracy và loss qua các epoch cho cả train và validation.
- **main()**: Điều phối toàn bộ pipeline training.

### `evaluate.py`

- Load model đã train từ file `.h5`
- Chạy inference trên toàn bộ tập test (không shuffle để giữ thứ tự)
- Tính loss và accuracy tổng thể
- Tạo classification report chi tiết (precision, recall, F1 cho từng class)
- Vẽ confusion matrix để trực quan hóa kết quả

### `predict.py`

- Nhận đường dẫn ảnh từ command line arguments
- Load và resize ảnh về 224x224, áp dụng preprocessing giống lúc train
- Chạy model.predict để lấy xác suất cho mỗi class
- Hiển thị kết quả và ảnh bằng matplotlib
