import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.datasets import load_files
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Đo thời gian training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Đo thời gian dự đoán
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Training Time': training_time,
        'Prediction Time': prediction_time
    }


# 1. Phân lớp dữ liệu IRIS
print("=== Phân lớp dữ liệu IRIS ===")

# Tải dữ liệu IRIS
iris = load_iris()
X = iris.data
y = iris.target

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo các mô hình
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf'),
    'ANN': MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
}

# Đánh giá các mô hình trên dữ liệu IRIS
results_iris = []
for model_name, model in models.items():
    result = evaluate_model(model, X_train_scaled, X_test_scaled,
                            y_train, y_test, model_name)
    results_iris.append(result)

# In kết quả cho IRIS
print("\nKết quả phân lớp dữ liệu IRIS:")
print(pd.DataFrame(results_iris))

# 2. Phân lớp ảnh động vật (giả lập dữ liệu ảnh)
print("\n=== Phân lớp ảnh động vật ===")

# Tạo dữ liệu ảnh mô phỏng
n_samples = 1000
n_features = 784  # 28x28 pixels
X_images = np.random.rand(n_samples, n_features)
y_images = np.random.randint(0, 3, n_samples)  # 3 classes of animals

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X_images, y_images,
                                                    test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Đánh giá các mô hình trên dữ liệu ảnh
results_images = []
for model_name, model in models.items():
    result = evaluate_model(model, X_train_scaled, X_test_scaled,
                            y_train, y_test, model_name)
    results_images.append(result)

# In kết quả cho ảnh động vật
print("\nKết quả phân lớp ảnh động vật:")
print(pd.DataFrame(results_images))