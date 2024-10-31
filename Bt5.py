import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import math


# Hàm tính Information Gain cho ID3
def calculate_entropy(y):
    classes = np.unique(y)
    entropy = 0
    for cls in classes:
        p = len(y[y == cls]) / len(y)
        entropy -= p * np.log2(p)
    return entropy


def calculate_information_gain(X, y, feature):
    entropy_parent = calculate_entropy(y)
    values = np.unique(X[:, feature])
    weighted_entropy = 0

    for value in values:
        subset_indices = X[:, feature] == value
        subset_size = len(X[subset_indices])
        weighted_entropy += (subset_size / len(X)) * calculate_entropy(y[subset_indices])

    information_gain = entropy_parent - weighted_entropy
    return information_gain


# Lớp ID3 Decision Tree
class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng
        if (self.max_depth is not None and depth >= self.max_depth) or n_labels == 1:
            return np.argmax(np.bincount(y))

        # Tìm feature tốt nhất dựa trên Information Gain
        best_gain = -1
        best_feature = None

        for feature in range(n_features):
            gain = calculate_information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        if best_gain == -1:
            return np.argmax(np.bincount(y))

        # Xây dựng cây quyết định
        tree = {'feature': best_feature}
        feature_values = np.unique(X[:, best_feature])

        for value in feature_values:
            mask = X[:, best_feature] == value
            if len(y[mask]) > 0:
                tree[value] = self._grow_tree(X[mask], y[mask], depth + 1)

        return tree

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x, tree=None):
        if tree is None:
            tree = self.tree

        if isinstance(tree, (int, np.integer)):
            return tree

        feature = tree['feature']
        value = x[feature]

        if value not in tree:
    # Nếu giá trị không có trong cây, trả về giá trị phổ biến nhất


return max(tree.values(), key=lambda x: isinstance(x, (int, np.integer)))

return self._predict_one(x, tree[value])


# 1. Xử lý bộ dữ liệu IRIS
def process_iris_dataset():
    print("Processing IRIS dataset...")
    # Load dữ liệu IRIS
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CART với Gini Index
    print("\nTraining CART model...")
    cart = DecisionTreeClassifier(criterion='gini', random_state=42)
    cart.fit(X_train, y_train)
    cart_predictions = cart.predict(X_test)

    print("\nCART Results:")
    print("Accuracy:", accuracy_score(y_test, cart_predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, cart_predictions))

    # ID3 với Information Gain
    print("\nTraining ID3 model...")
    id3 = ID3DecisionTree(max_depth=5)
    id3.fit(X_train, y_train)
    id3_predictions = id3.predict(X_test)

    print("\nID3 Results:")
    print("Accuracy:", accuracy_score(y_test, id3_predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, id3_predictions))


# 2. Xử lý bộ dữ liệu ảnh nha khoa
def process_dental_images(image_dir):
    print("\nProcessing Dental Images...")

    def load_and_preprocess_image(image_path):
        # Đọc và xử lý ảnh
        img = cv2.imread(image_path)
        img = cv2.resize(img, (64, 64))  # Resize để giảm kích thước
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển sang ảnh xám
        return img.flatten()  # Làm phẳng ảnh thành vector

    # Giả sử cấu trúc thư mục: mỗi lớp nằm trong một thư mục con
    images = []
    labels = []

    for class_name in os.listdir(image_dir):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)
                    img_vector = load_and_preprocess_image(image_path)
                    images.append(img_vector)
                    labels.append(class_name)

    # Chuyển đổi sang numpy array
    X = np.array(images)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # CART với Gini Index
    print("\nTraining CART model...")
    cart = DecisionTreeClassifier(criterion='gini', random_state=42)
    cart.fit(X_train, y_train)
    cart_predictions = cart.predict(X_test)

    print("\nCART Results:")
    print("Accuracy:", accuracy_score(y_test, cart_predictions))
    print("\nClassification Report:")


print(classification_report(y_test, cart_predictions))

# ID3 với Information Gain
print("\nTraining ID3 model...")
id3 = ID3DecisionTree(max_depth=5)
id3.fit(X_train, y_train)
id3_predictions = id3.predict(X_test)

print("\nID3 Results:")
print("Accuracy:", accuracy_score(y_test, id3_predictions))
print("\nClassification Report:")
print(classification_report(y_test, id3_predictions))

# Thực thi chương trình
if __name__ == "__main__":
    # Xử lý bộ dữ liệu IRIS
    process_iris_dataset()

    # Xử lý bộ dữ liệu ảnh nha khoa
    # Thay đổi đường dẫn tới thư mục chứa ảnh nha khoa
    dental_images_dir = "path/to/dental/images"
    process_dental_images(dental_images_dir)