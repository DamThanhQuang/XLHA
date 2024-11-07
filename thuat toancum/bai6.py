# Import các thư viện cần thiết
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz

# Danh sách các tệp ảnh vệ tinh
image_files = [ '11111.jpg']

# Thiết lập số cụm
n_clusters = 2  # Số cụm có thể thay đổi tùy vào yêu cầu

# Khởi tạo figure với lưới 3 hàng và số ảnh x 3 cột (gốc, K-means, FCM)
fig, axes = plt.subplots(len(image_files), 3, figsize=(15, 5 * len(image_files)))

# Lặp qua từng ảnh để phân cụm
for idx, file in enumerate(image_files):
    # Đọc ảnh vệ tinh
    image = cv2.imread(file)
    if image is None:
        print(f"Lỗi khi đọc ảnh {file}")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh gốc
    axes[idx, 0].imshow(image_rgb)
    axes[idx, 0].set_title(f"Ảnh gốc {idx + 1}")
    axes[idx, 0].axis('off')

    # Chuyển ảnh thành mảng 2D với mỗi pixel là một điểm dữ liệu (cho K-means)
    pixels = image_rgb.reshape((-1, 3))

    # --- Phân cụm bằng K-means ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)

    # Gán nhãn cụm cho mỗi pixel và tạo ảnh phân cụm K-means
    labels_kmeans = kmeans.labels_
    segmented_image_kmeans = labels_kmeans.reshape(image_rgb.shape[:2])

    # Hiển thị kết quả phân cụm K-means
    axes[idx, 1].imshow(segmented_image_kmeans, cmap='viridis')
    axes[idx, 1].set_title(f"Phân cụm K-means {idx + 1}")
    axes[idx, 1].axis('off')

    # --- Phân cụm bằng Fuzzy C-means (FCM) ---
    pixels_fcm = pixels.T  # Chuyển vị để phù hợp với FCM
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(pixels_fcm, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None)

    # Xác định các nhãn cụm dựa trên độ thành viên cao nhất
    labels_fcm = np.argmax(u, axis=0)
    segmented_image_fcm = labels_fcm.reshape(image_rgb.shape[:2])

    # Hiển thị kết quả phân cụm FCM
    axes[idx, 2].imshow(segmented_image_fcm, cmap='viridis')
    axes[idx, 2].set_title(f"Phân cụm Fuzzy C-means {idx + 1}")
    axes[idx, 2].axis('off')

# Tăng khoảng cách giữa các hàng để dễ nhìn hơn
plt.tight_layout()
plt.show()
