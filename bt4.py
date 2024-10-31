import os
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Tuple, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageClassifier:
    def __init__(self, image_folder: str, label_file_path: str, image_size: Tuple[int, int] = (30, 30)):
        self.image_folder = image_folder
        self.label_file_path = label_file_path
        self.image_size = image_size
        self.labels_df = None
        self.models = {
            'SVM': SVC(kernel='linear'),
            'KNN': KNeighborsClassifier(n_neighbors=5),  # Increased from 1 to 5 for better generalization
            'Decision Tree': DecisionTreeClassifier(max_depth=5)
        }

    def load_data(self) -> None:
        """Load and validate the labels file"""
        try:
            self.labels_df = pd.read_csv(self.label_file_path)
            logging.info(f"Loaded {len(self.labels_df)} labels from {self.label_file_path}")
        except Exception as e:
            logging.error(f"Error loading labels file: {e}")
            raise

    def load_images_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess images with error handling"""
        images = []
        labels = []
        total_images = len(self.labels_df)
        processed = 0

        for _, row in self.labels_df.iterrows():
            try:
                image_path = os.path.join(self.image_folder, row['image_name'])
                if not os.path.exists(image_path):
                    logging.warning(f"Image not found: {image_path}")
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Failed to load image: {image_path}")
                    continue

                # Preprocess image
                image = cv2.resize(image, self.image_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(image.flatten())
                labels.append(row['label'])

                processed += 1
                if processed % 100 == 0:  # Progress update every 100 images
                    logging.info(f"Processed {processed}/{total_images} images")

            except Exception as e:
                logging.error(f"Error processing image {row['image_name']}: {e}")
                continue

        return np.array(images), np.array(labels)

    def evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a single model with detailed metrics"""
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = model.predict(X_test)

            return {
                'Time': training_time,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro', zero_division=1),
                'Recall': recall_score(y_test, y_pred, average='macro', zero_division=1)
            }
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            return None

    def display_results(self, image_size: Tuple[int, int] = (300, 300)) -> None:
        """Display images with labels in a grid"""
        try:
            labels_dict = dict(zip(self.labels_df["image_name"], self.labels_df["label"]))
            image_files = [f for f in os.listdir(self.image_folder)
                           if os.path.isfile(os.path.join(self.image_folder, f))]

            num_images = len(image_files)
            num_cols = 4
            num_rows = (num_images + num_cols - 1) // num_cols

            # Create combined image with padding
            combined_image = np.zeros((image_size[1] * num_rows + 20 * num_rows,
                                       image_size[0] * num_cols, 3), dtype=np.uint8)

            for idx, image_name in enumerate(image_files):
                try:
                    image_path = os.path.join(self.image_folder, image_name)
                    image = cv2.imread(image_path)

                    if image is not None:
                        image = cv2.resize(image, image_size)
                        label = labels_dict.get(image_name, "unknown")

                        # Add label with better visibility
                        cv2.rectangle(image, (0, 0), (200, 30), (0, 0, 0), -1)
                        cv2.putText(image, label, (5, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                        row = idx // num_cols
                        col = idx % num_cols
                        y_start = row * (image_size[1] + 20)
                        y_end = y_start + image_size[1]
                        x_start = col * image_size[0]
                        x_end = x_start + image_size[0]

                        combined_image[y_start:y_end, x_start:x_end] = image

                except Exception as e:
                    logging.error(f"Error processing image {image_name} for display: {e}")
                    continue

            cv2.imshow("Image Classification Results", combined_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error in display_results: {e}")


def main():
    # Initialize classifier
    classifier = ImageClassifier(
        image_folder=r"D:\Workspace\XuLyAnh\LocAnh\image_paths",
        label_file_path=r"D:\Workspace\XuLyAnh\LocAnh\labels\labels.csv"
    )

    try:
        # Load data
        classifier.load_data()
        X, y = classifier.load_images_and_labels()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Evaluate models
        results = {}
        for model_name, model in classifier.models.items():
            logging.info(f"Evaluating {model_name}...")
            results[model_name] = classifier.evaluate_model(
                model, X_train, X_test, y_train, y_test
            )

        # Print results
        for model_name, metrics in results.items():
            if metrics:
                print(f"\n{model_name} Results:")
                print(f"Training Time: {metrics['Time']:.4f} seconds")
                print(f"Accuracy: {metrics['Accuracy']:.4f}")
                print(f"Precision: {metrics['Precision']:.4f}")
                print(f"Recall: {metrics['Recall']:.4f}")

        # Display results
        classifier.display_results()

    except Exception as e:
        logging.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()