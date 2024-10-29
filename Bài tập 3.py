import numpy as np
from sklearn.datasets import load_flowers
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the flowers dataset
data = load_flowers()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM Classifier
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred, average='macro')
svm_recall = recall_score(y_test, svm_pred, average='macro')
svm_f1 = f1_score(y_test, svm_pred, average='macro')
svm_time = svm_clf.fit(X_train, y_train).predict(X_test).all_time

# KNN Classifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_pred = knn_clf.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_precision = precision_score(y_test, knn_pred, average='macro')
knn_recall = recall_score(y_test, knn_pred, average='macro')
knn_f1 = f1_score(y_test, knn_pred, average='macro')
knn_time = knn_clf.fit(X_train, y_train).predict(X_test).all_time

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
tree_precision = precision_score(y_test, tree_pred, average='macro')
tree_recall = recall_score(y_test, tree_pred, average='macro')
tree_f1 = f1_score(y_test, tree_pred, average='macro')
tree_time = tree_clf.fit(X_train, y_train).predict(X_test).all_time

# Print the results
print("Algorithm   Accuracy   Precision   Recall   F1-Score   Time")
print("-" * 70)
print(f"SVM        {svm_accuracy:.2f}        {svm_precision:.2f}        {svm_recall:.2f}        {svm_f1:.2f}        {svm_time:.2f}")
print(f"KNN        {knn_accuracy:.2f}        {knn_precision:.2f}        {knn_recall:.2f}        {knn_f1:.2f}        {knn_time:.2f}")
print(f"Decision Tree {tree_accuracy:.2f}        {tree_precision:.2f}        {tree_recall:.2f}        {tree_f1:.2f}        {tree_time:.2f}")