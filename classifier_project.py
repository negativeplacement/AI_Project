
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.semi_supervised import LabelSpreading

IMG_SIZE = 128

def dataset(dir):
  feature = []
  label = []
  if os.path.isdir(dir):
    for folder in os.listdir(dir):
      folder_path = os.path.join(dir, folder)
      if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
          if file.lower().endswith('.jpg'):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is not None:
              img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
              img_flat = img_resized.flatten()
              feature.append(img_flat)
              label.append(folder)
  return np.array(feature), np.array(label)

x_train, y_train = dataset('drive/MyDrive/dataset/train')
x_test, y_test = dataset('drive/MyDrive/dataset/test')
x_train_norm = x_train/255.0
x_test_norm = x_test/255.0
# Dimensionality Reduction (PCA)
pca = PCA(n_components=150, random_state=42)
x_train_pca = pca.fit_transform(x_train_norm)
x_test_pca = pca.transform(x_test_norm)
# Supervised Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=7, random_state=42, min_samples_split = 15, min_samples_leaf=5, ccp_alpha=0.005)
decision_tree.fit(x_train_pca, y_train)

y_pred = decision_tree.predict(x_test_pca)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred))

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
pca = PCA(n_components=150, random_state=42)
x_train_pca = pca.fit_transform(x_train_norm)
x_test_pca = pca.transform(x_test_norm)
# Supervised XGBoost
xgb_model = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, tree_method='hist', random_state=42)
xgb_model.fit(x_train_pca, y_train_encoded)

train_accuracy = xgb_model.score(x_train_pca, y_train_encoded)
test_accuracy = xgb_model.score(x_test_pca, y_test_encoded)
print(f"Train Accuracy: {train_accuracy:.2%}")
print(f"Test Accuracy: {test_accuracy:.2%}")

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# This takes 20% of your data as labeled and turns 80% into unlabeled (-1)
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
    x_train_pca, y_train_encoded, train_size=0.20, random_state=42, stratify=y_train_encoded
)
# Create the -1 target array for the unlabeled portion
y_unlabeled = np.full(shape=len(X_unlabeled), fill_value=-1)
# Recombine them into a single semi-supervised training set
x_semi_train = np.vstack((X_labeled, X_unlabeled))
y_semi_train = np.concatenate((y_labeled, y_unlabeled))
# Train the LabelSpreading model
semi_model = LabelSpreading(kernel='rbf', gamma=0.01, alpha=0.1)
semi_model.fit(x_semi_train, y_semi_train)
# Evaluate on your standard test set
y_semi_pred = semi_model.predict(x_test_pca)

print("\n--- Semi-Supervised LabelSpreading Results ---")
print(f"Test Accuracy: {accuracy_score(y_test_encoded, y_semi_pred):.2%}\n")
print(classification_report(y_test_encoded, y_semi_pred, target_names=label_encoder.classes_))