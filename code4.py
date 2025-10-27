from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

digits = datasets.load_digits()
X, y = digits.data, digits.target

fig, axes = plt.subplots(3, 10, figsize=(15, 6))
for ax, img, label in zip(axes.ravel(), digits.images, y):
    ax.imshow(img, cmap='gray_r'); ax.axis('off'); ax.set_title(str(label))
plt.show()

print("Sample flattened image:", digits.images[0].ravel()[:10], "...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=99, stratify=y
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
conf_mat = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, cmap='nipy_spectral_r').set_title('Confusion Matrix')
plt.show()

print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_test, y_pred):.4f}")
