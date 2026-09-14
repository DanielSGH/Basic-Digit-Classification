from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
import umap
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

train_in = pd.read_csv('./data/train_in.csv', header=None)
train_out = pd.read_csv('./data/train_out.csv', header=None)
test_in = pd.read_csv('./data/test_in.csv', header=None)
test_out = pd.read_csv('./data/test_out.csv', header=None)

# -------------------------------------------------------------------------------
#   Data dimensionality, distance-based classifiers
#   Task 1.1
# -------------------------------------------------------------------------------

centers = {}

for i in range(10):
    values = train_in[train_out[0] == i]
    centers[i] = values.mean(axis=0)

distance_matrix = [[0.0 for _ in range(10)]
                   for _ in range(10)]

for a in centers:
    for b in centers:
        distance = math.dist(centers[a], centers[b])
        distance_matrix[a][b] = distance

df_distances = pd.DataFrame(
    distance_matrix)

print(df_distances)

unique_pairs = []
for i in range(10):
    for j in range(10):
        if i == j or i > j:
            continue

        distance_value = distance_matrix[i][j]
        pair_data = {
            'digit_a': i,
            'digit_b': j,
            'distance': distance_value
        }
        unique_pairs.append(pair_data)


def get_distance_key(item):
    return item['distance']


unique_pairs.sort(key=get_distance_key)

print("Ranked Closest Digit Pairs:")
print("---------------------------")
for item in unique_pairs:
    a = item['digit_a']
    b = item['digit_b']
    dist = item['distance']
    print(f"Digits {a} and {b} are close -> Distance: {dist:.2f}")


# -------------------------------------------------------------------------------
#   Data dimensionality, distance-based classifiers
#   Task 1.2
# -------------------------------------------------------------------------------

# # --- PCA ---
# pca = PCA(n_components=2)
# X = pca.fit_transform(train_in)

# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# scatter = plt.scatter(X[:, 0], X[:, 1], c=train_out[0],
#                       cmap='tab10', edgecolor='k')
# plt.legend(*scatter.legend_elements(), title="Digit")
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.title('PCA of digits')
# plt.show()

# # --- t-SNE ---
# tsne = TSNE(n_components=2, random_state=42)
# X_tsne = tsne.fit_transform(train_in)

# plt.figure(figsize=(8, 8))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
#                       c=train_out[0], cmap='tab10', s=10)
# plt.legend(*scatter.legend_elements(), title="Digit")
# plt.xlabel('t-SNE 1')
# plt.ylabel('t-SNE 2')
# plt.title('t-SNE of digits')
# plt.show()

# # --- UMAP ---
# reducer = umap.UMAP(n_components=2, random_state=42)
# X_umap = reducer.fit_transform(train_in)

# plt.figure(figsize=(8, 8))
# scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
#                       c=train_out[0], cmap='tab10', s=10)
# plt.legend(*scatter.legend_elements(), title="Digit")
# plt.xlabel('UMAP 1')
# plt.ylabel('UMAP 2')
# plt.title('UMAP of digits')
# plt.show()

# -------------------------------------------------------------------------------
#   Data dimensionality, distance-based classifiers
#   Task 1.3
# -------------------------------------------------------------------------------

centers_matrix = np.array([centers[i] for i in range(10)])


def nearest_mean_predict(X, centers_matrix):
    predictions = []
    for image in X:
        distances = [math.dist(image, center) for center in centers_matrix]
        predictions.append(np.argmin(distances))
    return np.array(predictions)


train_preds = nearest_mean_predict(train_in.values, centers_matrix)
train_acc = (train_preds == train_out[0].values).mean()
print(f"Train accuracy: {train_acc*100:.2f}%")

test_preds = nearest_mean_predict(test_in.values, centers_matrix)
test_acc = (test_preds == test_out[0].values).mean()
print(f"Test accuracy: {test_acc*100:.2f}%")

knn = KNeighborsClassifier(n_neighbors=5)  # k/n is variable
knn.fit(train_in, train_out[0])

knn_train_preds = knn.predict(train_in)
knn_test_preds = knn.predict(test_in)

knn_train_acc = (knn_train_preds == train_out[0].values).mean()
knn_test_acc = (knn_test_preds == test_out[0].values).mean()

print(f"KNN Train accuracy: {knn_train_acc*100:.2f}%")
print(f"KNN Test accuracy: {knn_test_acc*100:.2f}%")
