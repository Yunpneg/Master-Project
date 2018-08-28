from keras.applications import VGG16, Xception
from sklearn.cluster import AgglomerativeClustering, KMeans
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED
import sys


base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
inputShape = (224, 224, 3)  # Assumes 3 channel image

dirname_path = 'test_data'
count = 0
for root, dirs, files in os.walk(dirname_path):
    for each in files:
        if each != '.DS_Store':
            count += 1

images_array = np.zeros((count, 224, 224, 3))

t = 0
cluster_number = 6
true_label = []
vgg16_feature_list = []
rawImage_list = []
dir_path_list = os.listdir(dirname_path)
if '.DS_Store' in dir_path_list:
    dir_path_list.remove('.DS_Store')
dir_path_list.sort()
class_item = 0
for dirname in dir_path_list:
    file_path = dirname_path + '/' + dirname
    file_path_list = os.listdir(file_path)
    if '.DS_Store' in file_path_list:
        file_path_list.remove('.DS_Store')
    file_path_list.sort()
    for filename in file_path_list:
        img_path = file_path + '/' + filename
        true_label.append(class_item)
        image = load_img(img_path, target_size=inputShape)
        image = img_to_array(image)  # shape is (224,224,3)
        images_array[t] = image  # (224, 224, 3). float64
        t += 1
        feature_vectors = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)
        feature_vectors = feature_vectors / 255.0
        image = image / 255.0
        preds = base_model.predict(feature_vectors)  # (1, 7, 7, 512). float32
        vgg16_feature_np = np.array(preds)  # (1, 7, 7, 512). array
        vgg16_feature_list.append(vgg16_feature_np.flatten())  # (n, 25088)
        rawImage_list.append(image.flatten())

    class_item += 1

vgg16_feature_list_np = np.array(vgg16_feature_list)
rawImage_list_np = np.array(rawImage_list)


clusters_features = AgglomerativeClustering(n_clusters=cluster_number).fit(vgg16_feature_list_np)
clusters_rawImages = AgglomerativeClustering(n_clusters=cluster_number).fit(rawImage_list_np)
labels_features = clusters_features.labels_
labels_rawImages = clusters_rawImages.labels_

print("original labels, features", labels_features)
print("original labels, rawImages", labels_rawImages)


# Munkres Algorithm
def get_matrix(y_pred, y_true):
    cm_array = confusion_matrix(y_pred, y_true)
    # print("matrix: ", cm_array)
    cm_array_new = [[0 for i in range(cluster_number)] for i in range(cluster_number)]
    for row in range(len(cm_array[:][0])):
        for col in range(len(cm_array[0][:])):
            cm_array_new[row][col] = cm_array[row][col]
    print("new_matrix: ", cm_array_new)

    # Calculating Profit, Rather than Cost
    cost_matrix = []
    for row in cm_array_new:
        cost_row = []
        for col in row:
            cost_row += [sys.maxsize - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    print_matrix(cm_array_new, msg='Highest profit through this matrix:')
    total = 0
    prediction = []
    for row, column in indexes:
        value = cm_array_new[row][column]
        prediction.append(column)
        total += value
        print('(%d, %d) -> %d' % (row, column, value))

    print('total profit=%d' % total)

    return prediction


prediction_features = get_matrix(labels_features, true_label)
prediction_rawImages = get_matrix(labels_rawImages, true_label)

for i in range(len(labels_features)):
    labels_features[i] = prediction_features[labels_features[i]]

for j in range(len(labels_rawImages)):
    labels_rawImages[j] = prediction_rawImages[labels_rawImages[j]]

print('Truth: ', true_label)
print('Converted features Labels: ', labels_features)
print('Converted rawImages Labels: ', labels_rawImages)


# Clustering accracy: raw images
j = 0
arry_1 = np.array(0)
while (j < len(true_label)):
    if true_label[j] == labels_features[j]:
        arry_1 += 1
    j += 1

accuracy_1 = arry_1/len(true_label)
print("Predicted Accuracy for feature vectors: ", accuracy_1)

# Clustering accracy: Feature vectors
t = 0
arry_2 = np.array(0)
while (t < len(true_label)):
    if true_label[t] == labels_rawImages[t]:
        arry_2 += 1
    t += 1

accuracy_2 = arry_2/len(true_label)
print("Predicted Accuracy for raw Images: ", accuracy_2)


# Plot the confusion matrix: raw images
def plot_confusion_matrix_rawImage(y_pred, y_true):
    cm_array = confusion_matrix(y_pred, y_true)
    sns.heatmap(cm_array, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Raw Images", fontsize=30)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Classification label', fontsize=12)


plot_confusion_matrix_rawImage(labels_rawImages, true_label)
plt.show()

# Plot the confusion matrix: Feature vectors
def plot_confusion_matrix_features(y_pred, y_true):
    cm_array = confusion_matrix(y_pred, y_true)
    sns.heatmap(cm_array, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Feature Vectors", fontsize=30)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Classification label', fontsize=12)

plot_confusion_matrix_features(labels_features, true_label)
plt.show()






