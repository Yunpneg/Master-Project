from keras.applications import VGG16, Xception
from sklearn.cluster import AgglomerativeClustering, KMeans
from keras.preprocessing import image
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import seaborn as sns
# import argparse
import os
from skimage import transform
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED
import sys
from PIL import Image
import imagehash
import cv2

class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        计算图片的dHash值
        :param image: PIL.Image
        :return: dHash值,string类型
        """
        difference = DHash.__difference(image)
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value为0, 不用计算, 程序优化
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # 每8位的结束
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0
        return difference #hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        计算两张图片的汉明距离(基于dHash算法)
        :param first: Image或者dHash值(str)
        :param second: Image或者dHash值(str)
        :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
        """
        # A. dHash值计算汉明距离
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

        # # B. image计算汉明距离
        # hamming_distance = 0
        # image1_difference = DHash.__difference(first)
        # image2_difference = DHash.__difference(second)
        # for index, img1_pix in enumerate(image1_difference):
        #     img2_pix = image2_difference[index]
        #     if img1_pix != img2_pix:
        #         hamming_distance += 1
        # return hamming_distance

    @staticmethod
    def __difference(image):
        """
        *Private method*
        计算image的像素差值
        :param image: PIL.Image
        :return: 差值数组。0、1组成
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. 灰度化 Grayscale
        grayscale_image = smaller_image.convert("L")

        # 3. 比较相邻像素
        pixels = list(grayscale_image.getdata())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference

    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        根据dHash值计算hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: 汉明距离(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")


def getGray(image_file):
    tmpls = []
    for h in range(0, image_file.size[1]):  # h
        for w in range(0, image_file.size[0]):  # w
            tmpls.append(image_file.getpixel((w, h)))

    return tmpls


def getAvg(ls):  # 获取平均灰度值
    return sum(ls) / len(ls)


def getMH(a, b):  # 比较100个字符有几个字符相同
    dist = 0
    for i in range(0, len(a)):
        if a[i] == b[i]:
            dist = dist + 1
    return dist


def getImgHash(fne):
    image_file = Image.open(fne)  # 打开
    image_file = image_file.resize((8, 8))  #
    image_file = image_file.convert("L")  # 转256灰度图
    Grayls = getGray(image_file)  # 灰度集合

    avg = getAvg(Grayls)  # 灰度平均值
    bitls = ''  # 接收获取0或1
    # 除去变宽1px遍历像素
    for h in range(1, image_file.size[1] - 1):  # h
        for w in range(1, image_file.size[0] - 1):  # w
            if image_file.getpixel((w, h)) >= avg:  # 像素的值比较平均值 大于记为1 小于记为0
                bitls = bitls + '1'
            else:
                bitls = bitls + '0'
    return bitls


inputShape = (224, 224, 3)  # Assumes 3 channel image
dirname_path = 'test_data'
count = 0
for root, dirs, files in os.walk(dirname_path):  # 遍历统计
    for each in files:
        if each != '.DS_Store':
            count += 1  # 统计文件夹下文件个数

images_array = np.zeros((count, 224, 224, 3))

t = 0
cluster_number = 6
true_label = []
aHash_list = []
dHash_list = []
pHash_list = []
dir_path_list = os.listdir(dirname_path)
if '.DS_Store' in dir_path_list:
    dir_path_list.remove('.DS_Store')
dir_path_list.sort()
class_item = 0  # 类的下标
for dirname in dir_path_list:  # listdir的参数是文件夹的路径
    file_path = dirname_path + '/' + dirname
    file_path_list = os.listdir(file_path)
    if '.DS_Store' in file_path_list:
        file_path_list.remove('.DS_Store')
    file_path_list.sort()
    for filename in file_path_list:
        img_path = file_path + '/' + filename
        true_label.append(class_item)
        x = getImgHash(img_path)
        aHash_list.append(np.array(x).flatten())
        y = DHash.calculate_hash(Image.open(img_path))
        dHash_list.append(np.array(y).flatten())

        # a = imagehash.average_hash(Image.open(img_path), hash_size=8)
        # b = imagehash.dhash(Image.open(img_path),hash_size=8)
        # aHash_list.append(np.array(a).flatten())
        # dHash_list.append(np.array(b).flatten())




    class_item += 1

aHash_list_np = np.array(aHash_list)
dHash_list_np = np.array(dHash_list)
# pHash_list_np = np.array(pHash_list)

clusters_ahash = KMeans(n_clusters=cluster_number).fit(aHash_list_np)
clusters_dhash = KMeans(n_clusters=cluster_number).fit(dHash_list_np)
# clusters_phash = KMeans(n_clusters=cluster_number).fit(pHash_list_np)
# labels_phash = clusters_phash.labels_
labels_ahash = clusters_ahash.labels_
labels_dhash = clusters_dhash.labels_


# 正确配对
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


prediction_ahash = get_matrix(labels_ahash, true_label)
prediction_dhash = get_matrix(labels_dhash, true_label)
# prediction_phash = get_matrix(labels_phash, true_label)

for i in range(len(labels_ahash)):
    labels_ahash[i] = prediction_ahash[labels_ahash[i]]

for i in range(len(labels_dhash)):
    labels_dhash[i] = prediction_dhash[labels_dhash[i]]

# for i in range(len(labels_phash)):
#     labels_phash[i] = prediction_phash[labels_phash[i]]

print('Truth: ', true_label)
print('Converted ahash Labels: ', labels_ahash)
print('Converted dhash Labels: ', labels_dhash)
# print('Converted phash Labels: ', labels_phash)

# 预测正确率
j = 0
arry_1 = np.array(0)
while (j < len(true_label)):
    if true_label[j] == labels_ahash[j]:
        arry_1 += 1
    j += 1

accuracy_1 = arry_1/len(true_label)
print("Predicted Accuracy for ahash: ", accuracy_1)

# 预测正确率

arry_2 = np.array(0)
t = 0
while (t < len(true_label)):
    if true_label[t] == labels_dhash[t]:
        arry_2 += 1
    t += 1
accuracy_2 = arry_2/len(true_label)
print("Predicted Accuracy for dhash: ", accuracy_2)

# # 预测正确率
# j = 0
# arry_3 = np.array(0)
# while (j < len(true_label)):
#     if true_label[j] == labels_phash[j]:
#         arry_3 += 1
#     j += 1
#
# accuracy_3 = arry_3/len(true_label)
# print("Predicted Accuracy for phash: ", accuracy_3)



# Plot the confusion matrix
def plot_confusion_matrix_aHash(y_pred, y_true):
    cm_array = confusion_matrix(y_pred, y_true)
    sns.heatmap(cm_array, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Average Hash", fontsize=30)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Classification label', fontsize=12)


plot_confusion_matrix_aHash(labels_ahash, true_label)
plt.show()

# Plot the confusion matrix
def plot_confusion_matrix_dHash(y_pred, y_true):
    cm_array = confusion_matrix(y_pred, y_true)
    sns.heatmap(cm_array, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Difference Hash", fontsize=30)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Classification label', fontsize=12)


plot_confusion_matrix_aHash(labels_dhash, true_label)
plt.show()





