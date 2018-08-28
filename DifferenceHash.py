from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from PIL import Image


class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        :param image: PIL.Image
        :return: dHash,string
        """
        difference = DHash.__difference(image)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        :param first:dHash(str)
        :param second: dHash(str)
        :return: hamming distance.
        """
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

    @staticmethod
    def __difference(image):
        """
        *Private method*
        :param image: PIL.Image
        :return: 0 and 1
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. Grayscale
        grayscale_image = smaller_image.convert("L")

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
        hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: (int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")


# file path we want to test
dirname_path = 'test_data'
# Get total number of images
count = 0
for root, dirs, files in os.walk(dirname_path):
    for each in files:
        if each != '.DS_Store':
            count += 1
images_array = np.zeros((count, 224, 224, 3))

t = 0
path_list = []
true_label = [] # Ground Truth list
name_list = []  # file name list
dir_path_list = os.listdir(dirname_path)
if '.DS_Store' in dir_path_list:
    dir_path_list.remove('.DS_Store')
dir_path_list.sort()
class_item = 0  # index of category
for dirname in dir_path_list:
    file_path = dirname_path + '/' + dirname
    file_path_list = os.listdir(file_path)
    if '.DS_Store' in file_path_list:
        file_path_list.remove('.DS_Store')
    file_path_list.sort()
    for filename in file_path_list:
        img_path = file_path + '/' + filename
        name_list.append(filename)
        true_label.append(class_item)
        path_list.append(img_path)
        # load the images
        image = load_img(img_path, target_size=(224, 224, 3))
        image = img_to_array(image)  # shape is (224,224,3)
        images_array[t] = image  # (224, 224, 3). float64
        t += 1

    class_item += 1  # category number
# create 9 lists to contain the index of normal flower and the index of eight types of mutant flower.
normal_index = []
image_index =[]
mutation_index_1 =[]
mutation_index_2 =[]
mutation_index_3 =[]
mutation_index_4 =[]
mutation_index_5 =[]
# mutation_index_6 =[]
# mutation_index_7 =[]
# mutation_index_8 =[]

normal_index.append([t for t, x in enumerate(true_label) if x == 0])
mutation_index_1.append([t for t, x in enumerate(true_label) if x == 1])
mutation_index_2.append([t for t, x in enumerate(true_label) if x == 2])
mutation_index_3.append([t for t, x in enumerate(true_label) if x == 3])
mutation_index_4.append([t for t, x in enumerate(true_label) if x == 4])
mutation_index_5.append([t for t, x in enumerate(true_label) if x == 5])
# mutation_index_6.append([t for t, x in enumerate(true_label) if x == 6])
# mutation_index_7.append([t for t, x in enumerate(true_label) if x == 7])
# mutation_index_8.append([t for t, x in enumerate(true_label) if x == 8])

image_index.append(normal_index)
image_index.append(mutation_index_1)
image_index.append(mutation_index_2)
image_index.append(mutation_index_3)
image_index.append(mutation_index_4)
image_index.append(mutation_index_5)
# mutation_index.append(mutation_index_6)
# mutation_index.append(mutation_index_7)
# mutation_index.append(mutation_index_8)

# Difference Hash
# The average distance of intra-class
intra_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_1 = 0
        ave_distance = 0
        image01 = Image.open(path_list[j])
        dHash_mutation = DHash.calculate_hash(image01)
        for k in image_index[i][0]:
            if j != k:
                num_1 += 1
                image02 = Image.open(path_list[k])
                dHash_normal = DHash.calculate_hash(image02)
                ave_distance += DHash.hamming_distance(dHash_mutation, dHash_normal)
        single_sample.append(ave_distance / num_1)
    intra_distance.append(int(sum(single_sample)) / len(single_sample))
intra_ave_feature = int(sum(intra_distance)) / len(intra_distance)
print("Intra_class average distance for Difference Hash", ("%.2f" % intra_ave_feature))

# The average distance of Inter-class
inter_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_2 = 0
        ave_distance = 0
        index = 0
        image01 = Image.open(path_list[j])
        dHash_mutation = DHash.calculate_hash(image01)
        while index < len(image_index):
            if index != i:
                for k in image_index[index][0]:
                    num_2 += 1
                    image02 = Image.open(path_list[k])
                    dHash_normal = DHash.calculate_hash(image02)
                    ave_distance += DHash.hamming_distance(dHash_mutation, dHash_normal)
            index += 1
        single_sample.append(ave_distance / num_2)
    inter_distance.append(int(sum(single_sample)) / len(single_sample))
inter_ave_feature = int(sum(inter_distance)) / len(inter_distance)
print("Inter_class average distance for Difference Hash", ("%.2f" % inter_ave_feature))