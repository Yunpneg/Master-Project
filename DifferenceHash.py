import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        Calculate the difference hash of images
        :param image: PIL.Image
        :return: dHash,string format
        """
        difference = DHash.__difference(image)
        # Convert to Hex
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
        Calculate the Hamming distance between two images
        :param first: dHash(str)
        :param second: dHash(str)
        :return: hamming distance. The value is more larger, the difference between two images is bigger
        """
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)


    @staticmethod
    def __difference(image):
        """
        *Private method*
        Calculate the difference of pixels for images
        :param image: PIL.Image
        :return: consisting of 1 and 0
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. Grayscale
        grayscale_image = smaller_image.convert("L")

        # 3. compare with neighboring pixel
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
        Calculate the hamming distance based on dHash
        :param dhash1: str
        :param dhash2: str
        :return: Hamming distance (int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")



dirname_path = 'test_data'
true_label = []
path_list = []
file_list = []
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
        file_list.append(filename)
        true_label.append(class_item)
        path_list.append(img_path)
    class_item += 1

normal_index = []
mutation_index =[]
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

mutation_index.append(mutation_index_1)
mutation_index.append(mutation_index_2)
mutation_index.append(mutation_index_3)
mutation_index.append(mutation_index_4)
mutation_index.append(mutation_index_5)
# mutation_index.append(mutation_index_6)
# mutation_index.append(mutation_index_7)
# mutation_index.append(mutation_index_8)


whole_same_list = []
for t in range(len(mutation_index)):
    mutation_file_list = []
    same_list = []
    for i in mutation_index[t][0]:
        same = 0
        num = 0
        image01 = Image.open(path_list[i])
        dHash_mutation = DHash.calculate_hash(image01)
        mutation_file_list.append(file_list[i])
        for j in normal_index[0]:
            image02 = Image.open(path_list[j])
            dHash_normal = DHash.calculate_hash(image02)
            same += DHash.hamming_distance(dHash_mutation, dHash_normal)
            num += 1

        same_list.append(float("%.2f" % (same / num)))

    whole_same_list.append(same_list)
    print("file name", mutation_file_list)

print("Difference between mutation and normal flower: ", whole_same_list)


# Baseline
base_list = []
for i in normal_index[0]:
    image01 = Image.open(path_list[i])
    dHash_mutation = DHash.calculate_hash(image01)
    base = 0
    num_ = 0
    for j in normal_index[0]:
        if i != j:
            image02 = Image.open(path_list[j])
            dHash_normal = DHash.calculate_hash(image02)
            base += DHash.hamming_distance(dHash_mutation, dHash_normal)
            num_ += 1
    base_list.append(float(base/num_))
ave_base = "%.2f" % (int(sum(base_list))/len(base_list))

print("baseline", ave_base)


