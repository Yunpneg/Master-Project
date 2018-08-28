import pytesseract
from pytesser3 import *
from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import hashlib

def getGray(image_file):
    tmpls = []
    for h in range(0, image_file.size[1]):  # h
        for w in range(0, image_file.size[0]):  # w
            tmpls.append(image_file.getpixel((w, h)))

    return tmpls


def getAvg(ls):  # Get the average grayscale value
    return sum(ls) / len(ls)


def getMH(a, b):  # Compare the 100 characters and find the number of same characters between two strings
    dist = 0
    for i in range(0, len(a)):
        if a[i] == b[i]:
            dist = dist + 1
    return dist


def getImgHash(fne):
    image_file = Image.open(fne)
    image_file = image_file.resize((8, 8))
    image_file = image_file.convert("L")  # Convert to grayscale image
    Grayls = getGray(image_file)

    avg = getAvg(Grayls)  # Average value of grayscale
    bitls = ''
    for h in range(1, image_file.size[1] - 1):  # height
        for w in range(1, image_file.size[0] - 1):  # width
            if image_file.getpixel((w, h)) >= avg:
                # compare with the average value
                # if it is greater than the average value, recoded as 1; otherwise, recorded as 0
                bitls = bitls + '1'
            else:
                bitls = bitls + '0'
    return bitls


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

# The difference between mutant flowers and normal flowers
whole_same_list = []
for t in range(len(mutation_index)):
    mutation_file_list = []
    same_list = []
    for i in mutation_index[t][0]:
        same = 0
        num = 0
        a = getImgHash(path_list[i])
        mutation_file_list.append(file_list[i])
        for j in normal_index[0]:
            b = getImgHash(path_list[j])
            same += getMH(a, b)
            num += 1

        same_list.append(float("%.2f" % (same / num)))

    whole_same_list.append(same_list)
    print("file name", mutation_file_list)

print("Difference between mutation and normal flower: ", whole_same_list)


# Baseline
base_list = []
for i in normal_index[0]:
    a = getImgHash(path_list[i])
    base = 0
    num_ = 0
    for j in normal_index[0]:
        if i != j:
            b = getImgHash(path_list[j])
            base += getMH(a, b)
            num_ += 1
    base_list.append(float(base/num_))
ave_base = "%.2f" % (int(sum(base_list))/len(base_list))
print("baseline", ave_base)






