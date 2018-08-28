from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import numpy as np


def getGray(image_file):
    tmpls = []
    for h in range(0, image_file.size[1]):  # h
        for w in range(0, image_file.size[0]):  # w
            tmpls.append(image_file.getpixel((w, h)))

    return tmpls


def getAvg(ls):
    return sum(ls) / len(ls)


def getMH(a, b):  
    dist = 0
    for i in range(0, len(a)):
        if a[i] == b[i]:
            dist = dist + 1
    return dist


def getImgHash(fne):
    image_file = Image.open(fne)
    image_file = image_file.resize((8, 8))
    image_file = image_file.convert("L")
    Grayls = getGray(image_file)
    avg = getAvg(Grayls)
    bitls = ''
    for h in range(1, image_file.size[1] - 1):  # height
        for w in range(1, image_file.size[0] - 1):  # width
            if image_file.getpixel((w, h)) >= avg:
                bitls = bitls + '1'
            else:
                bitls = bitls + '0'
    return bitls

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

# Average Hash
# The average distance of intra-class
intra_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_1 = 0
        ave_distance = 0
        a = getImgHash(path_list[j])
        for k in image_index[i][0]:
            if j != k:
                num_1 += 1
                b = getImgHash(path_list[k])
                ave_distance += getMH(a, b)
        single_sample.append(ave_distance / num_1)
    intra_distance.append(int(sum(single_sample)) / len(single_sample))
intra_ave_feature = int(sum(intra_distance)) / len(intra_distance)
print("Intra_class average distance for Average Hash", ("%.2f" % intra_ave_feature))

# The average distance of Inter-class
inter_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_2 = 0
        ave_distance = 0
        index = 0
        a = getImgHash(path_list[j])
        while index < len(image_index):
            if index != i:
                for k in image_index[index][0]:
                    num_2 += 1
                    b = getImgHash(path_list[k])
                    ave_distance += getMH(a, b)
            index += 1
        single_sample.append(ave_distance / num_2)
    inter_distance.append(int(sum(single_sample)) / len(single_sample))
inter_ave_feature = int(sum(inter_distance)) / len(inter_distance)
print("Inter_class average distance for Average Hash", ("%.2f" % inter_ave_feature))





