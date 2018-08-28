from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# 1) Raws Images

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

# Raw Images
# The average distance of intra-class
intra_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_1 = 0
        ave_distance = 0
        for k in image_index[i][0]:
            if j != k:
                num_1 += 1
                ave_distance += np.linalg.norm(np.array(img_to_array(images_array[j])) -
                                                  np.array(img_to_array(images_array[k])))
        single_sample.append(ave_distance / num_1)
    intra_distance.append(int(sum(single_sample)) / len(single_sample))
intra_ave_images = int(sum(intra_distance)) / len(intra_distance)
print("Intra_class average distance for RawImages", ("%.2f" % intra_ave_images))

# The average distance of Inter-class
inter_distance = []
for i in range(len(image_index)):
    single_sample = []
    for j in image_index[i][0]:
        num_2 = 0
        ave_distance = 0
        index = 0
        while index < len(image_index):
            if index != i:
                for k in image_index[index][0]:
                    num_2 += 1
                    ave_distance += np.linalg.norm(np.array(img_to_array(images_array[j])) -
                                                      np.array(img_to_array(images_array[k])))
            index += 1
        single_sample.append(ave_distance / num_2)
    inter_distance.append(int(sum(single_sample)) / len(single_sample))
inter_ave_images = int(sum(inter_distance)) / len(inter_distance)
print("Inter_class average distance for RawImages", ("%.2f" % inter_ave_images))
