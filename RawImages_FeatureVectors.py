from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# 1) Raws Images
# 2) Feature vectors from model without classifier

# Load the VGG16 model
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
inputShape = (224, 224, 3)  # Assumes 3 channel image

# file path we want to test
dirname_path = 'test_data'
# Get the total number of images
count = 0
for root, dirs, files in os.walk(dirname_path):
    for each in files:
        if each != '.DS_Store':
            count += 1
images_array = np.zeros((count, 224, 224, 3))

t = 0
vgg16_feature_list = []
true_label = []
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
        image = load_img(img_path, target_size=inputShape)
        image = img_to_array(image)  # shape is (224,224,3)
        images_array[t] = image  # (224, 224, 3). float64
        t += 1
        image = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)
        image = image / 255.0
        preds = base_model.predict(image)  # (1, 7, 7, 512). float32
        vgg16_feature_np = np.array(preds)  # (1, 7, 7, 512). array
        vgg16_feature_list.append(vgg16_feature_np.flatten())  # (n, 25088)

    class_item += 1  # category number

# put all feature vector into one list
vgg16_feature_list_np = np.array(vgg16_feature_list)
# create 9 lists to contain the index of normal flower and the index of eight types of mutant flower.
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

# Raw Images and Feature Vectors
# The average/minimum distance between each abnormal flower and all normal flower
whole_ave_image = []
whole_ave_feature = []
for t in range(len(mutation_index)):
    sig_ave_image = []
    sig_ave_feature = []
    mutation_list = []
    for i in mutation_index[t][0]:
        num_1 = 0
        ave_dis_image = 0
        ave_dis_feature = 0
        mutation_list.append(name_list[i])
        for j in normal_index[0]:
            num_1 += 1
            ave_dis_image += np.linalg.norm(np.array(img_to_array(images_array[i])) -
                                              np.array(img_to_array(images_array[j])))
            ave_dis_feature += np.linalg.norm(vgg16_feature_list_np[i] -
                                          vgg16_feature_list_np[j])
        sig_ave_image.append(float("%.2f" % (ave_dis_image/num_1)))
        sig_ave_feature.append(float("%.2f" % (ave_dis_feature/num_1)))

    whole_ave_image.append(sig_ave_image)
    whole_ave_feature.append(sig_ave_feature)
    print("file name: ", mutation_list)
print("eight mutant groups, average, raw images: ", whole_ave_image)
print("eight mutant groups, average, feature vectors: ", whole_ave_feature)


# Baseline: The average distance between each parif of normal flowers
normal_ave_image = []
normal_ave_feature = []
for i in normal_index[0]:
    noraml_distance_image = 0
    noraml_distance_feature = 0
    num_2 = 0
    for j in normal_index[0]:
        if i != j:
            num_2 += 1
            noraml_distance_feature += np.linalg.norm(vgg16_feature_list_np[i] -
                                      vgg16_feature_list_np[j])
            noraml_distance_image += np.linalg.norm(np.array(img_to_array(images_array[i])) -
                                  np.array(img_to_array(images_array[j])))
    normal_ave_image.append(noraml_distance_image / num_2)
    normal_ave_feature.append(noraml_distance_feature / num_2)

ave_rawImage = "%.2f" % (int(sum(normal_ave_image))/len(normal_ave_image))
ave_featureVectors = "%.2f" % (int(sum(normal_ave_feature))/len(normal_ave_feature))
print("baseline, average, raw image: ", ave_rawImage)
print("baseline, average, feature vectors: ", ave_featureVectors)


