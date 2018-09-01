from keras.applications import VGG16, Xception
from keras.preprocessing import image
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Feature vectors with classifier (9 categories)

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Load the trained weights by Pre-trained VGG16 model
model.load_weights("GAP_D_Drop_D_D_6.h5")

inputShape = (224, 224) # Assumes 3 channel image

dirname_path = 'test_data'

count = 0
for root,dirs,files in os.walk(dirname_path):    
    for each in files:
        if each != '.DS_Store':
            count += 1

images_array = np.zeros((count, 224, 224, 3))

t = 0
true_label = []
pred_label = []
vgg16_feature_list = []
dir_path_list = os.listdir(dirname_path)
if '.DS_Store' in dir_path_list:
    dir_path_list.remove('.DS_Store')
dir_path_list.sort()
class_item = 0
fileName_list = []
for dirname in dir_path_list:
    file_path = dirname_path + '/' + dirname
    file_path_list = os.listdir(file_path)
    if '.DS_Store' in file_path_list:
        file_path_list.remove('.DS_Store')
    file_path_list.sort()
    for filename in file_path_list:
        img_path = file_path + '/' + filename
        fileName_list.append(filename)
        true_label.append(class_item)
        image = load_img(img_path, target_size=inputShape)
        image = img_to_array(image)
        images_array[t] = image
        t += 1
        image = np.expand_dims(image, axis=0)  # Now shape is (1,224,224,3)
        image = image / 255.0
        preds = model.predict(image)
        index_ = preds.argmax()
        pred_label.append(index_)
        vgg16_feature_np = np.array(preds)  # (1, 7, 7, 512). array
        vgg16_feature_list.append(vgg16_feature_np.flatten())  # (n, 25088)

    class_item += 1

vgg16_feature_list_np = np.array(vgg16_feature_list)
print(vgg16_feature_list_np)
normal_index = []
mutation_index = []
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


# The average distance between each mutant flower image and whole normal images
whole_ave_feature = []
for t in range(len(mutation_index)):
    sig_ave_feature = []
    mutation_list = []
    for i in mutation_index[t][0]:
        num_1 = 0
        ave_dis_feature = 0
        mutation_list.append(fileName_list[i])
        for j in normal_index[0]:
            ave_dis_feature += np.linalg.norm(np.array(vgg16_feature_list_np[i]) -
                                                np.array(vgg16_feature_list_np[j]))
            num_1 += 1

        sig_ave_feature.append(float("%.8f" % (ave_dis_feature / num_1)))
    whole_ave_feature.append(sig_ave_feature)
    print("file name: ", mutation_list)
print("Five mutant groups, average, feature vectors: ", whole_ave_feature)

# Baseline
normal_ave_feature = []
for i in normal_index[0]:
    noraml_distance_feature = 0
    num_2 = 0
    for j in normal_index[0]:
        if i != j:
            num_2 += 1
            noraml_distance_feature += np.linalg.norm(np.array(vgg16_feature_list_np[i]) -
                                            np.array(vgg16_feature_list_np[j]))


    normal_ave_feature.append(noraml_distance_feature / num_2)

ave_featureVectors = "%.2f" % (int(sum(normal_ave_feature))/len(normal_ave_feature))
print("baseline, average, feature vectors: ", ave_featureVectors)







