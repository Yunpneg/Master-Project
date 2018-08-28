from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights("new_weights/GAP_D_Drop_D_D_6.h5")

inputShape = (224, 224) # Assumes 3 channel image

dirname_path = 'data/2/test'

# Get the total number of images
count = 0
for root,dirs,files in os.walk(dirname_path):    #遍历统计
    for each in files:
        if each != '.DS_Store':
            count += 1   #统计文件夹下文件个数

images_array = np.zeros((count, 224, 224, 3))

t = 0
true_label = []     # Ground Truth
pred_label = []     # Prediction labels
dir_path_list = os.listdir(dirname_path)
if '.DS_Store' in dir_path_list:
    dir_path_list.remove('.DS_Store')
dir_path_list.sort()
class_item = 0  # index of category
for dirname in dir_path_list:  # listdir的参数是文件夹的路径
    file_path = dirname_path + '/' + dirname
    file_path_list = os.listdir(file_path)
    if '.DS_Store' in file_path_list:
        file_path_list.remove('.DS_Store')
    file_path_list.sort()
    for filename in file_path_list:
        img_path = file_path + '/' + filename
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

    class_item += 1

print("Ground Turth: ", true_label)
print("predicted label: ", pred_label)

# Get the predicted accuracy
j = 0
arry_same = []
while(j < len(true_label)):
    if true_label[j] == pred_label[j]:
        arry_same.append(true_label[j])
    j += 1
print(len(arry_same)/len(true_label))


# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm_array = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm_array, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Classification label', fontsize=12)


plot_confusion_matrix(true_label, pred_label)
plt.show()
