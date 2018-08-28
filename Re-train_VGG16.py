from keras.applications import VGG16
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time


img_size=224 # size of image
batch_size=10  # number of images be process each time

# Data Augmentation
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        shear_range=0.2,
        zoom_range=0.2
)

# Data Augmentation
test_datagen = image.ImageDataGenerator(rescale=1. / 255)

#
train_dir = 'data/2/train'
valid_dir = 'data/2/validation'


train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')


############ Create VGG-16 network graph without the last layers and load imagenet pretrained weights
print('loading the model and the pre-trained weights...')
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
for layer in base_model.layers:
    layer.trainable = False

# ############ Add the top as per number of classes in our dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

############ Specify the complete model input and output, optimizer and loss
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# save the optimal weights
filepath = 'new_weights/GAP_D_Drop_D_D_6.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min',period=1)
callbacks_list = [checkpoint, tensorboard]

model.fit_generator(
        train_generator,
        epochs=20,
        callbacks=callbacks_list,
        validation_data=validation_generator
        )
