###########################################################
###  PRACTICAL MACHINE LEARNING, AI UNIBUC              ###
###  UNSUPERVISED LEARNING - IMAGE CLUSTERING           ###
###  STUDENT: SOTIR ANCA-NICOLETA (GROUP 407)           ###
###########################################################

#=========================================================#
# RETINAL OPTICAL COHERENCE TOMOGRAPHY IMAGES DATASET     #
# SUPERVISED APPROACH: CONVOLUTIONAL NEURAL NETWORK       #
#=========================================================#



!mkdir ~/.kaggle
!cp /content/drive/MyDrive/PML/kaggle.json ~/.kaggle/ # kaggle.json is required to use the kaggle api on colab
!chmod 600 ~/.kaggle/kaggle.json # change permissions

!kaggle datasets download -d paultimothymooney/kermany2018 # download the retinal otc images dataset

!mkdir retinal_oct_images # the folder in which to unzip
import zipfile
with zipfile.ZipFile('/content/kermany2018.zip', 'r') as zip_file:
    zip_file.extractall('/content/retinal_oct_images')
!rm /content/kermany2018.zip # delete the archive after unzipping to free space

# organize folders and data
!rm /content/retinal_oct_images/oct2017 -r
!mv "/content/retinal_oct_images/OCT2017 " /content/retinal_oct_images/OCT2017
!mv /content/retinal_oct_images/OCT2017/val/CNV/* /content/retinal_oct_images/OCT2017/test/CNV
!mv /content/retinal_oct_images/OCT2017/val/DME/* /content/retinal_oct_images/OCT2017/test/DME
!mv /content/retinal_oct_images/OCT2017/val/DRUSEN/* /content/retinal_oct_images/OCT2017/test/DRUSEN
!mv /content/retinal_oct_images/OCT2017/val/NORMAL/* /content/retinal_oct_images/OCT2017/test/NORMAL
!rm /content/retinal_oct_images/OCT2017/val -r
!mv /content/retinal_oct_images/OCT2017/test /content/retinal_oct_images/OCT2017/valid



import os
import random
import numpy as np
import warnings; warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from skimage.io import ImageCollection, imread, imshow, imsave
from skimage.transform import resize
from skimage.restoration import denoise_bilateral
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import data
from sklearn.metrics import accuracy_score



sample_imgs_path = [] # the path to some sample images for visualization
sample_imgs_path.append('/content/retinal_oct_images/OCT2017/train/CNV/CNV-1016042-1.jpeg')
sample_imgs_path.append('/content/retinal_oct_images/OCT2017/train/DME/DME-1072015-1.jpeg')
sample_imgs_path.append('/content/retinal_oct_images/OCT2017/train/DRUSEN/DRUSEN-1001666-1.jpeg')
sample_imgs_path.append('/content/retinal_oct_images/OCT2017/train/NORMAL/NORMAL-1001666-1.jpeg')

sample_imgs = [imread(path) for path in sample_imgs_path]

# plot the sample images
fig, axs = plt.subplots(1, 4, figsize = (20, 3))
axs[0].imshow(sample_imgs[0])
axs[0].set_title('CNV')
axs[1].imshow(sample_imgs[1])
axs[1].set_title('DME')
axs[2].imshow(sample_imgs[2])
axs[2].set_title('DRUSEN')
axs[3].imshow(sample_imgs[3])
axs[3].set_title('NORMAL')
fig.show()



# file organization:

# /content/retinal_oct_images/OCT2017/train
# |------ CNV
# |       |---- img1.jpeg
# |       |---- img2.jpeg ...
# |------ DME
# |       |---- img1.jpeg
# |       |---- img2.jpeg ...
# |------ DRUSEN
# |       |---- img1.jpeg
# |       |---- img2.jpeg ...
# |------ NORMAL
#         |---- img1.jpeg
#         |---- img2.jpeg ...


# read the data
train_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/train', 
                                            labels = 'inferred', # deduces the labels based on the subfolder
                                            label_mode = 'categorical',
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 16,
                                            validation_split = 0.2, # 20% of training data is reserved for validation
                                            subset = 'training', # extract the training data for this call
                                            seed = 5) # seed must be fixed so no overlapping occurs on the next call to extract validation set from the same directory
print()
valid_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/train', 
                                            labels = 'inferred', # deduces the labels based on the subfolder
                                            label_mode = 'categorical',
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 16,
                                            validation_split = 0.2, # 20% of training data is reserved for validation
                                            subset = 'validation', # extract the validation data for this call
                                            seed = 5) # set the same seed as for the call above so no overlapping occurs

test_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/valid', 
                                            labels = 'inferred', # deduces the labels based on the subfolder
                                            label_mode = 'categorical',
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 16)

preprocess_layer = Rescaling(1./255) # used to get pixel values in range [0, 1]

train_dataset = train_dataset.map(lambda X, y: (preprocess_layer(X), y))
valid_dataset = valid_dataset.map(lambda X, y: (preprocess_layer(X), y))
test_dataset = test_dataset.map(lambda X, y: (preprocess_layer(X), y))


# organize checkpoint dirs
!mkdir model_data
!mkdir /content/model_data/checkpoints
!mkdir /content/model_data/logs


# %load_ext tensorboard
# import datetime, os



import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop 
from tensorflow.math import confusion_matrix
from tensorflow.keras.models import load_model

# define checkpoints location for model and for tensorboard
checkpoint_dir = '/content/drive/MyDrive/PML/Project2/cnn/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(checkpoint_dir, 'checkpoints/model.{epoch:05d}.hdf5')
)

log_dir = os.path.join('/content/model_data/logs/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

# %tensorboard --logdir /content/model_data/logs



# define the convolutional network
model = Sequential([
  BatchNormalization(),
  Conv2D(filters = 16, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'), 
  Conv2D(filters = 16, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Flatten(),
  Dropout(0.4),
  Dense(4, activation='softmax') 
])


# fitting the model
optimizer = Adam()

model.compile(optimizer = optimizer,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_dataset,
          epochs = 50, batch_size = 16, initial_epoch = 0,
          callbacks = [checkpoint_callback, tensorboard_callback],
          validation_data = valid_dataset)


# predict on test data
X_test = test_dataset.map(lambda X, y: X)

y_test = [] # true labels for the test data
for X, y in test_dataset:
  y_test.append(y)
y_test = np.argmax(np.concatenate(np.asarray(y_test)), axis = 1) # get the class from the one hot representation inferred by image_dataset_from_directory()

best_epoch = 5
best_model = load_model('/content/drive/MyDrive/PML/Project2/cnn/checkpoints/model.%05d.hdf5' % best_epoch)
y_pred = np.argmax(best_model.predict(X_test), axis = 1) # get the class with the highest probability

# y_pred = np.argmax(model.predict(X_test), axis = 1)

conf_matrix = confusion_matrix(y_test, y_pred, num_classes=4)

accuracy_score(y_test, y_pred)

# visualizing the confusion matrix
import seaborn as sns
counts = ['{0:0.0f}'.format(value) for value in np.asarray(conf_matrix).flatten()] # number of entries in each cell of the matrix
percentages = ['{0:.2%}'.format(value) for value in np.asarray(conf_matrix).flatten()/np.asarray(conf_matrix).sum()] # percentage in each cell of the total number of entries
labels = np.asarray([f'{v1}\n{v2}' for v1, v2 in zip(counts, percentages)]).reshape(4, 4) # write number of entries and percentage in each cell
categories = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Purples', xticklabels = categories, yticklabels = categories)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.yticks(rotation = 0)

