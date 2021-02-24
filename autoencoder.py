###########################################################
###  PRACTICAL MACHINE LEARNING, AI UNIBUC              ###
###  UNSUPERVISED LEARNING - IMAGE CLUSTERING           ###
###  STUDENT: SOTIR ANCA-NICOLETA (GROUP 407)           ###
###########################################################

#=========================================================#
# RETINAL OPTICAL COHERENCE TOMOGRAPHY IMAGES DATASET     #
# AUTOENCODER: UNSUPERVISED FEATURE EXTRACTION MODEL      #
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
                                            label_mode = None, # no need for labels
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 1,
                                            validation_split = 0.2, # 20% of training data is reserved for validation
                                            subset = 'training', # extract the training data for this call
                                            seed = 5) # seed must be fixed so no overlapping occurs on the next call to extract validation set from the same directory
print()
valid_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/train', 
                                            label_mode = None, # no need for labels
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 1,
                                            validation_split = 0.2, # 20% of training data is reserved for validation
                                            subset = 'validation', # extract the validation data for this call
                                            seed = 5) # set the same seed as for the call above so no overlapping occurs

test_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/valid', 
                                            label_mode = None, # no need for labels
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 1)

preprocess_layer = Rescaling(1./255) # used to get pixel values in range [0, 1]

train_dataset = train_dataset.map(lambda X: preprocess_layer(X))
valid_dataset = valid_dataset.map(lambda X: preprocess_layer(X))
test_dataset = test_dataset.map(lambda X: preprocess_layer(X))


# organize checkpoint dirs
!mkdir model_data
!mkdir /content/model_data/checkpoints
!mkdir /content/model_data/logs


# %load_ext tensorboard
# import datetime, os



import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D, Flatten, Dropout, Dense, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.math import confusion_matrix
from tensorflow.keras.losses import MeanSquaredError

# define checkpoints location for model and for tensorboard
checkpoint_dir = '/content/drive/MyDrive/PML/Project2/autoencoder/'
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = os.path.join(checkpoint_dir, 'checkpoints/model.{epoch:05d}.hdf5')
)

log_dir = os.path.join('/content/model_data/logs/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

# %tensorboard --logdir /content/model_data/logs


# convert datasets to numpy arrays
X_train = np.stack(list(train_dataset))
X_valid = np.stack(list(valid_dataset))
X_test = np.stack(list(test_dataset))

# reshape the arrays for the autoencoder
X_train = X_train.reshape(len(X_train), 128, 128, 1)
X_valid = X_valid.reshape(len(X_valid), 128, 128, 1)
X_test = X_test.reshape(len(X_test), 128, 128, 1)



# define the autoencoder (the encoder is similar to the convolutional network)
model = Sequential([
  BatchNormalization(input_shape = (128, 128, 1)),

  # encoder: 4 sequences of 2 convolutions followed by avg pooling
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2D(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  AvgPool2D(pool_size=2, strides=(2, 2), padding='same'),
  Flatten(),

  Dense(100), # encoder output; these are the features which will be extracted

  # decoder: "mirror" the encoder - 4 sequences of upsampling and 2 transposed convolutions
  Dense(2048),
  Reshape((8, 8, 32)),
  UpSampling2D((2,2)),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  UpSampling2D((2,2)),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  UpSampling2D((2,2)),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  UpSampling2D((2,2)),
  Conv2DTranspose(filters = 32, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same'),
  Conv2DTranspose(filters = 1, kernel_size = (3, 3), kernel_initializer = 'he_uniform', activation = 'relu', padding = 'same')
])


# fitting the model
model.compile(loss = MeanSquaredError(), optimizer = Adam())

model.fit(X_train, X_train,
          epochs = 50, initial_epoch = 0, batch_size = 16,
          callbacks = [checkpoint_callback, tensorboard_callback],
          validation_data = (X_valid, X_valid))


# display layer info (name and output shape)
print('Layer index'.ljust(25) + 'Layer name'.ljust(25) + 'Layer output shape')
print('-'*75)
for i in range(len(model.layers)):
  layer = model.get_layer(index = i)
  print(str(i).ljust(25), end = '')
  print(layer.name.ljust(25), end = '')
  print(str(layer.output_shape))


best_epoch = 50
best_model = load_model('/content/drive/MyDrive/PML/Project2/autoencoder/checkpoints/model.%05d.hdf5' % best_epoch)
test_output = best_model.predict(X_test)


# show autoencoder results by plotting some original vs reconstructed images
fig, axs = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
  axs[0][i].imshow(X_test[i].reshape(128, 128))
  axs[0][i].set_title('Original image')
  axs[1][i].imshow(test_output[i].reshape(128, 128))
  axs[1][i].set_title('Reconstructed image')
plt.tight_layout()
plt.show()


# define a function that discards the decoder to have acces to the encoder output only
def extract_encoder(autoencoder, middle_layer_idx = 14):
  encoder = Sequential() # create a new model
  for i in range(middle_layer_idx + 1): # only copy layers until the middle layer
    encoder.add(autoencoder.get_layer(index = i)) # copy layer in the new model
  return encoder


encoder = extract_encoder(best_model)
encoder_output = encoder.predict(X_test)


# show autoencoder and encoder results
fig, axs = plt.subplots(3, 5, figsize = (20, 13))
for i in range(5):
  axs[0][i].imshow(X_test[i].reshape(128, 128))
  axs[0][i].set_title('Original image')
  axs[1][i].imshow(test_output[i].reshape(128, 128))
  axs[1][i].set_title('Reconstructed image')
  axs[2][i].imshow(encoder_output[i].reshape(10, 10))
  axs[2][i].set_title('Encoder result')
  axs[2][i].set_xticks([])
  axs[2][i].set_yticks([])
plt.show()


# save the model to be used for clustering
save_model(best_model, '/content/drive/MyDrive/PML/Project2/autoencoder/autoencoder.hdf5')

# how to load the model later, when clustering
# loaded_encoder = extract_encoder(load_model('/content/drive/MyDrive/PML/Project2/autoencoder/autoencoder.hdf5'))
# loaded_encoder_output = loaded_encoder.predict(X_test)

