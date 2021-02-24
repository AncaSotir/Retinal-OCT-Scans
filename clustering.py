###########################################################
###  PRACTICAL MACHINE LEARNING, AI UNIBUC              ###
###  UNSUPERVISED LEARNING - IMAGE CLUSTERING           ###
###  STUDENT: SOTIR ANCA-NICOLETA (GROUP 407)           ###
###########################################################

#=========================================================#
# RETINAL OPTICAL COHERENCE TOMOGRAPHY IMAGES DATASET     #
#                                                         #
# CLUSTERING:                                             #
#   * K-means                                             #
#   * DBSCAN                                              #
#                                                         #
# Feature extraction methods:                             #
#   * Pixel values                                        #
#   * Autoencoder                                         #
#   * Histogram of Oriented Gradients (HOG)               #
#   * Pre-trained VGG16                                   #
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
                                            batch_size = 1,
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
                                            batch_size = 1,
                                            validation_split = 0.2, # 20% of training data is reserved for validation
                                            subset = 'validation', # extract the validation data for this call
                                            seed = 5) # set the same seed as for the call above so no overlapping occurs

test_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/valid', 
                                            labels = 'inferred', # deduces the labels based on the subfolder
                                            label_mode = 'categorical',
                                            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                            color_mode = 'grayscale', # adds one channel to the images instead of three
                                            image_size = (128, 128), # resizes images
                                            batch_size = 1)

preprocess_layer = Rescaling(1./255) # used to get pixel values in range [0, 1]

train_dataset = train_dataset.map(lambda X, y: (preprocess_layer(X), y))
valid_dataset = valid_dataset.map(lambda X, y: (preprocess_layer(X), y))
test_dataset = test_dataset.map(lambda X, y: (preprocess_layer(X), y))



# PLOTTING FUNCTIONS FROM THE CLUSTERING LAB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns; sns.set(style='white')

from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

from pylab import rcParams
rcParams['figure.figsize'] = 20, 20

COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']
MARKERS = ['o', 'v', 's', '<', '>', '8', '^', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

def plot2d(X, y_pred, y_true, mode=None, centroids=None):
    transformer = None
    X_r = X
    
    if mode is not None:
        transformer = mode(n_components=2)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 2, 'plot2d only works with 2-dimensional data'


    plt.grid()
    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        plt.plot(ix[0], ix[1], 
                    c=COLORS[iyp], 
                    marker=MARKERS[iyt])
        
    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            plt.plot(cx[0], cx[1], 
                        marker=MARKERS[-1], 
                        markersize=10,
                        c='red')
            
    plt.savefig('/content/2d.png', bbox_inches='tight')
    plt.close()

def plot3d(X, y_pred, y_true, mode=None, centroids=None):
    transformer = None
    X_r = X
    if mode is not None:
        transformer = mode(n_components=3)
        X_r = transformer.fit_transform(X)

    assert X_r.shape[1] == 3, 'plot2d only works with 3-dimensional data'

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.elev = 30
    ax.azim = 120

    for ix, iyp, iyt in zip(X_r, y_pred, y_true):
        ax.plot(xs=[ix[0]], ys=[ix[1]], zs=[ix[2]], zdir='z',
                    c=COLORS[iyp], 
                    marker=MARKERS[iyt])
        
    if centroids is not None:
        C_r = centroids
        if transformer is not None:
            C_r = transformer.fit_transform(centroids)
        for cx in C_r:
            ax.plot(xs=[cx[0]], ys=[cx[1]], zs=[cx[2]], zdir='z',
                        marker=MARKERS[-1], 
                        markersize=10,
                        c='red')
            
    plt.savefig('/content/3d.png', bbox_inches='tight')
    plt.close()


# define a function to display both plot2d and plot3d results
def plot_both(X, y_pred, y_true, mode2d = None, centroids2d = None, mode3d = None, centroids3d = None):
  plot2d(X, y_pred, y_true, mode2d, centroids2d)
  plot3d(X, y_pred, y_true, mode3d, centroids3d)
  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(plt.imread('/content/2d.png'))
  axs[1].imshow(plt.imread('/content/3d.png'))
  axs[0].set_xticks([])
  axs[0].set_yticks([])
  axs[1].set_xticks([])
  axs[1].set_yticks([])
  axs[0].spines["top"].set_visible(False)
  axs[0].spines["right"].set_visible(False)
  axs[0].spines["bottom"].set_visible(False)
  axs[0].spines["left"].set_visible(False)
  axs[1].spines["top"].set_visible(False)
  axs[1].spines["right"].set_visible(False)
  axs[1].spines["bottom"].set_visible(False)
  axs[1].spines["left"].set_visible(False)
  plt.show()




from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = [] # load pixel values
y = [] # load labels
for (xi, yi) in test_dataset:
  X.append(np.asarray(xi).reshape(128*128))
  y.append(np.argmax(yi)) # extract class value from one hot representation returned by image_dataset_from_directory
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# define a function for the heuristic used to choose DBSCAN hyperparameters
def dbscan_tuning(X):
  dist_matrix = pairwise_distances(X, X) # distance from each entry to all other entries
  ks = [3, 4, 5] # 3rd, 4th, 5th neighbor
  plt.figure(figsize = (10,10))
  for k in ks:
    # calculate the distance from each point to the kth nearest neighbor
    kth_neighbor_dist = [dist_matrix[i][np.argsort(dist_matrix[i])[k]] for i in range(len(X))]
    plt.grid(True, 'both') # display gridlines on the plot
    plt.plot(np.sort(kth_neighbor_dist), label = f'k = {k}') # plot the distances in ascending order
  plt.xlabel('Point index')
  plt.ylabel(r'Distance to the $k^{th}$ nearest neighbor')
  plt.title(f'DBSCAN hyperparameter tuning: heuristic for choosing eps and m = k')
  plt.legend()
  plt.show()


# define a function to calculate the accuracy of a clustering model
# it takes into account all possible class assignments and returns the greatest accuracy found (and also the class index combo)
def find_accuracy(y_true, y_pred):
  max_accuracy = 0
  max_acc_label_assign = None # this is the class index combination which leads to the best accuracy
  for i in range(4): # assignment for the first class
    for j in range(4): # assignment for the second class
      if i==j: continue # two classes cannot have the same index
      for k in range(4): # assignment for the third class
        if i==k or j==k: continue # the third class can't be equal with one of the first two
        l = 6 - i - j - k # infer the fourth class from the first three assigned
        label_assign = {-1: -1, 0: i, 1: j, 2: k, 3: l} # outliers will not be counted towards an existing class (-1: -1)
        pred = [label_assign[yi] for yi in y_pred] # transform the predictions based on the label assignment
        acc = np.sum([1 for i in range(len(y_true)) if y_true[i] == pred[i]])/len(y_true) # compute the accuracy
        if acc > max_accuracy: # retain only the best accuracy value and label assignment
          max_accuracy = acc
          max_acc_label_assign = label_assign
  return max_accuracy, max_acc_label_assign



# FEATURES -> PIXEL VALUES

# visualize pixel value features in 2D and 3D
plot_both(X, y, y, mode2d = TSNE, mode3d = PCA)

# apply K-means on pixel value features and visualize results
kmeans_pixel = KMeans(n_clusters = 4, random_state = 5).fit(X)
plot_both(X, kmeans_pixel.labels_, y, mode2d = TSNE, mode3d = PCA, centroids3d = kmeans_pixel.cluster_centers_)
print(f'NMI: {normalized_mutual_info_score(y, kmeans_pixel.labels_)}')
print(f'ACC: {find_accuracy(y, kmeans_pixel.labels_)}')
# find DBSCAN parameters
dbscan_tuning(X)
# apply DBSCAN on pixel value features and visualize results
dbscan_pixel = DBSCAN(eps = 130, min_samples = 5).fit(X)
plot_both(X, dbscan_pixel.labels_, y, mode2d = TSNE, mode3d = PCA)
print(f'NMI: {normalized_mutual_info_score(y, dbscan_pixel.labels_)}')
print(f'ACC: {find_accuracy(y, dbscan_pixel.labels_)}')



# FEATURES -> ENCODER

import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential


# define a function that discards the decoder to have acces to the encoder output only
def extract_encoder(autoencoder, middle_layer_idx = 14):
  encoder = Sequential() # create a new model
  for i in range(middle_layer_idx + 1): # only copy layers until the middle layer
    encoder.add(autoencoder.get_layer(index = i)) # copy layer in the new model
  return encoder

# load the trained autoencoder and extract the encoder layers to transform the inputs
encoder = extract_encoder(load_model('/content/drive/MyDrive/PML/Project2/autoencoder/autoencoder.hdf5'))
features = encoder.predict(test_dataset)

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features) # standardize the features

# visualize encoder features in 2D and 3D
plot_both(features, y, y, mode2d = TSNE, mode3d = PCA)

# apply K-means on encoder features and visualize results
kmeans_encoder = KMeans(n_clusters = 4, random_state = 5).fit(features)
plot_both(features, kmeans_encoder.labels_, y, mode2d = TSNE, mode3d = PCA, centroids3d = kmeans_encoder.cluster_centers_)
print(f'NMI: {normalized_mutual_info_score(y, kmeans_encoder.labels_)}')
print(f'ACC: {find_accuracy(y, kmeans_encoder.labels_)}')
# find DBSCAN parameters
dbscan_tuning(features)
# apply DBSCAN on encoder features and visualize results
dbscan_encoder = DBSCAN(eps = 11, min_samples = 3).fit(features)
plot_both(features, dbscan_encoder.labels_, y, mode2d = TSNE, mode3d = PCA)
print(f'NMI: {normalized_mutual_info_score(y, dbscan_encoder.labels_)}')
print(f'ACC: {find_accuracy(y, dbscan_encoder.labels_)}')



# FEATURES -> HISTOGRAM OF ORIENTED GRADIENTS ON RECONSTRUCTED IMAGES (AUTOENCODER)

from skimage.feature import hog
from skimage import data, exposure

# reloading the autoencoder model and transforming the images
autoencoder = load_model('/content/drive/MyDrive/PML/Project2/autoencoder/autoencoder.hdf5')
X_reconstructed = autoencoder.predict(test_dataset)

# visualize the original and reconstructed images
fig, axs = plt.subplots(2, 5, figsize = (20, 10))
for i in range(5):
  axs[0][i].imshow(X_reconstructed[i].reshape(128, 128), cmap = 'viridis')
  axs[0][i].set_title('Reconstructed image')
  hog_vals, hog_image = hog(X_reconstructed[i].reshape(128, 128), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, multichannel=False)
  axs[1][i].imshow(exposure.rescale_intensity(hog_image, in_range=(0, 10)), cmap = 'viridis')
  axs[1][i].set_title('HOG image')
plt.show()

# compute the HOG values for all images
hog_features = [hog(reconstructed_img.reshape(128, 128), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=False, multichannel=False) for reconstructed_img in X_reconstructed]

# visualize HOG features in 2D and 3D
plot_both(hog_features, y, y, mode2d = TSNE, mode3d = PCA)

# apply K-means on HOG features and visualize results
kmeans_hog_autoencoder = KMeans(n_clusters = 4, random_state = 5).fit(hog_features)
plot_both(hog_features, kmeans_hog_autoencoder.labels_, y, mode2d = TSNE, mode3d = PCA, centroids3d = kmeans_hog_autoencoder.cluster_centers_)
print(f'NMI: {normalized_mutual_info_score(y, kmeans_hog_autoencoder.labels_)}')
print(f'ACC: {find_accuracy(y, kmeans_hog_autoencoder.labels_)}')
# find DBSCAN parameters
dbscan_tuning(hog_features)
# apply DBSCAN on HOG features and visualize results
dbscan_hog_autoencoder =1 DBSCAN(eps = 11, min_samples = 3).fit(hog_features)
plot_both(hog_features, dbscan_hog_autoencoder.labels_, y, mode2d = TSNE, mode3d = PCA)
print(f'NMI: {normalized_mutual_info_score(y, dbscan_hog_autoencoder.labels_)}')
print(f'ACC: {find_accuracy(y, dbscan_hog_autoencoder.labels_)}')



# FEATURES -> PRE-TRAINED VGG16 WITHOUT FULLY CONNECTED LAYERS

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications import VGG16
from keras.preprocessing.image import img_to_array

# reloading test data with different shape and three channels (required vgg16 input is 224x224x3)
test_dataset = image_dataset_from_directory('/content/retinal_oct_images/OCT2017/valid', 
                                             labels = 'inferred',
                                             label_mode = 'categorical',
                                             class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL'],
                                             image_size = (224, 224),
                                             batch_size = 1)

# building arrays for data and labels
X = []
y = []
for (xi, yi) in test_dataset:
  X.append(np.asarray(xi).reshape(224, 224, 3))
  y.append(np.argmax(yi))

X = np.asarray(X)
X = preprocess_input(X) # the images must be preprocessed like the ones from imagenet used to train vgg16

vgg_model = VGG16( # create the vgg model
    include_top = False, # eliminate the fully connected layers which classify
    input_shape = (224, 224, 3),
    weights = "imagenet", # pre-trained model on imagenet
    input_tensor = None,
    pooling = 'avg' # final average pooling added
)

# show model architecture (layer names and output shapes)
print('Layer index'.ljust(25) + 'Layer name'.ljust(25) + 'Layer output shape')
print('-'*75)
for i in range(len(vgg_model.layers)):
  layer = vgg_model.get_layer(index = i)
  print(str(i).ljust(25), end = '')
  print(layer.name.ljust(25), end = '')
  print(str(layer.output_shape))

# get the vgg features for the test dataset
vgg_features = vgg_model.predict(X)

# visualize vgg features in 2D and 3D
plot_both(vgg_features, y, y, mode2d = TSNE, mode3d = PCA)
# apply K-means on vgg features and visualize results
kmeans_vgg = KMeans(n_clusters = 4, random_state = 5).fit(vgg_features)
plot_both(vgg_features, kmeans_vgg.labels_, y, mode2d = TSNE, mode3d = PCA, centroids3d = kmeans_vgg.cluster_centers_)
print(f'NMI: {normalized_mutual_info_score(y, kmeans_vgg.labels_)}')
print(f'ACC: {find_accuracy(y, kmeans_vgg.labels_)}')
# find DBSCAN parameters
dbscan_tuning(vgg_features)
# apply DBSCAN on vgg features and visualize results
dbscan_vgg = DBSCAN(eps = 40, min_samples = 3).fit(vgg_features)
plot_both(vgg_features, dbscan_vgg.labels_, y, mode2d = TSNE, mode3d = PCA)
print(f'NMI: {normalized_mutual_info_score(y, dbscan_vgg.labels_)}')
print(f'ACC: {find_accuracy(y, dbscan_vgg.labels_)}')
