import os

import adapt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.datasets import make_moons

from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Flatten, Reshape, GaussianNoise, BatchNormalization
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.regularizers import l2

from adapt.feature_based import DANN, ADDA, DeepCORAL, CORAL, MCD, MDD, WDGRL, CDAN
import glob

#%% data

# simulation data
data = []
filename = []
path = '.../Metals 03182024/'
for file_name in glob.glob(path+'*.csv'):
    x = pd.read_csv(file_name)
    data.append(x.values.tolist())
    filename.append(file_name[44:-4])

loaded_images = np.array(data)
crystal_system = [x[:3] for x in filename]
crystal_system_label = (np.array(crystal_system)=='BCC')*0 + (np.array(crystal_system)=='FCC')*1 + (np.array(crystal_system)=='HCP')*2

for i in range(1):
    plt.imshow(loaded_images_resized[i], cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()


import cv2
resize_scale = 10

loaded_images_resized = np.zeros((loaded_images.shape[0], round(loaded_images.shape[1] / resize_scale), round(loaded_images.shape[2] / resize_scale)))
for i in range(len(loaded_images)):
    loaded_images_resized[i] = cv2.resize(np.array(loaded_images[i], dtype='uint8'), (round(loaded_images.shape[2] / resize_scale), round(loaded_images.shape[1] / resize_scale)), interpolation=cv2.INTER_AREA)


Xs = np.repeat(loaded_images_resized[:, :, :, np.newaxis], 1, axis=3)
Xs = Xs/np.max(Xs)
ys = crystal_system_label



# synthetic data
# noise
noise = np.random.normal(0, 1, loaded_images.shape) 
loaded_images_noised = loaded_images + noise


loaded_images_noised_resized = np.zeros((loaded_images.shape[0], round(loaded_images.shape[1] / resize_scale), round(loaded_images.shape[2] / resize_scale)))
for i in range(len(loaded_images)):
    loaded_images_noised_resized[i] = cv2.resize(np.array(loaded_images_noised[i], dtype='uint8'), (round(loaded_images.shape[2] / resize_scale), round(loaded_images.shape[1] / resize_scale)), interpolation=cv2.INTER_AREA)

Xt = np.repeat(loaded_images_noised_resized[:, :, :, np.newaxis], 1, axis=3)
Xt = Xt/np.max(Xt)
yt = ys


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(sparse_output=False)
one.fit(np.array(ys).reshape(-1, 1))
ys_lab = one.transform(np.array(ys).reshape(-1, 1))
yt_lab = one.transform(np.array(yt).reshape(-1, 1))



#%% Discriminator
def get_discriminator():
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))
    # model.compile(optimizer=Adam(0.001), loss='mse')
    return model


#%% Encoder
class DeepResNet(tf.keras.Model):
  """Defines a multi-layer residual network."""
  def __init__(self, num_classes, kernel_size=2, num_conv2d_layers=3, num_dense_layers=3,
                num_conv2d_hidden=64, num_dense_hidden=64, dropout_rate=0.1, spec_norm_bound=0.9, **classifier_kwargs):
    super().__init__()
    # Defines class meta data.
    self.kernel_size = kernel_size
    self.num_conv2d_hidden = num_conv2d_hidden
    self.num_dense_hidden = num_dense_hidden
    self.num_conv2d_layers = num_conv2d_layers
    self.num_dense_layers = num_dense_layers
    self.dropout_rate = dropout_rate
    self.classifier_kwargs = classifier_kwargs
    self.spec_norm_bound = spec_norm_bound
    
    # Defines the hidden layers.
    self.input_layer = tf.keras.layers.Conv2D(filters=self.num_conv2d_hidden, kernel_size=self.kernel_size, trainable=False) #
    self.conv2d_layers = [self.make_conv2d_layer() for _ in range(num_conv2d_layers)]

    
  def call(self, inputs, return_latent=False):
    # Projects the 2d input data to high dimension.
    hidden = self.input_layer(inputs)
    
    # Computes the ResNet hidden representations.
    for i in range(self.num_conv2d_layers):
        resid = self.conv2d_layers[i](hidden)
        hidden = tf.keras.layers.MaxPooling2D((2,2), strides=1, padding='same')(resid) #
        hidden += resid
    
    hidden = tf.keras.layers.Flatten()(hidden)

    if return_latent:
        return hidden
    
    return hidden
    
  def make_conv2d_layer(self):
    """Uses the Conv2d layer as the hidden layer."""
    conv2d_layer = tf.keras.layers.Conv2D(filters=self.num_conv2d_hidden, kernel_size=self.kernel_size, strides=1, padding='same', activation='relu') 
    return  spectral_normalization.SpectralNormalizationConv2D( 
        conv2d_layer, norm_multiplier=self.spec_norm_bound)# 




def get_encoder():  
    resnet_config = dict(num_classes=1, kernel_size=(3,3),
                         num_conv2d_layers=3, num_dense_layers=2, 
                         num_conv2d_hidden=32, num_dense_hidden=64)
    
    resnet_model = DeepResNet(**resnet_config)
    return resnet_model

#%% Task
# GP
class DeepResNet_GP(tf.keras.Model):
  """Defines a multi-layer residual network."""
  def __init__(self, num_classes, num_layers=3, num_hidden=128,
               dropout_rate=0.1, **classifier_kwargs):
    super().__init__()
    self.classifier_kwargs = classifier_kwargs


    # Defines the output layer.
    self.classifier = self.make_output_layer(num_classes)

  def call(self, inputs):
    # Projects the 2d input data to high dimension.
    hidden = inputs

    return self.classifier(hidden)


  def make_output_layer(self, num_classes):
    """Uses the Dense layer as the output layer."""
    return tf.keras.layers.Dense(
        num_classes, **self.classifier_kwargs)

# The SNGP model
# Define SNGP model

import sys
import gaussian_process
import spectral_normalization

class DeepResNetSNGP(DeepResNet_GP):
  def __init__(self, spec_norm_bound=0.95, **kwargs):
    self.spec_norm_bound = spec_norm_bound
    super().__init__(**kwargs)

  def make_output_layer(self, num_classes):
    """Uses Gaussian process as the output layer."""
    return gaussian_process.RandomFeatureGaussianProcess( 
        num_classes,
        gp_cov_momentum=-1,
        **self.classifier_kwargs)#nlp_layers.gaussian_process.

  def call(self, inputs, training=False, return_covmat=False):
    # Gets logits and a covariance matrix from the GP layer.
    logits, covmat = super().call(inputs)

    # Returns only logits during training.
    if not training and return_covmat:
      return logits, covmat

    return logits



class ResetCovarianceCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs=None):
    """Resets covariance matrix at the beginning of the epoch."""
    if epoch > 0:
      self.model.classifier.reset_covariance_matrix()

class DeepResNetSNGPWithCovReset(DeepResNetSNGP):
  def fit(self, *args, **kwargs):
    """Adds ResetCovarianceCallback to model callbacks."""
    kwargs["callbacks"] = list(kwargs.get("callbacks", []))
    kwargs["callbacks"].append(ResetCovarianceCallback())

    return super().fit(*args, **kwargs)



def get_task():
    resnet_config = dict(num_classes=3, num_layers=6, num_hidden=128)
    sngp_model = DeepResNetSNGPWithCovReset(**resnet_config)
    
    return sngp_model


#%% ADDA

adda = ADDA(get_encoder(), get_task(), get_discriminator(),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer=Adam(0.00001, beta_1=0.5),
            copy=True, metrics=["acc"], random_state=0)


adda.fit(Xs, ys_lab, Xt, yt_lab, epochs=100, batch_size=32, verbose=1);
pd.DataFrame(adda.history_).plot(figsize=(8, 5))
plt.title("Training history", fontsize=14); plt.xlabel("Epochs"); plt.ylabel("Scores")
plt.legend(ncol=2)
plt.show()


sngp_logits, sngp_covmat = adda.task_(adda.encoder_.predict(Xt), return_covmat=True)
sngp_variance = tf.linalg.diag_part(sngp_covmat)[:, None]
sngp_logits_adjusted = sngp_logits / tf.sqrt(1. + (np.pi / 8.) * sngp_variance)
sngp_probs = tf.nn.softmax(sngp_logits_adjusted, axis=-1)
yt_pred = np.argmax(sngp_probs, axis=1)
acc = accuracy_score(yt, yt_pred)

