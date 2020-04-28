# File Name : GAN_train.py
# Main Reference : https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

from library.directory_handle import DirectoryHandle
import library.image_handle as ImageHandle
import library.data_handle  as DataHandle
import library.command_handle as CommandHandle

# Import library for plot image
import matplotlib.pyplot as plt

# Import library for manage model part Core Layers
from keras.layers import Input, Flatten, Dense, Reshape, Lambda
# Import library for manage model part Convolution Layers
from keras.layers import Conv2D, Conv2DTranspose
# Import library for mange model part activatin
from keras.layers import Activation, ReLU, LeakyReLU
# Import Library for manage model part Model Object
from keras.models import Model, Sequential
# Import Library for manage model part optimizer
from keras.optimizers import Adam
# Import Library about model 
from keras.utils import plot_model
# Import library for load model
from keras.models import load_model
# Import library operation in Keras tensor object
from keras import backend as K
K.clear_session()
#Import library for normal process
import numpy as np

# ================> Part Parameter Program
_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/AnimeFaceData"
_CROP = True
_COLOR = True
_RATIO = 8
_EPOCHES = 30
_LATENT_SIZE = 1024
_ACTIVATION = "relu"
_MODEL_NAME = "GAN3L1024D" # This will use to save model
if _ACTIVATION != None : _MODEL_NAME += _ACTIVATION
_LEARNING_RATE = 0.0005 # For use in optimizer
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result
_MEAN = 0
_STDDEV = 1
_LOSS_FACTOR = 1000 

# ================> Part Function Creater Model
## function model generator have arguments are 
#### input_dim          is latent size
#### input_shape        is shape before flattent in case convolution operation to dense
#### l_filters          is list filter in keras.layers.Conv2DTranspose
#### l_kernel           is list kernel in keras.layers.Conv2DTranspose
#### l_strides          is list strides in keras.layers.Conv2DTranspose
#### l_padding          is list padding in keras.layers.Conv2DTranspose
#### prefix             is string be prefix name in each layer and model name
#### activation         is string or None to assign activation will use each layer in this model
def model_generator( input_dim , input_shape,
        l_filters, l_kernels, l_strides, l_padding,
        prefix = "generator_", activation = "relu "):
    generator_input = Input( shape = input_dim )
    generator = Dense( np.prod( input_shape ),
            name = prefix + "prelayer" )( generator )
    generator = Reshape( input_shape,
            name = prefix + "prelayer_reshape" )( generator )
    count = 0
    for filters, kernels, strides, padding in zip( l_filters, l_kernels, l_strides, l_padding ):
        count += 1
        generator = Conv2DTranspose( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2dt" + str( count ) )( generator )
        if acitvation == None:
            None
        elif activation == "LeakyReLU":
            generator = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( generator )
        elif activation == "ReLU" :
            generator = ReLU( max_value = 1,
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( generator )
        else:
            generator = Activation( activation, 
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( generator )

    generator_output = generator
    generator_model = Model( generator_input , generator_output )
    generator_model.name = predix + "model"

    return generator_input, generator, generator_output, generator_model

## function model discriminator have arguments are
#### input_dim          is shape of image
#### l_filters          is list filter in keras.layers.Conv2D
#### l_kernel           is list kernel in keras.layers.Conv2D
#### l_strides          is list strides in keras.layers.Conv2D
#### l_padding          is list padding in keras.layers.Conv2D
#### activation_conv2d  is activation for conv2D layers
#### l_units            is list units in keras.layers.Dense
#### l_dropout          is list dropout rate 0 - 1 if None is don't have
#### activation_dense   is activation for dense layers
#### prefix             is string be prefix name in each layer and model name
def model_discriminator( input_dim , l_filters, l_kernel, l_strides, l_padding, activation_conv2d,
        l_units, l_dropout, activation_dense, prefix = "discriminator_" ):
    discriminator_input = Input( shape = input_dim )
    # Add part convolution
    discriminator = discriminator_input
    count = 0
    for filters, kernel, strides, padding in zip( l_filters, l_kernel, l_strides, l_padding ):
        count += 1
        discriminator = Conv2D( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2d" + str( count ) )( discriminator )
        if acitvation == None:
            None
        elif activation == "LeakyReLU":
            discriminator = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( discriminator )
        elif activation == "ReLU" :
            discriminator = ReLU( max_value = 1,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( discriminator )
        else:
            discriminator = Activation( activation, 
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( discriminator )
    # Finish part convolution
