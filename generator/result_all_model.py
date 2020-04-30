# File Name : result_all_model.py
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

from variational_autoencoder_train import total_loss, kl_loss, r_loss, model_vae_encoder, model_decoder
from GAN_train import model_discriminator, model_GAN, model_generator

_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/AnimeFaceData"
_CROP = True
_COLOR = True

_LATENT_SIZE = 1024
_MODEL_VAE_NAME = "VAE3L1024D_1000relu" # This will use to save model
_MODEL_GAN_NAME = "GAN3L1024Drelu"
_ACTIVATION = 'relu'
_SHOW_SIZE = False
_MEAN = 0
_STDDEV = 1

if __name__ == "__main__":
    directory_handle = DirectoryHandle( _PATH_DATA )
    list_file = directory_handle.get_all_file()

    width, height = ImageHandle.read_size( list_file )
    width = np.array( width )
    height = np.array( height )
    if _SHOW_SIZE :
        CommandHandle.plot_scatter( width , height, 
                "width (pixel)" , "height (pixel)", 
                figname = "picture_size" )

    if width.min() < height.min():
        square_size = int( np.ceil( width.min() ) )
    else:
        square_size = int( np.ceil( height.min() ) )
    square_size = square_size if square_size % 2 == 0 else square_size + 1
    print( f'This program parameter to input image is\n\tColor Image : {_COLOR}\n\tCrop Image : {_CROP}\n\tSquare size : {square_size}')
    plt.show()

    input_dim = ( square_size , square_size , 3 if _COLOR else 1 )

    print( f'Setup Model Variational Autoencoder')
    vae_encoder_input, vae_encoder, vae_encoder_output, vae_encoder_model, shape_before_flatten, mean_layer, variance_layer= model_vae_encoder(
            input_dim = input_dim,
            output_dim = _LATENT_SIZE,
            l_filters = [ 64, 32, 16 ], 
            l_kernels = [ (3,3), (3,3), (3,3) ],
            l_strides = [ 1, 2, 1 ], 
            l_padding = ['same', 'same', 'same'],
            prefix = "vae_encoder_",
            activation = _ACTIVATION )
    vae_encoder_model.summary()

    decoder_input, decoder, decoder_output, decoder_model = model_decoder(
            input_dim = _LATENT_SIZE,
            shape_before_flatten = shape_before_flatten,
            output_channel = input_dim[2],
            l_filters = [ 16, 32, 64 ], 
            l_kernels = [ (3,3), (3,3), (3,3) ],
            l_strides = [ 1, 2, 1 ], 
            l_padding = ['same', 'same', 'same'],
            prefix = "decoder_",
            activation = _ACTIVATION )
    decoder_model.summary()

    vae_autoencoder_model = Model( vae_encoder_input, decoder_model( vae_encoder_model( vae_encoder_input ) ) )
    vae_autoencoder_model.name = _MODEL_VAE_NAME
    vae_autoencoder_model.summary()
    print( f'Download wegiths from {_MODEL_VAE_NAME}_weights.h5' )
    vae_autoencoder_model.load_weights( _MODEL_VAE_NAME + "_weights.h5" )

    print( f'Setup Model GAN' )
    group_generator , group_discriminator, GAN_model = model_GAN( input_dim )
    discriminator_model = group_discriminator[3]
    generator_model = group_generator[3]
    generator_model.summary()
    discriminator_model.summary()
    GAN_model.summary()
    print( f"Download weights from {_MODEL_GAN_NAME}_weights.h5" )
    GAN_model.load_weights( _MODEL_GAN_NAME + "_weights.h5" )

    latent_random = np.random.normal( _MEAN, _STDDEV , size = ( 20 , _LATENT_SIZE ) )
    CommandHandle.plot_compare_model( latent_random , decoder_model, generator_model , dest_type = float )

    plt.show() 
