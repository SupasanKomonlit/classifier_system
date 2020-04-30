# File Name : tf_GAN_train.py
# Main Reference : https://towardsdatascience.com/generating-new-faces-with-variational-autoencoders-d13cfcb5f0a8

from library.directory_handle import DirectoryHandle
import library.image_handle as ImageHandle
import library.data_handle  as DataHandle
import library.command_handle as CommandHandle

# Import library for plot image
import matplotlib.pyplot as plt

# Import library for manage model part Core Layers
from keras.layers import Input, Flatten, Dense, Reshape, Lambda, Dropout
# Import library for manage model part Convolution Layers
from keras.layers import Conv2D, Conv2DTranspose
# Import library for mange model part activatin
from keras.layers import Activation, ReLU, LeakyReLU
# Import Library for manage model part Model Object
from keras.models import Model, Sequential
# Import Library for manage model part optimizer
from keras.optimizers import Adam, RMSprop
# Import Library about model 
from keras.utils import plot_model
# Import library for load model
from keras.models import load_model
# Import library operation in Keras tensor object
from keras import backend as K
K.clear_session()
_CLEAR_SESSION = False
#Import library for normal process
import numpy as np
import cv2

import tensorflow as tf

from GAN_train import model_generator, model_discriminator

# ================> Part Parameter Program
_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/AnimeFaceData"
_CROP = True
_COLOR = True
_RATIO = 8
_EPOCHES_GENERATE = 2
_EPOCHES_DISCRIMINATOR = 1
_LATENT_SIZE = 1024
_ACTIVATION = "relu"
_MODEL_NAME = "GAN3L1024D" # This will use to save model
if _ACTIVATION != None : _MODEL_NAME += _ACTIVATION
_LEARNING_RATE_GAN = 0.001 # For use in optimizer
_LEARNING_RATE_DISCRIMINATOR = 0.0005
_LEARNING_RATE_GENERATOR = 0.001
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result
_MEAN = 0
_STDDEV = 1
_CHECKPOINT_WEIGHTS = "GANWeightsCheckpoint.h5"
_PREFIX_CHECKPOINT = "checkpoint_"

_ALL_ROUNDS = 20
_CONTINUE_TRAIN = True 
_OFFSET_ROUND = 0
_SAMPLE_RESULT = 5
_SAMPLE_BATCH = int( _SAMPLE_RESULT * 2 ) # Save Example Picture
_CHECKPOINT_BATCH = int( _SAMPLE_BATCH * 4 ) # Save Model Weights
_SAMPLE_IMAGE = 3
_BATCH_SIZE = 2048

# Part Function Operations
## Part Loss function
LossHandle = tf.keras.losses.BinaryCrossentropy( from_logits = True )

def discriminator_loss( real_output, fake_output ):
    real_loss = LossHandle( tf.ones_like( real_output ), real_output )
    fake_loss = LossHandle( tf.zeros_like( fake_output ), fake_output )
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss( fake_output ):
    return LossHandle( tf.ones_like( fake_output ) , fake_output )

## Part Opimizer variable
generator_optimizer = Adam( lr = _LEARNING_RATE_GENERATOR )
discriminator_optimizer = Adam( lr = _LEARNING_RATE_DISCRIMINATOR )

## Part Survey Image
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

picture_shape = ( square_size , square_size , 3 if _COLOR else 1 )

# Part Setup Model
group_discriminator , shape_before_flatten = model_discriminator( input_dim = picture_shape,
        l_filters = [64, 32, 16 ],
        l_kernel = [(3,3), (3,3), (3,3)],
        l_strides = [1, 2, 1],
        l_padding = ['same', 'same', 'same' ],
        activation_conv2d = _ACTIVATION,
        l_units = [ _LATENT_SIZE ],
        l_dropout = [ 0.2 ],
        output_dropout = 0.2,
        activation_dense = 'sigmoid',
        prefix = "discriminator_" )
#discriminator = Sequential.from_config( group_discriminator[ 3 ].get_config() )
discriminator = group_discriminator[ 3 ]
discriminator.summary()
group_generator = model_generator( input_dim = (_LATENT_SIZE, ),
        input_shape = shape_before_flatten,
        l_filters = [16, 32, 64], 
        l_kernels = [(3,3), (3,3), (3,3)],
        l_strides = [1, 2, 1],
        l_padding = ['same', 'same', 'same'],
        o_filters = 3 if _COLOR else 1,
        o_kernels = (3,3),
        o_strides = 1,
        o_padding = 'same',
        o_activation = _ACTIVATION,
        prefix = "generator_",
        activation = _ACTIVATION )
generator = group_generator[ 3 ]
#generator = Sequential.from_config( group_generator[ 3 ].get_config() )
generator.summary()

## Part Prepare Data
print( f'Prepare Data')
print( f'\tDowloading....' )
X_data = ImageHandle.read_all_data( list_file , square_size , color = _COLOR , crop = _CROP )

@tf.function
def train_step( real_images ):
    latent_vector = tf.random.normal( [ _BATCH_SIZE, _LATENT_SIZE ] )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#        fake_images = generator( latent_vector , training = True )
#        real_output = discriminator( real_images , training = True )
#        fake_output = discriminator( fake_images , training = True )
#        fake_images = generator.predict( latent_vector )
#        real_output = discriminator.predict( real_images )
#        fake_output = discriminator.predict( fake_images )
        gen_tape.watch( generator.trainable_weights )
        disc_tape.watch( discriminator.trainable_weights )
        fake_images = generator( latent_vector )
        real_output = discriminator( real_images )
        fake_output = discriminator( fake_images )

        gen_loss = generator_loss( fake_output )
        disc_loss = discriminator_loss( real_output , fake_output )

    generator_gradients = gen_tape.gradient( gen_loss , generator.trainable_weights )
    discriminator_gradients = disc_tape.gradient( disc_loss , discriminator.trainable_weights )

    print( "Generator " , end = "")
    print( generator.trainable_variables[0])
    generator_optimizer.apply_gradients( zip( generator_gradients , generator.trainable_weights ) )
    discriminator_optimizer.apply_gradients( zip( discriminator_gradients , discriminator.trainable_weights ) )
    print( "Generator " , end = "")
    print( generator.trainable_variables[0])

## Part Loop Train
round_batch = int( np.ceil( len( X_data ) / _BATCH_SIZE ) )
size_latent_vector = ( _BATCH_SIZE , _LATENT_SIZE )
real_images = np.array( X_data ).astype( np.float32 ) / 255

for count_round in range( _ALL_ROUNDS ):

    start = 0
    count_batch = 0

    while( start < real_images.shape[0] ):

        stop = start + _BATCH_SIZE
        if stop > real_images.shape[ 0 ] : stop = real_images.shape[0]

        data_train = real_images[ start : stop ]
        print( data_train.shape ) 

        train_step( data_train )

        start = stop
        count_batch += 1
    
        if count_batch % _CHECKPOINT_BATCH == 0 :
            print( "Part Save Weights")
            generator.save_weights( _PREFIX_CHECKPOINT + generator.name + "_weights.h5" )
            discriminator.save_weights( _PREFIX_CHECKPOINT + discriminator.name + "_weights.h5" )
    
        if count_batch % _SAMPLE_BATCH == 0 :
            print( f'Part Save Sample Image {count_round + 1 }/{_ALL_ROUNDS} round and {count_batch}/{round_batch} batch' )
            latent_vector = np.random.normal( _MEAN, _STDDEV , size = ( _SAMPLE_IMAGE , _LATENT_SIZE ) )
            fake_image = generator( latent_vector )
            for count_image in range( 0 , _SAMPLE_IMAGE ):
                name = "generator_" + str( count_round + 1 ) + "_" + str( count_batch ) + "_" + str( count_image + 1 ) + ".jpg"
                cv2.imwrite( name , ImageHandle.normalize_image( fake_image[ count_image ], copy = True, dest_type = int) )

    # End subloop train in batch
    print( f'End {count_round + 1 + _OFFSET_ROUND}/{_ALL_ROUNDS + _OFFSET_ROUND}' )
    generator.save_weights( _PREFIX_CHECKPOINT + generator.name + "_weights.h5" )
    discriminator.save_weights( _PREFIX_CHECKPOINT + discriminator.name + "_weights.h5" )
    latent_vector = np.random.normal( _MEAN, _STDDEV , size = ( _SAMPLE_IMAGE , _LATENT_SIZE ) )
    fake_image = generator( latent_vector )
    for count_image in range( 0 , _SAMPLE_IMAGE ):
        name = "generator_end_" + str( count_round + 1 ) + "_" + str( count_batch ) + "_" + str( count_image + 1 ) + ".jpg"
        cv2.imwrite( name , ImageHandle.normalize_image( fake_image[ count_image ], copy = True, dest_type = int) )

# End Index loop count round
