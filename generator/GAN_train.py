# File Name : GAN_train.py
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
from keras.utils.generic_utils import Progbar
#Import library for normal process
import numpy as np
import cv2

# Import tensorflow for train by my self
import tensorflow as tf

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
_LEARNING_DISCRIMINATOR = 0.0005
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result
_MEAN = 0
_STDDEV = 1
_CHECKPOINT_WEIGHTS = "GANWeightsCheckpoint.h5"

_ALL_ROUNDS = 10
_CONTINUE_TRAIN = True 
_OFFSET_ROUND = 10
_SAMPLE_RESULT = 5
_SAMPLE_BATCH = int( _SAMPLE_RESULT * 2 ) # Save Example Picture
_CHECKPOINT_BATCH = int( _SAMPLE_BATCH * 4 ) # Save Model Weights
_SAMPLE_IMAGE = 1
_BATCH_SIZE = 32

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
        o_filters, o_kernels, o_strides, o_padding, o_activation = None,
        prefix = "generator_", activation = "relu "):

    generator_input = Input( shape = input_dim ,
            name = prefix + "input" )
    generator = Dense( np.prod( input_shape ),
            name = prefix + "prelayer" )( generator_input )
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
        if activation == None:
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

    generator_output = Conv2DTranspose( filters = o_filters,
            kernel_size = o_kernels,
            strides = o_strides,
            padding = o_padding,
            name = prefix + "_output" )( generator )
    if o_activation == None:
        None
    else:
        generator_output = Activation( o_activation,
                name = prefix + "_output_" + o_activation )( generator_output )

    generator_model = Model( generator_input , generator_output )
    generator_model.name = prefix + "model"

    return ( generator_input, generator, generator_output, generator_model )

## function model discriminator have arguments are
#### input_dim          is shape of image
#### l_filters          is list filter in keras.layers.Conv2D
#### l_kernel           is list kernel in keras.layers.Conv2D
#### l_strides          is list strides in keras.layers.Conv2D
#### l_padding          is list padding in keras.layers.Conv2D
#### activation_conv2d  is activation for conv2D layers
#### l_units            is list units in keras.layers.Dense
#### l_dropout          is list dropout rate 0 - 1 if None is don't have use before dense
#### output_dropout     is dropout rate before go to output layers
#### activation_dense   is activation for dense layers
#### prefix             is string be prefix name in each layer and model name
def model_discriminator( input_dim , l_filters, l_kernel, l_strides, l_padding, activation_conv2d,
        l_units, l_dropout, activation_dense, output_dropout = None, prefix = "discriminator_" ):
    discriminator_input = Input( shape = input_dim ,
            name = prefix + "input")
    # Add part convolution
    discriminator = discriminator_input
    count = 0
    for filters, kernels, strides, padding in zip( l_filters, l_kernel, l_strides, l_padding ):
        count += 1
        discriminator = Conv2D( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2d" + str( count ) )( discriminator )
        if activation_conv2d == None:
            None
        elif activation_conv2d == "LeakyReLU":
            discriminator = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation_conv2d )( discriminator )
        elif activation_conv2d == "ReLU" :
            discriminator = ReLU( max_value = 1,
                    name = prefix + "conv2d" + str( count ) + "_" + activation_conv2d )( discriminator )
        else:
            discriminator = Activation( activation_conv2d, 
                    name = prefix + "conv2d" + str( count ) + "_" + activation_conv2d )( discriminator )
    # Finish part convolution
    shape_before_flatten = tuple( discriminator.shape[1:] )
    discriminator = Flatten( name = prefix + "flatten" )( discriminator )
    # Add part connected
    count = 0
    for units, dropout in zip( l_units, l_dropout ):
        count += 1
        if dropout != None:
            discriminator = Dropout( dropout , name = prefix + "dense" + str( count ) + "_dropout" )( discriminator )
        
        discriminator = Dense( units,
                name = prefix + "dense" + str( count ) )( discriminator )

        if activation_dense == None:
            None
        else:
            discriminator = Activation( activation_dense,
                    name = prefix + "dense" + str( count ) + "_" + activation_dense )( discriminator )
    # Finish part Connected
    if output_dropout != None:
        discriminator = Dropout( output_dropout,
                name = prefix + "output_dropout" )( discriminator )

    discriminator_output = Dense( 1,
            name = prefix + "output" )( discriminator )
    discriminator_output = Activation( "sigmoid",
            name = prefix + "output_sigmoid" )( discriminator_output )

    discriminator_model = Model( discriminator_input , discriminator_output )
    discriminator_model.name = prefix + "model"

    return ( discriminator_input, discriminator, discriminator_output, discriminator_model) , shape_before_flatten

def model_GAN( picture_shape ):
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
    group_GAN_discriminator , shape_before_flatten = model_discriminator( input_dim = picture_shape,
            l_filters = [64, 32, 16 ],
            l_kernel = [(3,3), (3,3), (3,3)],
            l_strides = [1, 2, 1],
            l_padding = ['same', 'same', 'same' ],
            activation_conv2d = _ACTIVATION,
            l_units = [ _LATENT_SIZE ],
            l_dropout = [ 0.2 ],
            output_dropout = 0.2,
            activation_dense = 'sigmoid',
            prefix = "GAN_discriminator_" )
#    group_discriminator[3].summary()
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
#    group_generator[3].summary()
    group_GAN_discriminator[ 3 ].trainable = False
    GAN_model = Model( group_generator[0] , group_GAN_discriminator[3]( group_generator[3]( group_generator[0] ) ) )
    GAN_model.name = _MODEL_NAME
#    GAN_model.summary()

    return group_generator, group_discriminator, GAN_model

def model_GAN_compile( GAN_model, discriminator_model , GAN_optimizer , discriminator_optimizer ):
    GAN_model.compile( optimizer = GAN_optimizer,
            loss = 'binary_crossentropy',
            metrics = [ 'accuracy' ] )
    discriminator_model.compile( optimizer = discriminator_optimizer,
            loss = 'binary_crossentropy',
            metrics = [ 'accuracy' ] )

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

## START PART SETUP MODEL
    print( f'Part Setup Model Object')
    input_dim = ( square_size , square_size , 3 if _COLOR else 1 )

    group_generator , group_discriminator, GAN_model = model_GAN( input_dim )
    discriminator_model = group_discriminator[3]
    generator_model = group_generator[3]
    generator_model.summary()
    discriminator_model.summary()
    GAN_model.summary()

    print( f'GAN_model layers    : {GAN_model.layers}' )
    print( f'generator_model     : {generator_model}' )
    print( f'discriminator_model : {discriminator_model}' )

    GAN_optimizer = RMSprop( lr = _LEARNING_RATE_GAN )
    discriminator_optimizer = RMSprop( lr = _LEARNING_DISCRIMINATOR )

    model_GAN_compile( GAN_model, discriminator_model, GAN_optimizer, discriminator_optimizer )
#    GAN_model.compile( optimizer = GAN_optimizer, loss = 'binary_crossentropy', metrics = [ 'accuracy' ] )
## END PART SETUP MODEL

    print( f'Prepare Data')
    print( f'\tDowloading....' )
    X_data = ImageHandle.read_all_data( list_file , square_size , color = _COLOR , crop = _CROP )

    print( f'Start train model')

    if _CONTINUE_TRAIN:
        print( f'Download initial weights from {_CHECKPOINT_WEIGHTS}' )
        GAN_model.load_weights( _CHECKPOINT_WEIGHTS )

    round_batch = int( np.ceil( len( X_data ) / _BATCH_SIZE ) )
    size_latent_vector = ( _BATCH_SIZE , _LATENT_SIZE )
    real_image = np.array( X_data ).astype( np.float ) / 255

    discriminator_history = { 'loss' : [] , 'accuracy' : [] }
    gan_history = { 'loss' : [], 'accuracy' : [] }
    for count_round in range( _ALL_ROUNDS ):

        discriminator_loss = []
        discriminator_accuracy = []
        gan_loss = []
        gan_accuracy = []
    
        start = 0
        count_batch = 0
        progress_bar = Progbar( target = round_batch )
        while( start < real_image.shape[0] ):
            latent_vector = np.random.normal( _MEAN, _STDDEV , size = size_latent_vector )
            fake_image = generator_model.predict( latent_vector )
            if count_batch % _SAMPLE_BATCH == 0 :
#                print( 'Save sample picture')
                for count_image in range( 0 , _SAMPLE_IMAGE ):
                    name = "gan_image_round" + str( count_round + 1 + _OFFSET_ROUND ) + "_batch" + str( count_batch ) + "_" + str( count_image + 1 ) + ".jpg"
                    cv2.imwrite( name , ImageHandle.normalize_image( fake_image[ count_image ], copy = True, dest_type = int) )

            stop = start + _BATCH_SIZE
            if stop > real_image.shape[ 0 ] : stop = real_image.shape[0]
            
            all_image = np.concatenate( [ real_image[ start : stop ] , fake_image ] )
            label_image = np.concatenate( [ np.ones( ( stop - start , 1 ) ) , np.zeros( ( _BATCH_SIZE , 1 ) ) ] )
            # Train discriminator_model
#            d_history = discriminator_model.fit( all_image , label_image, epochs = _EPOCHES_DISCRIMINATOR , verbose = _VERBOSE )
            #print( "Train discriminator")
            #print( discriminator_model.get_weights()[0][0][0][0] )
            d_history = discriminator_model.train_on_batch( all_image , label_image )
            #print( "After train discriminator")
            #print( discriminator_model.get_weights()[0][0][0][0] )
            # Disable train discriminator_model
#            discriminator_model.trainable = False
#            model_GAN_compile( GAN_model, discriminator_model , GAN_optimizer, discriminator_optimizer )
            # Train generator_model
            GAN_model.layers[ -1 ].set_weights( discriminator_model.get_weights() )
            #print( GAN_model.layers[-1].get_weights()[0][0][0][0] )
            latent_vector = np.random.normal( _MEAN, _STDDEV , size = size_latent_vector )
#            g_history = GAN_model.fit( latent_vector , np.ones( ( _BATCH_SIZE , 1 ) ), epochs = _EPOCHES_GENERATE, verbose = _VERBOSE )
            g_history = GAN_model.train_on_batch( latent_vector , np.ones( ( _BATCH_SIZE , 1 ) ) )
            #print( "After train GAN")
            #print( discriminator_model.get_weights()[0][0][0][0] )
            # Enable train discriminator_model
#            discriminator_model.trainable = True
#            model_GAN_compile( GAN_model, discriminator_model , GAN_optimizer, discriminator_optimizer )

            start = stop
            count_batch += 1
            progress_bar.update( count_batch )
        
#            discriminator_loss += d_history.history[ 'loss' ]
#            discriminator_accuracy += d_history.history[ 'accuracy' ]
#            gan_loss += g_history.history[ 'loss' ]
#            gan_accuracy += g_history.history[ 'accuracy' ]
            discriminator_loss.append( d_history[0] )
            discriminator_accuracy.append( d_history[1] )
            gan_loss.append( g_history[0] )
            gan_accuracy.append( g_history[1] )

            if count_batch % _CHECKPOINT_BATCH == 0 :
#                print( f'Save weights to {_CHECKPOINT_WEIGHTS}' )
                GAN_model.save_weights( _CHECKPOINT_WEIGHTS )

#            if count_batch % _SAMPLE_RESULT == 0 :
#                print( f'Summary mean value on {count_round + 1 + _OFFSET_ROUND}/{_ALL_ROUNDS +_OFFSET_ROUND} round in {count_batch+1}/{round_batch} batch' )
#                print( f'==>Discriminator : Loss {np.mean( discriminator_loss ):10.5f} Accuracy {np.mean( discriminator_accuracy ):10.5f}' )
#                print( f'==>Generator     : Loss {np.mean( gan_loss ):10.5f} Accuracy {np.mean( gan_accuracy ):10.5f}' )
        # End subloop train in batch

        print( f'End {count_round + 1 + _OFFSET_ROUND}/{_ALL_ROUNDS + _OFFSET_ROUND} accuracy point {np.mean( discriminator_accuracy ) } in discriminator and {np.mean( gan_accuracy ) } in generator' )
        print( f'\tSaveing weights to {_CHECKPOINT_WEIGHTS}' )
        GAN_model.save_weights( _CHECKPOINT_WEIGHTS )
        print( f'\tSaveing sample picture')
        for count_image in range( 0 , _SAMPLE_IMAGE ):
            name = "gan_image_end_round" + str( count_round + 1 + _OFFSET_ROUND ) + "_" + str( count_image + 1 ) + ".jpg"
            cv2.imwrite( name , ImageHandle.normalize_image( fake_image[ count_image ], copy = True, dest_type = int) )
        gan_history['loss'].append( gan_loss )
        gan_history['accuracy'].append( gan_accuracy )
        discriminator_history['loss'].append( discriminator_loss )
        discriminator_history['accuracy'].append( discriminator_accuracy )

        if _CLEAR_SESSION :
            print( "Clear Session on Resetup Model")
            K.clear_session()
            group_generator , group_discriminator, GAN_model = model_GAN( input_dim )
            print( f'Download weights from {_CHECKPOINT_WEIGHTS}')
            GAN_model.load_weights( _CHECKPOINT_WEIGHTS )
            discriminator_model = group_discriminator[3]
            generator_model = group_generator[3]
#            print( f'GAN_model layers    : {GAN_model.layers}' )
#            print( f'generator_model     : {generator_model}' )
#            print( f'discriminator_model : {discriminator_model}' )
            model_GAN_compile( GAN_model, discriminator_model, GAN_optimizer, discriminator_optimizer )
    # End Index loop count round

    print( f'Finish train save weight to {_MODEL_NAME}.h5' )
    GAN_model.save_weights( _MODEL_NAME + "_weights.h5" )
    latent_random = np.random.normal( _MEAN, _STDDEV , size = ( 10 , _LATENT_SIZE ) )
    CommandHandle.plot( latent_random , generator_model , dest_type = float ) 
# Ploting part
    fig_history = []
    for rounds in range( _ALL_ROUNDS ):
        fig_history.append( plt.figure( "History Training GAN Model " + _MODEL_NAME + " on Round " + str( rounds + _OFFSET_ROUND + 1 ) ) )
        fig_history[ rounds ].subplots_adjust( hspace=0.8 , wspace=0.1 )
        sub = fig_history[ rounds ].add_subplot( 2 , 1 , 1 )
        sub.plot( gan_history['accuracy'][rounds] )
        sub.plot( discriminator_history['accuracy'][rounds] )
        sub.set_title('Model Accuracy Round ' + str( rounds + _OFFSET_ROUND + 1 ) )
        sub.set_ylabel('Accuracy')
        sub.set_xlabel('Order Batch')
        sub.legend(['Generator', 'Discriminator'], loc='upper left')
        sub = fig_history[ rounds ].add_subplot( 2 , 1 , 2 )
        sub.plot( gan_history['loss'][rounds] )
        sub.plot( discriminator_history['loss'][rounds] )
        sub.set_title('Model Loss Round ' + str( rounds + _OFFSET_ROUND + 1 ) )
        sub.set_ylabel('Loss')
        sub.set_xlabel('Order Batch')
        sub.legend(['Generator', 'Discriminator'], loc='upper left')
        plt.show( block = False )
# End ploting part
    plt.show()
