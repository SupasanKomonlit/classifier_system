# File Name : variational_autoencoder_train.py
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
_EPOCHES = 20
_LATENT_SIZE = 1024
_ACTIVATION = "relu"
_MODEL_NAME = "VAE3L1024D_1000" # This will use to save model
if _ACTIVATION != None : _MODEL_NAME += _ACTIVATION
_LEARNING_RATE = 0.0005 # For use in optimizer
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result
_MEAN = 0
_STDDEV = 1
_LOSS_FACTOR = 1000 

# ================> Part Function Creator Model

def sampling( args ): # Function output of Variational Encoder
    mean, variance = args
    epsilon = K.random_normal( shape = K.shape( mean ), mean = _MEAN, stddev = _STDDEV )
    return mean + K.exp( variance / 2 ) * epsilon

def kl_loss( y_true , y_pred ):
    loss = -0.5 * K.sum( 1 + variance_layer - K.square( mean_layer ) - K.exp( variance_layer ),
            axis = 1 )
    return loss

def r_loss( y_true , y_pred ):
    return K.mean( K.square( y_true - y_pred ), axis = [ 1 , 2 , 3 ] )

def total_loss( y_true , y_pred ):
    return _LOSS_FACTOR * r_loss( y_true , y_pred ) + kl_loss( y_true , y_pred )    

def model_vae_encoder( input_dim, output_dim, 
        l_filters, l_kernels, l_strides, l_padding, 
        prefix = "vae_encoder_" , activation = None ):
    vae_encoder_input = Input( shape = input_dim , name = prefix + "input" )
    vae_encoder = vae_encoder_input
    count = 0
    for filters , kernels, strides, padding in zip( l_filters, l_kernels, l_strides, l_padding ):
        count += 1
        vae_encoder = Conv2D( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2d" + str( count ) )( vae_encoder )
        if activation == None :
            None
        elif activation == "LeakyReLU":
            vae_encoder = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( vae_encoder )
        elif activation == "ReLU":
            vae_encoder = ReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( vae_encoder )
        else:
            vae_encoder = Activation( activation ,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( vae_encoder )

    vae_encoder = Flatten( name = prefix + "flatten" )( vae_encoder )

    mean_layer = Dense( output_dim , name = prefix + "mean" )( vae_encoder )
    variance_layer = Dense( output_dim , name = prefix + "variance" )( vae_encoder )    

    vae_encoder_output = Lambda( sampling, name = prefix + "output" )( [ mean_layer, variance_layer] )

    vae_encoder_model = Model( vae_encoder_input , vae_encoder_output )
    vae_encoder_model.name = prefix + "model"
    shape_before_flatten = vae_encoder_model.layers[ -5 ].output_shape[1:]

    return vae_encoder_input, vae_encoder, vae_encoder_output, vae_encoder_model, shape_before_flatten, mean_layer, variance_layer

def model_decoder( input_dim, shape_before_flatten, output_channel,
        l_filters, l_kernels, l_strides, l_padding, 
        prefix = "decoder_" , activation = None ):
    decoder_input = Input( shape = (input_dim,) , name = prefix + "input" )
    decoder = Dense( np.prod( shape_before_flatten ),
            name = prefix + "input_post")( decoder_input )
    decoder = Reshape( shape_before_flatten,
            name = prefix + "input_reshape" )( decoder )
    count = 0
    for filters , kernels, strides, padding in zip( l_filters, l_kernels, l_strides, l_padding ):
        count += 1
        decoder = Conv2DTranspose( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2dt" + str( count ) )( decoder )
        if activation == None :
            None
        elif activation == "LeakyReLU":
            decoder = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder )
        elif activation == "ReLU":
            decoder = ReLU( alpha = 0.3,
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder )
        else:
            decoder = Activation( activation ,
                    name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder )
    decoder_output = Conv2DTranspose( filters = output_channel,
            kernel_size = (3,3),
            strides = 1,
            padding = padding,
            name = prefix + "output" )( decoder )
    if activation == None :
        None
    elif activation == "LeakyReLU":
        decoder_output = LeakyReLU( alpha = 0.3,
                name = prefix + "output_" + activation )( decoder_output )
    elif activation == "ReLU":
        decoder_output = ReLU( alpha = 0.3,
                name = prefix + "output_" + activation )( decoder_output )
    else:
        decoder_output = Activation( activation ,
                name = prefix + "output_" + activation )( decoder_output )

    decoder_model = Model( decoder_input , decoder_output )
    decoder_model.name = prefix + "model"

    return decoder_input , decoder , decoder_output, decoder_model 

# ===================> PART MAIN PROGRAM

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

    vae_autoencoder_model = Model( vae_encoder_input, 
            decoder_model( vae_encoder_model( vae_encoder_input ) ) )
    vae_autoencoder_model.name = _MODEL_NAME
    vae_autoencoder_model.summary()

## END PART SETUP MODEL

    print( f'Prepare Data')
    print( f'\tDowloading....' )
    X_data = ImageHandle.read_all_data( list_file , square_size , color = _COLOR , crop = _CROP )
    print( f'\tSplitting.....' )
    X_train, X_test = DataHandle.split_data( X_data , 18 )
    X_train = np.array( X_train ).astype( float ) / 255
    X_test = np.array( X_test ).astype( float ) / 255

    print( f'Start Training Model' )
    optimizer = Adam( lr = _LEARNING_RATE )
    vae_autoencoder_model.compile( optimizer = optimizer,
            loss = total_loss,
            metrics = [ r_loss , kl_loss ] )

    history = vae_autoencoder_model.fit( [ X_train ],
            [X_train],
            validation_data = ( [X_test] , [X_test] ),
            epochs = _EPOCHES,
            verbose = _VERBOSE )

    vae_autoencoder_model.save_weights( _MODEL_NAME + "_weights.h5" )

    fig_history_loss = plt.figure( "History Loss of Training Model " + _MODEL_NAME )
    # Plot traing & validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show( block = False )

    # Plot training & validation Mean Square loss values
    fig_history_rms_loss = plt.figure( "History RMS Loss of Training Model " + _MODEL_NAME )
    plt.plot(history.history['r_loss'])
    plt.plot(history.history['val_r_loss'])
    plt.title('Model Mean Square loss')
    plt.ylabel('Mean Square Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show( block = False )

    fig_history_kl_loss = plt.figure( "History KL Loss of Training Model " + _MODEL_NAME )
    # Plot training & validation kl loss values
    plt.plot(history.history['kl_loss'])
    plt.plot(history.history['val_kl_loss'])
    plt.title('Model KL loss')
    plt.ylabel('KL Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show( block = False )

    sample_index = [ x for x in range( 0 , len( X_test ) , int( np.ceil( len( X_test ) / 4000 ) ) ) ]
    data = []
    for index in sample_index :
        data.append( X_test[ index ] )
    data = np.array( data )
    CommandHandle.plot_compare( data, vae_autoencoder_model,
        figname = "Compare Result Autoencoder Model " + vae_autoencoder_model.name,
        dest_type = np.float )

    latent_random = np.random.normal( _MEAN , _STDDEV , size = ( 30, _LATENT_SIZE ) )
    CommandHandle.plot( latent_random , decoder_model , dest_type = float )

    plt.show()
