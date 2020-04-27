# Import Library help operation
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
#from keras import backend as K
#K.clear_session()
#Import library for normal process
import numpy as np

# ============================== MODEL CREATER FUNCTION ==========================================
def model_encoder( input_dim, output_dim, 
        l_filters, l_kernels, l_strides, l_padding, 
        prefix = "encoder_" , activation = None ):
    encoder_input = Input( shape = input_dim , name = prefix + "input" )
    encoder = encoder_input
    count = 0
    for filters , kernels, strides, padding in zip( l_filters, l_kernels, l_strides, l_padding ):
        count += 1
        encoder = Conv2D( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2d" + str( count ) )( encoder )
        if activation == None :
            None
        elif activation == "LeakyReLU":
            decoder = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( encoder )
        elif activation == "ReLU":
            encoder = ReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( encoder )
        else:
            encoder = Activation( activation ,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( encoder )
    encoder = Flatten( name = prefix + "flatten" )(encoder)
    encoder_output = Dense( output_dim,
                name = prefix + "output" )( encoder )

    encoder_model = Model( encoder_input , encoder_output )
    encoder_model.name = prefix + "model"
    shape_before_flatten = encoder_model.layers[ -3 ].output_shape[1:]

    return encoder_input, encoder, encoder_output, encoder_model, shape_before_flatten

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
                name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder_output )
    elif activation == "ReLU":
        decoder_output = ReLU( alpha = 0.3,
                name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder_output )
    else:
        decoder_output = Activation( activation ,
                name = prefix + "conv2dt" + str( count ) + "_" + activation )( decoder_output )

    decoder_model = Model( decoder_input , decoder_output )
    decoder_model.name = prefix + "model"

    return decoder_input, decoder, decoder_output, decoder_model

# ============================ MAIN FUNCTION TO RUN PROGRAM =====================================+
# =====> PARAMETER
_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"
_CROP = True
_COLOR = True
_RATIO = 8
_EPOCHES = 30
_LATENT_SIZE = 1024
_MODEL_NAME = "autoencoder3L1024D" # This will use to save model
_LEARNING_RATE = 0.0005
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result
_ACTIVATION = None

if __name__=="__main__":
    print( "Survey directory of data")
    directory_handle = DirectoryHandle( _PATH_DATA )
    list_label , list_data = directory_handle.group_data()
    list_dictionary = directory_handle.group_dictionary()

    if _SHOW_SIZE : 
        width = []
        height = []
        for data in list_data:
            width , height = ImageHandle.read_size( data , width, height )

        CommandHandle.plot_scatter( width , height, 
                "width (pixel)" , "height (pixel)", 
                figname = "picture_size")

    square_size = ImageHandle.min_all_square_size1( list_data )
    square_size = square_size if square_size % 2 == 0 else square_size - 1
    print( f'This program parameter to input image is\n\tColor Image : {_COLOR}\n\tCrop Image :{_CROP}\n\tSquare size : {square_size}')

    input_dim = ( square_size , square_size , 3 if _COLOR else 1 )
    print( "Part Setup Model")
    encoder_input, encoder, encoder_output, encoder_model, shape_before_flatten = model_encoder(
            input_dim = input_dim,
            output_dim = _LATENT_SIZE,
            l_filters = [ 64, 32, 16 ], 
            l_kernels = [ (3,3), (3,3), (3,3) ],
            l_strides = [ 1, 2, 1 ], 
            l_padding = ['same', 'same', 'same'],
            prefix = "encoder_",
            activation = _ACTIVATION )
    encoder_model.summary()

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

    autoencoder_model = Model( encoder_input , decoder_model( encoder_model( encoder_input ) ) )
    autoencoder_model.name = _MODEL_NAME
    autoencoder_model.summary()

    print( "\nPart Prepare Data\n\tDownloading Data" )
    X_data, Y_data = ImageHandle.prepare_label_data( list_label, list_data, square_size, 
            color = _COLOR , crop = _CROP )
    print( "\tSpliiting Data")
    (X_train,Y_train) , (X_test,Y_test) = DataHandle.train_test_split( X_data , Y_data , _RATIO )
    X_train = np.array( X_train ).astype( np.float ) / 255
    X_test = np.array( X_test ).astype( np.float ) / 255

    # ========> Train autoencoder model
    print( "\nPart Training Model")
    optimizer = Adam( lr = _LEARNING_RATE )
    autoencoder_model.compile( optimizer = optimizer,
            loss = 'mean_squared_error',
            metrics = ['accuracy'] )
    history = autoencoder_model.fit( [X_train],
            [X_train],
            validation_data = ( [X_test] , [X_test] ),
            epochs = _EPOCHES,
            verbose = _VERBOSE )
    
    print( f'Save Model to {autoencoder_model.name}.h5' )
    autoencoder_model.save( autoencoder_model.name + ".h5" )

    fig_history_autoencoder = plt.figure( "History Training Autoencoder Model " + _MODEL_NAME )
    fig_history_autoencoder.subplots_adjust( hspace=0.8 , wspace=0.1 )
    sub = fig_history_autoencoder.add_subplot( 2 , 1 , 1 )
    sub.plot( history.history['accuracy'] )
    sub.plot( history.history['val_accuracy'] )
    sub.set_title('Model accuracy')
    sub.set_ylabel('Accuracy')
    sub.set_xlabel('Epoch')
    sub.legend(['Train', 'Test'], loc='upper left')
    sub = fig_history_autoencoder.add_subplot( 2 , 1 , 2 )
    sub.plot( history.history['loss'] )
    sub.plot( history.history['val_loss'] )
    sub.set_title('Model loss')
    sub.set_ylabel('Loss')
    sub.set_xlabel('Epoch')
    sub.legend(['Train', 'Test'], loc='upper left')
    plt.show( block = False )

#    random_index = []
#    for _ in range(0,10):
#        random_index.append( np.random.randint( len( X_test)))
#    random_index = tuple( set( random_index ) )
    random_index = [ x for x in range( 0 , len( X_test ) , 50 ) ]
    data = []
    for index in random_index :
        data.append( X_test[ index ] )
    data = np.array( data )
    CommandHandle.plot_compare( data, autoencoder_model,
        figname = "Compare Result Autoencoder Model " + autoencoder_model.name,
        dest_type = np.float )

    plt.show() 
