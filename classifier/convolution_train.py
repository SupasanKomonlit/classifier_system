# FILE      : convolution_train.py

# Import Library help operation
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
# Import library for manage model part Pooling Layers
from keras.layers import MaxPooling2D, AveragePooling2D
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
def model_convolution( input_dim, output_dim, 
        l_filters, l_kernels, l_strides, l_padding,
        pool_type = "MaxPooling2D", pool_size = (2,2) , pool_padding = "valid", pool_strides = None,
        prefix = "convolution_" , activation = None ):
    convolution_input = Input( shape = input_dim , name = prefix + "input" )
    convolution = convolution_input
    count = 0
    for filters , kernels, strides, padding in zip( l_filters, l_kernels, l_strides, l_padding ):
        count += 1
        convolution = Conv2D( filters = filters,
                kernel_size = kernels,
                strides = strides,
                padding = padding,
                name = prefix + "conv2d" + str( count ) )( convolution )
        if activation == None :
            None
        elif activation == "LeakyReLU":
            convolution = LeakyReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( convolution )
        elif activation == "ReLU":
            convolution = ReLU( alpha = 0.3,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( convolution )
        else:
            convolution = Activation( activation ,
                    name = prefix + "conv2d" + str( count ) + "_" + activation )( convolution )
    if pool_type == "MaxPooling2D" :
        convolution = MaxPooling2D( pool_size = pool_size,
                padding = pool_padding,
                strides = pool_strides,
                name = prefix + pool_type )( convolution )
    elif pool_type == "AveragePooling2D":
        convolution = AveragePooling2D( pool_size = pool_size,
                padding = pool_padding,
                strides = pool_strides,
                name = prefix + pool_type )( convolution )
    else:
        None
        
    convolution_output = Flatten( name = prefix + "flatten" )(convolution)

    convolution_model = Model( convolution_input , convolution_output )
    convolution_model.name = prefix + "model"

    shape_before_flatten = convolution_model.layers[ -2 ].output_shape[1:]

    return convolution_input, convolution, convolution_output, convolution_model, shape_before_flatten

def model_connected( input_dim, output_dim, output_activation,
        l_units,  
        prefix = "connected_" , activation = None, drop_rate = None ):
    connected_input = Input( shape = (input_dim,) , name = prefix + "input" )
    connected = connected_input
    count = 0
    for units in l_units:
        count += 1
        connected = Dense( units,
                name = prefix + "connected" + str( count ) )( connected )
        if activation == None :
            None
        elif activation == "LeakyReLU":
            connected = LeakyReLU( alpha = 0.3,
                    name = prefix + "connected" + str( count ) + "_" + activation )( connected )
        elif activation == "ReLU":
            connected = ReLU( alpha = 0.3,
                    name = prefix + "connected" + str( count ) + "_" + activation )( connected )
        else:
            connected = Activation( activation ,
                    name = prefix + "connected" + str( count ) + "_" + activation )( connected )

    if drop_rate != None :
        connected = Dropout( drop_rate ,
                name = prefix + "connected_dropout" )( connected )
    connected_output = Dense( output_dim,
            name = prefix + "output" )( connected )
    connected_output = Activation( output_activation,
                name = prefix + "output_" + activation )( connected_output )

    connected_model = Model( connected_input , connected_output )
    connected_model.name = prefix + "model"

    return connected_input, connected, connected_output, connected_model

# ============================ MAIN FUNCTION TO RUN PROGRAM =====================================+
# =====> PARAMETER
_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"
_CROP = True
_COLOR = True
_RATIO = 8
_EPOCHES = 50
_LATENT_SIZE = 64
_MODEL_NAME = "classifier_cnn3L64Drelu" # This will use to save model
_LEARNING_RATE = 0.0005
_DROP_RATE = 0.2
_SHOW_SIZE = False
_VERBOSE = 1 # 0 is silence 1 is process bar and 2 is result

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
    print( f'This program parameter to input image is\n\tColor Image : {_COLOR}\n\tCrop Image : {_CROP}\n\tSquare size : {square_size}')

    input_dim = ( square_size , square_size , 3 if _COLOR else 1 )

    print( "Part Setup Model")

    convolution_input, convolution, convolution_output, convolution_model, shape_before_flatten = model_convolution(
            input_dim = input_dim,
            output_dim = _LATENT_SIZE, 
            l_filters = [ 64, 32, 16 ], 
            l_kernels = [ (3,3), (3,3), (3,3) ],
            l_strides = [ 1, 2, 1 ], 
            l_padding = ['same', 'same', 'same'],
            pool_type = "MaxPooling2D", 
            pool_size = (2,2) , pool_padding = "valid", 
            pool_strides = None,
            prefix = "convolution_",
            activation = "relu" )
    convolution_model.summary()

    connected_input, connected, connected_output, connected_model = model_connected(
            input_dim = np.prod( shape_before_flatten ), 
            output_dim = len( list_dictionary ),  output_activation = "softmax",
            l_units = [ _LATENT_SIZE ],
            prefix = "connected_" , activation = "sigmoid" , drop_rate = _DROP_RATE )
    connected_model.summary()

    cnn_classifier_model = Model( convolution_input ,
            connected_model( convolution_model( convolution_input ) ) )
    cnn_classifier_model.name = _MODEL_NAME
    cnn_classifier_model.summary()

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
    cnn_classifier_model.compile( optimizer = optimizer,
            loss = 'categorical_crossentropy',
            metrics = ['accuracy'] )
    history = cnn_classifier_model.fit( [X_train],
            [Y_train],
            validation_data = ( [X_test] , [Y_test] ),
            epochs = _EPOCHES,
            verbose = _VERBOSE )

    cnn_classifier_model.save( cnn_classifier_model.name + ".h5")

    fig_history_autoencoder = plt.figure( "History Training CNN Classifier Model " + _MODEL_NAME )
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
    
    Y_predict = cnn_classifier_model.predict( X_test )
    DataHandle.result_classifier( Y_predict, np.array( Y_test ), list_dictionary )

    plt.show()
