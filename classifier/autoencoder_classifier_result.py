# File Name : autoencoder_classifier_result.py

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
# Import library for normal process
import numpy as np

# Import function for create connected model
from convolution_train import model_connected

_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"
_CROP = True
_COLOR = True
_RATIO = 8
_EPOCHES = 50
_LATENT_SIZE = 256
_MODEL_AUTOENCODER = "autoencoder3L256D"
_MODEL_NAME = "classifier_" + _MODEL_AUTOENCODER
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
    print( f'This program parameter to input image is\n\tColor Image : {_COLOR}\n\tCrop Image :{_CROP}\n\tSquare size : {square_size}')

    input_dim = ( square_size , square_size , 3 if _COLOR else 1 )

    print( "Part Setup Model")
    print( "Download autoencoder model from " + _MODEL_AUTOENCODER + ".h5")
    autoencoder_model = load_model( _MODEL_AUTOENCODER + ".h5")
    encoder_input = Input( shape = input_dim , name = "encoder_input")
    encoder_model = autoencoder_model.layers[ -2 ]
    encoder_model.trainable = False
    encoder_model.summary()
    
    connected_input, connected, connected_output, connected_model = model_connected(
            input_dim = _LATENT_SIZE, 
            output_dim = len( list_dictionary ),  output_activation = "softmax",
            l_units = [],
            prefix = "connected_" , activation = "sigmoid" , drop_rate = _DROP_RATE )
    connected_model.summary()

    print( "Download classifier autoencoder model from " + _MODEL_NAME + ".h5")
    classifier_model = load_model( _MODEL_NAME + ".h5")

    print( "\nPart Prepare Data\n\tDownloading Data" )
    X_data, Y_data = ImageHandle.prepare_label_data( list_label, list_data, square_size, 
            color = _COLOR , crop = _CROP )
    print( "\tSpliiting Data")
    (X_train,Y_train) , (X_test,Y_test) = DataHandle.train_test_split( X_data , Y_data , _RATIO )
    X_train = np.array( X_train ).astype( np.float ) / 255
    X_test = np.array( X_test ).astype( np.float ) / 255
    X_data = np.array( X_data ).astype( np.float ) / 255

    print( "\nPart Result Model\n\tautoencoder model")
    random_index = [ x for x in range( 0 , len( X_test ) , 50 ) ]
    data = []
    for index in random_index :
        data.append( X_test[ index ] )
    data = np.array( data )
    CommandHandle.plot_compare( data, autoencoder_model,
        figname = "Compare Result Autoencoder Model " + autoencoder_model.name,
        dest_type = np.float )

    Y_predict = classifier_model.predict( X_data )
    accuracy =  DataHandle.get_accuracy_classifier( Y_predict, 
                np.array( Y_data ),
                list_dictionary )

    print( f'====> Result of Model from {len(list_label)} label and {X_data.shape[0]} data')
    fig = plt.figure( "Accuracy of Classifier Model Name " + _MODEL_NAME )
    plt.plot( accuracy )
    plt.xlabel( "Name Data Set")
    plt.ylabel( "Accuracy")
    plt.title( "Grap Accuracy Each Model")
    plt.draw()

    DataHandle.result_classifier( Y_predict , np.array( Y_data ) , list_dictionary )

    plt.show()
