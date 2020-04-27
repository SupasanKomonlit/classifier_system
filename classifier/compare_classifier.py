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

_ACTIVATION_TYPE = "relu"
_PREFIX_MODEL = "classifier_"
_LIST_MODEL = (
    "autoencoder3L64D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 64 Latent Vector
    "autoencoder3L128D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 128 Latent Vector
    "autoencoder3L256D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 256 Latent Vector
    "autoencoder3L512D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 512 Latent Vector
    "autoencoder3L1024D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 1024 Latent Vector
    "cnn3L64D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 64 Latent Vector
    "cnn3L128D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 128 Latent Vector
    "cnn3L256D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 256 Latent Vector
    "cnn3L512D" + _ACTIVATION_TYPE, # Classifier on Autoencoder 3 Layer and 512 Latent Vector
    "cnn3L1024D" + _ACTIVATION_TYPE # Classifier on Autoencoder 3 Layer and 1024 Latent Vector
)

# Readme This file will can use only case model use all layers function for activation and calculate
#   Base on Keras library
if __name__ == "__main__":
    model_classifier = []
    print( "Downloading model")
    for model_name in _LIST_MODEL:
        print( f'\tDownload model : {_PREFIX_MODEL}{model_name}.h5' )
        model_classifier.append( load_model( _PREFIX_MODEL + model_name + ".h5" ) )

    print( "Survey directory of data")
    directory_handle = DirectoryHandle( _PATH_DATA )
    list_label , list_data = directory_handle.group_data()
    list_dictionary = directory_handle.group_dictionary()

    print( "\nPart Prepare Data\n\tDownloading Data" )
    X_data, Y_data = ImageHandle.prepare_label_data( list_label, list_data, square_size, 
            color = _COLOR , crop = _CROP )
    X_data = np.array( X_data ).astype( np.float ) / 255
    Y_data = np.araay( Y_data )

    accuracy_result = []
    for model in model_classifier:
        Y_predict = model.predict()
        accuracy_result.append( DataHandle.get_accuracy_classifier( Y_predict, 
                Y_data,
                list_dictionary ) )

    fig = plt.figure( "Compare Accuracy of Classifier Model" )
    for accuracy in accuracy_result :
        plt.plot( accuracy )
    plt.legend( _LIST_MODEL )
    plt.xlabel( "Name Data Set")
    plt.ylable( "Accuracy")
    plt.tile( "Grap Accuracy Each Model")
