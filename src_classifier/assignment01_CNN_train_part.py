# README
# This file will use to access all file in directory and manage about file name for get path

# REFERENCE
# ref01 : https://docs.python.org/3/tutorial/inputoutput.html
# ref02 : https://keras.io/layers/convolutional/
# ref03 : https://keras.io/getting-started/faq/#how-can-i-install-hdf5-or-h5py-to-save-my-models-in-keras

from directory_handle import DirectoryHandle
import image_handle as ImageHandle
import data_handle as DataHandle

import matplotlib.pyplot as plt # Plot image

import numpy as np

# Import Library for CNN 
from keras.models import Sequential
## Part Core Layers
from keras.layers import Dense, Flatten, Dropout
## Part Convolutional Layers
from keras.layers import Conv2D
## Part Pooling Layers
from keras.layers import MaxPooling2D

from keras.utils import to_categorical

PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"

if __name__ == "__main__":

# First get all file name in patter full path
    print( "Focus on : " + PATH_DATA )
    directory_handle = DirectoryHandle( PATH_DATA )
    list_label , list_data = directory_handle.group_data()
    print(f'\tData {len(list_label)} groups and {sum([len(data) for data in list_data])} pictures') 

    smallest_size = ImageHandle.min_all_square_size1( list_data )
    smallest_size = smallest_size if smallest_size % 2 == 0 else smallest_size - 1
    print(f'Assignment 01 : CNN Will manage by resize image to smallest size {smallest_size} square')

    print(f'Assignment 01 : Downloading all image and resize original image to square image')
    color = ( int(input("\tLoad image in GRAYSCALE enter 0, Color other : " )) != 0 )
    X_all , Y_all = ImageHandle.prepare_label_data( list_label , list_data , smallest_size , color )
    print(f'Assignment 01 : Summary have data {len(Y_all[0])} lable and {len(X_all)} pictures in mode {"RGB" if color else "GRAYSCALE"}')

#    temp_number = np.random.randint( len(X_all) )
#    print(f'Example picture index {temp_number}')
#    plt.figure()    
#    imgplot = plt.imshow( X_all[ temp_number ] )
#    plt.show()
    ( X_train , Y_train ) , ( X_test , Y_test ) = DataHandle.train_test_split(
            X_all , Y_all , 8 )
    print(f'From {len(X_all)} datas split for train {len(X_train)} and test {len(X_test)}')

    dropout = int( input("Assignment 01 : enter 0 case don\'t use dropout, otherwise case use : "))
    dropout = (dropout != 0)

# Part Start Setup Model
    model = Sequential()
    model.name = "classifier_pokemon"
    # Add all layers
    ## Add First layer is Convolution Layer
    model.add( Conv2D( 64 ,
            (3,3),
            activation='relu',
            padding='same',
            input_shape=(smallest_size , smallest_size , 3 if color else 1),
            data_format='channels_last'))
    ## Add Second layer is Convolution Layer
    model.add( Conv2D( 32 , 
            (3,3),
            padding='same',
            activation='relu') )
    ## Add Third layer is Pooling Layer
    model.add( MaxPooling2D( (2,2),
            padding='same',
            data_format='channels_last') )
    ## Add Forth Layer is Flatten Layer 
    ### This layer will manage only output shape to one dimension
    model.add( Flatten( data_format='channels_last' ) )
    ## May Add this layer for prevent overfitting 
    if dropout : model.add( Dropout( 0.2) )
    ## Add Fifth Layer is Fully Connected layer use activation softmax
    model.add( Dense( len(list_label), activation='softmax' ) )
    # Finish add layers
    model.compile( optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    print( model.summary() )

# Part Start Train Model
    rounds = int( input("Please input round for train : ") )
    model.fit( [X_train] , [Y_train],
             validation_data=( [X_test] , [Y_test] ),
             epochs = rounds )

    name_model = input( "Please input name file model : " )
    model.save( name_model + ".h5" )
