# README
# This file will use to access all file in directory and manage about file name for get path

# REFERENCE
#   ref01 : https://docs.python.org/3/tutorial/inputoutput.html

import os
from directory_handle import DirectoryHandle
import image_handle as ImageHandle

# For operating
import numpy as np
from numpy import floor, ceil

# For use function mping requirement to install package pillow
import matplotlib.image as mping # This will bring image file to represent by numpy array
import matplotlib.pyplot as plt # Plot image
import cv2

PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"

if __name__ == "__main__":

# First get all file name in patter full path
    print( "Focus on : " + PATH_DATA )
    directory_handle = DirectoryHandle( PATH_DATA )
    list_label , list_data = directory_handle.group_data()
    print( f'\tData {len(list_label)} groups and {sum([len(data) for data in list_data])} pictures') 

    print( f'Read data {list_data[0][0]}')
    img = mping.imread( list_data[0][0] )
#   img = cv2.imread( list_data[0][0] , cv2.IMREAD_UNCHANGED )

#    cv2.imshow( "original_image" , img )
    print(f'\noriginal shape : {img.shape}' )
    plt.figure()    
    imgplot = plt.imshow( img )


    success , scale_img = ImageHandle.read_crop_reshape_square( list_data[0][0] , 50 , False )
#    cv2.imshow( "resize image" , scale_img )
    print(f'scale shape : {scale_img.shape}')
#    plt.figure()    
#    imgplot = plt.imshow( scale_img )

    plt.show()
#    cv2.waitKey( 0 )
