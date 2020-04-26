# README

# REFERENCE
# ref01 : https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory

import cv2
import numpy as np
from PIL import Image
from numpy import floor, ceil
import matplotlib.image as mping # This will bring image file to represent by numpy array

def read_size( list_name , width = [] , height = []):
    for name in list_name:
        img = Image.open( name )
        width.append( img.size[0] )
        height.append( img.size[1] )
        img.close()
    return width , height 

def min_all_square_size( list_name ):
    for name in list_name :
        try:
            img = Image.open( name )
            ans = img.size[0]
            img.close()
            break
        except:
#            print( f'Faliure on start\t{name}' )
            None

    for name in list_name :
        try:
            img = Image.open( name )
            ans = img.size[0] if img.size[0] < ans else img.size[1] if img.size[1] < ans else ans
            img.close()
        except:
            print(f'Faliure on\t{name}')
    return ans

def min_all_square_size1( list_list_name ):
    img = Image.open( list_list_name[0][0] )
    ans = img.size[0]
    img.close()
    for list_name in list_list_name:
        temp = min_all_square_size( list_name )
        ans = temp if temp < ans else ans
    return ans

def crop_reshape_square( src , target_size ):
    # Create picture to square picture
    if( src.shape[ 0 ] == src.shape[ 1 ] ):
        crop = src
    elif( src.shape[ 0 ] > src.shape[ 1 ] ):
        temp = int(floor( ( src.shape[ 0 ] - src.shape[ 1 ] ) / 2 ))
        crop = src[ temp : src.shape[ 0 ] - temp , :  , : ]
    else:
        temp = int(floor( ( src.shape[ 1 ] - src.shape[ 0 ] ) / 2 ))
        crop = src[ : , temp : src.shape[ 1 ] - temp , : ]

    return cv2.resize( crop , ( target_size , target_size ) )

def read_reshape_square( name , target_size , color = True ):
    success = True
    try:
        if color :
            src = cv2.imread( name , cv2.IMREAD_COLOR )
            src = cv2.cvtColor( src , cv2.COLOR_BGR2RGB )
        else :
            src = cv2.imread( name , cv2.IMREAD_GRAYSCALE)
        answer = cv2.resize( src , ( target_size , target_size ) )
        if not color :
            new_answer = []
            for row in answer:
                temp = []
                for data in row:
                    temp.append( [ data ] )
                new_answer.append( temp )
            answer = np.array(new_answer)
    except:
        print(f'Problem on file {name}')
        success = False
        answer = None
    return success , answer

def read_crop_reshape_square( name , target_size , color = True ):
    success = True
    try:
        if color :
            src = cv2.imread( name , cv2.IMREAD_COLOR )
            src = cv2.cvtColor( src , cv2.COLOR_BGR2RGB )
        else :
            src = cv2.imread( name , cv2.IMREAD_GRAYSCALE)
        # Create picture to square picture
        if( src.shape[ 0 ] == src.shape[ 1 ] ):
            crop = src
        elif( src.shape[ 0 ] > src.shape[ 1 ] ):
            temp = int(floor( ( src.shape[ 0 ] - src.shape[ 1 ] ) / 2 ))
            if color :
                crop = src[ temp : src.shape[ 0 ] - temp , :  , : ]
            else:
                crop = src[ temp : src.shape[ 0 ] - temp , : ]
        else:
            temp = int(floor( ( src.shape[ 1 ] - src.shape[ 0 ] ) / 2 ))
            if color :
                crop = src[ : , temp : src.shape[ 1 ] - temp , : ]
            else:
                crop = src[ : , temp : src.shape[ 1 ] - temp ]
        answer = cv2.resize( crop , ( target_size , target_size ) )
        if not color :
            new_answer = []
            for row in answer:
                temp = []
                for data in row:
                    temp.append( [ data ] )
                new_answer.append( temp )
            answer = np.array(new_answer)
    except:
        print(f'Problem on file {name}')
        success = False
        answer = None
    return success , answer

def prepare_label_data( list_label , list_data , square_size , color = True , crop = True ):
    # This step will get list_label and list_data to data for input and data output
    # Order od output will same index of list_label
    length_output = len( list_label )
    X = []
    Y = []
    for run in range( 0 , length_output ):
        result = np.zeros( length_output , dtype=np.float64 )
        result[ run ] = 1.0
        for name in list_data[ run ]:
            ok , data = read_crop_reshape_square( name , square_size , color ) if crop else read_reshape_square( name , square_size , color )
            if ok:
                X.append( data )
                Y.append( result )
    return X , Y

def read_all_data( list_name , squater_size , color = True , crop = True ):
    X = []
    for name in list_name:
        ok , data = read_crop_reshape_square( name , squater_size , color ) if crop else read_reshape_square( name , squater_size , color )
        if ok:
            X.append( data )
    return X
