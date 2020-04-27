# README

# REFERENCE

# X is list_of_data
# Y is list_of_result
# ratio is propotional of train and test data

import matplotlib.pyplot as plt
from numpy import argmax, zeros, floor, ceil

def train_test_split( X , Y , ratio ):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    count = 1
    for run in range( 0 , len( X ) ):
        if( count == 0 ):
            X_test.append( X[ run ] )
            Y_test.append( Y[ run ] ) 
        else:
            X_train.append( X[ run ] )
            Y_train.append( Y[ run ] ) 
        count = ( count + 1 )%ratio
    return ( X_train , Y_train ) , (X_test , Y_test )

def split_data( datas , ratio ):
    data1 = []
    data2 = []
    count = 1
    for data in datas :
        if( count == 0 ):
            data2.append( data )
        else:
            data1.append( data )
        count = (count+1)%ratio
    return data1 , data2

def plot( data , model , figsize = None , dest_type = int ):
    result = model.predict( data ).astype( dest_type )
    
    n_to_show = result.shape[0]
    n_column = 5
    n_offset = 5
    n_row = int( (floor( n_to_show / n_column ) + 1 ) * 2 ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = result[run].shape[0] if width < result[run].shape[0] else width
        height = result[run].shape[1] if height < result[run].shape[1] else height

    fig = plt.figure( figsize = figsize if figsize != None else ( width , height ) )
    fig.subplots_adjust( hspace=0.1 , wspace=0.1 )
                               
    for i in range(n_to_show): 
        img = result[i].squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)        
                               
def plot_compare( data , model , figsize = None , dest_type = int , picture = True ):
    n_to_show = len( data[0] ) 
    result = model.predict( data ).astype( dest_type )
    
    n_to_show = data.shape[0]
    n_column = 5
    n_offset = 10
    n_row = int( (floor( n_to_show / n_column ) + 1 ) * 2 ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = data[run].shape[0] if width < data[run].shape[0] else width
        height = data[run].shape[1] if height < data[run].shape[1] else height

    fig = plt.figure( figsize = figsize if figsize != None else ( width , height ) )
    fig.subplots_adjust( hspace=0.1 , wspace=0.1 )
                               
    for i in range(n_to_show): 
        img = data[i].squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)        
                               
    for i in range(n_to_show): 
        img = result[i].squeeze()
        sub = fig.add_subplot( n_row , n_column,
                int( floor( i / n_column)*n_offset ) + ( n_column + ( i % n_column ) + 1 ) )
        sub.axis('off')        
        sub.imshow(img) 

def result_classifier( predict , actual , dictionary ):
    print(f'Report classifier system {predict.shape[0]} datas')
    recall = zeros( len( dictionary ) ).astype( np.float ) # Count data have revel
    precision = zeros( len(dictionary ) ).astype( np.float ) # Count data have to predict
    correct = zeros( len(dictionary ) ).astype( np.float ) # Count collect data
    for run in range( 0 , predict.shape[0] ):
        index_predict = argmax( predict[run] )
        index_actual = argmax( actual[run] )
        if index_predict == index_actual :
            correct[ index_predict ] += 1.
        recall[ index_actual ] += 1.
        precision[ index_predict ] += 1.
    print(f'Summary correct {sum(correct)} datas from {sum(precision)} datas')
    print(f'{"Name":25}|{"":6}PRECISION |{"":9}RECALL')
    print('-------------------------------------------------------------------------------')
    for run in range( 0 , len(dictionary) ):
        if( np.equal( recall[ run ] , 0 ) ): continue
        precision[ run ] = correct[run] / precision[ run ] 
        recall[ run ] = correct[ run ] / recall[ run ]
        print(f'{dictionary[run]:25}|{precision[run]:15.5f} |{recall[run]:15.5f}')

def get_accuracy_classifier( predict , actual , dictionary ):
    correct = zeros( len( dictionary ) )
    recall = zeros( len( dictionary ) )
    for run in range( 0 , predict.shape[0] ):
        index_predict = argmax( predict[ run ] )
        index_actual = argmax( actual[ run ] )
        if index_predict == index_actual:
            correct[ index_predict ] += 1
        recall[ index_actual ] += 1
    return correct / recall
