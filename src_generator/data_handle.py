# README

# REFERENCE

# X is list_of_data
# Y is list_of_result
# ratio is propotional of train and test data

import matplotlib.pyplot as plt

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

def plot_compare( data , model , figsize = ( 25 , 10 ) ):
    n_to_show = len( data[0] )
    result = model.predict( data ).astype( int )

    n_to_show = data.shape[0]

    fig = plt.figure( figsize = figsize )
    fig.subplots_adjust( hspace=0.4 , wspace=0.4 )

    for i in range(n_to_show):
        img = data[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

    for i in range(n_to_show):
        img = result[i].squeeze()
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)   
