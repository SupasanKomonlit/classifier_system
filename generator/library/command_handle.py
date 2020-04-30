# README

# REFERENCE

import matplotlib.pyplot as plt
from numpy import floor, ceil
import numpy as np

def normalize_image( src , copy = True , dest_type = float ):
    result = np.copy( src ) if copy else src
    min_value = np.min( result )
    result -= min_value
    max_value = np.max( result )
    result = result.astype( np.float ) 
    result /= max_value
    if dest_type == float:
        None
    elif dest_type == int :
        result *= 255
        result = result.astype( np.int )
    else:
        print( "Fatal can't return image" )
    return result 

def plot_scatter( x , y , xlabel , ylabel , figname = None , figsize = None ):
    fig_scatter = plt.figure( figname , figsize = figsize )
    plt.scatter( x , y )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.draw()
#    plt.show( block = False )

def plot_compare( data , model , figname = None , figsize = None , dest_type = int , picture = True ):
    n_to_show = len( data[0] ) 
    result = model.predict( data ).astype( dest_type )
    
    n_to_show = data.shape[0]
    n_column = 10
    n_offset = n_column * 2 
    n_row = int( (ceil( n_to_show / n_column ) ) * 2 ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = data[run].shape[0] if width < data[run].shape[0] else width
        height = data[run].shape[1] if height < data[run].shape[1] else height

    fig = plt.figure( figname,
            figsize = figsize if figsize != None else ( width  , height ) )
    fig.subplots_adjust( hspace=0.01 , wspace=0.01 )
                               
    for i in range(n_to_show): 
#        img = data[i].squeeze()
        img = normalize_image( data[i] , copy = True , dest_type = dest_type ).squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)        
                               
    for i in range(n_to_show): 
#        img = result[i].squeeze()
        img = normalize_image( result[i] , copy = True , dest_type = dest_type ).squeeze()
        sub = fig.add_subplot( n_row , n_column,
                int( floor( i / n_column)*n_offset ) + ( n_column + ( i % n_column ) + 1 ) )
        sub.axis('off')        
        sub.imshow(img) 
    plt.draw()
#    plt.show( block = False )

def plot_compare_model( data , model01, model02, figname = None , figsize = None , dest_type = int , picture = True ):
    n_to_show = len( data[0] ) 
    result01 = model01.predict( data ).astype( dest_type )
    result02 = model02.predict( data ).astype( dest_type )
    
    n_to_show = data.shape[0]
    n_column = 10
    n_offset = n_column * 2 
    n_row = int( (ceil( n_to_show / n_column ) ) * 2 ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = result01[run].shape[0] if width < result01[run].shape[0] else width
        height = result01[run].shape[1] if height < result02[run].shape[1] else height

    fig = plt.figure( figname,
            figsize = figsize if figsize != None else ( width  , height ) )
    fig.subplots_adjust( hspace=0.01 , wspace=0.01 )
                               
    for i in range(n_to_show): 
#        img = data[i].squeeze()
        img = normalize_image( result01[i] , copy = True , dest_type = dest_type ).squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)        
                               
    for i in range(n_to_show): 
#        img = result[i].squeeze()
        img = normalize_image( result02[i] , copy = True , dest_type = dest_type ).squeeze()
        sub = fig.add_subplot( n_row , n_column,
                int( floor( i / n_column)*n_offset ) + ( n_column + ( i % n_column ) + 1 ) )
        sub.axis('off')        
        sub.imshow(img) 
    plt.draw()
#    plt.show( block = False )

def plot( data , model , figsize = None , dest_type = int , save = None ):
    result = model.predict( data ).astype( dest_type )
    
    n_to_show = result.shape[0]
    n_column = 10
    n_offset = n_column
    n_row = int( ceil( n_to_show / n_column )  ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = result[run].shape[0] if width < result[run].shape[0] else width
        height = result[run].shape[1] if height < result[run].shape[1] else height

    fig = plt.figure( "Result of model name " + model.name, 
            figsize = figsize if figsize != None else ( width , height ) )
    fig.subplots_adjust( hspace=0.01 , wspace=0.01 )
                               
    for i in range(n_to_show): 
        img = normalize_image( result[i] , copy = True , dest_type = dest_type ).squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)
    if save != None:
        plt.savefig( save )
        plt.show( block = False )
    plt.draw() 
