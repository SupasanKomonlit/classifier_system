# README

# REFERENCE

import matplotlib.pyplot as plt
import image_handle as ImageHandle
from numpy import floor, ceil

def plot_scatter( x , y , xlabel , ylabel , figname = None , figsize = None ):
    fig_scatter = plt.figure( figname , figsize = figsize )
    plt.scatter( x , y )
    plt.xlabel( xlabel )
    plt.ylabel( ylabel )
    plt.show( block = False )
    plt.draw()

def plot_compare( data , model , figname = None , figsize = None , dest_type = int , picture = True ):
    n_to_show = len( data[0] ) 
    result = model.predict( data ).astype( dest_type )
    
    n_to_show = data.shape[0]
    n_column = 10
    n_offset = n_column * 2 
    n_row = int( (floor( n_to_show / n_column ) + 1 ) * 2 ) 
    
    width = 0
    height = 0
    for run in range( n_column ):
        width = data[run].shape[0] if width < data[run].shape[0] else width
        height = data[run].shape[1] if height < data[run].shape[1] else height

    fig = plt.figure( figname,
            figsize = figsize if figsize != None else ( width , height ) )
    fig.subplots_adjust( hspace=0.1 , wspace=0.1 )
                               
    for i in range(n_to_show): 
#        img = data[i].squeeze()
        img = ImageHandle.normalize_image( data[i] , copy = True , dest_type = float ).squeeze()
        sub = fig.add_subplot( n_row , n_column, 
                int( floor( i / n_column)*n_offset ) + ( i % n_column ) + 1)
        sub.axis('off')        
        sub.imshow(img)        
                               
    for i in range(n_to_show): 
#        img = result[i].squeeze()
        img = ImageHandle.normalize_image( result[i] , copy = True , dest_type = float ).squeeze()
        sub = fig.add_subplot( n_row , n_column,
                int( floor( i / n_column)*n_offset ) + ( n_column + ( i % n_column ) + 1 ) )
        sub.axis('off')        
        sub.imshow(img) 
    plt.show( block = False )
