# README
# This file will use to access all file in directory and manage about file name for get path

# REFERENCE
#   ref01 : https://docs.python.org/3/tutorial/inputoutput.html

from directory_handle import DirectoryHandle

import image_handle 

PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"

if __name__ == "__main__":

# First get all file name in patter full path
    print( "Focus on : " + PATH_DATA )
    directory_handle = DirectoryHandle( PATH_DATA )
    list_label , list_data = directory_handle.group_data()
    print( f'\tData {len(list_label)} groups and {sum([len(data) for data in list_data])} pictures') 

    smallest_size = image_handle.min_all_square_size1( list_data )
    print( f'\tSuggest square size small at size {smallest_size}')
