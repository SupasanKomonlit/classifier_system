# README
# This file will use to access all file in directory and manage about file name for get path

# REFERENCE
#   ref01 : https://docs.python.org/3/tutorial/inputoutput.html

from library.directory_handle import DirectoryHandle
import library.image_handle as ImageHandle
import library.command_handle as CommandHandle
import matplotlib.pyplot as plt

_PATH_DATA = "/home/zeabus/Documents/supasan/2019_deep_learning/PokemonData"
_SHOW_SIZE = True

if __name__ == "__main__":

# First get all file name in patter full path
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

    plt.show()
