# README
# This file will use to access all file in directory and manage about file name for get path

# REFERENCE
#   ref01 : https://docs.python.org/3.7/library/pathlib.html

import os # Import function os.path.join
from pathlib import Path

class DirectoryHandle:

    def __init__( self , path ):
        self.path = Path( path )

    # Group path will follow to manage only directory in dept equal 1
    def group_data( self , only_name = False ):
        list_name = []
        list_file = []
        # First will collect all directory in depth
        for x in self.path.iterdir():
            if( x.is_dir() ): # That mean hit directory collect them
                list_name.append( x.name )
                if only_name:
                    temp = [ temp.name for temp in Path( x ).iterdir() if not temp.is_dir() ]
                else:
                    temp = [ str(temp) for temp in Path( x ).iterdir() if not temp.is_dir() ]
                list_file.append( temp )
            else:
                None

        return list_name , list_file

    def get_all_file( self, only_name = False ):
        list_file = []
        for x in self.path.iterdir():
            if not x.is_dir() :
                list_file.append( x.name if only_name else str( x ) )
        return list_file
