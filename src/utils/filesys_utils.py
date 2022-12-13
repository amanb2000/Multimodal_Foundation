""" Utility functions for things we need to do with the 
file system. 
 - Create output directory. 
 - `tee` class for simultaneously logging stdio & stderr to terminal AND to log files. 
"""

import datetime 
import sys 
import os 

class tee :
    def __init__(self, _fd1, _fd2) :
        self.fd1 = _fd1
        self.fd2 = _fd2

    def __del__(self) :
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
            self.fd2.close()

    def write(self, text) :
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self) :
        self.fd1.flush()
        self.fd2.flush()

def make_output_folder(output_folder): 
    NOW = None
    if output_folder.endswith('{now}'):
        output_folder = output_folder[:-5]
        print(output_folder)
        ct = str(datetime.datetime.now()).replace(' ', '_')
        ct = ct.split('.')[0]
        print("\n",ct)
        NOW = ct
        output_folder = os.path.join(output_folder, ct)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    return NOW, output_folder