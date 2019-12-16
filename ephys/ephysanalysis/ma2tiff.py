#!/usr/bin/env python3
from __future__ import print_function
"""
Translate metaarray file to tiff file (for images)
11 July 2018 PBM

Specify path in this file, and then pass the filename on the command line
The output file will be written to the same directory as the input file.

Note we set bigtiff true because some of the tiff files will be > 4 GB

"""

import sys
python_version = sys.version_info[0]
if python_version < 3:
    raise ValueError('Need python > 3 to run this')
    exit()

import fnmatch
from ephysanalysis import metaarray as MA
from pathlib import Path
import pylibrary.tifffile as tf
import numpy as np

    
def convertfiles():
# define path to data
    basepath = Path("/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het/2017.12.13_000/slice_001/cell_000") # Documents/data/MRK_Pyramidal/2017.10.04_000/slice_000"
    bpexists = basepath.is_dir()
    print(bpexists)
    matches = list(basepath.glob('**/video_*.ma'))
    
    print('files: ', matches)

    for m in matches:

        fullfile = str(m)
        data = MA.MetaArray(file=fullfile)
        ofile = Path(m.parent, m.name).with_suffix('.tif') # cleaner with pathlib... 
        print('output file: ', ofile)
        # print(data.shape)
        data = np.max(data, axis=0)
        # print('   > ', data.shape)
        fh = open(ofile, 'wb')
        tf.imsave(fh, data.view(np.ndarray).astype('float32'), imagej=True, bigtiff=False, software='acq4')
        fh.close()
        
if __name__ == '__main__':
    convertfiles()



