ACQ4 Data Processing Pipelines
==============================


Required Packages
-----------------

#  pyqtgraph
#  pylibrary (github.com/pbmanis/pylibrary)
#  ephysanalysis (github.com/pbmanis/ephysanalysis)
#  mapanalysistools (github.com/pbmanis/mapanalysistools)
#  mini_analysis (github.com/pbmanis/mini_analysis)


General Packages
----------------
pandas
numpy
scipy
mahotas
termcolor

Although the code will run under Python 2.7, we strongly recommend running it under a recent version of Python 3.


Processing pipeline
-------------------

The processing pipeline is flexible, and is meant to be created by writing short python scripts that make use of the packages and their various classes. 
The recommended order of processing is as follows.

1. Run ephysanlysis/ephysanalysis/dir_check.py on the main data directory. This script examines the directory structure and tests for "out-of-place" data sets, providing a color printout that includes the structure of the entire directory (thus providing an overview of the dataset). If you wish to analyze the "out-of-place" datasets, use the DataManager in acq4 to reorganize the directory structure as needed. You must use the DataManager, as there are hidden files ('.index') that will be modified to reflect the contents at each directory level. Using a simple system file manager (or Explorer) program will not update these files, and the data will be consider to be unmanaged and likely cannot be analyzed. 

dir_check.py::
    usage: dir_check.py [-h] [-r] basedir

    Generate Data Summaries from acq4 datasets

    positional arguments:
      basedir     Base Directory

    optional arguments:
      -h, --help  show this help message and exit
      -r, --read  just read the protocol
 
2. Next, run ephysanlysis/ephysanalysis/dirSummary.py directory name, flags



