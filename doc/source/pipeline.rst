ACQ4 Data Processing Pipelines
==============================


Required Packages
-----------------

#  pyqtgraph
#  pylibrary (github.com/pbmanis/pylibrary)
#  ephysanalysis (github.com/pbmanis/ephysanalysis)
#  mapanalysistools (github.com/pbmanis/mapanalysistools)
#  mini_analysis (github.com/pbmanis/mini_analysis)
#  montager ()

General Packages
These can all be "conda install... " into the environment
----------------
matplotlib
pandas
numpy
scipy
mahotas
termcolor
seaborn
tifffile
numba
openpyxl (and it's dependencies)




Although the code might run under Python 2.7, we strongly recommend running it under a recent version of Python 3.

We also recommend setting up a python3 environment in anaconda (or however you prefer) so that there is a clean
environment to work in.

Processing pipeline
-------------------

The processing pipeline is flexible, and is meant to be created by writing short python scripts that make use of the packages and their various classes. Many of the modules have a main entry point that allows you to call from the command line to perform many of the functions, but the design is such that the code can be imported into another script

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


2. Next, run ephysanlysis/ephysanalysis/dirSummary.py directory name, flags. This routine can generate a database for the analysis of the data directory, 

dataSummary.py::

    usage: dataSummary.py [-h] [-o {terminal,pandas,excel,tabfile}] [-r] [-w]
                          [--daylist DAYLIST] [-a AFTER] [-b BEFORE] [--dry-run]
                          [--no-inspect] [-d {days,slices,cells,protocols,all}]
                          basedir

    Generate Data Summaries from acq4 datasets

    positional arguments:
      basedir               Base Directory

    optional arguments:
      -h, --help            show this help message and exit
      -o {terminal,pandas,excel,tabfile}, --output {terminal,pandas,excel,tabfile}
                            Specify output dataplan key for one entry to process
      -r, --read            just read the summary table
      -w, --write           janalyze and write the data summary
      --daylist DAYLIST     Specify daylistfile
      -a AFTER, --after AFTER
                            only analyze dates on or after a date
      -b BEFORE, --before BEFORE
                            only analyze dates on or before a date
      --dry-run             Do a dry run, reporting only directories
      --no-inspect          Do not inspect protocols, only report directories
      -d {days,slices,cells,protocols,all}, --depth {days,slices,cells,protocols,all}
                            Specify depth for --dry-run

3. Finally, run a specific script for the dataset that you are working on. vgat_ivs.py is an example of such a script 

Example:

python (path to ephysanalysis)/dir_check.py path/to/topofdirs -o dirfilename.tex

    Generates a LaTeX source file (run pdflatex dirfilename to make a pdf version of the file). This file contains information about the day/experiment/slice/cell, protocols and most of the other ancillary information in the file. It also indicates where the organizaiton is "out of compliance".

python (path to ephysanalysis)/dataSummary.py path/to/topdirofdays -o pandas -w -f filename.pkl --no-inspect

This will produce an output to a file stored as a pickled pandas dataframe. The --no-inspect flag speeds up the processing by not checking whether protocols were completed or aborted. Removing this flag slows down the processing considerably, as each data file in a protocol is checked; however this is necessary to build the "protocols_completed" column in the dataframe.

python (path to ephysanalysis)/dataSummary.py path/to/topdirofdays -o pandas -w -f filename.pkl
Same as above, but gets more detail. Recent changes to the dataSummary code allow the introspection to be faster (checking only for the presence of files, but not for actual data.)

python vgat_ivs.py path/to/topdirofdays -f pklfilename -o pdffilename [--VC --IV --maps]  - does some analysis and prints out traces in a multipage PDF file based on the pandas dataframe information, for all valide files of the selected data type (voltage-clamp, current-clamp or maps).


ToDo:
1. We need a separate table to indicate which images go with which maps. This might be able to be abstracted by looking at the position for the maps and the images and selecting the images that are closest to the map. 

2. More sophisticated VCIV analysis. 

3. lots more..... 


Using the Pipeline
==================

Note: When manipulating data file structures on disk, **always** use Acq4's DataManager.
 
Clean up the data
-----------------
The semi-automated analysis of large datasets requires (and relies) on the data being "pristine" and in the correct structure. Unfortuately, ACQ4 has significant flexibility and it is all to easy to store the data into the wrong level/folder, fail to open a slice or cell folder, etc. In addition, users sometimes store non-acq4 data in some of the directories. Such files might include bits of screenshots or exports of a window, SQL databases, etc.

Therefore, the first step is to clean up the data structures. To do this, first get a report about the data directory structure from dir_check.py. Examine the output (the pdf file produced in latex) for any red lines. These indicate directories or files that are "out of compliance". Use ACQ4's DataManager to move the files into place. Use the timestamps to rename protocols to keep the sequence in temporal order if possible. For example, a VCIV might have been incorrectly stored at the "slice" level, but additional ones were stored at the "cell level". The folders however might have the same name. In DataManager, in the "cell" directory rename the protocols such that you can insert the one from the "slice" level in sequence. However, watch the sequence. Sometimes, a protocol was run without creating a cell, and then the recording was abandoned, in which case the protocol does not belong to an exsting cell. Careful inspection of the timestamps may help to interpret this. If this is the case, it is best to create a new cell (whose number will be out of order) and to put the protocol in that cell directory. Then, make a note on the cell regarding the actions taken to restructure the directory.

Once the data structure is cleaned, the dir_check output should have no red text. Note that there might be some datasets that are still flagged because some aspect of the structure is corrupted (usually, the .index files). These datasets should be move (with the DataManager) to a separate higher-level folder outside the main folder, and the problems fixed before adding them to the primary data set.







