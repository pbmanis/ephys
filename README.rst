ephys overview
==============

ephys is a collection of modules for analysis of electrophysiology data. It current consists of 3 modules:

ephysanalysis
-------------
This module provides basic data reading for acq4, matdatac, datac, and nwb files. 
It also provides "standard" analysis of IV curves, including measures of input resistance,
membrane time constant, etc, and measure of spike rate and spike shapes.

mini_analyses
-------------
This module provides functions for analyzing miniature synaptic potentials and evoked synaptic
potentials and currents. Detection with Clements-Bekkers, and Andrade-Jonas methods are provided,
along with a simple zero-crossing method.

mapanalysistools
----------------
This module provides tools for analyzing laser scanning photostimulation data from acq4.


Requirements
------------
Python 3.13, matplotlib, scipy, numpy, seaborn, pyqtgraph.
pylibrary (github.com/pbmanis/pylibrary)
and many other things... 

Installation
------------
Clone the repository and run the following command in the root directory:
`uv venv``  # build the virtual environment
`source .venv/bin/activate``  # activate the virtual environment
`uv build`` # build the package if you want it in that format
`python ephys/ephys_analysis/tests/hh_sim.py`` # build the test data .pkl files
# The tests test the mini analysis detection and spike shape analysis routines.
Then run the tests:
`python test.py`

ephys is meant to be used as an imported library, or use datatables to load data and run analyses.
